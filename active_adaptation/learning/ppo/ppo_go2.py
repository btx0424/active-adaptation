# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import warnings
import functools

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import VecNorm
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase, 
    TensorDictModule as Mod,
    TensorDictSequential as Seq
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *

torch.set_float32_matmul_precision('high')

CMD_KEY = "command_"

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_go2.PPOPolicy"
    name: str = "ppo_go2"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.006
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = field(default_factory=lambda: [CMD_KEY, OBS_KEY, OBS_PRIV_KEY, "height_scan"])


cs = ConfigStore.instance()
cs.store("ppo_go2", node=PPOConfig, group="algo")


class MixedEncoder(nn.Module):
    def __init__(self, mlp_out=256, cnn_out=32):
        super().__init__()
        self.mlp_out = mlp_out
        self.cnn_out = cnn_out
        self.mlp_encoder = make_mlp([self.mlp_out])
        # self.cnn_encoder = nn.Sequential(
        #     make_conv([8, 8, 8], flatten=True),
        #     nn.LazyLinear(32),
        #     nn.LayerNorm(32),
        #     nn.Mish(),
        # )
        self.cnn_encoder = nn.Sequential(
            nn.Flatten(start_dim=-3),
            make_mlp([self.cnn_out])
        )
        self.linear = nn.LazyLinear(256)
        self.activation = nn.Sequential(nn.LayerNorm(256), nn.Mish())

        self.register_buffer("u_sum", torch.zeros(self.mlp_out + self.cnn_out))
        self.register_buffer("o_sum", torch.tensor(0.))
        self.register_buffer("cnt", torch.tensor(0.))
        self.u_sum: torch.Tensor
        self.o_sum: torch.Tensor
        self.cnt: torch.Tensor
        self.update_ema = True
    
    @property
    def u_ratios(self):
        ratios = (self.u_sum / self.u_sum.mean()).split([self.mlp_out, self.cnn_out])
        return ratios

    def forward(self, mlp_inp, cnn_inp, mask_cnn=None):
        mlp_feature = self.mlp_encoder(mlp_inp)
        cnn_feature = self.cnn_encoder(cnn_inp)
        if mask_cnn is not None:
            cnn_feature = cnn_feature * mask_cnn
        mix_feature = torch.cat([mlp_feature, cnn_feature], dim=-1)
        o = self.linear(mix_feature)
        if self.update_ema:
            assert o.ndim == 2
            w = self.linear.weight.data.detach()
            u = (mix_feature.abs().unsqueeze(1) * w.abs()).sum(1)
            self.u_sum.mul_(0.99).add_(u.sum(0))
            self.cnt.mul_(0.99).add_(u.shape[0])
        return self.activation(o)


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        self.vecnorm = VecNorm(
            in_keys=[OBS_KEY, OBS_PRIV_KEY, "height_scan"],
            shapes=[fake_input[OBS_KEY].shape[-1],
                    fake_input[OBS_PRIV_KEY].shape[-1],
                    fake_input["height_scan"].shape[-2:]]
        ).to(self.device)
        self.vecnorm(fake_input)
        
        _actor = nn.Sequential(make_mlp([128]), Actor(self.action_dim))
        self.actor_encoder = MixedEncoder()
        actor_module = Seq(
            CatTensors([CMD_KEY, OBS_KEY, OBS_PRIV_KEY], "mlp_inp", sort=False),
            Mod(self.actor_encoder, ["mlp_inp", "height_scan", "cnn_mask"], ["actor_feature"]),
            Mod(_actor, ["actor_feature"], ["loc", "scale"]),
            Mod(nn.LazyLinear(1), ["actor_feature"], ["aux_pred"])
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        _critic = nn.Sequential(make_mlp([128]), nn.LazyLinear(1))
        self.critic_encoder = MixedEncoder()
        self.critic = Seq(
            CatTensors([CMD_KEY, OBS_KEY, OBS_PRIV_KEY], "mlp_inp", sort=False),
            Mod(self.critic_encoder, ["mlp_inp", "height_scan", "cnn_mask"], ["critic_feature"]),
            Mod(_critic, ["critic_feature"], ["state_value"])
        ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        self.opt = torch.optim.AdamW(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr,
            weight_decay=0.02
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
            # if isinstance(module, nn.Conv2d):
            #     nn.init.orthogonal_(module.weight, 0.01)
            #     nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)
    
    def get_rollout_policy(self, mode: str="train"):
        if mode == "train":
            vecnorm = self.vecnorm
        else:
            vecnorm = self.vecnorm.to_observation_norm()
        policy = Seq(vecnorm, self.actor)
        return policy

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        self.actor_encoder.update_ema = False
        self.critic_encoder.update_ema = False

        tensordict = tensordict.copy()
        infos = []

        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(TensorDict(self._update(minibatch), []))
        
        with torch.no_grad(), torch.device(self.device):
            a = self.critic(tensordict.replace(cnn_mask=torch.zeros(*tensordict.shape, 1)))
            b = self.critic(tensordict.replace(cnn_mask=torch.ones(*tensordict.shape, 1)))
            value_diff = F.mse_loss(a["state_value"], b["state_value"])
            a = self.actor.get_dist(tensordict.replace(cnn_mask=torch.zeros(*tensordict.shape, 1)))
            b = self.actor.get_dist(tensordict.replace(cnn_mask=torch.ones(*tensordict.shape, 1)))
            policy_diff = F.mse_loss(a.mean, b.mean)

        infos = {k: v.mean().item() for k, v in torch.stack(infos).items()}
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        
        mlp_ratios, cnn_ratios = self.actor_encoder.u_ratios
        infos["actor/cnn_ratio"] = cnn_ratios.mean().item()
        infos["actor/mlp_ratio"] = mlp_ratios.mean().item()
        mlp_ratios, cnn_ratios = self.critic_encoder.u_ratios
        infos["critic/cnn_ratio"] = cnn_ratios.mean().item()
        infos["critic/mlp_ratio"] = mlp_ratios.mean().item()

        infos["actor/policy_diff"] = policy_diff.item()
        infos["critic/value_diff"] = value_diff.item()
        self.actor_encoder.update_ema = True
        self.critic_encoder.update_ema = True
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: Mod, 
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        self.vecnorm.freeze()
        with tensordict.view(-1) as tensordict_flat:
            critic(tensordict_flat)
            self.vecnorm(tensordict_flat["next"])
            critic(tensordict_flat["next"])
        self.vecnorm.unfreeze()

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values)
        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    # @torch.compile
    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]))
        entropy_loss = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()

        aux_loss = F.mse_loss(tensordict["aux_pred"], b_returns)
        
        loss = policy_loss + entropy_loss + value_loss + 1.0 * aux_loss
        self.opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return {
            "actor/policy_loss": policy_loss,
            "actor/entropy": entropy,
            "actor/noise_std": tensordict["scale"].mean(),
            "actor/grad_norm": actor_grad_norm,
            'actor/approx_kl': ((ratio - 1) - log_ratio).mean(),
            "actor/aux_loss": aux_loss,
            "critic/value_loss": value_loss,
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        # self.actor_encoder.cnn_encoder.apply(init_)
        # nn.init.orthogonal_(self.actor_encoder.linear.weight[:, -32:], 0.01)
        return failed_keys
    
    def learn(self, env, cfg):
        import os
        import wandb
        import logging

        from torchrl.collectors import SyncDataCollector
        from tqdm import tqdm
        from omegaconf import OmegaConf

        run = wandb.init(
            job_type=cfg.wandb.job_type,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        run.config.update(OmegaConf.to_container(cfg))

        frames_per_batch = self.cfg.train_every * env.num_envs
        total_iters = int(cfg.total_frames / frames_per_batch)
        save_interval = cfg.save_interval
        log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
        collector = SyncDataCollector(
            env,
            policy=self.get_rollout_policy("train"),
            frames_per_batch=frames_per_batch,
            total_frames=cfg.total_frames,
            device=cfg.sim.device,
            return_same_td=True,
        )

        pbar = tqdm(collector, total=total_iters)

        stats_keys = [
            k for k in env.reward_spec.keys(True, True) 
            if isinstance(k, tuple) and k[0] == "stats"
        ]
        episode_stats = EpisodeStats(stats_keys)

        ckpt_path = None
        for i, data in enumerate(pbar):
            info = {}
            episode_stats.add(data)

            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()
                info.update(env.extra)
            
            info.update(self.train_op(data))
            
            if save_interval > 0  and i % save_interval == 0:
                checkpoint_name = f"checkpoint_{i}"
                ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
                state_dict = self.state_dict()
                torch.save(state_dict, ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            
            info["env_frames"] = collector._frames
            run.log(info)
            
            print(OmegaConf.to_yaml(info))
            table_print(env.stats_ema)
            
            if ckpt_path is not None:
                print(f"Latest checkpoint path: {ckpt_path}")


from prettytable import PrettyTable
def table_print(info):
    pt = PrettyTable()
    nrow = max(len(v) for v in info.values())
    for k, v in info.items():
        data = [f"{kk}:{vv:.3f}" for kk, vv in v.items()]
        data += [" "] * (nrow - len(data))
        pt.add_column(k, data)
    print(pt)


from typing import Sequence
class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    def add(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict = tensordict["next"]
        done = next_tensordict["done"]
        if done.any():
            done = done.squeeze(-1)
            self._episodes += done.sum().item()
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats.extend(
                next_tensordict[done].clone().unbind(0)
            )
        return len(self)
    
    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats)
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
