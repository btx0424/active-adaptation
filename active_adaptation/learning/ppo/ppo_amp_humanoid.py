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
import wandb
from pathlib import Path

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs import EnvBase, CatTensors, VecNorm
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase as ModBase,
    TensorDictModule as Mod,
    TensorDictSequential
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union, List, Tuple
from collections import OrderedDict

import wandb.sdk

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *
from active_adaptation.utils.motion import MotionDataset



@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_amp_humanoid.PPOPolicy"
    name: str = "ppo_amp"

    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.001
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    data_path: Union[str, None] = "/home/btx0424/lab45/retarget/simple-retarget/data/h2"
    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY)


cs = ConfigStore.instance()
cs.store("ppo_amp_humanoid", node=PPOConfig, group="algo")


class PPOPolicy(ModBase):

    JOINT_NAMES_ISAAC: List[str] = None
    BODY_NAMES_ISAAC: List[str] = None

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        # if observation_spec.get("amp_obs", None) is None:
        #     raise ValueError
        
        self.cfg = cfg
        self.device = device

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        
        self.value_norm = ValueNormFake(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()

        actor_module = TensorDictSequential(
            CatTensors([CMD_KEY, OBS_KEY, OBS_PRIV_KEY], "_actor_obs", sort=False),
            Mod(make_mlp([512, 256, 256]), ["_actor_obs"], ["_actor_feature"]),
            Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"])
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        self.critic = TensorDictSequential(
            CatTensors([CMD_KEY, OBS_KEY, OBS_PRIV_KEY], "_critic_obs", sort=False),
            Mod(make_mlp([512, 256, 256]), ["_critic_obs"], ["_critic_feature"]),
            Mod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
        ).to(self.device)

        self.disc = nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)
        # self.disc(fake_input["amp_obs"])

        self.opt = torch.optim.Adam()
        self.opt.add_param_group({"params": self.actor.parameters(), "lr": 5e-4})
        self.opt.add_param_group({"params": self.critic.parameters(), "lr": 1e-3})
        
        self.opt_amp = torch.optim.Adam(self.disc.parameters(), lr=1e-3)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)
        # self.disc.apply(init_)
    
    def get_rollout_policy(self, mode: str="train"):
        return self.actor

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret")
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(TensorDict(self._update(minibatch), []))
        
        # for batch in make_batch(tensordict, self.cfg.num_minibatches):
        #     motion = self.motion_dataset.sample_transitions(batch.shape, 1).to(self.device)
            
        #     score_real = self.disc(motion)
        #     score_fake = self.disc(batch["amp_obs"])
        #     loss_amp = (score_real - 1.).square().mean() + (score_fake + 1.).square().mean()
        #     self.opt_amp.zero_grad()
        #     loss_amp.backward()
        #     self.opt_amp.step()
        
        infos = {k: v.mean().item() for k, v in torch.stack(infos).items()}
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        # infos["amp/disc_loss"] = loss_amp.item()
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: Mod, 
        adv_key: str="adv",
        ret_key: str="ret",
    ):
        with tensordict.view(-1) as tensordict_flat:
            critic(tensordict_flat)
            critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True).clamp_min(0.)
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values)
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
        
        loss = policy_loss + entropy_loss + value_loss
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
            "critic/value_loss": value_loss,
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }
    
    @classmethod
    def create(cls, env: EnvBase, cfg: PPOConfig):
        policy = cls(cfg, env.observation_spec, env.action_spec, env.reward_spec)
        policy.BODY_NAMES_ISAAC = env.scene["robot"].body_names
        policy.JOINT_NAMES_ISAAC = env.scene["robot"].joint_names
        return policy

    def learn(self, env: EnvBase, cfg, run: wandb.sdk.wandb_run.Run):
        import os
        import logging

        from active_adaptation.utils.torchrl import SyncDataCollector
        from active_adaptation.utils.helpers import table_print
        from tqdm import tqdm
        from omegaconf import OmegaConf

        if self.cfg.data_path is not None:
            self.motion_dataset = MotionDataset.create_from_path(self.cfg.data_path)
        
        asset = env.scene["robot"]
            
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

        episode_ema_sum = {}
        for key in env.reward_spec["stats"].keys():
            episode_ema_sum[key] = torch.tensor(0., device=env.device)
        episode_ema_cnt = torch.tensor(0., device=env.device)

        ckpt_path = None
        for i, data in enumerate(pbar):
            info = {}
            
            done = data["next", "done"]
            if done.any():
                s = data["next", "stats"][done.squeeze(-1)]
                for key in episode_ema_sum.keys():
                    episode_ema_sum[key].add_(s[key].sum())
                episode_ema_cnt.add_(done.sum())

            if i % log_interval == 0:
                for k, v in sorted(episode_ema_sum.items()):
                    key = "train/stats/" + k
                    info[key] = (v / episode_ema_cnt).item()
                    v.zero_()
                episode_ema_cnt.zero_()
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



def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
