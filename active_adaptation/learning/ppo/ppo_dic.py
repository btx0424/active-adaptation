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
import einops
import copy

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from ..modules.rnn import GRU, set_recurrent_mode
from .common import *

torch.set_float32_matmul_precision('high')

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_dic.PPODICPolicy"
    name: str = "ppo_dic"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    # entropy_coef: float = 0.004
    # entropy_coef: float = 0.002
    entropy_coef_start: float = 0.001
    entropy_coef_end: float = 0.001
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    phase: str = "train"
    vecnorm: Union[str, None] = None
    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = field(default_factory=lambda: [CMD_KEY, OBS_KEY, OBS_PRIV_KEY, "ext", "ext_"])

cs = ConfigStore.instance()
cs.store("ppo_dic_train", node=PPOConfig(phase="train", vecnorm="train"), group="algo")
cs.store("ppo_dic_adapt", node=PPOConfig(phase="adapt", vecnorm="eval"), group="algo")
cs.store("ppo_dic_finetune", node=PPOConfig(phase="finetune", vecnorm="eval"), group="algo")

class GRU(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        allow_none: bool = False,
        burn_in: bool = False
    ) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.allow_none = allow_none
        self.burn_in = burn_in

    def forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        if x.ndim == 2: # single step

            N = x.shape[0]
            if hx is None and self.allow_none:
                hx = torch.zeros(N, self.gru.hidden_size, device=x.device)
            # assert (hx[is_init.squeeze()] == 0.).all()
            output = hx = self.gru(x, hx)
            output = self.ln(output)
            return output, hx

        elif x.ndim == 3: # multi-step

            N, T = x.shape[:2]
            if hx is None and self.allow_none:
                hx = torch.zeros(N, self.gru.hidden_size, device=x.device)
            else:
                hx = hx[:, 0]
            output = []
            reset = 1. - is_init.float().reshape(N, T, 1)
            for i, x_t, reset_t in zip(range(T), x.unbind(1), reset.unbind(1)):
                hx = self.gru(x_t, hx * reset_t)
                if self.burn_in and i < T // 4:
                    hx = hx.detach()
                output.append(hx)
            output = torch.stack(output, dim=1)
            output = self.ln(output)
            return output, einops.repeat(hx, "b h -> b t h", t=T)


class GRUModule(nn.Module):
    def __init__(self, dim: int, split):
        super().__init__()
        self.split = split
        self.mlp = make_mlp([128, 128])
        self.gru = GRU(128, hidden_size=128)
        self.out = nn.LazyLinear(dim)
    
    def forward(self, x, is_init, hx):
        out1 = self.mlp(x)
        out2, hx = self.gru(out1, is_init, hx)
        out3 = self.out(out2 + out1)
        out = torch.split(out3, self.split, dim=-1)
        return out + (hx.contiguous(),)


class PolicyUpdateInferenceMod:
    def __init__(self, actor: ProbabilisticActor, encoder: TensorDictModule=None) -> None:
        self.actor = actor
        self.encoder = encoder
    
    def __call__(self, tensordict: TensorDictBase):
        # TODO@botian: write to tensordict?
        if self.encoder is not None:
            self.encoder(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()
        return log_probs, entropy


class PPODICPolicy(TensorDictModuleBase):
    """
    
    version: 0.1.0, 2024.9.22 @botian
    * cleanup imitation stuff
    * add finetune phase
    * report ext_rec_error instead of ext_rec_loss
    * fix explicit force est

    """
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
        self.observation_spec = observation_spec
        assert self.cfg.phase in ["train", "adapt", "finetune"]

        self.entropy_coef = self.cfg.entropy_coef_start
        self.max_grad_norm = 1.0
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.ext_dim = observation_spec["ext_"].shape[-1]
        self.gae = GAE(0.99, 0.95)
        self.reg_lambda = 0.0
        self.ext_rec_lambda = 0.1
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        
        self.encoder_priv = TensorDictSequential(
            TensorDictModule(nn.Sequential(make_mlp([128]), nn.LazyLinear(128)), [OBS_PRIV_KEY], ["_priv_feature"]),
            TensorDictModule(nn.Sequential(make_mlp([32]), nn.LazyLinear(32)), ["ext"], ["_ext_feature"]),
        ).to(self.device)

        self.adapt_module =  TensorDictModule(
            GRUModule(128 + 32 + 6, split=[128, 32, 6]), 
            [OBS_KEY, "is_init", "adapt_hx"], 
            ["priv_pred", "ext_pred", "ext_rec", ("next", "adapt_hx")]
        ).to(self.device)
        
        in_keys = [CMD_KEY, OBS_KEY, "_priv_feature", "_ext_feature"]
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                CatTensors(in_keys, "_actor_feature", del_keys=False, sort=False),
                TensorDictModule(
                    nn.Sequential(make_mlp([512, 256, 256]), Actor(self.action_dim)),
                    ["_actor_feature"], ["loc", "scale"])
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        in_keys = [CMD_KEY, OBS_KEY, "priv_pred", "ext_pred"]
        self.actor_adapt: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                CatTensors(in_keys, "_actor_feature", del_keys=False, sort=False),
                TensorDictModule(
                    nn.Sequential(make_mlp([512, 256, 256]), Actor(self.action_dim)), 
                    ["_actor_feature"], ["loc", "scale"])
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        _critic = nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(1))
        self.critic = TensorDictSequential(
            CatTensors([CMD_KEY, OBS_KEY, OBS_PRIV_KEY, "ext"], "_critic_input", del_keys=False),
            TensorDictModule(_critic, ["_critic_input"], ["state_value"])
        ).to(self.device)

        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["adapt_hx"] = torch.zeros(fake_input.shape[0], 128)

        self.encoder_priv(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        self.adapt_module(fake_input)
        self.actor_adapt(fake_input)

        self.policy_train_inference = PolicyUpdateInferenceMod(self.actor, self.encoder_priv)
        self.policy_adapt_inference = PolicyUpdateInferenceMod(self.actor_adapt, None)
        
        self.adapt_ema = copy.deepcopy(self.adapt_module)
        self.adapt_ema.requires_grad_(False)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
                {"params": self.encoder_priv.parameters()},
            ],
            lr=cfg.lr
        )

        self.opt_adapt = torch.optim.Adam(
            [
                {"params": self.adapt_module.parameters()},
            ],
            lr=cfg.lr
        )

        self.opt_finetune = torch.optim.Adam(
            [
                {"params": self.actor_adapt.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)
        self.encoder_priv.apply(init_)
        self.adapt_module.apply(init_)
        self.num_updates = 0
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        spec = UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)
        return TensorDictPrimer({"adapt_hx": spec}, reset_key="done")

    def get_rollout_policy(self, mode: str="train"):
        modules = []
        
        if self.cfg.phase == "train":
            modules.append(self.encoder_priv)
            modules.append(self.actor)
            modules.append(self.adapt_module)
        elif self.cfg.phase == "adapt":
            modules.append(self.adapt_module)
            modules.append(self.actor_adapt)
        elif self.cfg.phase == "finetune":
            modules.append(self.adapt_ema)
            modules.append(self.actor_adapt)
        
        policy = TensorDictSequential(*modules)
        return policy
    
    def step_schedule(self, progress: float):
        self.reg_lambda = progress
        self.entropy_coef = self.cfg.entropy_coef_start + (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start) * progress

    def train_op(self, tensordict: TensorDict):
        info = {}
        if self.cfg.phase == "train":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "adapt":
            info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "finetune":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_adapt(tensordict.copy()))
        self.num_updates += 1
        return info
    
    # @torch.compile
    def train_policy(self, tensordict: TensorDict):    
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        policy_inference = self.policy_train_inference if self.cfg.phase == "train" else self.policy_adapt_inference
        opt = self.opt if self.cfg.phase == "train" else self.opt_finetune

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                info = self._update(minibatch, policy_inference, opt)
                infos.append(TensorDict(info, []))
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        return infos
    
    def train_adapt(self, tensordict: TensorDict):
        infos = []

        with torch.no_grad():
            self.encoder_priv(tensordict)

        with set_recurrent_mode(True):
            for epoch in range(2):
                for minibatch in make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every):
                    self.adapt_module(minibatch)
                    priv_loss = F.mse_loss(minibatch["priv_pred"], minibatch["_priv_feature"], reduce="none")
                    priv_loss = (priv_loss * (~minibatch["is_init"])).mean()
                    ext_loss = F.mse_loss(minibatch["ext_pred"], minibatch["_ext_feature"], reduce="none")
                    ext_loss = (ext_loss * (~minibatch["is_init"])).mean()
                    if self.ext_rec_lambda > 0:
                        ext_rec_error = F.mse_loss(minibatch["ext_rec"], minibatch["ext_"], reduce="none")
                        ext_rec_error = (ext_rec_error * (~minibatch["is_init"])).mean()
                    else:
                        ext_rec_error = 0.
                    self.opt_adapt.zero_grad()
                    (priv_loss + ext_loss + self.ext_rec_lambda * ext_rec_error).backward()
                    self.opt_adapt.step()
                    infos.append(TensorDict({
                        "adapt/priv_loss": priv_loss,
                        "adapt/ext_loss": ext_loss,
                        "adapt/ext_rec_loss": ext_rec_error,
                    }, []))
        
        soft_copy_(self.adapt_module, self.adapt_ema, 0.05)
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        return infos

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TensorDictModule, 
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        with tensordict.view(-1) as tensordict_flat:
            critic(tensordict_flat)
            critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)
        # dones = tensordict["next", "done"]
        # rewards = torch.where(dones, rewards + values * self.gae.gamma, rewards)
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
    def _update(self, tensordict: TensorDict, policy_inference: PolicyUpdateInferenceMod, opt: torch.optim.Optimizer):
        log_probs, entropy = policy_inference(tensordict)

        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]))
        entropy_loss = - self.entropy_coef * entropy

        # tensordict[CMD_KEY].requires_grad_(True)
        # tensordict[OBS_KEY].requires_grad_(True)
        # tensordict["priv"].requires_grad_(True)
        # tensordict["ext"].requires_grad_(True)

        # policy_inference.encoder(tensordict)
        # dist = policy_inference.actor.get_dist(tensordict)
        # action = dist.rsample()

        # gradient = torch.autograd.grad(
        #     outputs=action,
        #     inputs=[tensordict[CMD_KEY], tensordict[OBS_KEY], tensordict["priv"], tensordict["ext"]],
        #     grad_outputs=torch.ones_like(action),
        #     retain_graph=True,
        #     create_graph=True,
        # )
        # gradient = torch.cat(gradient, dim=-1)
        # gradient_penalty = gradient.square().sum(-1).mean()
        gradient_penalty = 0.

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()

        if self.cfg.phase == "train" and self.reg_lambda > 0:
            reg_loss = F.mse_loss(tensordict["_priv_feature"], tensordict["priv_pred"], reduce="none")
            reg_loss = self.reg_lambda * (reg_loss * (~tensordict["is_init"])).mean()
        else:
            reg_loss = 0.
            
        loss = policy_loss + entropy_loss + value_loss + reg_loss + gradient_penalty * 0.002
        
        opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        opt.step()
        
        explained_var = 1 - value_loss / b_returns[~tensordict["is_init"]].var()
        info = {
            "actor/policy_loss": policy_loss,
            "actor/entropy": entropy,
            "actor/noise_std": tensordict["scale"].mean(),
            "actor/grad_norm": actor_grad_norm,
            'actor/approx_kl': ((ratio - 1) - log_ratio).mean(),
            "actor/gradient_penalty": gradient_penalty,
            "adapt/reg_loss": reg_loss,
            "critic/value_loss": value_loss,
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }
        return info

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
        hard_copy_(self.actor, self.actor_adapt)
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
