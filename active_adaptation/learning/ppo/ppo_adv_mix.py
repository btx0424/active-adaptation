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
from torchrl.envs.transforms import CatTensors, VecNorm
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, Tuple
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *

torch.set_float32_matmul_precision('high')

@dataclass
class PPOAdvMixConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_adv_mix.PPOPolicy"
    name: str = "ppo"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.001
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    reward_groups: Tuple[str, ...] = field(default_factory=lambda: ('loco', 'manip'))
    group_action_dims: Tuple[int, ...] = field(default_factory=lambda: (12, 6))
    mixing_schedule: Tuple[float, int, int] = field(default_factory=lambda: (1.0, 500, 1000))

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_adv_mix", node=PPOAdvMixConfig, group="algo")


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOAdvMixConfig, 
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
        
        def make_cnn():
            cnn = nn.Sequential(
                nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.LazyLinear(64),
                nn.LayerNorm(64),
            )
            return cnn
        
        def make_encoder(out_key: str):
            if "height_scan" in observation_spec.keys(True, True):
                modules = [
                    TensorDictModule(make_cnn(), ["height_scan"], ["_cnn"]),
                    TensorDictModule(make_mlp([256]), [OBS_KEY], ["_mlp"]),
                    CatTensors(["_cnn", "_mlp"], out_key),
                ]
            else:
                modules = [
                    TensorDictModule(make_mlp([256]), [OBS_KEY], [out_key])
                ]
            return modules

        _actor = nn.Sequential(make_mlp([256, 128]), Actor(self.action_dim))
        actor_module = TensorDictSequential(
            *make_encoder("_actor_feature"),
            TensorDictModule(_actor, ["_actor_feature"], ["loc", "scale"])
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        self.reward_groups = cfg.reward_groups
        self.group_action_dims = tuple(cfg.group_action_dims)
        _critic = nn.Sequential(make_mlp([256, 128]), nn.Linear(128, len(self.reward_groups)))
        self.critic = TensorDictSequential(
            *make_encoder("_critic_feature"),
            TensorDictModule(_critic, ["_critic_feature"], ["state_value"])
        ).to(self.device)

        self.mixing_schedule = cfg.mixing_schedule
        self.counter = 0

        self.actor(fake_input)
        self.critic(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
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
    
    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(
            self.actor,
        )
        return policy

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        # reconstruct the sample log prob
        dist = D.Normal(tensordict['loc'], tensordict['scale'])
        sample_log_probs = dist.log_prob(tensordict['action'])
        sample_log_prob_groups = torch.split(sample_log_probs, self.group_action_dims, dim=-1)
        tensordict['sample_log_prob_grouped'] = torch.stack([group.sum(dim=-1) for group in sample_log_prob_groups], dim=-1)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(TensorDict(self._update(minibatch), []))
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
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

        rewards = tensordict[REWARD_KEY]
        # dones = tensordict["next", "done"]
        # rewards = torch.where(dones, rewards + values * self.gae.gamma, rewards)
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def get_value_mixing_ratio(self):
        return min(max((self.counter - self.mixing_schedule[1]) / self.mixing_schedule[2], 0), 1) * self.mixing_schedule[0]

    # @torch.compile
    def _update(self, tensordict: TensorDict):
        self.counter += 1
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.base_dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()
        log_prob_groups = torch.split(log_probs, self.group_action_dims, dim=-1)
        log_prob_grouped = torch.stack([group.sum(dim=-1) for group in log_prob_groups], dim=-1)
        ratio_grouped = torch.exp(log_prob_grouped - tensordict["sample_log_prob_grouped"])

        adv = tensordict["adv"]
        assert adv.shape[-1] == 2
        # do advantage mixing
        value_mixing_ratio = self.get_value_mixing_ratio()
        adv_mixed = torch.empty_like(adv)
        adv_mixed[:, 0] = adv[:, 0] + value_mixing_ratio * adv[:, 1]
        adv_mixed[:, 1] = adv[:, 1] + value_mixing_ratio * adv[:, 0]

        surr1 = adv_mixed * ratio_grouped
        surr2 = adv_mixed * ratio_grouped.clamp(1.-self.clip_param, 1.+self.clip_param)
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
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    dims = tuple(range(x.dim() - 1))
    if subtract_mean:
        return (x - x.mean(dims)) / x.std(dims).clamp(min=1e-7)
    else:
        return x / x.std(dims).clamp(min=1e-7)