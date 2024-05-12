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

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, VecNorm
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *

@dataclass
class PPOConfig:
    name: str = "ppo"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.2
    recompute_adv: bool = False
    value_norm: bool = False

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo", node=PPOConfig, group="algo")


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        vecnorm: VecNorm,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.vecnorm = vecnorm

        self.entropy_coef = 0.001
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.HuberLoss(delta=10, reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        
        actor_module=TensorDictModule(
            nn.Sequential(
                make_mlp([256, 256, 256], nn.Mish), 
                Actor(self.action_dim)
            ),
            [OBS_KEY], ["loc", "scale"]
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        def make_critic():
            return nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1))
        
        self.critic = TensorDictModule(
            make_critic(), [OBS_KEY], ["state_value"]
        ).to(self.device)

        self.critic_priv = TensorDictSequential(
            CatTensors(
                [OBS_KEY, OBS_PRIV_KEY], "policy_priv", del_keys=False
            ),
            TensorDictModule(make_critic(), ["policy_priv"], ["state_value"])
        ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)
        self.critic_priv(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
                {"params": self.critic_priv.parameters()}
            ],
            lr=cfg.lr
        )
        
        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            
            self.actor.apply(init_)
            self.critic.apply(init_)
            self.critic_priv.apply(init_)
    
    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(
            self.actor,
        ).select_out_keys(*self.actor.in_keys, ACTION_KEY, "sample_log_prob", "collector", "is_init")
        return policy

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        self._compute_advantage(tensordict, self.critic_priv, "adv_priv", "ret_priv", update_value_norm=False)
        priv_adv_larger = (tensordict["adv_priv"] > tensordict["adv"])
        priv_ret_larger = (tensordict["ret_priv"] > tensordict["ret"])
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["value_mean"] = tensordict["ret"].mean().item()
        infos["priv_adv_larger"] = priv_adv_larger.float().mean().item()
        infos["priv_ret_larget"] = priv_ret_larger.float().mean().item()
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
        values = critic(tensordict)["state_value"]
        next_values = critic(tensordict["next"])["state_value"]

        rewards = tensordict[REWARD_KEY]
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

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()

        b_returns_priv = tensordict["ret_priv"]
        values_priv = self.critic_priv(tensordict)["state_value"]
        value_loss_priv = self.critic_loss_fn(b_returns_priv, values_priv)
        value_loss_priv = (value_loss_priv * (~tensordict["is_init"])).mean()
        
        loss = policy_loss + entropy_loss + value_loss + value_loss_priv
        self.opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 10)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 10)
        critic_priv_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic_priv.parameters(), 10)
        self.opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        explained_var_priv = 1 - F.mse_loss(values_priv, b_returns_priv) / b_returns_priv.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "value_loss_priv": value_loss_priv,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "critic_priv_grad_norm": critic_priv_grad_norm,
            "explained_var": explained_var,
            "explained_var_priv": explained_var_priv
        }, [])

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        if "vecnorm" in state_dict:
            state_dict["vecnorm"]["_extra_state"]["lock"] = None # TODO: check with torchrl
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        # self.num_frames = state_dict.get("num_frames", 0)
        # self.last_phase = state_dict.get("phase", None)
        # if "vecnorm._extra_state" in state_dict:
        #     vecnorm_td = state_dict["vecnorm._extra_state"]["td"]
        #     state_dict["lock"] = None
        #     state_dict["vecnorm._extra_state"]["td"] = vecnorm_td.to(self.device)
        # return super().load_state_dict(state_dict, strict=strict)
        succeed_keys = []
        failed_keys = []
        if "vecnorm" in state_dict:
            state_dict["vecnorm"]["_extra_state"]["td"] = state_dict["vecnorm"]["_extra_state"]["td"].to(self.device)
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
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
