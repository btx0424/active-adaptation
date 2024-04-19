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

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import *

@dataclass
class PPOConfig:
    name: str = "ppo_asy"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.2
    recompute_adv: bool = False

    adv_key: str = "adv"
    marginal_loss: bool = True
    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_asy", node=PPOConfig, group="algo")


class PPOAsyPolicy(TensorDictModuleBase):

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

        self.entropy_coef = 0.001
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.HuberLoss(delta=10, reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        self.value_norm = ValueNorm1(input_shape=1).to(self.device)
        self.value_norm_priv = ValueNorm1(input_shape=1).to(self.device)

        self.observation_spec = observation_spec
        fake_input = observation_spec.zero()
        
        self.encoder_priv = TensorDictModule(
            # make_mlp([128]), 
            nn.LazyLinear(128),
            [OBS_PRIV_KEY], ["context_expert"]
        ).to(self.device)

        def make_actor(context_key: str) -> ProbabilisticActor:
            actor_module = nn.Sequential(make_mlp([512, 256, 256]), Actor(self.action_dim, True))
            actor = ProbabilisticActor(
                module=TensorDictSequential(
                    CatTensors([OBS_KEY, context_key], "actor_feature", del_keys=False),
                    TensorDictModule(actor_module, ["actor_feature"], ["loc", "scale"])
                ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True
            ).to(self.device)
            return actor
        
        self.actor_expert = make_actor("context_expert")
        
        def make_critic(output_dim: int):
            return nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(output_dim))
        
        self.critic = TensorDictModule(
            make_critic(1), [OBS_KEY], ["state_value"]
        ).to(self.device)

        self.critic_priv = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "policy_priv", del_keys=False),
            TensorDictModule(
                nn.Sequential(make_critic(2), Chunk(2)), 
                ["policy_priv"], ["state_value", "_state_value"]
            )
        ).to(self.device)

        self.critic_marg = TensorDictSequential(
            TensorDictModule(make_critic(1), ["policy_priv"], ["value_marg_error"])
        ).to(self.device)

        self.encoder_priv(fake_input)
        self.actor_expert(fake_input)
        self.critic(fake_input)
        self.critic_priv(fake_input)
        self.critic_marg(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.encoder_priv.parameters()},
                {"params": self.actor_expert.parameters()},
                {"params": self.critic.parameters()},
                {"params": self.critic_priv.parameters()},
                {"params": self.critic_marg.parameters()}
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
            
            self.encoder_priv.apply(init_)
            self.actor_expert.apply(init_)
            self.critic.apply(init_)
            self.critic_priv.apply(init_)
            self.critic_marg.apply(init_)
    
    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(
            self.encoder_priv,
            self.actor_expert,
        )
        return policy

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        infos = []
        self._compute_advantage(
            tensordict, 
            self.critic,
            self.value_norm,
            "adv", 
            "ret",
            update_value_norm=True
        )
        values = tensordict["state_value"]
        self._compute_advantage(
            tensordict, 
            self.critic_priv,
            self.value_norm_priv,
            "adv_priv", 
            "ret_priv", 
            update_value_norm=True
        )
        if self.cfg.marginal_loss:
            tensordict["marg_error"] = (tensordict["_state_value"] -  values).square()
            self._compute_advantage(
                tensordict,
                self.critic_marg,
                None,
                "adv_marg",
                "ret_marg",
                "marg_error",
                "value_marg_error",
                update_value_norm=True
            )
            tensordict["adv_mixed"] = normalize(tensordict["adv_priv"] + tensordict["adv_marg"], subtract_mean=True)
        priv_adv_larger = (tensordict["adv_priv"] > tensordict["adv"])
        priv_ret_larger = (tensordict["ret_priv"] > tensordict["ret"])
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)
        tensordict["adv_priv"] = normalize(tensordict["adv_priv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["value_mean"] = self.value_norm.denormalize(tensordict["ret"]).mean().item()
        infos["value_priv_mean"] = self.value_norm_priv.denormalize(tensordict["ret_priv"]).mean().item()
        infos["asy/priv_adv_larger"] = priv_adv_larger.float().mean().item()
        infos["asy/priv_ret_larget"] = priv_ret_larger.float().mean().item()
        if self.cfg.marginal_loss:
            infos["value_marg_mean"] = tensordict["ret_marg"].mean().item()
        return infos

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TensorDictModule, 
        value_norm: ValueNorm1,
        adv_key: str="adv",
        ret_key: str="ret",
        rew_key: str=REWARD_KEY,
        value_key: str="state_value",
        update_value_norm: bool=True,
    ):
        values = critic(tensordict)[value_key]
        next_values = critic(tensordict["next"])[value_key]

        rewards = tensordict[rew_key]
        dones = tensordict[DONE_KEY]
        if value_norm is not None:
            values = value_norm.denormalize(values)
            next_values = value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        if value_norm is not None:
            if update_value_norm:
                value_norm.update(ret)
            ret = value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        self.encoder_priv(tensordict)
        dist = self.actor_expert.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        losses = {}

        adv = tensordict[self.cfg.adv_key]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        losses["policy_loss"] = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        losses["entropy_loss"] = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        losses["value_loss"] = (self.critic_loss_fn(b_returns, values) * (~tensordict["is_init"])).mean()

        b_returns_priv = tensordict["ret_priv"]
        values_priv = self.critic_priv(tensordict)["state_value"]
        losses["value_loss_priv"] = (self.critic_loss_fn(b_returns_priv, values_priv) * (~tensordict["is_init"])).mean()
        
        if self.cfg.marginal_loss:
            marginal_loss = self.critic_loss_fn(tensordict["_state_value"], b_returns)
            losses["asy/marginal_loss"] = (marginal_loss * (~tensordict["is_init"])).mean()
            
            values_marg = self.critic_marg(tensordict)["value_marg_error"]
            losses["asy/value_loss_marg"] = (
                self.critic_loss_fn(values_marg, tensordict["ret_marg"])
                * (~tensordict["is_init"])
            ).mean()

        self.opt.zero_grad()
        sum(v for k, v in losses.items()).backward()
        losses["actor_grad_norm"] = nn.utils.clip_grad.clip_grad_norm_(self.actor_expert.parameters(), 10)
        losses["critic_grad_norm"] = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 10)
        losses["critic_priv_grad_norm"] = nn.utils.clip_grad.clip_grad_norm_(self.critic_priv.parameters(), 10)
        self.opt.step()
        losses["explained_var"] = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        losses["explained_var_priv"] = 1 - F.mse_loss(values_priv, b_returns_priv) / b_returns_priv.var()
        losses["entropy"] = entropy
        return TensorDict(losses, [])


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
