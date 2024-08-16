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
from typing import List

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, VecNorm
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field, MISSING
from typing import Union, Any
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *


@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.critics.Critics"
    
    name: str = "critics"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.002

    adv_key: str = "adv_priv"
    in_keys: List[str] = field(default_factory=lambda: [OBS_KEY, OBS_PRIV_KEY, "params"])

cs = ConfigStore.instance()
cs.store("critics", node=PPOConfig, group="algo")


class Critics(TensorDictModuleBase):

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

        self.max_grad_norm = 2.0
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)
        self.action_dim = action_spec.shape[-1]
        value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)
        
        fake_input = observation_spec.zero()
        
        actor_module = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "_actor", del_keys=False, sort=False),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 128]), Actor(self.action_dim)),
                ["_actor"], ["loc", "scale"]
            )
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic_obs = TensorDictModule(
            nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(1)),
            [OBS_KEY], ["value_obs"]
        ).to(self.device)
        self.critic_priv = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "_critic_priv", del_keys=False, sort=False),
            TensorDictModule(
                nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(1)),
                ["_critic_priv"], ["value_priv"]
            )
        ).to(self.device)
        self.critic_both = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY, "params"], "_critic_both", del_keys=False, sort=False),
            TensorDictModule(
                nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(1)),
                ["_critic_both"], ["value_both"]
            )
        ).to(self.device)

        self.actor(fake_input)
        self.critic_obs(fake_input)
        self.critic_priv(fake_input)
        self.critic_both(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic_obs.parameters()},
                {"params": self.critic_priv.parameters()},
                {"params": self.critic_both.parameters()},
            ],
            lr=cfg.lr
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic_obs.apply(init_)
        self.critic_priv.apply(init_)
        self.critic_both.apply(init_)
    
    def get_rollout_policy(self, mode: str="train"):
        return self.actor

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        infos = []
        self._compute_advantage(tensordict, self.critic_obs, "adv_obs", "ret_obs", "value_obs")
        self._compute_advantage(tensordict, self.critic_priv, "adv_priv", "ret_priv", "value_priv")
        self._compute_advantage(tensordict, self.critic_both, "adv_both", "ret_both", "value_both")

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                losses = {}

                self.critic_obs(minibatch)
                self.critic_priv(minibatch)
                self.critic_both(minibatch)

                dist = self.actor.get_dist(minibatch)
                log_probs = dist.log_prob(minibatch[ACTION_KEY])
                entropy = dist.entropy().mean()

                adv = normalize(minibatch[self.cfg.adv_key], subtract_mean=True)
                log_ratio = (log_probs - minibatch["sample_log_prob"]).unsqueeze(-1)
                ratio = torch.exp(log_ratio)
                surr1 = adv * ratio
                surr2 = adv * ratio.clamp(1 - self.cfg.clip_param, 1 + self.cfg.clip_param)
                losses["actor/policy_loss"] = -torch.mean(torch.min(surr1, surr2) * (~minibatch["is_init"]).float())
                losses["actor/entropy_loss"] = -entropy * self.cfg.entropy_coef

                for key in ("obs", "priv", "both"):
                    loss = self.critic_loss_fn(minibatch[f"value_{key}"], minibatch[f"ret_{key}"])
                    loss = (loss * (~minibatch["is_init"])).mean()
                    losses[f"critic/value_loss_{key}"] = loss

                loss = sum(losses.values())
                self.opt.zero_grad()
                loss.backward()
                losses["actor/grad_norm"] = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                for key in ("obs", "priv", "both"):
                    critic: TensorDictModule = getattr(self, f"critic_{key}")
                    losses[f"critic/grad_norm_{key}"] = nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
                self.opt.step()

                # compute explained variance
                for key in ("obs", "priv", "both"):
                    explained_var = 1 - F.mse_loss(minibatch[f"value_{key}"], minibatch[f"ret_{key}"]) / minibatch[f"ret_{key}"].var()
                    losses[f"critic/explained_var_{key}"] = explained_var
                
                losses["actor/entropy"] = entropy
                losses["actor/approx_kl"] = ((ratio - 1) - log_ratio).mean()
                infos.append(TensorDict(losses, []))
        
        infos = infos[-self.cfg.num_minibatches:]
        infos = {k: v.mean().item() for k, v in torch.stack(infos).items()}
        for key in ("obs", "priv", "both"):
            infos[f"critic/value_mean_{key}"] = tensordict[f"ret_{key}"].mean().item()
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TensorDictModule, 
        adv_key: str="adv",
        ret_key: str="ret",
        value_key: str="value"
    ):
        with tensordict.view(-1) as tensordict_flat:
            critic(tensordict_flat)
            critic(tensordict_flat["next"])

        values = tensordict[value_key]
        next_values = tensordict["next", value_key]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

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
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
