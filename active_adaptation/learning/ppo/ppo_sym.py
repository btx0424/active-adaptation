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
import copy
from typing import List

from torchrl.data import CompositeSpec, TensorSpec, TensorDictReplayBuffer, LazyTensorStorage
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, VecNorm
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *

torch.set_float32_matmul_precision('high')

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_sym.PPOPolicy"
    name: str = "ppo_sym"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.001
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = field(default_factory=lambda: [OBS_KEY, "symmetry"])

cs = ConfigStore.instance()
cs.store("ppo_sym", node=PPOConfig, group="algo")


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
        
        _critic = nn.Sequential(make_mlp([256, 128]), nn.Linear(128, 1))
        self.critic = TensorDictSequential(
            *make_encoder("_critic_feature"),
            TensorDictModule(_critic, ["_critic_feature"], ["state_value"])
        ).to(self.device)

        self.symmetry = nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)).to(self.device)
        self.symmetry(fake_input["symmetry"])

        self.random = nn.Sequential(make_mlp([256, 256], norm=None), nn.LazyLinear(32)).to(self.device)
        self.random(fake_input["symmetry"])
        self.random.requires_grad_(False)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 1.414) # sqrt(2)
        
        self.random.apply(init_)
        self.random_target = nn.Sequential(make_mlp([256, 256], norm=None), nn.LazyLinear(32)).to(self.device)
        self.random_target(fake_input["symmetry"])

        self.actor(fake_input)
        self.critic(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )

        self.opt_symmetry = torch.optim.Adam(self.symmetry.parameters(), lr=cfg.lr)
        self.opt_random = torch.optim.Adam(self.random_target.parameters(), lr=cfg.lr)
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)
        self.symmetry.apply(init_)

        self.symmetry_ema = copy.deepcopy(self.symmetry)
        self.symmetry_ema.requires_grad_(False)
        self.symmetry_reward_coef = 0.
    
    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(
            self.actor,
        )
        return policy
    
    def step_schedule(self, progress: float):
        self.symmetry_reward_coef = min(2 * progress, 1.)

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        infos = []

        with torch.no_grad():
            left_obs, right_obs = tensordict["symmetry"].unbind(2)
            left_score = self.symmetry(left_obs)
            right_score = self.symmetry(right_obs)
            symmetry_reward = (1 - (right_score - 1).square())
            tensordict[REWARD_KEY] += self.symmetry_reward_coef * symmetry_reward

            rnd_error_left = (self.random_target(left_obs) - self.random(left_obs)).square()
            rnd_error_right = (self.random_target(right_obs) - self.random(right_obs)).square()
            # tensordict[REWARD_KEY] += rnd_reward

        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(TensorDict(self._update(minibatch), []))

        infos_symmetry = []
        for iter in range(2):
            infos_symmetry.append(TensorDict(self._update_symmetry(tensordict.reshape(-1)), []))
        
        infos = collect_info(infos)
        infos.update(collect_info(infos_symmetry))

        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["symmetry/reward"] = symmetry_reward.mean().item()
        infos["symmetry/ema_acc"] = ((left_score > 0) & (right_score < 0)).float().mean().item()
        infos["symmetry/score"] = right_score.mean().item()
        infos["symmetry/rnd_error_left"] = rnd_error_left.mean().item()
        infos["symmetry/rnd_error_right"] = rnd_error_right.mean().item()
        return dict(sorted(infos.items()))

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
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        # dones = tensordict["next", "done"]
        # rewards = torch.where(dones, rewards + values * self.gae.gamma, rewards)
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
    
    def _update_symmetry(self, tensordict: TensorDict):

        left_obs, right_obs = tensordict["symmetry"].unbind(1)
        left_obs.requires_grad_(True)
        left_score = self.symmetry(left_obs)
        right_score = self.symmetry(right_obs)
        valid = (~tensordict["is_init"]).float()
        loss_left = (left_score - torch.ones_like(left_score)).square()
        loss_right = (right_score + torch.ones_like(right_score)).square()
        symmetry_loss = torch.mean((loss_left + loss_right) * valid)

        grad = torch.autograd.grad(
            left_score,
            left_obs, 
            torch.ones_like(left_score),
            retain_graph=True,
            create_graph=True
        )[0]
        gradient_penalty = torch.mean(grad.square().sum(dim=-1))
        
        self.opt_symmetry.zero_grad()
        (symmetry_loss + 5 * gradient_penalty).backward()
        self.opt_symmetry.step()

        rnd_target = self.random(left_obs)
        rnd_loss = F.mse_loss(self.random_target(left_obs), rnd_target)

        self.opt_random.zero_grad()
        rnd_loss.backward()
        self.opt_random.step()

        return {
            "symmetry/loss": symmetry_loss,
            "symmetry/gradient_penalty": gradient_penalty,
            "symmetry/acc": ((left_score > 0) & (right_score < 0)).float().mean(),
            "symmetry/loss_rnd": rnd_loss
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
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
