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
import einops
import math

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from torchrl.objectives.utils import hold_out_net

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE, Actor, make_mlp, make_batch
from .ppo_rnn import GRU

from active_adaptation.utils.wandb import parse_path

OBS_KEY = "policy" # ("agents", "observation")
OBS_PRIV_KEY = "priv"
OBS_HIST_KEY = "policy_h"
ACTION_KEY = "action" # ("agents", "action")
REWARD_KEY = ("next", "reward") # ("agents", "reward")
# DONE_KEY = ("next", "done")
DONE_KEY = ("next", "terminated")


@dataclass
class PPOConfig:
    name: str = "ppo_roa"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 1e-3

    checkpoint_path: Union[str, None] = None

    adapt_reward: float = 0.0
    regularize: bool = True


cs = ConfigStore.instance()
cs.store("ppo_roa", node=PPOConfig, group="algo")


class TConv(nn.Module):
    def __init__(self, out_dim: int, activation=nn.Mish) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.tconv = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1), activation(),
            nn.LazyConv1d(64, kernel_size=7, stride=2), activation(),
            nn.LazyConv1d(64, kernel_size=5, stride=2), activation(),
        )
        self.mlp = make_mlp([256, out_dim], activation=nn.Mish)
    
    def forward(self, features: torch.Tensor):
        batch_shape = features.shape[:-2]
        features = features.reshape(-1, *features.shape[-2:])
        features_tconv = einops.rearrange(self.tconv(features), "b d t -> b (t d)")
        features = torch.cat([features_tconv, features[:, :, -1]], dim=1)
        features = self.mlp(features)
        return features.reshape(*batch_shape, *features.shape[1:])


class PPOROAPolicy(TensorDictModuleBase):

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
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        # observation_dim = observation_spec[OBS_KEY].shape[-1]
        # observation_priv_dim = observation_spec[self.OBS_PRIV_KEY].shape[-1]

        self.make_models()
        self.actor_critic = TensorDictSequential(self.actor, self.critic)

        self.encoder(fake_input)
        self.actor_critic(fake_input)
        self.adapt(fake_input)
        self.value_norm = ValueNorm1(input_shape=1).to(self.device)
        
        checkpoint_path = parse_path(self.cfg.checkpoint_path)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            self.load_state_dict(state_dict, strict=True)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
                if isinstance(module, nn.Conv1d):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            
            self.actor_critic.apply(init_)
            self.encoder.apply(init_)
            self.adapt.apply(init_)

        self.adapt_opt = torch.optim.Adam(self.adapt.parameters())
        self.opt = torch.optim.Adam([
            {"params": self.actor.parameters()},
            {"params": self.critic.parameters()},
            {"params": self.encoder.parameters()},
        ], lr=5e-4)

        self.mode = "expert"
        self.adaptation_loss = self.feature_mse
        self.lmbda = 0. # regularization
        self.train_adapt = False
    
    def step_schedule(self, progress: float):
        self.lmbda = max(0., min(2 * progress - 0.5, 1.))
        if self.lmbda > 1e-3:
            self.train_adapt = True

    def make_models(self):
        """
        Concat conditioning.
        """
        condition = lambda: CatTensors(["_context", "_feature"], "_feature", del_keys=False)
        context_dim = 128

        self.encoder = TensorDictModule(
            make_mlp([256, context_dim], nn.Mish),
            [OBS_PRIV_KEY],
            ["_context"]
        ).to(self.device)

        self.adapt = TensorDictModule(
            TConv(context_dim, nn.Mish),
            [OBS_HIST_KEY],
            ["_context"]
        ).to(self.device)

        self.actor = ProbabilisticActor(
            TensorDictSequential(
                TensorDictModule(make_mlp([512], nn.Mish), [OBS_KEY], ["_feature"]),
                condition(),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256], nn.Mish), Actor(self.action_dim)),
                    ["_feature"], 
                    ["loc", "scale"]
                )
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictSequential(
            TensorDictModule(make_mlp([512], nn.Mish), [OBS_KEY], ["_feature"]),
            condition(),
            TensorDictModule(
               nn.Sequential(make_mlp([256, 256], nn.Mish), nn.LazyLinear(1)),
                ["_feature"], 
                ["state_value"]
            )
        ).to(self.device)

        self.classifier_latent = nn.LazyLinear(1).to(self.device)
        self.classifier_latent_opt = torch.optim.Adam(self.classifier_latent.parameters(), lr=5e-4)

    def __call__(self, tensordict: TensorDict):
        if self.mode == "expert":
            self.actor_critic(self.encoder(tensordict))
        elif self.mode == "adapt":
            self.actor_critic(self.adapt(tensordict))
        tensordict.exclude("_feature", "_obs", "loc", "scale", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):

        next_tensordict = tensordict["next"]
        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = tensordict["state_value"]

        with torch.no_grad():
            next_tensordict = self.encoder(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]

        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = {k: torch.mean(v).item() for k, v in infos.items()}
        infos["lambda"] = self.lmbda
        return infos

    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, mode: str):
        assert mode in ("expert", "adapt")
        self._mode = mode
        
    def eval(self):
        super().eval()
        self.mode = "adapt"
    
    def train(self, mode: bool=True):
        super().train(mode=mode)
        self.mode = "expert"

    def _update(self, tensordict: TensorDict):
        self.encoder(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)


        loss = policy_loss + entropy_loss + value_loss
        if self.train_adapt:
            _td = tensordict.exclude("_context")
            adapt_loss = self.feature_mse(tensordict, self.adapt(_td))
            true = self.classifier_latent(tensordict["_context"].detach())
            false = self.classifier_latent(_td["_context"].detach())
            classifier_loss = (
                F.binary_cross_entropy_with_logits(true, torch.ones_like(true))
                + F.binary_cross_entropy_with_logits(false, torch.zeros_like(false))
            )
            classifier_acc = ((true > 0).float().mean() + (false < 0).float().mean()) / 2
            loss = loss + adapt_loss + classifier_loss
            self.adapt_opt.zero_grad(set_to_none=True)
            self.classifier_latent.zero_grad(set_to_none=True)
        else:
            adapt_loss = torch.tensor(0., device=self.device)
            classifier_loss = torch.tensor(0., device=self.device)
            classifier_acc = torch.tensor(0., device=self.device)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 10)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 10)
        self.opt.step()
        if self.train_adapt:
            self.adapt_opt.step()
            self.classifier_latent_opt.step()

        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        info = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "adapt_loss": adapt_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var,
            "classifier_loss": classifier_loss,
            "classifier_acc": classifier_acc,
        }
        return TensorDict(info, [])

    def feature_mse(self, tensordict_priv: TensorDict, tensordict_adapt: TensorDict):
        pred = tensordict_adapt["_context"]
        target = tensordict_priv["_context"]
        loss = F.mse_loss(pred, target.detach())
        if self.cfg.regularize:
            loss += self.lmbda * F.mse_loss(target, pred.detach())
        else:
            loss += self.lmbda * F.mse_loss(target.detach(), pred.detach())
        return loss

