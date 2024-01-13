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
from torch._tensor import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import einops

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, Transform
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE, Actor, make_mlp, Chunk, make_batch


class Detach(Transform):

    def _apply_transform(self, obs: Tensor) -> None:
        return obs.detach()


@dataclass
class PPOConfig:
    name: str = "ppo_contra"
    lr: float = 1e-3
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 32

    checkpoint_path: Union[str, None] = None


cs = ConfigStore.instance()
cs.store("ppo_contra", node=PPOConfig, group="algo")


class TConv(nn.Module):
    def __init__(self, out_dim: int, activation=nn.Mish) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.tconv = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1), activation(),
            nn.LazyConv1d(64, kernel_size=7, stride=2), activation(),
            nn.LazyConv1d(64, kernel_size=5, stride=2), activation(),
        )
        self.mlp = make_mlp([out_dim], activation=nn.Mish)
    
    def forward(self, features: torch.Tensor):
        batch_shape = features.shape[:-2]
        features = features.reshape(-1, *features.shape[-2:])
        features = einops.rearrange(self.tconv(features), "b d t -> b (t d)")
        features = self.mlp(features)
        return features.reshape(*batch_shape, *features.shape[1:])


OBS_KEY = "policy"
OBS_HIST_KEY = "policy_h"
ACTION_KEY = "action"
REWARD_KEY = ("next", "reward")
# DONE_KEY = ("next", "done")
DONE_KEY = ("next", "terminated")

class PPOTConvPolicy(TensorDictModuleBase):

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

        self.encoder = TensorDictSequential(
            TensorDictModule(
                nn.Sequential(TConv(256), nn.LazyLinear(256), Chunk(2)),
                [OBS_HIST_KEY], 
                ["_feature_hist", "_context"]
            ),
            Detach(["_context"], ["_context_detached"])
        ).to(self.device)
        # self.encoder = TensorDictSequential(
        #     TensorDictModule(
        #         nn.Sequential(TConv(128), nn.LazyLinear(128)),
        #         [OBS_HIST_KEY], ["_feature_hist"]
        #     ),
        #     TensorDictModule(
        #         nn.Sequential(TConv(128), nn.LazyLinear(128)),
        #         [OBS_HIST_KEY], ["_context"]
        #     ),
        #     Detach(["_context"], ["_context_detached"])
        # ).to(self.device)
        
        actor = TensorDictSequential(
            TensorDictModule(make_mlp([256, 256]), [OBS_KEY], ["_feature"]),
            CatTensors(["_feature_hist", "_feature", "_context_detached"], "_feature", del_keys=False),
            TensorDictModule(
                nn.Sequential(make_mlp([256]), Actor(self.action_dim)), 
                ["_feature"], ["loc", "scale"]
            )
        )
        
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictSequential(
            TensorDictModule(make_mlp([256, 256]), [OBS_KEY], ["_feature"]),
            CatTensors(["_feature_hist", "_feature", "_context_detached"], "_feature", del_keys=False),
            TensorDictModule(
                nn.Sequential(make_mlp([256]), nn.LazyLinear(1)), 
                ["_feature"], ["state_value"]
            )
        ).to(self.device)

        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        
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

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(1).to(self.device)

        self.exclude_keys = ["_feature_hist", "_feature", "_context", "_context_detached"]
    
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def __call__(self, tensordict: TensorDict):
        self.encoder(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude(*self.exclude_keys, inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            self.encoder(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = tensordict["state_value"]
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
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

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

        # contrastive learning
        context = tensordict["_context"]
        next_context = tensordict["next", "_context"].detach()
        logits = torch.einsum("ik, jk -> ij", context, next_context)
        I = torch.eye(logits.shape[0], device=logits.device)
        contra_loss = F.binary_cross_entropy_with_logits(logits, I)
        contra_acc = ((logits.detach() > 0).float() == I).float().mean()
        
        loss = policy_loss + entropy_loss + value_loss + contra_loss
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        self.encoder_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()

        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var,
            "contra_loss": contra_loss,
            "contra_acc": contra_acc,
        }, [])

