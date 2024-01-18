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
from .common import Actor, GAE, make_mlp, make_batch

@dataclass
class PPOConfig:
    name: str = "ppo_static"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 1e-3

    priv_actor: bool = False
    priv_critic: bool = False

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_static", node=PPOConfig, group="algo")


class PPOStaticPolicy(TensorDictModuleBase):

    OBS_KEY = "policy"
    ACTION_KEY = "action"
    REWARD_KEY = ("next", "reward")
    # DONE_KEY = ("next", "done")
    DONE_KEY = ("next", "terminated")

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
        observation_dim = observation_spec[self.OBS_KEY].shape[-1]
        
        self.embed = nn.Embedding(4, 16).to(self.device)

        actor_module=TensorDictModule(
            nn.Sequential(
                # nn.LayerNorm(observation_dim),
                make_mlp([512, 256, 256], nn.Mish), 
                Actor(self.action_dim)
            ),
            ["full_obs"], ["loc", "scale"]
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[self.ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        self.critic = TensorDictModule(
            nn.Sequential(
                # nn.LayerNorm(observation_dim),
                make_mlp([512, 256, 256], nn.Mish), 
                nn.LazyLinear(1)
            ),
            ["full_obs"], ["state_value"]
        ).to(self.device)

        self._embed(fake_input)
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

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        self.embed_opt = torch.optim.Adam(self.embed.parameters(), lr=0.01)
        self.value_norm = ValueNorm1(input_shape=1).to(self.device)
    
    # @torch.compile
    def __call__(self, tensordict: TensorDict):
        self._embed(tensordict)
        tensordict = self.actor(tensordict)
        tensordict = self.critic(tensordict)
        tensordict = tensordict.exclude("loc", "scale", "feature")
        return tensordict

    def _embed(self, tensordict: TensorDict):
        static_embed = self.embed(tensordict["static_embed"].squeeze().long())
        tensordict.set("full_obs", torch.cat([tensordict[self.OBS_KEY], static_embed], dim=-1))

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        with torch.no_grad():
            next_tensordict = tensordict["next"]
            self._embed(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[self.REWARD_KEY]
        dones = tensordict[self.DONE_KEY]
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
        self._embed(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[self.ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * entropy

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)
        embed_reg_l1 = self.embed.weight.abs().mean() * 1e-2

        loss = policy_loss + entropy_loss + value_loss + embed_reg_l1
        self.opt.zero_grad()
        self.embed_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 10)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 10)
        self.opt.step()
        self.embed_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])

