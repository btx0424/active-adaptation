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

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuous
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import TensorDictPrimer, ExcludeTransform, VecNorm
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase as ModBase,
    TensorDictModule as Mod,
    TensorDictSequential
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from ..modules.temporal import GRU, TConv
from .common import *


@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_ji.PPOPolicy"
    name: str = "ppo_ji"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 4
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.002

    orthogonal_init: bool = True
    value_norm: bool = False

    se_arch: str = "rnn"
    hack: bool = False # debug option, which gives actor access to the privileged information
    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = field(default_factory=lambda: ["command_", OBS_KEY, OBS_PRIV_KEY, "ext"])

cs = ConfigStore.instance()
cs.store("ppo_ji", node=PPOConfig, group="algo")
cs.store("ppo_ji_hack", node=PPOConfig(hack=True), group="algo")


class GRUModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = make_mlp([128, 128])
        self.gru = GRU(128, hidden_size=128, allow_none=False)
        self.out = nn.LazyLinear(dim)
    
    def forward(self, x, is_init, hx):
        x = self.mlp(x)
        x, hx = self.gru(x, is_init, hx)
        x = self.out(x)
        return x, hx.contiguous()


class PPOPolicy(ModBase):
    """
    
    Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion
    https://arxiv.org/abs/2202.05481

    """
    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        vecnorm: VecNorm=None,
        device: str="cuda:0"
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.vecnorm = vecnorm

        self.entropy_coef = self.cfg.entropy_coef
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)

        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        self.num_frames = 0
        self.num_updates = 0

        self.observation_spec = observation_spec
        fake_input = observation_spec.zero()

        # the state estimator
        if self.cfg.se_arch == "rnn":
            self.state_estimator = Mod(
                GRUModule(observation_spec["priv"].shape[-1]),
                [OBS_KEY, "is_init", "estimator_hx"],
                ["priv_estimate", ("next", "estimator_hx")]
            ).to(self.device)
        else:
            self.state_estimator = Mod(
                TConv(observation_spec["priv"].shape[-1]),
                [OBS_HIST_KEY], ["priv_estimate"]
            ).to(self.device)

        if self.cfg.hack:
            actor_in_keys = ["command_", OBS_KEY, OBS_PRIV_KEY]
        else:
            actor_in_keys = ["command_", OBS_KEY, "priv_estimate"]
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                CatTensors(actor_in_keys, "policy_estimate", del_keys=False),
                Mod(make_mlp([512, 256, 256]), ["policy_estimate"], ["actor_input"]),
                Mod(Actor(self.action_dim), ["actor_input"], ["loc", "scale"]),
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        critic_module = nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(1))
        self.critic = TensorDictSequential(
            CatTensors(["command_", OBS_KEY, OBS_PRIV_KEY], "policy_priv", del_keys=False),
            Mod(critic_module, ["policy_priv"], ["state_value"])
        ).to(self.device)

        # lazy initialization
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["estimator_hx"] = torch.zeros(fake_input.shape[0], 128)

        self.state_estimator(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.state_estimator.parameters()},
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        if self.cfg.orthogonal_init:
            self.state_estimator.apply(init_)
            self.actor.apply(init_)
            self.critic.apply(init_)
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        return TensorDictPrimer(
            {"estimator_hx": UnboundedContinuous((num_envs, 128), device=self.device)},
            reset_key="done"
        )
    
    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(
            self.state_estimator,
            self.actor,
            ExcludeTransform("priv_estimate", "loc", "scale")
        )
        return policy

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret")

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every)
            for minibatch in batch:
                infos.append(TensorDict(self._update(minibatch), []))
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["critic/value_priv"] = self.value_norm.denormalize(tensordict["ret"]).mean().item()
        self.num_frames += tensordict.numel()
        self.num_updates += 1
        return infos

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: Mod, 
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
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values)
        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        adv = normalize(adv, subtract_mean=True)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        losses = {}
        self.state_estimator(tensordict)
        losses["state_est_loss"] = F.mse_loss(
            tensordict["priv_estimate"],
            tensordict[OBS_PRIV_KEY]
        )

        tensordict = tensordict.detach()
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        losses["actor/policy_loss"] = - torch.mean(torch.min(surr1, surr2))
        losses["actor/entropy_loss"] = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        losses["critic/value_loss_priv"] = (value_loss * (~tensordict["is_init"])).mean()
        
        loss = sum(losses.values())
        self.opt.zero_grad()
        loss.backward()
        losses["state_est_grad_norm"] = nn.utils.clip_grad_norm_(self.state_estimator.parameters(), 5.)
        losses["actor/grad_norm"] = nn.utils.clip_grad_norm_(self.actor.parameters(), 2.)
        losses["critic/grad_norm"] = nn.utils.clip_grad_norm_(self.critic.parameters(), 2.)
        self.opt.step()
        
        losses["critic/explained_var"] = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        losses["actor/noise_std"] = tensordict["scale"].mean()
        losses["actor/entropy"] = entropy
        return losses

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["num_frames"] = self.num_frames
        return state_dict
    
    def load_state_dict(self, state_dict, strict=False):
        self.num_frames = state_dict.pop("num_frames", 0)
        return super().load_state_dict(state_dict, strict=strict)

def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
