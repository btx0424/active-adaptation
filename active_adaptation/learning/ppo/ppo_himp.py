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
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *

torch.set_float32_matmul_precision('high')

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_himp.PPOHIMPolicy"
    name: str = "ppo_himp"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.002
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    fix: bool = False
    short_history: int = 6
    num_prototypes: int = 256
    
    temperature: float = 3.0
    checkpoint_path: Union[str, None] = None

    in_keys: List[str] = field(default_factory=lambda: [
        CMD_KEY, OBS_KEY, OBS_PRIV_KEY, "aux_target_"])

cs = ConfigStore.instance()
cs.store("ppo_himp", node=PPOConfig, group="algo")


class PPOHIMPolicy(TensorDictModuleBase):

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
        if "aux_target_" in observation_spec.keys(True, True):
            # target for explicit estimation, e.g., base velocity
            self.aux_target_dim = observation_spec["aux_target_"].shape[-1]
        else:
            raise ValueError("Specify aux_target_ to use HIM.")

        self.gae = GAE(0.99, 0.95)
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        print(fake_input)
        
        actor_keys = [CMD_KEY, OBS_HIST_KEY, "aux_pred"]
        actor_module=Seq(
            CatTensors(actor_keys, "_actor_input", del_keys=False, sort=False),
            Mod(make_mlp([512, 256, 256]), ["_actor_input"], ["_actor_feature"]),
            Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"])
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        self.critic = Seq(
            CatTensors([CMD_KEY, OBS_KEY, OBS_PRIV_KEY], "_critic_input", del_keys=False),
            Mod(make_mlp([512, 256, 128]), ["_critic_input"], ["_critic_feature"]),
            Mod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
        ).to(self.device)

        # HIM estimation module
        def _make_mlp(num_units):
            return nn.Sequential(make_mlp(num_units[:-1]), nn.LazyLinear(num_units[-1]))
        
        latent_dim = 32
        self.encoder = Mod(
            _make_mlp([128, 64, self.aux_target_dim]),
            [OBS_HIST_KEY], ["aux_pred"]
        ).to(self.device)

        self._proto = nn.Embedding(self.cfg.num_prototypes, latent_dim).to(self.device)
        self._source = _make_mlp([64, latent_dim]).to(self.device)
        self._target = _make_mlp([64, latent_dim]).to(self.device)

        nn.init.orthogonal_(self._proto.weight)

        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
                # {"params": self._proto.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self._source.parameters()},
                {"params": self._target.parameters()},
            ],
            lr=cfg.lr,
        )

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

        # compile_mode = "reduce-overhead"
        # self._update = torch.compile(self._update, mode=compile_mode)
        # self._update_estimation = torch.compile(self._update_estimation, mode=compile_mode)
    
    def get_rollout_policy(self, mode: str="train"):
        policy = Seq(self.encoder, self.actor)
        return policy

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)
        del tensordict["_critic_input"]
        
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
        critic: Mod, 
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        values = critic(tensordict)["state_value"]
        next_values = critic(tensordict["next"])["state_value"]

        rewards = tensordict[REWARD_KEY]
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

    def _update(self, tensordict: TensorDict):
        self.encoder(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]))
        entropy_loss = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()
        
        estimation_loss = F.mse_loss(tensordict["aux_pred"], tensordict["aux_target_"])
        
        z_s = self._source(tensordict["_actor_feature"])
        z_t = self._target(tensordict["_critic_feature"])

        with torch.no_grad():
            w = self._proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self._proto.weight.copy_(w)

        score_s = F.normalize(z_s, dim=-1, p=2) @ self._proto.weight.T
        score_t = F.normalize(z_t, dim=-1, p=2) @ self._proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.cfg.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.cfg.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()

        loss = (
            policy_loss + entropy_loss + value_loss
            + estimation_loss + swap_loss
        )
        self.opt.zero_grad()
        loss.backward()
        self._proto.zero_grad()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()
        
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        assignment_s = q_s.argmax(-1)
        assignment_t = q_t.argmax(-1)
        
        return {
            "actor/policy_loss": policy_loss,
            "actor/entropy": entropy,
            "actor/noise_std": tensordict["scale"].mean(),
            "actor/grad_norm": actor_grad_norm,
            "critic/value_loss": value_loss,
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
            "adapt/estimation_loss": estimation_loss,
            "adapt/swap_loss": swap_loss,
            "adapt/same_assignment": (assignment_s == assignment_t).float().mean(),
            "adapt/similarity": (self._proto(assignment_s) * self._proto(assignment_t)).sum(-1).mean(),
            "adapt/entropy_s": (-log_p_s.exp() * log_p_s).sum(-1).mean(),
            "adapt/entropy_t": (-log_p_t.exp() * log_p_t).sum(-1).mean(),
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

@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T