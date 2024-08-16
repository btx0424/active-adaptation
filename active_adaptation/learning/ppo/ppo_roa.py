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

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, TensorDictPrimer, ExcludeTransform, VecNorm

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from ..modules.temporal import GRU, TConv
from .common import *

from active_adaptation.utils.wandb import parse_path

@dataclass
class LinearSchedule:
    start_value: float
    end_value: float
    start_step: float 
    end_step: float

    def compute(self, progress: float):
        """
        progress: float, in [0, 1]
        """
        progress = max(0., progress - self.start_step) / (self.end_step - self.start_step)
        return progress * (self.end_value - self.start_value) + self.start_value


@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_roa.PPOROAPolicy"
    name: str = "ppo_roa"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.002

    actor_predict_std: bool = False
    orthogonal_init: bool = True
    value_norm: bool = False

    checkpoint_path: Union[str, None] = None

    adapt_arch: str = "rnn"
    context_dim: int = 128
    regularize: bool = True
    adapt_update_interval: int = 1
    lambda_schedule: tuple = (0., 1., 0., 1.)


cs = ConfigStore.instance()
cs.store("ppo_roa", node=PPOConfig, group="algo")
cs.store("ppo_roa_noreg", node=PPOConfig(regularize=False), group="algo")


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


class PPOROAPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device: str="cuda:0"
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = self.cfg.entropy_coef
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        # self.critic_loss_fn = nn.HuberLoss(delta=10, reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.max_grad_norm = 2.0
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
        # for lazy initialization
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["context_adapt_hx"] = torch.zeros(fake_input.shape[0], 128)
        
        self.encoder_priv = TensorDictModule(
            # nn.Sequential(make_mlp([self.cfg.context_dim]), nn.LazyLinear(self.cfg.context_dim)), 
            make_mlp([self.cfg.context_dim]),
            [OBS_PRIV_KEY], ["context_priv"]
        ).to(self.device)
        
        if self.cfg.adapt_arch == "rnn":
            self.adapt_module = TensorDictModule(
                GRUModule(self.cfg.context_dim), 
                [OBS_KEY, "is_init", "context_adapt_hx"], 
                ["context_adapt", ("next", "context_adapt_hx")]
            ).to(self.device)
        else:
            self.adapt_module = TensorDictModule(
                TConv(self.cfg.context_dim),
                [OBS_HIST_KEY], ["context_adapt"]
            ).to(self.device)

        def make_actor(context_key: str) -> ProbabilisticActor:
            _actor = Actor(self.action_dim, self.cfg.actor_predict_std)
            actor = ProbabilisticActor(
                module=TensorDictSequential(
                    CatTensors([OBS_KEY, context_key], "actor_input", del_keys=False),
                    TensorDictModule(make_mlp([256, 256, 256]), ["actor_input"], ["actor_feature"]),
                    TensorDictModule(_actor, ["actor_feature"], ["loc", "scale"])
                ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True
            ).to(self.device)
            return actor
        
        self.actor_expert = make_actor("context_priv")
        self.actor_adapt = make_actor("context_adapt")
        
        critic_module = nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(1))
        self.critic = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "policy_priv", del_keys=False),
            TensorDictModule(critic_module, ["policy_priv"], ["value_priv"])
        ).to(self.device)
        
        self.encoder_priv(fake_input)
        self.adapt_module(fake_input)
        self.actor_expert(fake_input)
        self.actor_adapt(fake_input)
        self.critic(fake_input)
        
        self.actor_adapt.requires_grad_(False)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
            elif isinstance(module, nn.GRUCell):
                nn.init.orthogonal_(module.weight_hh)
            if isinstance(module, nn.Conv1d):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.opt = torch.optim.Adam([
            {"params": self.encoder_priv.parameters()},
            {"params": self.actor_expert.parameters()},
            {"params": self.critic.parameters()},
        ], lr=self.cfg.lr)
        self.opt_adapt = torch.optim.Adam(self.adapt_module.parameters(), lr=self.cfg.lr)

        if self.cfg.orthogonal_init:
            self.encoder_priv.apply(init_)
            self.actor_expert.apply(init_)
            self.critic.apply(init_)
            self.adapt_module.apply(init_)

        # self.mode = "expert"
        self.lmbda = 0. # regularization
        self.lmbda_schedule = LinearSchedule(*self.cfg.lambda_schedule)
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        return TensorDictPrimer(
            {"context_adapt_hx": UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)},
            reset_key="done"
        )
    
    def step_schedule(self, progress: float):
        self.lmbda = self.lmbda_schedule.compute(progress)

    def get_rollout_policy(self, mode: str="train"):
        if mode == "train":
            policy = TensorDictSequential(
                self.encoder_priv,
                self.actor_expert,
                self.adapt_module,
                ExcludeTransform("actor_feature", "loc", "scale")
            )
        elif mode == "eval":
            class _ActionKL(TensorDictModuleBase):
                in_keys = ["context_priv", "context_adapt"]
                out_keys = ["action_kl"]
                def forward(_, tensordict: TensorDictBase):
                    kl = self._action_kl(tensordict, reduce=False)
                    tensordict["action_kl"] = kl
                    return tensordict
            
            policy = TensorDictSequential(
                self.encoder_priv,
                self.adapt_module,
                self.actor_adapt,
                self.critic,
                _ActionKL(),
                ExcludeTransform("actor_feature", "loc", "scale")
            )
        return policy

    def train_op(self, tensordict: TensorDict):
        infos = {}
        infos.update(self.train_expert(tensordict.copy()))
        infos.update(self.train_adaptation(tensordict.copy()))      
        self.num_frames += tensordict.numel()
        self.num_updates += 1
        infos["lambda"] = self.lmbda
        return infos
    
    def train_expert(self, tensordict: TensorDictBase):
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                minibatch["adv"] = normalize(minibatch["adv"], subtract_mean=True)
                infos.append(self._update_policy(minibatch))
        hard_copy_(self.actor_expert, self.actor_adapt)

        infos = infos[-self.cfg.num_minibatches:]
        infos = {k: v.mean().item() for k, v in torch.stack(infos).items()}
        infos["critic/value_priv"] = self.value_norm.denormalize(tensordict["ret"]).mean().item()
        infos["adapt/context_priv"] = tensordict["context_priv"].norm(dim=-1).mean().item()
        return dict(sorted(infos.items()))
    
    def train_adaptation(self, tensordict: TensorDictBase):
        infos = []
        with torch.no_grad():
            self.encoder_priv(tensordict)
        
        for epoch in range(2):
            batch = make_batch(tensordict, 8, self.cfg.train_every)
            for minibatch in batch:
                infos.append(self._update_adaptation(minibatch))
        
        infos = ({k: v.mean().item() for k, v in sorted(torch.stack(infos).items())})
        with torch.no_grad():
            infos["adapt/adapt_module_a_kl"] = self._action_kl(tensordict).item()
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
        values = critic(tensordict)["value_priv"]
        next_values = critic(tensordict["next"])["value_priv"]

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

    def _update_policy(self, tensordict: TensorDictBase):
        losses = {}
        self.encoder_priv(tensordict)
        with torch.no_grad():
            self.adapt_module(tensordict)
        priv_reg_loss = self.lmbda * F.mse_loss(tensordict["context_priv"], tensordict["context_adapt"])
        if not self.cfg.regularize:
            priv_reg_loss = priv_reg_loss.detach()
        losses["priv_reg_loss"] = priv_reg_loss

        dist = self.actor_expert.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        losses["actor/policy_loss"] = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]))
        losses["actor/entropy_loss"] = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["value_priv"]
        value_loss = self.critic_loss_fn(b_returns, values)
        losses["critic/value_loss_priv"] = (value_loss * (~tensordict["is_init"])).mean()
        
        loss = sum(losses.values())
        self.opt.zero_grad()
        loss.backward()
        losses["actor/encoder_grad_norm"] = nn.utils.clip_grad_norm_(self.encoder_priv.parameters(), self.max_grad_norm)
        losses["actor/grad_norm"] = nn.utils.clip_grad_norm_(self.actor_expert.parameters(), self.max_grad_norm)
        losses["critic/grad_norm"] = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()

        losses["critic/explained_var"] = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        losses["actor/noise_std"] = tensordict["scale"].mean()
        losses["actor/entropy"] = entropy
        losses["actor/approx_kl"] = ((ratio - 1) - log_ratio).mean()
        return TensorDict(losses, [])
    
    def _update_adaptation(self, tensordict: TensorDictBase):
        losses = {}
        losses["adapt/adaptation_loss"] = F.mse_loss(
            self.adapt_module(tensordict)["context_adapt"],
            tensordict["context_priv"]
        )
        loss = sum(losses.values())
        self.opt_adapt.zero_grad()
        loss.backward()
        losses["adapt/adapt_module_grad_norm"] = nn.utils.clip_grad_norm_(self.adapt_module.parameters(), 10)
        self.opt_adapt.step()
        return TensorDict(losses, [])
    
    def _action_kl(self, tensordict: TensorDict, reduce: bool=True):
        with torch.no_grad():
            dist1 = self.actor_expert.get_dist(tensordict)
        dist2 = self.actor_adapt.get_dist(self.adapt_module(tensordict))
        kl = D.kl_divergence(dist2, dist1).unsqueeze(-1)
        if reduce:
            kl = kl.mean()
        return kl
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["num_frames"] = self.num_frames
        return state_dict
    
    def load_state_dict(self, state_dict, strict=False):
        self.num_frames = state_dict.get("num_frames", 0)
        return super().load_state_dict(state_dict, strict=strict)