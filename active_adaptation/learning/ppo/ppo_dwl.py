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

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *
from ..modules.rnn import GRU, set_recurrent_mode


RECON_KEY = "recon_target"

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_dwl.PPODWLPolicy"
    name: str = "ppo_dwl"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.001
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    checkpoint_path: Union[str, None] = None
    dwl_latent_dim: int = 48
    dwl_weight: float = 0.2
    l1_reg: float = dwl_weight / 500

    in_keys: List[str] = field(default_factory=lambda: ["policy", "priv", RECON_KEY])

cs = ConfigStore.instance()
cs.store("ppo_dwl", node=PPOConfig, group="algo")


class DWL(nn.Module):
    def __init__(self, latent_dim: int = 24):
        super().__init__()
        self.inp = nn.LazyLinear(64)
        self.gru = GRU(64, 256)
        self.out = nn.Sequential(nn.Mish(), nn.Linear(256, latent_dim))

    def forward(self, obs: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        x = self.inp(obs)
        x, hx = self.gru(x, is_init, hx)
        return self.out(x), hx.contiguous()


class PPODWLPolicy(TensorDictModuleBase):
    """
    Denoising World Model Learning as described in the paper:
    https://roboticsconference.org/program/papers/58/

    """
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
        self.observation_spec = observation_spec
        self.action_dim = action_spec.shape[-1]
        self.hidden_dim = 256
        self.gae = GAE(0.99, 0.95)
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()

        _actor = nn.Sequential(make_mlp([64]), Actor(self.action_dim))
        self.dwl_enc = TensorDictModule(
            DWL(latent_dim=self.cfg.dwl_latent_dim), 
            [OBS_KEY, "is_init", "estimator_hx"], 
            ["_latent", ("next", "estimator_hx")]
        ).to(self.device)
        
        self.recon_dim = self.observation_spec[RECON_KEY].shape[-1]
        self.dwl_dec = TensorDictModule(
            nn.Sequential(make_mlp([64]), nn.LazyLinear(self.recon_dim)),
            ["_latent"],
            ["recon"]
        ).to(self.device)
        
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictModule(_actor, ["_latent"], ["loc", "scale"]),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        _critic = nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(1))
        self.critic = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "_critic_feature", del_keys=False),
            TensorDictModule(_critic, ["_critic_feature"], ["state_value"])
        ).to(self.device)

        # lazy initialization
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["estimator_hx"] = torch.zeros(fake_input.shape[0], self.hidden_dim)

        self.dwl_enc(fake_input)
        self.dwl_dec(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        self.opt = torch.optim.Adam([
            {"params": self.actor.parameters()},
            {"params": self.critic.parameters()},
            {"params": self.dwl_enc.parameters()},
            {"params": self.dwl_dec.parameters()},
        ], lr=cfg.lr)
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)
        self.dwl_enc.apply(init_)
        self.dwl_dec.apply(init_)
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        spec = UnboundedContinuousTensorSpec((num_envs, self.hidden_dim), device=self.device)
        return TensorDictPrimer({"estimator_hx": spec}, reset_key="done")
    
    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(
            self.dwl_enc,
            self.actor,
        )
        return policy

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        with set_recurrent_mode(True):
            for epoch in range(self.cfg.ppo_epochs):
                batch = make_batch(tensordict, self.cfg.num_minibatches, tensordict.shape[1])
                for minibatch in batch:
                    infos.append(TensorDict(self._update(minibatch), []))

        infos = {k: v.mean().item() for k, v in torch.stack(infos).items()}
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
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
        # dones = tensordict["next", "done"]
        # rewards = torch.where(dones, rewards + values * self.gae.gamma, rewards)
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

    # @torch.compile
    def _update(self, tensordict: TensorDict):
        self.dwl_enc(tensordict)
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

        recon = self.dwl_dec(tensordict)["recon"]
        recon_target = tensordict[RECON_KEY]
        dwl_loss = self.cfg.dwl_weight * F.mse_loss(recon, recon_target)
        # l1-norm regularization
        dwl_reg = self.cfg.l1_reg * torch.mean(tensordict["_latent"].abs().sum(-1))

        loss = policy_loss + entropy_loss + value_loss + dwl_loss + dwl_reg
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
            "critic/value_loss": value_loss,
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
            "adapt/dwl_loss": dwl_loss,
            "adapt/reg_loss": dwl_reg
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
    dim = tuple(range(x.ndim))
    if subtract_mean:
        return (x - x.mean(dim)) / x.std(dim).clamp(1e-7)
    else:
        return x  / x.std(dim).clamp(1e-7)
