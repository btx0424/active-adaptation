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
from torchrl.envs.transforms import CatTensors, TensorDictPrimer, ExcludeTransform
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
    name: str = "ppo_ji"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.2
    recompute_adv: bool = False

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_ji", node=PPOConfig, group="algo")

class GRU(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        allow_none: bool = False,
        burn_in: bool = False
    ) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.allow_none = allow_none
        self.burn_in = burn_in

    def forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        if x.ndim == 2: # single step

            N = x.shape[0]
            if hx is None and self.allow_none:
                hx = torch.zeros(N, self.gru.hidden_size, device=x.device)
            assert (hx[is_init.squeeze()] == 0.).all()
            output = hx = self.gru(x, hx)
            output = self.ln(output)
            return output, hx

        elif x.ndim == 3: # multi-step

            N, T = x.shape[:2]
            if hx is None and self.allow_none:
                hx = torch.zeros(N, self.gru.hidden_size, device=x.device)
            else:
                hx = hx[:, 0]
            output = []
            reset = 1. - is_init.float().reshape(N, T, 1)
            for i, x_t, reset_t in zip(range(T), x.unbind(1), reset.unbind(1)):
                hx = self.gru(x_t, hx * reset_t)
                if self.burn_in and i < T // 4:
                    hx = hx.detach()
                output.append(hx)
            output = torch.stack(output, dim=1)
            output = self.ln(output)
            return output, einops.repeat(hx, "b h -> b t h", t=T)


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


class PPOPolicy(TensorDictModuleBase):
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

        self.observation_spec = observation_spec
        fake_input = observation_spec.zero()
        
        # the state estimator
        self.state_estimator = TensorDictModule(
            GRUModule(observation_spec["priv"].shape[-1]),
            [OBS_KEY, "is_init", "estimator_hx"],
            ["priv_estimate", ("next", "estimator_hx")]
        ).to(self.device)

        actor_module = nn.Sequential(make_mlp([512, 256, 256], nn.Mish), Actor(self.action_dim))
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                CatTensors([OBS_KEY, "priv_estimate"], "policy_estimate", del_keys=False),
                TensorDictModule(actor_module, ["policy_estimate"], ["loc", "scale"])
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        critic_module = nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(1))
        self.critic = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "policy_priv", del_keys=False),
            TensorDictModule(critic_module, ["policy_priv"], ["state_value"])
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
        
        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            
            self.state_estimator.apply(init_)
            self.actor.apply(init_)
            self.critic.apply(init_)
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        return TensorDictPrimer(
            {"estimator_hx": UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)},
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
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["value_mean"] = self.value_norm.denormalize(tensordict["ret"]).mean().item()
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
        values = critic(tensordict)["state_value"]
        next_values = critic(tensordict["next"])["state_value"]

        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        adv = normalize(adv, subtract_mean=True)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        losses = {}
        losses["state_est_loss"] = F.mse_loss(
            self.state_estimator(tensordict)["priv_estimate"],
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
        losses["policy_loss"] = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        losses["entropy_loss"] = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        losses["value_loss"] = (value_loss * (~tensordict["is_init"])).mean()
        
        loss = sum(losses.values())
        self.opt.zero_grad()
        loss.backward()
        losses["state_est_grad_norm"] = nn.utils.clip_grad.clip_grad_norm_(self.state_estimator.parameters(), 10)
        losses["actor_grad_norm"] = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 10)
        losses["critic_grad_norm"] = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 10)
        self.opt.step()
        losses["explained_var"] = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict(losses, [])


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
