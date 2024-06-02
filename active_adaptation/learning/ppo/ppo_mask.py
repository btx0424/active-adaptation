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

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, ExcludeTransform
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *

@dataclass
class PPOConfig:
    name: str = "ppo_mask"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.2
    recompute_adv: bool = False
    value_norm: bool = False

    context_dim: int = 128
    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_mask", node=PPOConfig, group="algo")


class PPOMaskPolicy(TensorDictModuleBase):

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
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        obs_dim = fake_input[OBS_PRIV_KEY].shape[-1]
        
        MASK_KEY = OBS_PRIV_KEY + "_mask"
        MASKED_KEY = OBS_PRIV_KEY + "_masked"
        assert MASK_KEY in fake_input.keys(True, True)

        def _mask_with_embedding():
            return TensorDictModule(
                MaskWithEmbedding(obs_dim),
                [OBS_PRIV_KEY, MASK_KEY], [MASKED_KEY]
            )
        
        self.encoder = TensorDictSequential(
            _mask_with_embedding(),
            TensorDictModule(
                nn.Sequential(make_mlp([self.cfg.context_dim]), nn.LazyLinear(cfg.context_dim)),
                [MASKED_KEY], ["context"]
            )
        ).to(self.device)

        actor_module = TensorDictSequential(
            CatTensors([OBS_KEY, "context"], "actor_input", del_keys=False),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), Actor(self.action_dim)),
                ["actor_input"], ["loc", "scale"]
            )
        )

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        def make_critic():
            return nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(1))
        
        self.critic_priv_a = TensorDictSequential(
            _mask_with_embedding(),
            CatTensors([OBS_KEY, OBS_PRIV_KEY, MASKED_KEY], "policy_priv", del_keys=False),
            TensorDictModule(make_critic(), ["policy_priv"], ["state_value"])
        ).to(self.device)

        self.critic_priv_b = TensorDictSequential(
            CatTensors([OBS_KEY, MASKED_KEY], "policy_priv", del_keys=False),
            TensorDictModule(make_critic(), ["policy_priv"], ["state_value"])
        ).to(self.device)

        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic_priv_a(fake_input)
        self.critic_priv_b(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "weight_decay": 0.02},
                {"params": self.actor.parameters()},
                {"params": self.critic_priv_a.parameters()},
                {"params": self.critic_priv_b.parameters()}
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
            
            self.encoder.apply(init_)
            self.actor.apply(init_)
            self.critic_priv_a.apply(init_)
            self.critic_priv_b.apply(init_)
    
    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(
            self.encoder,
            self.actor,
            ExcludeTransform("actor_input", "loc", "scale")
        )
        return policy

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        infos = []
        self._compute_advantage(tensordict, self.critic_priv_a, "adv_a", "ret_priv_a")
        self._compute_advantage(tensordict, self.critic_priv_b, "adv_b", "ret_priv_b")

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                minibatch["adv_a"] = normalize(minibatch["adv_a"], True)
                minibatch["adv_b"] = normalize(minibatch["adv_b"])
                infos.append(self._update(minibatch))
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["value_priv"] = tensordict["ret_priv_a"].mean().item()
        infos["value_obs"] = tensordict["ret_priv_b"].mean().item()
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

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        self.encoder(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv_a"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * entropy

        b_returns_priv = tensordict["ret_priv_a"]
        values_priv = self.critic_priv_a(tensordict)["state_value"]
        value_loss_priv = self.critic_loss_fn(b_returns_priv, values_priv)
        value_loss_priv = (value_loss_priv * (~tensordict["is_init"])).mean()

        b_returns = tensordict["ret_priv_b"]
        values = self.critic_priv_b(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()

        
        loss = policy_loss + entropy_loss + value_loss + value_loss_priv
        self.opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 10)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic_priv_a.parameters(), 10)
        critic_priv_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic_priv_b.parameters(), 10)
        self.opt.step()
        explained_var_obs = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        explained_var_priv = 1 - F.mse_loss(values_priv, b_returns_priv) / b_returns_priv.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss/obs": value_loss,
            "value_loss/priv": value_loss_priv,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "critic_priv_grad_norm": critic_priv_grad_norm,
            "value_loss/explained_var_obs": explained_var_obs,
            "value_loss/explained_var_priv": explained_var_priv
        }, [])

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
