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
from torchrl.envs.transforms import CatTensors, VecNorm
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *

torch.set_float32_matmul_precision('high')

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_cmd.PPOPolicy"
    name: str = "ppo_cmd"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.002
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_cmd", node=PPOConfig, group="algo")


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
        self.max_grad_norm = 2.0
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        
        self.action_dim = action_spec.shape[-1]
        self.joint_action_dim = self.action_dim - 6
        self.command_dim = 6

        self.gae = GAE(0.99, 0.95)
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        print(fake_input)

        self.encoder = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "_actor_in", del_keys=False),
            TensorDictModule(make_mlp([256, 256, 256]), ["_actor_in"], ["_actor_feature"]),
        ).to(self.device)
        
        self.actor_joint: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictModule(Actor(self.joint_action_dim), ["_actor_feature"], ["loc", "scale"]),
            in_keys=["loc", "scale"],
            out_keys=["action_joint"],
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="log_prob_joint"
        ).to(self.device)
        
        self.actor_cmd: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictModule(Actor(6), ["_actor_feature"], ["loc", "scale"]),
            in_keys=["loc", "scale"],
            out_keys=["action_cmd"],
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="log_prob_cmd"
        ).to(self.device)

        self.output = CatTensors(["action_joint", "action_cmd"], ACTION_KEY, del_keys=False, sort=False)
        
        _critic = nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(2))
        self.critic = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "_critic_in", del_keys=False),
            TensorDictModule(_critic, ["_critic_in"], ["state_value"])
        ).to(self.device)

        self.encoder(fake_input)
        self.actor_joint(fake_input)
        self.actor_cmd(fake_input)
        self.critic(fake_input)
        self.output(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.encoder.parameters()},
                {"params": self.actor_joint.parameters()},
                {"params": self.actor_cmd.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.encoder.apply(init_)
        self.actor_joint.apply(init_)
        self.actor_cmd.apply(init_)
        self.critic.apply(init_)

        self.train_cmd = True
        self.num_updates = 0
        self.cmd_weight = 1.0
    
    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(
            self.encoder,
            self.actor_joint,
            self.actor_cmd,
            # TensorDictModule(nn.Identity(), ["arm_velocity_"], ["action_cmd"]),
            self.output,
        )
        return policy

    def step_schedule(self, progress: float):
        if progress > 0.15:
            self.train_cmd = True

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        infos = []
        self._compute_advantage(tensordict, self.critic)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                minibatch["adv_joint"] = normalize(minibatch["adv_joint"], True)
                minibatch["adv_cmd"] = normalize(minibatch["adv_cmd"], True)
                infos.append(TensorDict(self._update(minibatch), []))
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["critic/value_mean"] = tensordict["ret"][..., 0].mean().item()
        infos["critic/value_mean_cmd"] = tensordict["ret"][..., 1].mean().item()

        if self.train_cmd and infos["critic/value_mean_cmd"] < infos["critic/value_mean"] * 0.5:
            self.cmd_weight = min(self.cmd_weight + 0.1, 2.0)
        infos["cmd_weight"] = self.cmd_weight
        self.num_updates += 1
        return infos

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TensorDictModule,
    ):
        with tensordict.view(-1) as tensordict_flat:
            critic(tensordict_flat)
            critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        adv_loco, adv_arm = adv.unbind(-1)
        tensordict.set("adv_joint", (adv_loco + self.cmd_weight * adv_arm).unsqueeze(-1))
        tensordict.set("adv_cmd", adv_loco.unsqueeze(-1))
        tensordict.set("ret", ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        
        losses = {}
        
        self.encoder(tensordict)
        action_joint = tensordict["action_joint"]
        action_cmd = tensordict["action_cmd"]

        dist = self.actor_joint.get_dist(tensordict)
        log_probs = dist.log_prob(action_joint)
        entropy_joint = dist.entropy().mean()
        adv = tensordict["adv_joint"]
        ratio = torch.exp(log_probs - tensordict["log_prob_joint"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        losses["actor/policy_loss"] = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]))
        losses["actor/entropy_loss"] = - self.entropy_coef * entropy_joint

        dist = self.actor_cmd.get_dist(tensordict)
        entropy_cmd = dist.entropy().mean()
        if not self.train_cmd:
            log_probs = dist.log_prob(tensordict["next", "arm_velocity_"])
            losses["actor_cmd/policy_loss"] = - torch.mean(log_probs.unsqueeze(-1) * (~tensordict["is_init"]))
        else:
            log_probs = dist.log_prob(action_cmd)
            adv = tensordict["adv_cmd"]
            ratio = torch.exp(log_probs - tensordict["log_prob_cmd"]).unsqueeze(-1)
            surr1 = adv * ratio
            surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
            losses["actor_cmd/policy_loss"] = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]))
        losses["actor_cmd/entropy_loss"] = - self.entropy_coef * entropy_cmd

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        losses["critic/value_loss"] = (value_loss * (~tensordict["is_init"])).mean()
        
        loss = sum(losses.values())
        self.opt.zero_grad()
        loss.backward()
        encoder_grad_norm = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor_joint.parameters(), self.max_grad_norm)
        actor_cmd_grad_norm = nn.utils.clip_grad_norm_(self.actor_cmd.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()

        explained_var_0 = 1 - F.mse_loss(values[..., 0], b_returns[..., 0]) / b_returns[..., 0].var()
        explained_var_1 = 1 - F.mse_loss(values[..., 1], b_returns[..., 1]) / b_returns[..., 1].var()
        losses["actor/encoder_grad_norm"] = encoder_grad_norm
        losses["actor/grad_norm"] = actor_grad_norm
        losses["actor/entropy"] = entropy_joint
        losses["actor_cmd/grad_norm"] = actor_cmd_grad_norm
        losses["actor_cmd/entropy"] = entropy_cmd
        losses["critic/grad_norm"] = critic_grad_norm
        losses["critic/explained_var"] = explained_var_0
        losses["critic/explained_var_cmd"] = explained_var_1
        return losses

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
