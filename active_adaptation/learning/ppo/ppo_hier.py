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
import termcolor

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, VecNorm
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List
from collections import OrderedDict

from active_adaptation.learning.ppo.common import *
from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal


@dataclass
class Config:
    _target_: str = "active_adaptation.learning.ppo.ppo_hier.HierarchicalPolicy"
    name: str = "test"
    mode: str = "high"
    train_every: int = 32

    in_keys: Union[List, None] = field(default_factory=lambda: [OBS_KEY, CMD_KEY])

cs = ConfigStore.instance()
cs.store("ppo_hier", node=Config, group="algo")


class Interface(TensorDictModuleBase):
    
    in_keys = [CMD_KEY, "action_high"]
    out_keys = [CMD_KEY, "action_high"]
    
    def __init__(self, horizon: int = 2) -> None:
        super().__init__()
        self.horizon = horizon

    def forward(self, tensordict: TensorDictBase):
        update_cmd = tensordict["step_count"] % self.horizon == 0
        command = torch.where(
            update_cmd, 
            tensordict["action_high"], 
            tensordict["command"]
        )
        tensordict["command"] = command
        return tensordict

assert TensorDictModuleBase.is_tdmodule_compatible(Interface(1))



class HierarchicalPolicy(TensorDictModuleBase):
    
    def __init__(
        self, 
        cfg: Config, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec

        self._make_low_level_policy()
        
    def _make_low_level_policy(self):
        from .ppo_low import LowPolicy, PPOConfig
        cfg = PPOConfig()
        policy = LowPolicy(
            cfg, 
            self.observation_spec, 
            self.action_spec, 
            self.reward_spec, 
            device=self.device
        )
        self.policy_low = policy
    
    def get_rollout_policy(self, mode: str="train"):
        if self.cfg.mode == "low":
            policy = self.policy_low.get_rollout_policy(mode)
        elif self.cfg.mode == "high":
            policy = TensorDictSequential(
                TensorDictModule(lambda x: x.clone(), ["command"], ["action_high"]),
                Interface(horizon=1),
                self.policy_low.get_rollout_policy(mode)
            )
        return policy

    def train_op(self, tensordict: TensorDictBase):
        info = self.policy_low.train_op(tensordict.copy())
        return info

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
