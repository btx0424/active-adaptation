import hydra
import torch
import torch.nn as nn
from torchrl.envs.transforms import (
    RenameTransform,
    CatTensors,
    MultiStepTransform
)
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential


class HierarchicalPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg,
        observation_spec,
        action_spec,
        reward_spec,
        device,
    ):
        super().__init__()
        
        def _policy_high(td: TensorDictBase):
            return td
        
        self.policy_high = _policy_high
        self.policy_low = hydra.utils.instantiate(cfg.policy_low)

    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(
            self.policy_low.get_rollout_policy(mode),
        )
        return policy
    
    def train_op(self, tensordict: TensorDictBase):
        return {}