import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec, TensorDictReplayBuffer, LazyTensorStorage, ListStorage
from torchrl.envs.transforms import CatTensors, ExcludeTransform, VecNorm, MultiStepTransform
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from typing import Mapping, Union

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

from .ppo.common import *
from .modules.distributions import IndependentNormal

@dataclass
class HBCConfig:
    name: str = "hbc"
    train_every: int = 32
    lr: float = 5e-4

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("hbc", node=HBCConfig, group="algo")

class HBC(TensorDictModuleBase):
    def __init__(
        self,
        cfg: HBCConfig,
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        vecnorm: VecNorm,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.vecnorm = vecnorm
        self.observation_spec = observation_spec
        self.action_spec = action_spec

        self.low_action_dim = self.action_spec.shape[-1]
        self.high_action_dim = 12

        fake_input = observation_spec.zero()

        self.actor_high: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                CatTensors([OBS_KEY, OBS_PRIV_KEY], "actor_input", del_keys=False),
                TensorDictModule(make_mlp([256, 256, 256]), ["actor_input"], ["actor_high_feature"]),
                TensorDictModule(Actor(self.high_action_dim, True), ["actor_high_feature"], ["loc", "scale"]),
            ),
            in_keys=["loc", "scale"],
            out_keys=["action_high"],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.actor_low: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                TensorDictModule(self.get_low_obs, [OBS_KEY, OBS_PRIV_KEY, "action_high"], "actor_low_input"),
                TensorDictModule(make_mlp([256, 256, 256]), ["actor_low_input"], ["actor_low_feature"]),
                TensorDictModule(Actor(self.low_action_dim, True), ["actor_low_feature"], ["loc", "scale"]),
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.actor_high(fake_input)
        self.actor_low(fake_input)

        self.opt_high = torch.optim.Adam(self.actor_high.parameters(), lr=self.cfg.lr)
        self.opt_low = torch.optim.Adam(self.actor_low.parameters(), lr=self.cfg.lr)
    
    def get_low_obs(self, obs, obs_priv, action_high):
        return torch.cat([obs[..., 4:], obs_priv, action_high], -1)

    def get_rollout_policy(self, mode: str):
        policy = TensorDictSequential(
            self.actor_high,
            self.actor_low,
            ExcludeTransform(
                "actor_high_input", "actor_high_feature",
                "actor_low_input", "actor_low_feature",
            )
        )
        return policy
    
    def train_op(self, tensordict: TensorDictBase):
        self._relabel(tensordict)

        infos = []
        for _ in range(4):
            for minibatch in make_batch(tensordict, 8):
                infos.append(self._update(minibatch))
        
        infos = collect_info(infos)
        return infos

    def _update(self, tensordict: TensorDictBase):
        losses = {}
        losses["actor_high_loss"] = self._bc(tensordict, self.actor_high, "action_high")
        losses["actor_low_loss"] = self._bc(tensordict, self.actor_low, ACTION_KEY)

        loss = sum(v.mean() for k, v in losses.items())
        self.opt_high.zero_grad()
        self.opt_low.zero_grad()
        loss.backward()
        losses["actor_high_grad_norm"] = nn.utils.clip_grad_norm_(self.actor_high.parameters(), 5.)
        losses["actor_high_grad_norm"] = nn.utils.clip_grad_norm_(self.actor_low.parameters(), 5.)
        self.opt_high.step()
        self.opt_low.step()

        return TensorDict(losses, [])
    
    def _relabel(self, tensordict: TensorDictBase):
        tensordict["action_high"] = tensordict["next", "feet_pos_b"]
    
    def _bc(self, tensordict: TensorDictBase, actor: ProbabilisticActor, action_key: str):
        dist = actor.get_dist(tensordict)
        log_prob = dist.log_prob(tensordict[action_key])
        return - log_prob

    def state_dict(self):
        state_dict = super().state_dict()
        if "vecnorm._extra_state" in state_dict:
            state_dict["vecnorm._extra_state"]["lock"] = None # TODO: check with torchrl
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, torch.Any], strict: bool = True):
        if "vecnorm._extra_state" in state_dict:
            vecnorm_td = state_dict["vecnorm._extra_state"]["td"]
            state_dict["lock"] = None
            state_dict["vecnorm._extra_state"]["td"] = vecnorm_td.to(self.device)
        return super().load_state_dict(state_dict, strict)