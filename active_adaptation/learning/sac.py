import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math

from torchrl.data import CompositeSpec, TensorSpec, TensorDictReplayBuffer, LazyTensorStorage, ListStorage
from torchrl.objectives import hold_out_net
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.envs.transforms import CatTensors, ExcludeTransform, VecNorm, MultiStepTransform
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Mapping, Union
from copy import deepcopy

from .ppo.common import *
from .modules.distributions import IndependentNormal

@dataclass
class SACConfig:
    name: str = "sac"
    train_every: int = 32
    warm_up_steps: int = 100000
    lr: float = 5e-4

    checkpoint_path: Union[str, None] = None
    context_dim: int = 128

cs = ConfigStore.instance()
cs.store("sac", node=SACConfig, group="algo")

class SAC(TensorDictModuleBase):
    def __init__(
        self,
        cfg: SACConfig,
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec

        fake_input = observation_spec.zero()
        self.action_dim = self.action_spec.shape[-1]
        self.target_entropy = 0 # - self.action_dim

        self.encoder_priv = TensorDictModule(
            make_mlp([self.cfg.context_dim]),
            [OBS_PRIV_KEY],
            ["context_expert"]
        ).to(self.device)

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                CatTensors([OBS_KEY, "context_expert"], "actor_input", del_keys=False),
                TensorDictModule(make_mlp([256, 256, 256]), ["actor_input"], ["actor_feature"]),
                TensorDictModule(Actor(self.action_dim, True), ["actor_feature"], ["loc", "scale"]),
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=TanhNormal,
            distribution_kwargs={"min": -2.0, "max": 2.0},
            return_log_prob=True
        ).to(self.device)

        def make_critic():
            return nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1))
        
        self.qs = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY, ACTION_KEY], "q_input", del_keys=False),
            TensorDictModule(make_critic(), ["q_input"], ["Q1"]),
            TensorDictModule(make_critic(), ["q_input"], ["Q2"]),
        ).to(self.device)

        self.dynamics = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY, ACTION_KEY], "dyn_input", del_keys=False),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(fake_input[OBS_KEY].shape[-1])),
                ["dyn_input"], [("next", OBS_KEY)]
            )
        ).to(self.device)

        self.encoder_priv(fake_input)
        self.actor(fake_input)
        self.qs(fake_input)
        self.dynamics(fake_input)

        self.qs_ema = deepcopy(self.qs)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)

        self.log_alpha = nn.Parameter(torch.tensor(0., device=self.device))
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=1e-2)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.opt_critic = torch.optim.Adam(self.qs.parameters(), lr=self.cfg.lr)
        self.opt_dyn = torch.optim.Adam(self.dynamics.parameters(), lr=cfg.lr)

        self.rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=2000000),
            batch_size=8192,
            prefetch=2,
        )
        self.multi_step = MultiStepTransform(3, gamma=0.99)

    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(
            self.encoder_priv,
            self.actor,
            ExcludeTransform("actor_feature", "loc", "scale")
        )
        return policy

    def train_op(self, tensordict: TensorDictBase):
        tensordict = tensordict.view(-1).cpu()
        self.rb.extend(tensordict)
        if len(self.rb) < self.cfg.warm_up_steps:
            return {"rb_size": len(self.rb)}
        
        infos = []

        for _ in range(self.cfg.train_every * 2):
            batch = self.rb.sample().to(self.device)
            infos.append(self.update(batch))
        
        soft_copy_(self.qs, self.qs_ema)

        infos = {k: v.float().mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["rb_size"] = len(self.rb)
        infos["alpha"] = self.log_alpha.exp().item()
        return infos
    
    def update(self, tensordict: TensorDictBase):
        losses = {}

        losses["dyn_loss"] = self._compute_dyn_loss(tensordict.copy())
        self.opt_dyn.zero_grad()
        losses["dyn_loss"].backward()
        self.opt_dyn.step()

        losses["critic_loss"] = self._compute_critic_loss(tensordict)
        self.opt_critic.zero_grad()
        losses["critic_loss"].backward()
        losses["qs_grad_norm"] = nn.utils.clip_grad_norm_(self.qs.parameters(), 2.)
        self.opt_critic.step()

        losses["actor_loss"] = self._compute_actor_loss(tensordict)
        self.opt_actor.zero_grad()
        losses["actor_loss"].backward()
        losses["actor_grad_norm"] = nn.utils.clip_grad_norm_(self.actor.parameters(), 2.)
        self.opt_actor.step()

        losses["alpha_loss"] = -(self.log_alpha.exp() * (tensordict["sample_log_prob"].detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        losses["alpha_loss"].backward()
        self.opt_alpha.step()

        losses["entropy"] = -tensordict["sample_log_prob"].mean()
        losses["q_taken"] = tensordict["q_taken"].mean()

        return TensorDict(losses, [])
    
    def _compute_actor_loss(self, tensordict: TensorDictBase):
        dist = self.actor.get_dist(tensordict)
        action = dist.rsample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        tensordict[ACTION_KEY] = action
        tensordict["sample_log_prob"] = log_prob
        with hold_out_net(self.qs):
            self.qs(tensordict)
        q = 0.5 * (tensordict["Q1"] + tensordict["Q2"])
        tensordict["q_taken"] = q
        actor_loss = (self.log_alpha.exp().detach() * log_prob - q).mean()

        return actor_loss
    
    def _compute_critic_loss(self, tensordict: TensorDictBase):
        with torch.no_grad():
            self.encoder_priv(tensordict["next"])
            self.actor(tensordict["next"])
            self.qs_ema(tensordict["next"])            
            next_q = (
                torch.min(tensordict["next", "Q1"], tensordict["next", "Q2"]) 
                - self.log_alpha.exp() * tensordict["next", "sample_log_prob"].unsqueeze(-1)
            )
            td_target = tensordict[REWARD_KEY] + 0.99 * (1 - tensordict[DONE_KEY].float()) * next_q
        
        self.qs(tensordict)
        
        critic_loss = (
            F.mse_loss(tensordict["Q1"], td_target)
            + F.mse_loss(tensordict["Q2"], td_target)
        )
        return critic_loss

    def _compute_dyn_loss(self, tensordict: TensorDictBase):
        pred = self.dynamics(tensordict.copy())["next", OBS_KEY]
        target = tensordict["next", OBS_KEY]
        dyn_loss = F.mse_loss(pred, target)
        return dyn_loss

    def state_dict(self):
        state_dict = super().state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict=strict)
