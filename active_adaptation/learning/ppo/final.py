import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase

from collections import defaultdict
import einops
from .common import *
from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal


from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, MISSING
from typing import Any, Mapping, Union, Sequence

@dataclass
class PPOConfig:
    name: str = "final"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4

    checkpoint_path: Union[str, None] = None
    train_model_with_policy: bool = False

cs = ConfigStore.instance()
cs.store(name="final", node=PPOConfig, group="algo")


class Dynamics(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            make_mlp([256, 256]),
            nn.LazyLinear(latent_dim + 1)
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor):
        transition = self.mlp(torch.cat([z, a], dim=-1))
        z_next, r = transition.split([self.latent_dim, 1], dim=-1)
        return z_next, r


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.obs_encoder = make_mlp([256, latent_dim // 2])
        self.height_scan_encoder = nn.Sequential(
            nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.LazyConv2d(16, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(),
            make_mlp([latent_dim // 2])
        )

    def forward(self, obs: torch.Tensor, obs_priv: torch.Tensor, height_scan: torch.Tensor):
        latent =  torch.cat([
            self.obs_encoder(torch.cat([obs, obs_priv], -1)),
            self.height_scan_encoder(height_scan)
        ], dim=-1)
        return latent


class ActorExpert(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.loc = nn.Sequential(make_mlp([256, 128]), nn.LazyLinear(action_dim))
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, z: torch.Tensor):
        loc = self.loc(z)
        scale = torch.exp(self.log_std)
        return loc, scale.expand_as(loc)

class ActorAdapt(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.loc = nn.Sequential(make_mlp([256, 128]), nn.LazyLinear(action_dim))
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, z: torch.Tensor, z_pred: torch.Tensor):
        loc = self.loc(torch.cat([z, z_pred], dim=-1))
        scale = torch.exp(self.log_std)
        return loc, scale.expand_as(loc)

class Policy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec,
        action_spec,
        reward_spec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.entropy_coef = 0.001
        self.clip_param = 0.2
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        self.encoder = TensorDictModule(
            Encoder(latent_dim=256),
            [OBS_KEY, OBS_PRIV_KEY, "height_scan"], ["latent"]
        ).to(self.device)

        self.dynamics = TensorDictModule(
            Dynamics(latent_dim=256),
            ["latent", ACTION_KEY], 
            [("next", "latent_pred"), ("next", "reward_pred")]
        ).to(self.device)

        self.actor = ProbabilisticActor(
            TensorDictModule(
                ActorExpert(action_spec.shape[-1]),       
                ["latent"], ["loc", "scale"]
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        self.actor_adapt = TensorDictModule(
            ActorAdapt(action_spec.shape[-1]),
            ["latent", "latent_pred"], ["action"]
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.Sequential(make_mlp([256, 128]), nn.LazyLinear(1)),
            ["latent"], ["state_value"]
        ).to(self.device)

        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        self.dynamics(fake_input)

        checkpoint_path = self.cfg.checkpoint_path
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            self.encoder.apply(init_)
            self.actor.apply(init_)
            self.critic.apply(init_)
            self.dynamics.apply(init_)

        self.value_norm = ValueNorm1(input_shape=1).to(self.device)

        self.opt = torch.optim.Adam([
            {"params": self.encoder.parameters()},
            {"params": self.dynamics.parameters()},
            {"params": self.actor.parameters()},
            {"params": self.actor_adapt.parameters()},
            {"params": self.critic.parameters()},
        ], lr=5e-4)

        self.train_model_with_policy = self.cfg.train_model_with_policy

    def __call__(self, tensordict: TensorDict):
        self.encoder(tensordict)
        self.actor(tensordict)
        return tensordict
    
    def train_op(self, tensordict: TensorDict):
        infos = []

        with torch.no_grad():
            self.encoder(tensordict.view(-1))
            self.encoder(tensordict["next"].view(-1))
            self._compute_advantage(tensordict, self.critic, self.value_norm)

        for epoch in range(self.cfg.ppo_epochs):
            for minimatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos.append(self._update_policy(minimatch))
        
        infos = collect_info(infos)
        return infos
    
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TensorDictModule,
        value_norm: ValueNorm1,
        reward_weights = 1.
    ):
        values = critic(tensordict)["state_value"]
        next_values = critic(tensordict["next"])["state_value"]

        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = value_norm.denormalize(values)
        next_values = value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = (adv * torch.as_tensor(reward_weights, device=adv.device)).sum(-1, True)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        value_norm.update(ret)
        ret = value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        
        return tensordict
    
    def _update_model(self, tensordict: TensorDict):

        return
    
    def _update_policy(self, tensordict: TensorDict):

        losses = TensorDict({}, [])
        self.encoder(tensordict)
        policy_loss, entropy_loss, entropy = compute_policy_loss(
            tensordict,
            self.actor, 
            self.clip_param, 
            self.entropy_coef
        )
        losses["policy_loss"] = policy_loss
        losses["entropy_loss"] = entropy_loss

        value_loss, explained_var = compute_value_loss(
            tensordict, 
            self.critic, 
            self.clip_param, 
            self.critic_loss_fn
        )
        losses["value_loss"] = value_loss

        if self.train_model_with_policy:
            self.dynamics(tensordict)
            dynamics_loss = F.mse_loss(tensordict["next", "latent_pred"], tensordict["next", "latent"])
            losses["dynamics"] = dynamics_loss

        loss = sum(losses.values())
        self.opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 10)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 10)
        self.opt.step()

        infos = losses.to_tensordict()
        infos["actor_grad_norm"] = actor_grad_norm
        infos["critic_grad_norm"] = critic_grad_norm
        infos["explained_var"] = explained_var
        infos["entropy"] = entropy
        return infos
    

def collect_info(info_list):
    return {k: v.mean().item() for k, v in torch.stack(info_list).items()}