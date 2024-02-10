import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from tensordict.nn import TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import *

@dataclass
class PPGConfig:
    name: str = "ppg"
    train_every: int = 32
    ppo_epochs: int = 4
    aux_epochs: int = 6
    beta_clone: float = 1.

    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.2
    recompute_adv: bool = False

cs = ConfigStore.instance()
cs.store(name="ppg", node=PPGConfig, group="algo")

class PPGPolicy:

    def __init__(
        self,
        cfg: PPGConfig,
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device
    ):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_spec.shape[-1]

        self.entropy_coef = 0.001
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.HuberLoss(delta=10, reduction="none")
        self.action_dim = action_spec.shape[-1]

        self.gae = GAE(0.99, 0.95).to(self.device)
        self.value_norm = ValueNorm1(1).to(self.device)

        fake_input = observation_spec.zero()

        self.actor: ProbabilisticActor = ProbabilisticActor(
            TensorDictSequential(
                TensorDictModule(make_mlp([256, 256, 128]), [OBS_KEY], ["_feature"]),
                TensorDictModule(nn.LazyLinear(1), ["_feature"], ["value_actor"]),
                TensorDictModule(Actor(self.action_dim), ["_feature"], ["loc", "scale"]),
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.Sequential(make_mlp([256, 256, 128]), nn.LazyLinear(1),),
            [OBS_KEY], ["state_value"]
        ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg.lr
        )

        if False:
            pass
        else:
            self.actor.apply(init_)
            self.critic.apply(init_)
        
        self.train_iter = 0

    def __call__(self, tensordict: TensorDictBase):
        self.actor(tensordict)
        return tensordict
    
    def train_op(self, tensordict: TensorDictBase):
        infos = {}

        infos_policy = []
        for epoch in range(self.cfg.ppo_epochs):
            if epoch == 0 or self.cfg.recompute_adv:
                self._compute_advantage(tensordict)
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos_policy.append(self._update(minibatch))
        
        infos.update(collect_info(infos_policy))
        
        if self.train_iter % 16 == 0 and self.cfg.aux_epochs > 0:
            infos_aux = []
            with torch.no_grad():
                self.actor.get_dist_params(tensordict)
            for epoch in range(self.cfg.aux_epochs):
                for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                    infos_aux.append(self._update_aux(minibatch))
            infos.update(collect_info(infos_aux))
        
        self.train_iter += 1
        return infos

    @torch.no_grad()
    def _compute_advantage(self, tensordict: TensorDict, subtract_mean: bool=False):
        values = self.critic(tensordict)["state_value"]
        next_values = self.critic(tensordict["next"])["state_value"]

        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        if subtract_mean:
            adv = (adv - adv_mean) / adv_std.clip(1e-7)
        else:
            adv /= adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        return tensordict
    
    def _update(self, tensordict: TensorDictBase):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * entropy

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped).mean()

        loss = policy_loss + entropy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 10)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 10)
        self.opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])
    
    def _update_aux(self, tensordict: TensorDictBase):
        dist_old = self.actor.build_dist_from_params(tensordict)
        dist_new = self.actor.get_dist(tensordict)
        kl = D.kl_divergence(dist_old, dist_new).mean()
        
        b_returns = tensordict["ret"]
        values_actor = tensordict["value_actor"]
        values_critic = self.critic(tensordict)["state_value"]

        aux_value_loss_actor = F.mse_loss(values_actor, b_returns)
        aux_value_loss = F.mse_loss(values_critic, b_returns)

        loss = aux_value_loss_actor + aux_value_loss + self.cfg.beta_clone * kl
        self.opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 10)
        self.opt.step()

        return TensorDict({
            "aux_kl": kl,
            "aux_value_loss": aux_value_loss,
            "aux_value_loss_actor": aux_value_loss_actor,
            "grad_norm": grad_norm
        }, [])
