import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict.nn import (
    TensorDictModule, 
    TensorDictSequential, 
    TensorDictModuleBase,
    utils
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
import einops

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import *
from .ppo_rnn import GRU
import functools


@dataclass
class PPGConfig:
    name: str = "ppg"
    train_every: int = 32
    ppo_epochs: int = 4
    aux_epochs: int = 6
    beta_clone: float = 1.

    rnn_actor: bool = False
    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.2
    recompute_adv: bool = True

cs = ConfigStore.instance()
cs.store(name="ppg", node=PPGConfig, group="algo")
cs.store(name="ppg_rnn", node=PPGConfig(rnn_actor=True), group="algo")


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
        self.mlp = make_mlp([256, 128])
        self.gru = GRU(128, hidden_size=128, allow_none=False)
        self.out = make_mlp([dim])
    
    def forward(self, x, is_init, hx):
        x = self.mlp(x)
        x, hx = self.gru(x, is_init, hx)
        x = self.out(x)
        return x, hx.contiguous()


class PPGPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg: PPGConfig,
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.action_dim = action_spec.shape[-1]

        self.entropy_coef = 0.001
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.action_dim = action_spec.shape[-1]

        self.gae = GAE(0.99, 0.95).to(self.device)
        self.value_norm = ValueNorm1(1).to(self.device)

        fake_input = observation_spec.zero()

        with utils._set_auto_make_functional(False):
            if self.cfg.rnn_actor:
                self.actor: ProbabilisticActor = ProbabilisticActor(
                    TensorDictSequential(
                        TensorDictModule(GRUModule(128), [OBS_KEY, "is_init", "hx"], ["_feature_actor", ("next", "hx")]),
                        TensorDictModule(nn.LazyLinear(1), ["_feature_actor"], ["value_actor"]),
                        TensorDictModule(nn.LazyLinear(5), ["_feature_actor"], ["linvel_pred"]),
                        TensorDictModule(Actor(self.action_dim), ["_feature_actor"], ["loc", "scale"]),
                    ),
                    in_keys=["loc", "scale"],
                    out_keys=[ACTION_KEY],
                    distribution_class=IndependentNormal,
                    return_log_prob=True
                ).to(self.device)
                fake_input["hx"] = torch.zeros(fake_input.shape[0], 128, device=self.device)

                from torchrl.envs.transforms.transforms import TensorDictPrimer
                from torchrl.data import UnboundedContinuousTensorSpec
                def make_tensordict_primer():
                    num_envs = observation_spec.shape[0]
                    return TensorDictPrimer({
                        "hx": UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)
                    })
                self.make_tensordict_primer = make_tensordict_primer

            else:
                self.actor: ProbabilisticActor = ProbabilisticActor(
                    TensorDictSequential(
                        TensorDictModule(make_mlp([256, 256, 128]), [OBS_KEY], ["_feature_actor"]),
                        TensorDictModule(nn.LazyLinear(1), ["_feature_actor"], ["value_actor"]),
                        TensorDictModule(Actor(self.action_dim), ["_feature_actor"], ["loc", "scale"]),
                    ),
                    in_keys=["loc", "scale"],
                    out_keys=[ACTION_KEY],
                    distribution_class=IndependentNormal,
                    return_log_prob=True
                ).to(self.device)

            self.critic = TensorDictSequential(
                CatTensors([OBS_KEY, OBS_PRIV_KEY], "full_obs", del_keys=False),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256, 128]), nn.LazyLinear(1),),
                    ["full_obs"], ["state_value"]
                ),
            ).to(self.device)

        self.action_enc = make_mlp([32]).to(self.device)
        self.dynamics = make_mlp([128, 128]).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)
        self.action_enc(fake_input[ACTION_KEY])
        self.dynamics(
            torch.cat([fake_input["_feature_actor"], self.action_enc(fake_input[ACTION_KEY])], -1)
        )

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
                {"params": self.dynamics.parameters()}
            ],
            lr=self.cfg.lr
        )

        if False:
            pass
        else:
            self.actor.apply(init_)
            self.critic.apply(init_)
            self.action_enc.apply(init_)
            self.dynamics.apply(init_)
        
        self.train_iter = 0

    def __call__(self, tensordict: TensorDictBase):
        self.actor(tensordict)
        return tensordict
    
    def train_op(self, tensordict: TensorDictBase):
        assert tensordict.device == self.device
        infos = {}

        infos_policy = []
        for epoch in range(self.cfg.ppo_epochs):
            if epoch == 0 or self.cfg.recompute_adv:
                self._compute_advantage(tensordict)
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every):
                infos_policy.append(self._update(minibatch))
        
        infos.update(collect_info(infos_policy))
        
        if self.cfg.aux_epochs > 0 and self.train_iter % 16 == 0:
            infos_aux = []
            with torch.no_grad():
                self.actor.get_dist_params(tensordict)
                self.actor(tensordict["next"])
            for epoch in range(self.cfg.aux_epochs):
                for minibatch in make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every):
                    infos_aux.append(self._update_aux(minibatch))
            infos.update(collect_info(infos_aux))
        
        self.train_iter += 1
        return infos

    @torch.no_grad()
    def _compute_advantage(self, tensordict: TensorDict, subtract_mean: bool=False):
        shape = (*tensordict.shape, -1)
        values = self.critic(tensordict.view(-1))["state_value"].reshape(shape)
        next_values = self.critic(tensordict["next"].view(-1))["state_value"].reshape(shape)

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
    
    # @functools.partial(torch.compile, mode="reduce-overhead")
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

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)

        loss = policy_loss + entropy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
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
        # values_critic = self.critic(tensordict)["state_value"]

        aux_value_loss_actor = F.mse_loss(values_actor, b_returns)
        # aux_value_loss = F.mse_loss(values_critic, b_returns)
        state_target = tensordict[OBS_PRIV_KEY][..., :5]
        state_est_loss = F.mse_loss(tensordict["linvel_pred"], state_target)

        # feature_pred = self.dynamics(
        #     torch.cat([tensordict["_feature_actor"], self.action_enc(tensordict[ACTION_KEY])], -1)
        # )
        # feature_next = tensordict["next", "_feature_actor"]
        # aux_model_loss = F.mse_loss(feature_pred, feature_next)

        loss = state_est_loss + aux_value_loss_actor + self.cfg.beta_clone * kl
        self.opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 10)
        self.opt.step()

        return TensorDict({
            "aux_kl": kl,
            "state_est_loss": state_est_loss,
            # "aux_value_loss": aux_value_loss,
            "aux_value_loss_actor": aux_value_loss_actor,
            # "aux_model_loss": aux_model_loss,
            "grad_norm": grad_norm
        }, [])
