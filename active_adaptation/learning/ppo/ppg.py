from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, TensorDictPrimer, ExcludeTransform
from torchrl.envs.utils import exploration_type, ExplorationType
from tensordict.nn import (
    TensorDictModule, 
    TensorDictSequential, 
    TensorDictModuleBase,
    utils
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from termcolor import colored
import einops

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *
from .ppo_rnn import GRU
from .ppo_roa import LinearSchedule

kl_divergence = D.kl._KL_REGISTRY[(D.Normal, D.Normal)]

@dataclass
class PPGConfig:
    name: str = "ppg"
    train_every: int = 32
    ppo_epochs: int = 4
    aux_epochs: int = -1 # 6
    beta_clone: float = 1.
    entropy_coef: float = 0.01
    value_norm: bool = False
    vecnorm: Union[str, None] = None
    
    # short_history: int = 5
    kl_prior: bool = False
    free_bits: float = 100
    rep_loss: float = 0.05

    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.1
    phase: str = "train"

cs = ConfigStore.instance()
cs.store(name="ppg_train", node=PPGConfig(vecnorm="train"), group="algo")
cs.store(name="ppg_adapt", node=PPGConfig(vecnorm="eval", phase="adapt"), group="algo")
cs.store(name="ppg_finetune", node=PPGConfig(vecnorm="eval", phase="finetune"), group="algo")

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
        obs_feature = self.mlp(x)
        rnn_feature, hx = self.gru(obs_feature.detach(), is_init, hx)
        x = self.out(torch.cat([obs_feature, rnn_feature], -1))
        return x, hx.contiguous()

class GRUModuleStoch(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = make_mlp([128, 128])
        self.gru = GRU(128, hidden_size=128, allow_none=False)
        self.out = nn.LazyLinear(dim * 2)
    
    def forward(self, x, is_init, hx):
        x = self.mlp(x)
        x, hx = self.gru(x, is_init, hx)
        x_loc, x_scale = self.out(x).chunk(2, -1)
        x_scale = F.softplus(x_scale)
        return x_loc, x_scale, hx.contiguous()
    
class NormalParams(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.LazyLinear(dim * 2)
    
    def forward(self, x):
        loc, scale = self.linear(x).chunk(2, -1)
        scale = F.softplus(scale)
        return loc, scale

def sample_normal(loc, scale, k: int=1, n: int=2):
    loc = loc.expand(k, n, *loc.shape)
    if exploration_type() == ExplorationType.MODE:
        samples = loc + 0.
    else:
        samples = loc + torch.randn_like(loc).clamp(-3., 3.) * scale
    if n > 1:
        samples = torch.cat(list(samples.unbind(1)), dim=-1)
    else:
        samples = samples.squeeze(1)
    return samples.squeeze(0)

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
        self.observation_spec = observation_spec

        self.entropy_coef = self.cfg.entropy_coef
        self.clip_param = self.cfg.clip_param
        # self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.context_dim = 128
        self.phase = self.cfg.phase
        # self.aux_target_dim = observation_spec["aux_target_"].shape[-1]

        self.gae = GAE(0.99, 0.95).to(self.device)
        
        self.beta = 0.
        self.beta_schedule = LinearSchedule(0., self.cfg.rep_loss, 0.2, 1.0)

        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["context_adapt_hx"] = torch.zeros(fake_input.shape[0], 128)
        print(fake_input)

        # self.encoder = TensorDictModule(
        #     nn.Sequential(make_mlp([128]), nn.LazyLinear(128)),
        #     [OBS_PRIV_KEY],
        #     ["context_priv"]
        # ).to(self.device)

        self.encoder = TensorDictModule(
            nn.Sequential(make_mlp([128]), NormalParams(128)),
            [OBS_PRIV_KEY],
            ["context_priv_loc", "context_priv_scale"]
        ).to(self.device)

        self.sample_context = TensorDictModule(
            sample_normal, ["context_priv_loc", "context_priv_scale"], ["context_priv"])
        self.sample_context_adapt = TensorDictModule(
            sample_normal, ["context_adapt_loc", "context_adapt_scale"], ["context_adapt"])

        def make_adapt_module(mode: str="deter"):
            if mode == "deter":
                module = TensorDictModule(
                    GRUModule(self.context_dim), 
                    [OBS_KEY, "is_init", "context_adapt_hx"], 
                    ["context_adapt", ("next", "context_adapt_hx")]
                ).to(self.device)
            elif mode == "stoch":
                module = TensorDictModule(
                    GRUModuleStoch(self.context_dim),
                    [OBS_KEY, "is_init", "context_adapt_hx"], 
                    ["context_adapt_loc", "context_adapt_scale", ("next", "context_adapt_hx")]
                ).to(self.device)
            return module
        self.adapt_module = make_adapt_module("stoch")
        
        def make_actor(context_key):
            actor = TensorDictSequential(
                CatTensors([OBS_KEY, context_key], "_actor_in", del_keys=False),
                TensorDictModule(make_mlp([256, 256, 256]), ["_actor_in"], ["_feature_actor"]),
                TensorDictModule(Actor(self.action_dim, True), ["_feature_actor"], ["loc", "scale"]),
                # TensorDictModule(nn.LazyLinear(self.aux_target_dim), ["_feature_actor"], ["actor_aux"]),
            )
            return actor
        
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=make_actor("context_priv"),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.actor_adapt: ProbabilisticActor = ProbabilisticActor(
            module=make_actor("context_adapt"),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic_priv = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "_critic_in", del_keys=False),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)), 
                ["_critic_in"], ["value_priv"]
            ),
        ).to(self.device)

        self.critic_adapt = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY, "context_adapt_loc"], "_critic_in", del_keys=False),
            TensorDictModule(
                nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(1)), 
                ["_critic_in"], ["value_adapt"]
            ),
        ).to(self.device)

        self.encoder(fake_input)
        self.sample_context(fake_input)
        self.adapt_module(fake_input)
        self.sample_context_adapt(fake_input)
        self.actor(fake_input)
        self.actor_adapt(fake_input)
        self.critic_priv(fake_input)
        self.critic_adapt(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.encoder.parameters()},
                {"params": self.actor.parameters()},
                {"params": self.critic_priv.parameters()},
            ],
            lr=self.cfg.lr
        )
        self.opt_aux = torch.optim.Adam(
            [
                {"params": self.encoder.parameters()},
                {"params": self.actor.parameters()},
                {"params": self.critic_priv.parameters()},
            ],
            lr=self.cfg.lr
        )
        self.opt_adapt = torch.optim.Adam(
            [
                {"params": self.adapt_module.parameters()}
            ],
            lr=self.cfg.lr
        )
        self.opt_target = torch.optim.Adam(
            [
                {"params": self.actor_adapt.parameters()},
                {"params": self.critic_priv.parameters()},
                {"params": self.critic_adapt.parameters()}
            ],
            lr=self.cfg.lr
        )

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.)

        self.encoder.apply(init_)
        self.actor.apply(init_)
        self.critic_priv.apply(init_)
        self.critic_adapt.apply(init_)
        self.adapt_module.apply(init_)
        print(self.encoder)
        print(self.actor)
        print(self.critic_priv)
        
        self.train_iter = 0

    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        spec = UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)
        return TensorDictPrimer({"context_adapt_hx": spec}, reset_key="done")
    
    def get_rollout_policy(self, mode: str):
        modules = []
        exclude_keys = ["loc", "scale"]
        if mode in ("train", "eval"):
            modules.append(self.encoder)
            modules.append(self.sample_context)
            modules.append(self.critic_priv)
        
        # def _noise_(x: torch.Tensor):
        #     return x + torch.randn_like(x) * 1.6
        # modules.append(TensorDictModule(_noise_, ["context_priv"], ["context_priv"]))
        # modules.append(TensorDictModule(_noise_, ["context_adapt"], ["context_adapt"]))
        
        if self.phase == "train":
            modules.append(self.adapt_module) # to compute prior
            modules.append(self.actor)
        elif self.phase in ("adapt", "finetune"):
            print(colored("[Info]: Use adapt policy.", "green"))
            modules.append(self.adapt_module)
            modules.append(self.sample_context_adapt)
            modules.append(self.actor_adapt)

        policy = TensorDictSequential(*modules, ExcludeTransform(*exclude_keys))
        return policy
        
    
    def step_schedule(self, progress: float):
        self.beta = self.beta_schedule.compute(progress)
        # self.gae.gamma.copy_(min(0.99 + progress * 0.005, 0.995))

    def train_op(self, tensordict: TensorDictBase):
        infos = {}
        if self.phase == "train":
            infos.update(self.train_policy_priv(tensordict.copy()))
        elif self.phase == "finetune":
            infos.update(self.train_policy_adapt(tensordict.copy()))
        infos.update(self.train_adaptation(tensordict.copy()))
        
        self.train_iter += 1
        return dict(sorted(infos.items()))

    def train_policy_priv(self, tensordict: TensorDictBase):
        infos = []

        self._compute_advantage(tensordict, "priv")

        with torch.no_grad():
            tensordict["sample_log_prob"] = self._compute_logprobs(tensordict, self.actor)
        
        for epoch in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every):
                minibatch["adv"] = normalize(minibatch["adv_priv"], True)
                infos.append(TensorDict(self._update(minibatch), []))
        
        hard_copy_(self.actor, self.actor_adapt)
        
        infos = collect_info(infos)
        infos["value_priv"] = self.value_norm.denormalize(tensordict["ret_priv"]).mean().item()
        infos["beta"] = self.beta
        return infos
    
    def train_policy_adapt(self, tensordict: TensorDictBase):
        infos = []

        with torch.no_grad(), tensordict["next"].view(-1) as td_next:
            self.adapt_module(td_next)
        self._compute_advantage(tensordict, key="priv")
        self._compute_advantage(tensordict, key="adapt")
        #     tensordict["sample_log_prob"] = self._compute_logprobs(tensordict, self.actor_adapt)
        
        adv_key = "adv_priv"
        for epoch in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every):
                minibatch["adv"] = normalize(minibatch[adv_key], True)
                infos.append(TensorDict(self._update_adapt(minibatch), []))
        
        infos = collect_info(infos)
        infos["value_priv"] = self.value_norm.denormalize(tensordict["ret_priv"]).mean().item()
        return infos

    # @torch.compile
    def train_adaptation(self, tensordict: TensorDictBase, reverse: bool=False):
        infos = []
        
        with torch.no_grad(): # update the contexts
            self.encoder(tensordict)

        for epoch in range(2):
            batch = make_batch(tensordict, 8, self.cfg.train_every)
            for minibatch in batch:
                losses = {}
                self.adapt_module(minibatch)
                context_dist_priv = D.Normal(minibatch["context_priv_loc"], minibatch["context_priv_scale"])
                context_dist_pred = D.Normal(minibatch["context_adapt_loc"], minibatch["context_adapt_scale"])
                if not reverse:
                    kl = kl_divergence(context_dist_priv, context_dist_pred)
                else:
                    kl = kl_divergence(context_dist_pred, context_dist_priv)
                losses["adapt/adapt_module_loss"] = kl.sum(-1).mean()

                loss = sum(losses.values())
                self.opt_adapt.zero_grad()
                loss.backward()
                losses["adapt/adapt_module_grad_norm"] = nn.utils.clip_grad_norm_(self.adapt_module.parameters(), 10.)
                self.opt_adapt.step()
                losses["adapt/context_adapt_std"] = minibatch["context_adapt_scale"].mean()
                infos.append(TensorDict(losses, []))
    
        infos = collect_info(infos)
        return infos
    
    def _compute_logprobs(self, tensordict: TensorDictBase, actor: ProbabilisticActor, k: int=4):
        actor_input = tensordict.select(OBS_KEY).unsqueeze(0).expand(k, *tensordict.shape)
        if actor is self.actor:
            actor_input["context_priv"] = sample_normal(tensordict["context_priv_loc"], tensordict["context_priv_scale"], k)
        else:
            actor_input["context_adapt"] = sample_normal(tensordict["context_adapt_loc"], tensordict["context_adapt_scale"], k)
        dist = actor.get_dist(actor_input)
        action = tensordict[ACTION_KEY]
        log_prob = dist.log_prob(action.expand(k, *action.shape))
        log_prob = torch.logsumexp(log_prob, 0) - torch.log(torch.tensor(k, device=self.device))
        return log_prob

    @torch.no_grad()
    def _compute_advantage(self, tensordict: TensorDict, key: str):
        critic = getattr(self, f"critic_{key}")
        values = critic(tensordict)[f"value_{key}"]
        next_values = critic(tensordict["next"])[f"value_{key}"]

        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)

        tensordict.set(f"adv_{key}", adv)
        tensordict.set(f"ret_{key}", ret)
        return tensordict
    
    def _compute_policy_loss(self, tensordict: TensorDictBase, actor: ProbabilisticActor):
        if actor is self.actor:
            log_probs = self._compute_logprobs(tensordict, actor)
        else:
            log_probs = actor.get_dist(tensordict).log_prob(tensordict[ACTION_KEY])
        entropy = - log_probs.mean()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]).float())
        entropy_loss = - self.entropy_coef * entropy
        return policy_loss, entropy_loss, entropy
    
    def _update(self, tensordict: TensorDictBase):
        self.encoder(tensordict)
        policy_loss, entropy_loss, entropy = self._compute_policy_loss(tensordict, self.actor)

        b_returns = tensordict["ret_priv"]
        values = self.critic_priv(tensordict)["value_priv"]
        value_loss = self.critic_loss_fn(b_returns, values) * (~tensordict["is_init"])
        value_loss = value_loss.mean()

        context_dist_priv = D.Normal(tensordict["context_priv_loc"], tensordict["context_priv_scale"])
        context_dist_pred = D.Normal(tensordict["context_adapt_loc"], tensordict["context_adapt_scale"])
        context_kl = kl_divergence(context_dist_priv, context_dist_pred).sum(-1)

        loss = policy_loss + entropy_loss + value_loss + self.beta * context_kl.clamp_min(128.).mean()
        self.opt.zero_grad()
        loss.backward()
        encoder_grad_norm = nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.)
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 2.)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic_priv.parameters(), 2.)
        self.opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "encoder_grad_norm": encoder_grad_norm,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "value_loss/value_loss": value_loss,
            "value_loss/explained_var": explained_var,
            "rep_loss": context_kl.mean(),
            "context_std": torch.mean(context_dist_priv.scale)
        }

    def _update_adapt(self, tensordict: TensorDictBase):
        policy_loss, entropy_loss, entropy = self._compute_policy_loss(tensordict, self.actor_adapt)

        b_returns = tensordict["ret_priv"]
        values = self.critic_priv(tensordict)["value_priv"]
        value_loss_priv = self.critic_loss_fn(b_returns, values) * (~tensordict["is_init"])
        value_loss_priv = value_loss_priv.mean()
        explained_var_priv = 1 - F.mse_loss(values, b_returns) / b_returns.var()

        b_returns = tensordict["ret_adapt"]
        values = self.critic_adapt(tensordict)["value_adapt"]
        value_loss_adapt = self.critic_loss_fn(b_returns, values) * (~tensordict["is_init"])
        value_loss_adapt = value_loss_adapt.mean()
        explained_var_adapt = 1 - F.mse_loss(values, b_returns) / b_returns.var()

        loss = policy_loss + entropy_loss + value_loss_priv + value_loss_adapt
        self.opt_target.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor_adapt.parameters(), 1)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic_adapt.parameters(), 10.)
        self.opt_target.step()
        return {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "value_loss/value_loss_priv": value_loss_priv,
            "value_loss/value_loss_adapt": value_loss_adapt,
            "value_loss/explained_var_priv": explained_var_priv,
            "value_loss/explained_var_adapt": explained_var_adapt
        }
    
    # def _update_aux(self, tensordict: TensorDictBase):
    #     losses = {}
    #     dist_old = self.actor.build_dist_from_params(tensordict)
    #     dist_new = self.actor.get_dist(self.encoder(tensordict))
        
    #     actor_aux = tensordict["actor_aux"]
    #     actor_aux_target = tensordict["aux_target_"]

    #     losses["aux/pred_loss"] = F.mse_loss(actor_aux[..., :-1], actor_aux_target)
    #     losses["aux/value_loss"] = F.mse_loss(actor_aux[..., -1:], tensordict["ret"])
    #     losses["aux/kl"] = self.cfg.beta_clone * D.kl_divergence(dist_old, dist_new).mean()
        
    #     context_loc, context_scale = tensordict["context_priv_loc"], tensordict["context_priv_scale"]
    #     context_dist = D.Normal(context_loc, context_scale)
    #     kl_prior = D.kl_divergence(
    #         context_dist,
    #         D.Normal(torch.zeros_like(context_loc), torch.ones_like(context_loc))
    #     ).sum(-1)
    #     if self.cfg.kl_prior:
    #         losses["aux/prior_loss"] = 0.1 * kl_prior.clamp_min(self.cfg.free_bits).mean()

    #     loss =  sum(losses.values())
    #     self.opt_aux.zero_grad()
    #     loss.backward()
    #     grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 10)
    #     self.opt_aux.step()
    #     losses["aux/grad_norm"] = grad_norm
    #     losses["aux/kl_prior"] = kl_prior.mean()
    #     losses["aux/context_entropy"] = context_dist.entropy().sum(-1).mean()
    #     return TensorDict(losses, [])

    def load_state_dict(self, state_dict, strict: bool = False):
        super().load_state_dict(state_dict, strict)
