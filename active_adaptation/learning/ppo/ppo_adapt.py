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
import einops
import functools
import warnings
import copy

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, TensorDictPrimer, ExcludeTransform, VecNorm
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Mapping, Union
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake, Normalizer
from ..modules.distributions import IndependentNormal
from .common import *

@dataclass
class PPOConfig:
    name: str = "ppo_adapt"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.1
    entropy_coef: float = 0.01

    actor_predict_std: bool = True
    orthogonal_init: bool = True
    layer_norm: bool = True
    value_norm: bool = False

    aux_reward: float = 0.
    aux_epochs: int = -1

    context_dim: int = 128
    adapt_module: str = "mse"
    adapt_reward: bool = False
    tune_alpha: bool = True
    target_kl: float = 1.0
    unbiased_critic: bool = True # whether use critic_adapt or critic_expert during finetuning

    phase: str = "train"
    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_adapt_train", node=PPOConfig(phase="train"), group="algo")
cs.store("ppo_adapt_adapt", node=PPOConfig(phase="adapt"), group="algo")
cs.store("ppo_adapt_finetune", node=PPOConfig(phase="finetune"), group="algo")

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
        self.mlp = make_mlp([128, 128])
        self.gru = GRU(128, hidden_size=128, allow_none=False)
        self.out = nn.LazyLinear(dim)
    
    def forward(self, x, is_init, hx):
        x = self.mlp(x)
        x, hx = self.gru(x, is_init, hx)
        x = self.out(x)
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
        x = x_loc + torch.randn_like(x_loc) * x_scale
        return x, x_loc, x_scale, hx.contiguous()


class Marginalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(make_mlp([64, 64], norm=None), nn.LayerNorm(64))
        # self.f.requires_grad_(False)
        self.f_marg = nn.Sequential(make_mlp([64, 64]), nn.LazyLinear(128))
    
    def forward(self, obs, state):
        full_obs = torch.cat([obs, state], -1)
        f = self.f(full_obs).detach()
        loc, scale = self.f_marg(obs).chunk(2, -1)
        nll = - D.Normal(loc, scale.exp()).log_prob(f).sum(-1, True)
        return nll


class PPOAdaptPolicy(TensorDictModuleBase):
    
    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        vecnorm: VecNorm=None,
        device: str="cuda:0"
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.vecnorm = vecnorm

        self.entropy_coef = self.cfg.entropy_coef
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.HuberLoss(delta=10, reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        if not cfg.layer_norm:
            global make_mlp
            make_mlp = functools.partial(make_mlp, norm=None)

        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        
        self.value_norms: Mapping[str, Normalizer] = nn.ModuleDict({
            "obs": value_norm_cls(input_shape=1),
            "priv": value_norm_cls(input_shape=1),
            "adapt": value_norm_cls(input_shape=1)
        }).to(self.device)
        
        self.phase = self.cfg.phase
        self.last_phase = None
        self.num_updates = 0
        self.num_frames = 0

        if self.phase != "train":
            self.vecnorm.eval() # stop updating during "adapt" and "finetune"

        self.observation_spec = observation_spec
        fake_input = observation_spec.zero()
        # lazy initialization
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["context_adapt_hx"] = torch.zeros(fake_input.shape[0], 128)
        print(fake_input.shapes)

        self.log_alpha = nn.Parameter(torch.tensor(0.))
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=1e-2)
        self.target_kl = self.cfg.target_kl
        self.context_dim = self.cfg.context_dim
        
        if "height_scan" in observation_spec.keys(True, True):
            # the conv_encoder is shared between actor and critic for efficiency
            self.context_dim = self.cfg.context_dim + 16
            conv_encoder = nn.Sequential(make_conv([8, 16], kernel_sizes=[5, 3]), make_mlp([16]))
            mlp_encoder = make_mlp([self.cfg.context_dim])
            self.encoder_priv = TensorDictSequential(
                TensorDictModule(mlp_encoder, [OBS_PRIV_KEY], ["_mlp"]),
                TensorDictModule(conv_encoder, ["height_scan"], ["cnn"]),
                CatTensors(["_mlp", "cnn"], "context_expert", del_keys=False)
            ).to(self.device)
        else:
            self.encoder_priv = TensorDictModule(
                make_mlp([self.cfg.context_dim]),
                [OBS_PRIV_KEY],
                ["context_expert"]
            ).to(self.device)
        
        self.encoder_priv(fake_input)
        
        def make_adapt_module(stoch: bool = False):
            if stoch:
                module = TensorDictModule(
                    GRUModuleStoch(self.context_dim),
                    [OBS_KEY, "is_init", "context_adapt_hx"], 
                    ["context_adapt", "context_adapt_loc", "context_adapt_std", ("next", "context_adapt_hx")]
                ).to(self.device)
            else:
                module = TensorDictModule(
                    GRUModule(self.context_dim), 
                    [OBS_KEY, "is_init", "context_adapt_hx"], 
                    ["context_adapt", ("next", "context_adapt_hx")]
                ).to(self.device)
            return module
        
        self.adapt_module_a = make_adapt_module()
        self.adapt_module_a(fake_input)
        self.adapt_module_b = make_adapt_module()
        self.adapt_module_b(fake_input)
        self.adapt_modules = {
            "mse": self.adapt_module_a,
            "action_kl": self.adapt_module_b,
        }
        self.adapt_module_ema = copy.deepcopy(self.adapt_modules[self.cfg.adapt_module])
        self.adapt_module_ema(fake_input)
        
        def make_actor(context_key: str) -> ProbabilisticActor:
            actor = ProbabilisticActor(
                module=TensorDictSequential(
                    CatTensors([OBS_KEY, context_key], "actor_input", del_keys=False),
                    TensorDictModule(make_mlp([256, 256, 256]), ["actor_input"], ["actor_feature"]),
                    TensorDictModule(Actor(self.action_dim, self.cfg.actor_predict_std), ["actor_feature"], ["loc", "scale"]),
                ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True
            ).to(self.device)
            return actor
        
        def make_critic(num_outputs: int=1):
            layers = [make_mlp([256, 256, 256]), nn.LazyLinear(num_outputs)]
            if num_outputs > 1:
                layers.append(Chunk(num_outputs))
            return nn.Sequential(*layers)
        
        # expert actor with privileged information
        self._actor_expert = make_actor("context_expert")
        # expert actor with estimated information used for compute estimation error
        self.__actor_expert = make_actor("context_adapt")
        # target actor with estimated information
        self._actor_adapt = make_actor("context_adapt")

        # expert critic with privileged information
        critic_priv_keys = [OBS_KEY, OBS_PRIV_KEY]
        if "params" in observation_spec.keys(True, True):
            critic_priv_keys.append("params")
        if "height_scan" in observation_spec.keys(True, True):
            critic_priv_keys.append("cnn")
        
        self._critic_obs = TensorDictModule(make_critic(), [OBS_KEY], ["value_obs"]).to(self.device)
        self._critic_priv = TensorDictSequential(
            CatTensors(critic_priv_keys, "critic_priv_input", del_keys=False),
            TensorDictModule(make_critic(), ["critic_priv_input"], ["value_priv"])
        ).to(self.device)
        # expert critic with both privilged and estimated information to be unbiased
        self._critic_adapt = TensorDictSequential(
            # CatTensors([OBS_KEY, OBS_PRIV_KEY, "context_adapt"], "critic_adapt_input", del_keys=False),
            CatTensors([*critic_priv_keys, "context_adapt"], "critic_adapt_input", del_keys=False),
            TensorDictModule(make_critic(), ["critic_adapt_input"], ["value_adapt"])
        ).to(self.device)

        self._critic_aux = TensorDictModule(
            make_critic(), ["critic_priv_input"], ["value_aux"]
        ).to(self.device)
        self._marg = TensorDictModule(
            Marginalizer(), [OBS_KEY, OBS_PRIV_KEY], ["nll"]
        ).to(self.device)
        self._marg(fake_input)

        self.actor_critic_priv = TensorDictSequential(
            self.encoder_priv,
            self._actor_expert,
            self._critic_priv
        )
        
        self._actor_expert(fake_input)
        self._actor_adapt(fake_input)
        self._critic_obs(fake_input)
        self._critic_priv(fake_input)
        self._critic_adapt(fake_input)
        self._critic_aux(fake_input)

        self.opt_expert = torch.optim.Adam(
            [
                {"params": self.encoder_priv.parameters()},
                {"params": self._actor_expert.parameters()},
                {"params": self._critic_priv.parameters()},
                {"params": self._critic_obs.parameters()},
                {"params": self._critic_aux.parameters()},
                {"params": self._marg.parameters()}
            ],
            lr=cfg.lr
        )

        self.opt_target = torch.optim.Adam(
            [
                {"params": self._actor_adapt.parameters()},
                {"params": self._critic_adapt.parameters()},
            ],
            lr=cfg.lr
        )

        self.opt_adapt = torch.optim.Adam(
            [
                {"params": self.adapt_module_a.parameters(), "name": "adapt_module_a", "max_grad_norm": 10.},
                {"params": self.adapt_module_b.parameters(), "name": "adapt_module_b", "max_grad_norm": 20.},
            ],
            lr=cfg.lr
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
            elif isinstance(module, nn.GRUCell):
                nn.init.orthogonal_(module.weight_hh)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=True)
        elif self.cfg.orthogonal_init:
            self.actor_critic_priv.apply(init_)
            self._critic_obs.apply(init_)
            self._critic_aux.apply(init_)
            self.adapt_module_a.apply(init_)
            self.adapt_module_b.apply(init_)
        
        self.__actor_expert(fake_input)
        self.__actor_expert.requires_grad_(False)
        hard_copy_(self._actor_expert, self.__actor_expert)

        self.adapt_module_ema.requires_grad_(False)
        hard_copy_(self.adapt_modules[self.cfg.adapt_module], self.adapt_module_ema)

        self.log_alpha.data.zero_()
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        return TensorDictPrimer(
            {"context_adapt_hx": UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)},
            reset_key="done"
        )

    def get_rollout_policy(self, mode: str="train"):
        if mode not in ("train", "eval", "deploy"):
            raise ValueError(mode)
        
        modules = []
        exclude_keys = ["actor_input", "actor_feature", "loc", "scale"]
        
        if mode in ("train", "eval"):
            modules.append(self.encoder_priv)
        modules.append(self.adapt_module_ema)

        if self.phase == "train":
            modules.append(self._actor_expert)
        elif self.phase == "adapt" or self.phase == "finetune":
            modules.append(self._actor_adapt)
        
        if mode == "eval":
            modules.append(self._critic_priv)
            modules.append(self._critic_adapt)
            exclude_keys.append("critic_priv_input")
            exclude_keys.append("critic_adapt_input")
        
        policy = TensorDictSequential(*modules, ExcludeTransform(*exclude_keys))
        return policy
    
    def train_op(self, tensordict: TensorDict):
        infos = {}
        match self.phase:
            case "train":
                infos.update(self.train_expert(tensordict.copy()))
                infos.update(self.train_adaptation(tensordict.copy()))
            case "adapt":
                infos.update(self.train_adaptation(tensordict.copy()))
                infos.update(self.train_target(tensordict.copy(), train_actor=False))
            case "finetune":
                infos.update(self.train_adaptation(tensordict.copy()))
                infos.update(self.train_target(tensordict.copy(), train_actor=True))
            case _:
                raise NotImplementedError
        self.num_updates += 1
        return infos
    
    def train_expert(self, tensordict: TensorDict):
        infos = []
        
        with torch.no_grad():
            with tensordict.view(-1) as _tensordict:
                self.encoder_priv(_tensordict["next"])
            
            self._compute_advantage(
                tensordict, self._critic_obs, "value_obs", "adv_obs", "ret_obs", value_norm=self.value_norms["obs"])
            self._compute_advantage(
                tensordict, self._critic_priv, "value_priv", "adv_priv", "ret_priv", value_norm=self.value_norms["priv"])
            
            value_obs = self.value_norms["obs"].denormalize(tensordict["value_obs"])
            value_priv = self.value_norms["priv"].denormalize(tensordict["value_priv"])
            value_gap = value_obs - value_priv
            adv_gap = tensordict["adv_obs"] - tensordict["adv_priv"]
            tensordict["value_gap"] = value_gap
            tensordict["adv_gap"] = adv_gap
        
        # save some memory?
        del tensordict["next"]
        del tensordict["critic_priv_input"]

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                self.encoder_priv(minibatch)
                losses = {}
                minibatch["adv_priv"] = normalize(minibatch["adv_mixed"], True)
                losses["policy_loss"], losses["entropy_loss"] = self._policy_loss(
                    minibatch, self._actor_expert, "adv_priv")
                losses["value_loss/obs"] = self._value_loss(
                    minibatch, self._critic_obs, "value_obs", "ret_obs").mean()
                losses["value_loss/priv"] = self._value_loss(
                    minibatch, self._critic_priv, "value_priv", "ret_priv").mean()

                loss = sum(v for k, v in losses.items())
                self.opt_expert.zero_grad(set_to_none=True)
                loss.backward()
                for param_group in self.opt_expert.param_groups:
                    nn.utils.clip_grad_norm_(param_group["params"], 2.)
                self.opt_expert.step()

                infos.append(TensorDict(losses, []))
        
        infos = {k: v.mean() for k, v in torch.stack(infos).items(True, True)}
        
        hard_copy_(self._actor_expert, self.__actor_expert)
        hard_copy_(self._actor_expert, self._actor_adapt)
        
        infos["value_gap"] = tensordict["value_gap"].square().mean()
        infos["adv_gap"] = tensordict["adv_gap"].square().mean()
        infos["value_obs"] = tensordict["value_obs"].mean()
        infos["value_priv"] = tensordict["value_priv"].mean()

        return {k: v.item() for k, v in sorted(infos.items())}
    
    def train_target(self, tensordict: TensorDict, train_actor: bool):
        infos = []
        keys = tensordict.keys(True, True)
        with torch.no_grad(), tensordict.view(-1) as _tensordict:
            self.encoder_priv(_tensordict)
            self.encoder_priv(_tensordict["next"])
            self.adapt_module_ema(_tensordict["next"])
            del _tensordict["next", "next"]
            action_kl = self._action_kl(_tensordict.copy(), self.adapt_module_a, reduce=False)
            if self.cfg.adapt_reward:
                _tensordict[REWARD_KEY] = _tensordict[REWARD_KEY] - self.log_alpha.exp() * action_kl

        self._compute_advantage(
            tensordict, self._critic_priv, "value_priv", "adv_priv", "ret_priv", value_norm=self.value_norms["priv"])
        self._compute_advantage(
            tensordict, self._critic_adapt, "value_adapt", "adv_adapt", "ret_adapt", value_norm=self.value_norms["adapt"])
        tensordict.select(*keys, inplace=True)

        # save some memory?
        del tensordict["next"]
        del tensordict["critic_priv_input"]

        if self.cfg.tune_alpha:
            # alpha_loss = - self.log_alpha * (action_kl - self.target_kl)
            alpha_loss = - self.log_alpha.exp() * (action_kl.mean() - self.target_kl)
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()

        actor = self._actor_adapt if train_actor else None
        if self.cfg.unbiased_critic:
            adv_key = "adv_adapt"
        else:
            adv_key = "adv_priv"
        
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                losses = {}
                if actor is not None:
                    minibatch[adv_key] = normalize(minibatch[adv_key])
                    losses["policy_loss"], losses["entropy_loss"] = self._policy_loss(
                        minibatch, self._actor_adapt, adv_key)
                losses["value_loss/priv"] = self._value_loss(
                    minibatch, self._critic_priv, "value_priv", "ret_priv").mean()
                losses["value_loss/adapt"] = self._value_loss(
                    minibatch, self._critic_adapt, "value_adapt", "ret_adapt").mean()
                loss = sum(v for k, v in losses.items() if "loss" in k)
                
                self.opt_expert.zero_grad(set_to_none=True)
                self.opt_target.zero_grad(set_to_none=True)
                loss.backward()
                for param_group in self.opt_expert.param_groups:
                    nn.utils.clip_grad_norm_(param_group["params"], 2.)
                self.opt_expert.step()
                self.opt_target.step()
                infos.append(TensorDict(losses, []))
        
        infos = {k: v.mean() for k, v in torch.stack(infos).items(True, True)}
        value_adapt = self.value_norms["adapt"].denormalize(tensordict["value_adapt"])
        value_priv = self.value_norms["priv"].denormalize(tensordict["value_priv"])
        infos["value_gap"] = (value_adapt - value_priv)[~tensordict["is_init"]].square().mean()
        infos["value_adapt"] = value_adapt.mean()
        infos["value_priv"] = value_priv.mean()
        infos["alpha"] = self.log_alpha.exp()
        return {k: v.item() for k, v in sorted(infos.items())}
    
    def train_adaptation(self, tensordict: TensorDict):
        infos = []
        with torch.no_grad():
            self.encoder_priv(tensordict)
            # tensordict["critic_feature"] = torch.cat([tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY]], dim=-1)
            # tensordict["action_kl"] = self._action_kl(tensordict.copy(), self.adapt_module_b, reduce=False)
        
        for epoch in range(2):
            batch = make_batch(tensordict, 8, self.cfg.train_every)
            for minibatch in batch:
                infos.append(self._update_adaptation(minibatch))
        
        # hard_copy_(self.adapt_modules[self.cfg.adapt_module], self.adapt_module_ema)
        soft_copy_(self.adapt_modules[self.cfg.adapt_module], self.adapt_module_ema, 0.05)
    
        infos = {f"adapt/{k}": v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        with torch.no_grad():
            infos["adapt/adapt_module_a_kl"] = self._action_kl(tensordict.copy(), self.adapt_module_a).item()
            infos["adapt/adapt_module_b_kl"] = self._action_kl(tensordict.copy(), self.adapt_module_b).item()
        return infos

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TensorDictModule, 
        value_key: str="value",
        adv_key: str="adv",
        ret_key: str="ret",
        rew_key: str=REWARD_KEY,
        value_norm: ValueNorm1=None,
    ):
        values = critic(tensordict)[value_key]
        next_values = critic(tensordict["next"])[value_key]

        rewards = tensordict[rew_key]
        dones = tensordict[DONE_KEY]
        if value_norm is not None:
            values = value_norm.denormalize(values)
            next_values = value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        # adv = normalize(adv, subtract_mean=True)

        if value_norm is not None:
            value_norm.update(ret)
            ret = value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update_adaptation(self, tensordict: TensorDictBase):
        losses = TensorDict({}, [])
        losses["adapt_module_a_loss"] = self._feature_mse(tensordict.copy(), self.adapt_module_a)
        # losses["adapt_module_a_loss"] = self._nll(tensordict.copy(), self.adapt_module_a)
        losses["adapt_module_b_loss"] = (
            self._action_kl(tensordict.copy(), self.adapt_module_b)
            + self._feature_mse(tensordict.copy(), self.adapt_module_b)
        )
        self.opt_adapt.zero_grad()
        sum(v for k, v in losses.items() if k.endswith("loss")).backward()
        for param_group in self.opt_adapt.param_groups:
            grad_norm = nn.utils.clip_grad_norm_(param_group["params"], param_group["max_grad_norm"])
            losses[param_group["name"] + "_grad_norm"] = grad_norm
        self.opt_adapt.step()
        return losses

    def _feature_mse(self, tensordict: TensorDictBase, adapt_module: TensorDictModule, reduce: bool=True):
        adapt_module(tensordict)
        if reduce:
            return F.mse_loss(tensordict["context_adapt"], tensordict["context_expert"])
        else:
            return F.mse_loss(tensordict["context_adapt"], tensordict["context_expert"], reduction="none").mean(-1, True)
    
    def _nll(self, tensordict: TensorDictBase, adapt_module: TensorDictModule, reduce: bool=True):
        adapt_module(tensordict)
        dist = D.Normal(tensordict["context_adapt_loc"], tensordict["context_adapt_std"])
        nll = -dist.log_prob(tensordict["context_expert"]).sum(-1, True)
        if reduce:
            nll = nll.mean()
        return nll

    def _action_kl(self, tensordict: TensorDictBase, adapt_module: TensorDictModule, reduce: bool=True):
        with torch.no_grad():
            dist1 = self._actor_expert.get_dist(tensordict)
        dist2 = self.__actor_expert.get_dist(adapt_module(tensordict))
        kl = D.kl_divergence(dist2, dist1).unsqueeze(-1)
        if reduce:
            kl = kl.mean()
        return kl

    def _policy_loss(
        self,
        tensordict: TensorDictBase,
        actor: ProbabilisticActor,
        adv_key: str
    ):
        dist = actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()
        adv = tensordict[adv_key]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * entropy
        return policy_loss, entropy_loss

    def _value_loss(
        self, 
        tensordict: TensorDictBase,
        critic: TensorDictModule,
        value_key: str,
        ret_key: str
    ):
        value_loss = self.critic_loss_fn(
            critic(tensordict)[value_key],
            tensordict[ret_key]
        ) * (~tensordict["is_init"])
        return value_loss

    def state_dict(self):
        # state_dict = super().state_dict()
        # if "vecnorm._extra_state" in state_dict:
        #     state_dict["vecnorm._extra_state"]["lock"] = None # TODO: check with torchrl
        # state_dict["num_frames"] = self.num_frames
        # state_dict["phase"] = self.phase
        # return state_dict
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        if "vecnorm" in state_dict:
            state_dict["vecnorm"]["_extra_state"]["lock"] = None # TODO: check with torchrl
        state_dict["num_frames"] = self.num_frames
        state_dict["phase"] = self.phase
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        # self.num_frames = state_dict.get("num_frames", 0)
        # self.last_phase = state_dict.get("phase", None)
        # if "vecnorm._extra_state" in state_dict:
        #     vecnorm_td = state_dict["vecnorm._extra_state"]["td"]
        #     state_dict["lock"] = None
        #     state_dict["vecnorm._extra_state"]["td"] = vecnorm_td.to(self.device)
        # return super().load_state_dict(state_dict, strict=strict)
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
        self.num_frames = state_dict["num_frames"]
        self.last_phase = state_dict["phase"]
        return failed_keys

