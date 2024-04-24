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

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, TensorDictPrimer, ExcludeTransform
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Mapping, Union

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
    clip_param: float = 0.2

    actor_predict_std: bool = True
    orthogonal_init: bool = True
    layer_norm: bool = True
    value_norm: bool = False

    aux_reward: float = 0.
    aux_epochs: int = -1

    context_dim: int = 128
    adapt_loss: str = "feature_mse"
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


class ClassifierA(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(1))
    
    def forward(self, obs, priv, action):
        return self.layers(torch.cat([obs, priv, action], dim=-1))


class ClassifierB(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embed = nn.Sequential(make_mlp([512, 256]), nn.LazyLinear(64))
        self.action_embed = nn.Sequential(make_mlp([128, 64]), nn.LazyLinear(64))
    
    def forward(self, obs, priv, action):
        input_embed = self.input_embed(torch.cat([obs, priv], dim=-1))
        action_embed = self.action_embed(action)
        input_embed = input_embed / input_embed.norm(dim=-1, keepdim=True).clamp(1e-7)
        action_embed = action_embed / action_embed.norm(dim=-1, keepdim=True).clamp(1e-7)
        return (input_embed * action_embed).sum(dim=-1, keepdim=True)


class PPOAdaptPolicy(TensorDictModuleBase):
    
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

        self.entropy_coef = 0.001
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

        self.observation_spec = observation_spec
        fake_input = observation_spec.zero()
        print(fake_input.shapes)

        self.log_alpha = nn.Parameter(torch.tensor(0.))
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=1e-2)
        self.target_kl = self.cfg.target_kl
        self.context_dim = self.cfg.context_dim
        
        if "height_scan" in observation_spec.keys(True, True):
            # the conv_encoder is shared between actor and critic for efficiency
            self.context_dim = self.cfg.context_dim + 16
            conv_encoder = nn.Sequential(make_conv([8, 16], kernel_sizes=[5, 3]), nn.LazyLinear(16), nn.Mish())
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
        
        def make_adapt_module():
            return TensorDictModule(
                GRUModule(self.context_dim), 
                [OBS_KEY, "is_init", "context_adapt_hx"], 
                ["context_adapt", ("next", "context_adapt_hx")]
            ).to(self.device)
        
        self.adapt_module_a = make_adapt_module()
        self.adapt_module_b = make_adapt_module()
        self.adapt_module_ema = make_adapt_module()
        
        def make_actor(context_key: str) -> ProbabilisticActor:
            actor = ProbabilisticActor(
                module=TensorDictSequential(
                    CatTensors([OBS_KEY, context_key], "actor_input", del_keys=False),
                    TensorDictModule(make_mlp([256, 256, 256]), ["actor_input"], ["actor_feature"]),
                    TensorDictModule(Actor(self.action_dim, self.cfg.actor_predict_std), ["actor_feature"], ["loc", "scale"]),
                    TensorDictModule(nn.LazyLinear(1), ["actor_feature"], ["actor_aux"]),
                ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True
            ).to(self.device)
            return actor
        
        def make_critic():
            return nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1))
        
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
        self._aux_pred = TensorDictModule(
            make_critic(), ["critic_priv_input"], ["aux_pred"]
        ).to(self.device)
        self._critic_aux = TensorDictModule(
            make_critic(), ["critic_priv_input"], ["value_aux"]
        ).to(self.device)

        # expert critic with both privilged and estimated information to be unbiased
        self._critic_adapt = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY, "context_adapt"], "critic_adapt_input", del_keys=False),
            TensorDictModule(make_critic(), ["critic_adapt_input"], ["value_adapt"])
        ).to(self.device)

        self.actor_critic_priv = TensorDictSequential(
            self.encoder_priv,
            self._actor_expert,
            self._critic_priv
        )

        # self.inverse_pred = TensorDictSequential(
        #     CatTensors([OBS_KEY, ACTION_KEY], "obs_action", del_keys=False),
        #     TensorDictModule(
        #         nn.Sequential(
        #             make_mlp([256, 256]), 
        #             nn.LazyLinear(2 * observation_spec["params"].shape[-1]),
        #             Chunk(2)
        #         ),
        #         ["obs_action"], ["params_loc", "params_scale"]
        #     )
        # ).to(self.device)
        # with torch.device(self.device):
        #     self.f_oa = nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(32))
        #     self.f_params = nn.Sequential(make_mlp([256]), nn.LazyLinear(32))
        #     self.f_oa(torch.zeros(self.observation_spec["policy"].shape[-1]+self.action_dim))
        #     self.f_params(torch.zeros(self.observation_spec["params"].shape[-1]))

        # lazy initialization
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["context_adapt_hx"] = torch.zeros(fake_input.shape[0], 128)
        
        self.encoder_priv(fake_input)
        self.adapt_module_a(fake_input)
        self.adapt_module_b(fake_input)
        self.adapt_module_ema(fake_input)
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
                {"params": self._aux_pred.parameters()},
                {"params": self._critic_priv.parameters()},
                {"params": self._critic_obs.parameters()},
                {"params": self._critic_aux.parameters()},
            ],
            lr=cfg.lr
        )

        self.opt_aux = torch.optim.Adam(
            [
                {"params": self.encoder_priv.parameters()},
                {"params": self._actor_expert.parameters()},
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
                {"params": self.adapt_module_a.parameters(), "name": "adapt_module_a", "max_grad_norm": 5.},
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
            self.load_state_dict(state_dict, strict=False)
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
        hard_copy_(self.adapt_module_a, self.adapt_module_ema)

        self.log_alpha.data.zero_()
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        return TensorDictPrimer(
            {"context_adapt_hx": UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)},
            reset_key="done"
        )

    def get_rollout_policy(self, mode: str="train"):
        if self.phase == "train":
            policy = TensorDictSequential(
                self.encoder_priv,
                self.adapt_module_ema,
                self._actor_expert,
                ExcludeTransform("actor_input", "actor_feature", "loc", "scale")
            )
        elif self.phase == "adapt" or self.phase == "finetune":
            policy = TensorDictSequential(
                self.adapt_module_ema,
                self._actor_adapt,
                ExcludeTransform("actor_input", "actor_feature", "loc", "scale")
            )
        else:
            raise NotImplementedError
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
        with torch.no_grad(), tensordict.view(-1) as _tensordict:
            self.encoder_priv(_tensordict["next"])

            self._compute_advantage(
                tensordict, self._critic_obs, "value_obs", "adv_obs", "ret_obs", value_norm=self.value_norms["obs"])
            self._compute_advantage(
                tensordict, self._critic_priv, "value_priv", "adv_priv", "ret_priv", value_norm=self.value_norms["priv"])
            
            value_obs = self.value_norms["obs"].denormalize(tensordict["value_obs"])
            value_priv = self.value_norms["priv"].denormalize(tensordict["value_priv"])
            aux_pred = self._aux_pred(tensordict)["aux_pred"]
            tensordict["value_gap"] = value_obs - value_priv
            tensordict["value_gap_error"] = (aux_pred.sign() != tensordict["value_gap"].sign())
            self._compute_advantage(
                tensordict, self._critic_aux, "value_aux", "adv_aux", "ret_aux", "value_gap_error")
            tensordict["adv_mixed"] = tensordict["adv_priv"] + self.cfg.aux_reward * tensordict["adv_aux"]
        
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
                losses["value_loss/aux"] = self._value_loss(
                    minibatch, self._critic_aux, "value_aux", "ret_aux").mean()
                losses["marg_pred_loss"] = F.mse_loss(
                    self._aux_pred(minibatch)["aux_pred"],
                    minibatch["value_gap"]
                )

                # self.inverse_pred(minibatch)
                # Q = D.Normal(minibatch["params_loc"], minibatch["params_scale"].exp())
                # losses["inverse_pred_loss"] = - Q.log_prob(minibatch["params"]).mean()

                # oa = torch.cat([minibatch[OBS_KEY], minibatch[ACTION_KEY]], -1)
                # e_oa = F.normalize(self.f_oa(oa))
                # e_params = F.normalize(self.f_params(minibatch["params"]))
                # # e_params_ = F.normalize(self.f_params(minibatch["params"] + torch.randn_like(minibatch["params"]) * 0.1))
                # e_params_ = F.normalize(self.f_params(minibatch["params"][torch.randperm(minibatch.shape[0], device=self.device)]))
                # losses["nce_loss"] = (
                #     F.binary_cross_entropy_with_logits(dot(e_oa, e_params), torch.ones(minibatch.shape[0], device=self.device))
                #     + F.binary_cross_entropy_with_logits(dot(e_oa, e_params_), torch.zeros(minibatch.shape[0], device=self.device))
                # )

                loss = sum(v for k, v in losses.items())
                self.opt_expert.zero_grad(set_to_none=True)
                loss.backward()
                for param_group in self.opt_expert.param_groups:
                    nn.utils.clip_grad_norm_(param_group["params"], 5)
                self.opt_expert.step()

                infos.append(TensorDict(losses, []))
        
        infos = {k: v.mean() for k, v in torch.stack(infos).items(True, True)}
        
        if self.cfg.aux_epochs > 0 and self.num_updates % 2 == 0:
            infos_aux = []
            with torch.no_grad():
                self.encoder_priv(tensordict)
                self._actor_expert.get_dist_params(tensordict)
            for epoch in range(self.cfg.aux_epochs):
                for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                    self.encoder_priv(minibatch)
                    infos_aux.append(self._update_aux(minibatch))
            infos_aux = {k: v.mean() for k, v in torch.stack(infos_aux).items(True, True)}
            infos.update(infos_aux)
        
        hard_copy_(self._actor_expert, self.__actor_expert)
        hard_copy_(self._actor_expert, self._actor_adapt)
        
        infos["value_gap"] = tensordict["value_gap"].square().mean()
        infos["value_obs"] = value_obs.mean()
        infos["value_priv"] = value_priv.mean()
        infos["value_aux"] = tensordict["value_aux"].mean()
        infos["value_gap_acc"] = 1. - tensordict["value_gap_error"].float().mean()
        return {k: v.item() for k, v in sorted(infos.items())}
    
    def train_target(self, tensordict: TensorDict, train_actor: bool):
        infos = []
        keys = tensordict.keys(True, True)
        with torch.no_grad(), tensordict.view(-1) as _tensordict:
            self.encoder_priv(_tensordict)
            self.encoder_priv(_tensordict["next"])
            self.adapt_module_ema(_tensordict["next"])
            del _tensordict["next", "next"]
            action_kl = self._action_kl(_tensordict.to_tensordict(), self.adapt_module_a, reduce=False)
            if self.cfg.adapt_reward:
                _tensordict[REWARD_KEY] = _tensordict[REWARD_KEY] - self.log_alpha.exp() * action_kl

        self._compute_advantage(
            tensordict, self._critic_priv, "value_priv", "adv_priv", "ret_priv", self.value_norms["priv"])
        self._compute_advantage(
            tensordict, self._critic_adapt, "value_adapt", "adv_adapt", "ret_adapt", self.value_norms["adapt"])
        tensordict.select(*keys, inplace=True)

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
                # with torch.no_grad():
                #     minibatch = self.encoder_priv(minibatch).detach()
                #     minibatch = self.adapt_module_ema(minibatch).detach()
                losses = {}
                if actor is not None:
                    losses["policy_loss"], losses["entropy_loss"] = self._policy_loss(
                        minibatch, self._actor_adapt, adv_key)
                losses["priv_value_loss"] = self._value_loss(
                    minibatch, self._critic_priv, "value_priv", "ret_priv").mean()
                losses["adapt_value_loss"] = self._value_loss(
                    minibatch, self._critic_adapt, "value_adapt", "ret_adapt").mean()
                loss = sum(v for k, v in losses.items() if k.endswith("loss"))
                
                self.opt_expert.zero_grad(set_to_none=True)
                self.opt_target.zero_grad(set_to_none=True)
                loss.backward()
                for param_group in self.opt_expert.param_groups:
                    nn.utils.clip_grad_norm_(param_group["params"], 5)
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
            tensordict["critic_feature"] = torch.cat([tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY]], dim=-1)
            tensordict["action_kl"] = self._action_kl(tensordict.copy(), self.adapt_module_b, reduce=False)
        
        for epoch in range(2):
            batch = make_batch(tensordict, 8, self.cfg.train_every)
            for minibatch in batch:
                infos.append(self._update_adaptation(minibatch))
        
        # hard_copy_(self.adapt_module_a, self.adapt_module_ema)
        soft_copy_(self.adapt_module_a, self.adapt_module_ema, 0.05)
    
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
        losses["adapt_module_a_loss"] = self._feature_mse(tensordict.to_tensordict(), self.adapt_module_a)
        losses["adapt_module_b_loss"] = (
            self._action_kl(tensordict.to_tensordict(), self.adapt_module_b)
            + self._feature_mse(tensordict.to_tensordict(), self.adapt_module_b)
        )
        self.opt_adapt.zero_grad()
        sum(v for k, v in losses.items() if k.endswith("loss")).backward()
        for param_group in self.opt_adapt.param_groups:
            grad_norm = nn.utils.clip_grad_norm_(param_group["params"], param_group["max_grad_norm"])
            losses[param_group["name"] + "_grad_norm"] = grad_norm
        self.opt_adapt.step()
        return losses
    
    def _update_aux(self, tensordict: TensorDictBase, ret_key: str="ret_expert"):
        losses = {}
        dist_old = self._actor_expert.build_dist_from_params(tensordict.to_tensordict())
        dist_new = self._actor_expert.get_dist(tensordict)
        losses["aux/kl_loss"] = D.kl_divergence(dist_old, dist_new)
        losses["aux/value_loss"] = (
            self.critic_loss_fn(tensordict["actor_value"], tensordict[ret_key]) * (~tensordict["is_init"])
        )
        loss = sum(v.mean() for k, v in losses.items())
        self.opt_aux.zero_grad()
        loss.backward()
        for param_group in self.opt_aux.param_groups:
            nn.utils.clip_grad_norm_(param_group["params"], 5)
        self.opt_aux.step()
        return TensorDict(losses, [])

    def _feature_mse(self, tensordict: TensorDictBase, adapt_module: TensorDictModule, reduce: bool=True):
        adapt_module(tensordict)
        if reduce:
            return F.mse_loss(tensordict["context_adapt"], tensordict["context_expert"])
        else:
            return F.mse_loss(tensordict["context_adapt"], tensordict["context_expert"], reduction="none").mean(-1, True)
    
    def _action_kl(self, tensordict: TensorDictBase, adapt_module: TensorDictModule, reduce: bool=True):
        with torch.no_grad():
            dist1 = self._actor_expert.get_dist(tensordict)
        dist2 = self.__actor_expert.get_dist(adapt_module(tensordict))
        kl = D.kl_divergence(dist2, dist1).unsqueeze(-1)
        if reduce:
            kl = kl.mean()
        return kl

    def _least_square(self, true: torch.Tensor, false: torch.Tensor):
        loss_true = F.mse_loss(true, torch.ones_like(true))
        loss_false = F.mse_loss(false, -torch.ones_like(false))
        acc = ((true > 0).float().mean() + (false < 0).float().mean()) / 2
        return loss_true + loss_false, acc
    
    def _cross_entropy(self, true: torch.Tensor, false: torch.Tensor):
        loss_true = F.binary_cross_entropy_with_logits(true, torch.ones_like(true))
        loss_false = F.binary_cross_entropy_with_logits(false, torch.zeros_like(false))
        acc = ((true.sigmoid() > 0.5).float().mean() + (false.sigmoid() < 0.5).float().mean()) / 2
        return loss_true + loss_false, acc

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
        state_dict = super().state_dict()
        state_dict["num_frames"] = self.num_frames
        state_dict["phase"] = self.phase
        del state_dict["log_alpha"]
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        self.num_frames = state_dict.get("num_frames", 0)
        self.last_phase = state_dict.get("phase", None)
        return super().load_state_dict(state_dict, strict=strict)

def dot(a, b):
    return (a * b).sum(-1)
