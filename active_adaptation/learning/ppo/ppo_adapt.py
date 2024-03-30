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

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor
from torchrl.objectives.utils import hold_out_net
from torchrl.envs.transforms.transforms import TensorDictPrimer
from torchrl.data import UnboundedContinuousTensorSpec, replay_buffers as rb

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    TensorDictModule, 
    TensorDictSequential, 
    TensorDictModuleBase,
    ProbabilisticTensorDictSequential,
    ProbabilisticTensorDictModule,
    utils
)
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, MISSING
from typing import Any, Iterator, Mapping, Union, Sequence

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .adaptation import Action, Value, ActionValue, MSE
from .ppo_rnn import GRU
from .common import *

from active_adaptation.utils.wandb import parse_path
import copy
import logging

torch.set_float32_matmul_precision('high')

@dataclass
class PPOConfig:
    name: str = "ppo_rma"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    recompute_adv: bool = True
    predict_std: bool = True
    discard_init: bool = True

    checkpoint_path: Union[str, None] = None
    phase: str = "train"
    condition_mode: str = "cat"
    expert_reg: float = 0.0
    norm_context: bool = False

    adapt_arch: str = "rnn"
    # what the adaptation module learns to predict
    adaptation_key: Any = "context"
    adaptation_loss: str = "mse" # mse, action_kl
    use_separate_critics: bool = False

    # coefficients for using adaptation error as exploration bonus or penalty
    exp_reward: float = 0.0
    reg_reward: float = 0.0

    def __post_init__(self):
        assert self.condition_mode.lower() in ("cat", "film")
        assert self.adaptation_key in ("context", OBS_HIST_KEY, "_feature")
        assert self.phase in ("train", "adapt", "finetune")

cs = ConfigStore.instance()
cs.store("rma_train", node=PPOConfig, group="algo")
cs.store("rma_adapt", node=PPOConfig(phase="adapt"), group="algo")
cs.store("rma_adapt_rnn", node=PPOConfig(adapt_arch="rnn", phase="adapt"), group="algo")
cs.store("rma_finetune", node=PPOConfig(phase="finetune"), group="algo")
cs.store("rma_finetune_rnn", node=PPOConfig(adapt_arch="rnn", phase="finetune"), group="algo")

class TConv(nn.Module):
    def __init__(self, out_dim: int, activation=nn.Mish) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.tconv = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1), activation(),
            nn.LazyConv1d(64, kernel_size=7, stride=2), activation(),
            nn.LazyConv1d(64, kernel_size=5, stride=2), activation(),
        )
        # self.mlp = make_mlp([256, out_dim], activation=nn.Mish)
        self.mlp = nn.Sequential(make_mlp([256]), nn.LazyLinear(out_dim))
    
    def forward(self, features: torch.Tensor):
        batch_shape = features.shape[:-2]
        features = features.reshape(-1, *features.shape[-2:])
        features_tconv = einops.rearrange(self.tconv(features), "b d t -> b (t d)")
        features = torch.cat([features_tconv, features[:, :, -1]], dim=1)
        features = self.mlp(features)
        return features.reshape(*batch_shape, *features.shape[1:])


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

class GRUStochModule(nn.Module):
    def __init__(self, output_dim: int, latent_dim: int = 64):
        super().__init__()
        self.mlp = make_mlp([128, 128])
        self.gru = GRU(128, 128, allow_none=False)
        self.proj = nn.LazyLinear(latent_dim * 2)
        self.decoder = nn.Sequential(make_mlp([128], nn.LazyLinear(output_dim)))
        
    def forward(self, x, is_init, hx):
        x = self.mlp(x)
        x, hx = self.gru(x, is_init, hx)
        mu, logvar = self.proj(x).chunk(2, dim=-1)
        sample = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return sample, mu, logvar, hx


class FiLM(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.f = nn.LazyLinear(feature_dim * 2)
        self.act = nn.ELU()
        self.ln = nn.LayerNorm(feature_dim)
    
    def forward(self, feature, context):
        w, b = self.f(context).chunk(2, dim=-1)
        feature = self.act(w * feature + b) + feature
        return feature


def make_adaptation_module(
    adapt_arch: str, 
    outpur_dim: int, 
    output_key="context_adapt"
):
    if adapt_arch == "tconv":
        module = TensorDictModule(TConv(outpur_dim), [OBS_HIST_KEY], [output_key])
    elif adapt_arch == "rnn":
        in_keys = [OBS_KEY, "is_init", f"{output_key}_hx"]
        out_keys = [output_key, ("next", f"{output_key}_hx")]
        gru = GRUModule(outpur_dim)
        module = TensorDictModule(gru, in_keys, out_keys)
    elif adapt_arch == "rnn_stoch":
        in_keys = [OBS_KEY, "is_init", f"{output_key}_hx"]
        out_keys = ["context_adapt", "mu", "logvar", ("next", f"{output_key}_hx")]
        module = TensorDictModule(GRUStochModule(outpur_dim), in_keys, out_keys)
    else:
        raise NotImplementedError(adapt_arch)
    return module


def make_state_estimator(arch: str, dim: int):
    if arch == "tconv":
        module = TensorDictModule(TConv(dim), [OBS_HIST_KEY], [OBS_PRIV_KEY])
    elif arch == "rnn":
        in_keys = [OBS_KEY, "is_init", "_hx"]
        out_keys = [OBS_PRIV_KEY, ("next", f"_hx")]
        gru = GRUModule(dim)
        return TensorDictModule(gru, in_keys, out_keys)
    return module


def condition(expert: bool, mode: str, dim: int=256):
    module = nn.Sequential(make_mlp([dim]))
    in_keys = ["_feature", "context_expert" if expert else "context_adapt"]
    # out_key = "_feature_expert" if expert else "_feature_adapt"
    out_key = "_feature"
    if mode == "cat":
        return TensorDictSequential(
            TensorDictModule(module, [OBS_KEY], ["_feature"]),
            CatTensors(in_keys, out_key, del_keys=False)
        )
    elif mode == "film":
        return TensorDictSequential(
            TensorDictModule(module, [OBS_KEY], ["_feature"]),
            TensorDictModule(FiLM(256), in_keys, [out_key])
        )

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1), nn.Mish(),
            nn.LazyConv2d(16, kernel_size=3, stride=2, padding=1), nn.Mish(),
            nn.Flatten(),
            make_mlp([128])
        )
        self.mlp = make_mlp([256, 128])

    def forward(self, propri, height_scan):
        enc = torch.cat([self.mlp(propri), self.conv(height_scan)], -1)
        return enc


class PPORMAPolicy(TensorDictModuleBase):
    
    @utils._set_auto_make_functional(False)
    def __init__(self,
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
        self.clip_param = 0.2
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.adaptation_key = self.cfg.adaptation_key
        self.phase = self.cfg.phase
        
        self.num_rewards = 5
        self.value_names = ["value_mean", "value_mse", "value_kl", "value_D(student)", "value_D(expert*)"]

        self.exp_reward = self.cfg.exp_reward
        self.reg_reward = self.cfg.reg_reward

        if not isinstance(self.adaptation_key, str):
            self.adaptation_key = tuple(self.adaptation_key)
        self.gae_expert = GAE((0.995, 0.99, 0.99), 0.95).to(self.device)
        self.gae_adapt = GAE((0.995,) + (0.99,) * (self.num_rewards - 1), 0.95).to(self.device)
        
        self.action_dim = action_spec.shape[-1]

        print(observation_spec)
        observation_priv_dim = observation_spec[OBS_PRIV_KEY].shape[-1]
        observation_dim = observation_spec[OBS_KEY].shape[-1]

        fake_input = observation_spec.zero()

        # create and initialize the modules
        if "height_scan" in observation_spec.keys():
            self.context_dim = 256
            self.encoder = TensorDictModule(
                Encoder(), [OBS_PRIV_KEY, "height_scan"], ["context_expert"]
            ).to(self.device)
        else:
            self.context_dim = 128
            self.encoder = TensorDictModule(
                make_mlp([256, self.context_dim]), [OBS_PRIV_KEY], ["context_expert"]
            ).to(self.device)

        self.actor_expert: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                condition(expert=True, mode=cfg.condition_mode),
                TensorDictModule(
                    nn.Sequential(
                        make_mlp([256, 256]), 
                        Actor(self.action_dim, predict_std=cfg.predict_std)
                    ), 
                    ["_feature"], ["loc", "scale"]
                )
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.actor_target: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                condition(expert=False, mode=self.cfg.condition_mode),
                TensorDictModule(
                    nn.Sequential(
                        make_mlp([256, 256]), 
                        Actor(self.action_dim, predict_std=self.cfg.predict_std)
                    ), 
                    ["_feature"], ["loc", "scale"]
                )
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        # the critics always observe priviledged information
        self.critic_expert = TensorDictSequential(
            condition(expert=True, mode=cfg.condition_mode),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 128]), nn.LazyLinear(3)), 
                ["_feature"], ["state_value"]
            )
        ).to(self.device)

        self.value_norm_expert = ValueNorm1(3).to(self.device)
        self.value_norm_adapt = ValueNorm1(self.num_rewards).to(self.device)
        
        self.adapt_module = (
            make_adaptation_module(self.cfg.adapt_arch, self.context_dim)
            .to(self.device)
        )
        
        if self.cfg.norm_context:
            logging.info("Normalize context.")
            self.encoder = TensorDictSequential(
                self.encoder,
                TensorDictModule(SimNorm(8), ["context_expert"], ["context_expert"])
            )
            self.adapt_module = TensorDictSequential(
                self.adapt_module,
                TensorDictModule(SimNorm(8), ["context_adapt"], ["context_adapt"])
            )
        
        if self.cfg.adaptation_loss == "mse":
            self.adaptation_loss = MSE(self.adapt_module).to(self.device)
        elif self.cfg.adaptation_loss == "action_kl":
            self.adaptation_loss = Action(
                self.adapt_module,
                self.actor_expert,
                self.actor_target,
                closed_kl=True
            ).to(self.device)
        elif self.cfg.adaptation_loss == "elbo":
            self.adaptation_loss = ELBO(self.adapt_module).to(self.device)
        else:
            raise ValueError(self.cfg.adaptation_loss)

        if "rnn" in self.cfg.adapt_arch:
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool, device=self.device)
            fake_input["context_adapt_hx"] = torch.zeros(fake_input.shape[0], 128, device=self.device)
        
        self.encoder(fake_input)
        self.actor_expert(fake_input)
        self.critic_expert(fake_input)
        self.adapt_module(fake_input)
        self.actor_target(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor_expert.parameters()},
                {"params": self.critic_expert.parameters()},
                {"params": self.encoder.parameters()},
            ], 
            lr=self.cfg.lr
        )
        self.adapt_opt = torch.optim.Adam(
            [
                {"params": self.adapt_module.parameters()},
                # {"params": self.actor_adapt.parameters()},
            ],
            lr=self.cfg.lr,
        )

        if self.phase in ("adapt", "finetune"):
            self.make_adapt_policy(fake_input)
            logging.info("Disable grad tracking for experts.")
            self.encoder.requires_grad_(False)
            self.actor_expert.requires_grad_(False)
            
            self.classifier = Classifier().to(self.device)
            self._compute_disc_loss(fake_input)
            self.classifier.apply(init_)
            self.adapt_opt.add_param_group({"params": self.classifier.parameters(), "lr": 1e-3})

        checkpoint_path = parse_path(self.cfg.checkpoint_path)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            state_dict = {}
            self.actor_expert.apply(init_)
            self.critic_expert.apply(init_)
            self.encoder.apply(init_)
        
        self.adapt_module_ema = copy.deepcopy(self.adapt_module)
        self.adapt_module_ema.requires_grad_(False)

        # initialize the adaptation actor
        if (
            self.phase in ("adapt", "finetune") 
            and not any(key.startswith("actor_adapt") for key in state_dict.keys())
        ):
            logging.warning("Initialize the adaptation actor with the expert actor.")
            hard_copy_(self.actor_expert, self.actor_adapt)
        
        # perturb_(self.actor_adapt, 0.2)
        # initialize the target actor
        hard_copy_(self.actor_expert, self.actor_target)
        self.actor_target.requires_grad_(False)
        
        if "rnn" in cfg.adapt_arch:
            def make_tensordict_primer():
                num_envs = observation_spec.shape[0]
                return TensorDictPrimer({
                    "context_adapt_hx": UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)
                })
            self.make_tensordict_primer = make_tensordict_primer
            self.make_batch = functools.partial(make_batch, seq_len=cfg.train_every)
        else:
            self.make_batch = make_batch
        
        self.exclude_keys = ["_feature", "_feature_expert", "_feature_adapt"]
        self.train_iter = 0

    def get_rollout_policy(self, mode):
        if mode == "eval":
            policy = TensorDictSequential(
                self.adapt_module,
                self.actor_adapt,
            )
        return copy.deepcopy(policy)

    def make_adapt_policy(self, fake_input: TensorDictBase):

        self.actor_adapt: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                condition(expert=False, mode=self.cfg.condition_mode),
                TensorDictModule(
                    nn.Sequential(
                        make_mlp([256, 256]), 
                        Actor(self.action_dim, predict_std=self.cfg.predict_std)
                    ), 
                    ["_feature"], ["loc", "scale"]
                ),
                # TensorDictModule(nn.LazyLinear(3), ["_feature"], ["state_value_actor"])
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic_adapt = TensorDictSequential(
            TensorDictModule(make_mlp([256]), [OBS_KEY], ["_feature"]),
            CatTensors(
                ["context_adapt", "context_expert", "_feature"], 
                "_feature_critic", 
                del_keys=False
            ),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 128]), nn.LazyLinear(self.num_rewards)), 
                ["_feature_critic"], 
                ["state_value"]
            )
        ).to(self.device)
        # self.critic_expert = TensorDictSequential(
        #     condition(expert=True, mode=self.cfg.condition_mode),
        #     TensorDictModule(
        #         nn.Sequential(make_mlp([256, 128]), nn.LazyLinear(self.num_rewards)), 
        #         ["_feature"], ["state_value"]
        #     )
        # ).to(self.device)
        
        self.actor_adapt(fake_input)
        # self.critic_target(fake_input)
        self.critic_adapt(fake_input)
        self.critic_adapt.apply(init_)
        
        # self.adapt_opt.add_param_group({"params": self.actor_adapt.parameters()})
        # self.adapt_opt.add_param_group({"params": self.critic_adapt.parameters()})
        self.opt.add_param_group({"params": self.actor_adapt.parameters()})
        self.opt.add_param_group({"params": self.critic_adapt.parameters()})

    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, value: str):
        assert value in ("train", "adapt", "finetune")
        self._phase = value

    def forward(self, tensordict: TensorDict):
        self.encoder(tensordict)
        if self.phase == "train":
            self.actor_expert(tensordict)
            if not self.training:
                self.critic_expert(tensordict)
        else:
            self.adapt_module_ema(tensordict)
            self.actor_adapt(tensordict)
            if not self.training:
                self.adaptation_loss(tensordict, tensordict, mean=False)
                self.critic_adapt(tensordict)
                # action_adapt = self.actor_adapt(tensordict.exclude(ACTION_KEY))[ACTION_KEY]
                # action_target = self.actor_target(tensordict.exclude(ACTION_KEY))[ACTION_KEY]
                # sa_adapt = torch.cat([tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY], action_adapt], -1)
                # sa_target = torch.cat([tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY], action_target], -1)
                # # score = (1. - 0.25 * (self.classifier(sa) - 1.).square()).clamp(0.)
                # score_adapt = self.classifier(sa_adapt)
                # score_target = self.classifier(sa_target)
                # tensordict.set("score_adapt", score_adapt)
                # tensordict.set("score_target", score_target)
        tensordict.exclude(*self.exclude_keys, inplace=True)
        if not self.training:
            tensordict.exclude("context_expert", "context_adapt", inplace=True)
        return tensordict

    # def step_schedule(self, progress: float):
    #     if self.phase == "train":
    #         self.exp_reward = self.cfg.exp_reward * (1. - progress)

    def train_op(self, tensordict: TensorDict):
        info = {}
        tensordict = tensordict.to_tensordict()
        if self.phase == "train":
            info.update(self._train_expert(tensordict))
        elif self.phase == "adapt":
            info.update(self._train_adaptation(tensordict))
        elif self.phase == "finetune":
            info.update(self._finetune(tensordict))
        self.train_iter += 1
        return info

    def _train_expert(self, tensordict: TensorDict):
        hard_copy_(self.actor_expert, self.actor_target)
        self._compute_rewards(tensordict)
        weights = (1., self.exp_reward, self.reg_reward)

        infos = {}
        infos_policy = []
        for epoch in range(self.cfg.ppo_epochs):
            if epoch == 0 or self.cfg.recompute_adv:
                self._compute_advantage(
                    tensordict, 
                    self.critic_expert, 
                    self.gae_expert, 
                    self.value_norm_expert, 
                    weights
                )
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos_policy.append(self._update(minibatch, self.critic_expert))
        infos.update(collect_info(infos_policy))
        
        infos_adapt = []
        for epoch in range(1):
            for minibatch in make_batch(tensordict, 8, self.cfg.train_every):
                infos_adapt.append(self._update_adaptation(minibatch, classifier=False))
        infos.update(collect_info(infos_adapt, "adapt/"))
        soft_copy_(self.adapt_module, self.adapt_module_ema)

        mean, var = self.value_norm_expert.running_mean_var()
        for name, value_mean in zip(
            ["value_mean", "value_mse", "value_kl"], 
            mean.unbind(-1),
            strict=True
        ):
            infos[name] = value_mean.item()
        infos["adapt/same_sign"] = tensordict["same_sign"].float().mean().item()
        infos["adapt/exp_c"] = self.exp_reward
        infos["adapt/reg_c"] = self.reg_reward
        return infos

    def _update(
        self, 
        tensordict: TensorDict, 
        critic: TensorDictModule,
    ):
        if self.phase == "train":
            tensordict = self.encoder(tensordict)
            actor = self.actor_expert
        elif self.phase == "finetune":
            # TODO: detach or not?
            tensordict = self.adapt_module_ema(tensordict).detach()
            actor = self.actor_adapt
        
        losses = {}

        policy_loss, entropy_loss, entropy = compute_policy_loss(
            tensordict,
            actor, 
            self.clip_param, 
            self.entropy_coef,
            self.cfg.discard_init
        )
        losses["policy_loss"] = policy_loss
        losses["entropy_loss"] = entropy_loss

        value_loss, explained_var = compute_value_loss(
            tensordict, 
            critic, 
            self.clip_param, 
            self.critic_loss_fn,
            self.cfg.discard_init
        )
        losses["value_loss"] = value_loss

        loss = sum(losses.values())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 10)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(critic.parameters(), 10)
        self.opt.step()

        infos = losses
        infos["actor_grad_norm"] = actor_grad_norm
        infos["critic_grad_norm"] = critic_grad_norm
        infos["explained_var"] = explained_var
        infos["entropy"] = entropy
        return TensorDict(infos, []).detach()
    
    # @functools.partial(torch.compile, mode="reduce-overhead")
    def _update_adaptation(self, tensordict: TensorDict, classifier: bool=True):
        losses = {}
        self.adaptation_loss(tensordict, losses, mean=True)
        if classifier:
            classifier_loss, classifier_acc = self._compute_disc_loss(tensordict)
            losses["classifier_loss"] = classifier_loss
            losses["classifier_acc"] = classifier_acc
        # value_loss_adapt, explained_var_adapt = compute_value_loss(
        #     tensordict.detach(), 
        #     self.critic_adapt, 
        #     self.clip_param, 
        #     self.critic_loss_fn
        # )
        # losses["value_loss"] = value_loss_adapt
        loss = sum(v for k, v in losses.items() if "loss" in k)
        self.adapt_opt.zero_grad()
        loss.backward()
        self.adapt_opt.step()
        return TensorDict(losses, []).detach()
    
    def _train_adaptation(self, tensordict: TensorDict):
        self._compute_rewards(tensordict, adapt_rewards=True)
        self._compute_advantage(
            tensordict,
            self.critic_adapt,
            self.gae_adapt,
            self.value_norm_adapt,
        )

        infos = []
        for epoch in range(2):
            for batch in self.make_batch(tensordict, 8):
                losses = TensorDict({}, [])
                self.adaptation_loss(batch, losses, mean=True)
                value_loss_adapt, explained_var_adapt = compute_value_loss(
                    batch.detach(), 
                    self.critic_adapt, 
                    self.clip_param, 
                    self.critic_loss_fn
                )
                classifier_loss, classifier_acc = self._compute_disc_loss(batch)
                losses["classifier_loss"] = classifier_loss
                losses["value_loss"] = value_loss_adapt
                loss = sum(losses.values())
                self.adapt_opt.zero_grad()
                self.opt.zero_grad()
                loss.backward()
                self.adapt_opt.step()
                self.opt.step()

                losses["explained_var"] = explained_var_adapt
                losses["classifier_acc"] = classifier_acc
                infos.append(losses)
        
        infos = collect_info(infos, "adapt/")
        soft_copy_(self.adapt_module, self.adapt_module_ema)

        denormed_values = tensordict["denormed_values"]
        for name, value in zip(self.value_names, denormed_values.unbind(-1), strict=True):
            infos[name] = value.mean().item()
        infos["adapt/score_adapt"] = tensordict["score_adapt"].mean().item()
        infos["adapt/score_target"] = tensordict["score_target"].mean().item()
        return infos

    def _finetune(self, tensordict: TensorDict):
        if self.cfg.use_separate_critics:
            self._compute_rewards(tensordict, adapt_rewards=True)
            weights = (1., 0., 0., 0., 0.)
            self._compute_advantage(
                tensordict, 
                self.critic_adapt,
                self.gae_adapt,
                self.value_norm_adapt,
                weights
            )
            critic = self.critic_adapt
        else:
            self._compute_rewards(tensordict, adapt_rewards=False)
            weights = (1., 0., 0.)
            self._compute_advantage(
                tensordict, 
                self.critic_expert,
                self.gae_expert,
                self.value_norm_expert,
                weights
            )
            critic = self.critic_expert
        
        infos = {}
        # update policy (actor and critic)
        if self.train_iter > 0:
            infos_policy = []
            for epoch in range(self.cfg.ppo_epochs):
                for minibatch in self.make_batch(tensordict, self.cfg.num_minibatches):
                    infos_policy.append(self._update(minibatch, critic))
            infos.update(collect_info(infos_policy))

        # update adaptation module
        infos_adapt = []
        for epoch in range(2):
            for minibatch in self.make_batch(tensordict, 8):
                infos_adapt.append(self._update_adaptation(minibatch))
        
        # update adaptation module ema
        infos.update(collect_info(infos_adapt, "adapt/"))
        soft_copy_(self.adapt_module, self.adapt_module_ema)

        denormed_values = tensordict["denormed_values"]
        for name, value in zip(self.value_names, denormed_values.unbind(-1)):
            infos[name] = value.mean().item()
        infos["adapt/score_adapt"] = tensordict["score_adapt"].mean().item()
        infos["adapt/score_target"] = tensordict["score_target"].mean().item()
        return infos
    
    def _compute_rewards(self, tensordict: TensorDict, adapt_rewards: bool=False):
        """
        Compute auxiliary rewards.
        """

        rewards = [tensordict[REWARD_KEY]]
        with torch.no_grad(), tensordict["next"].view(-1) as next_tensordict:
            self.encoder(next_tensordict)
            self.adapt_module_ema(tensordict)
            self.adapt_module_ema(next_tensordict)

            mse = F.mse_loss(tensordict["context_adapt"], tensordict["context_expert"], reduction="none").mean(-1, True)
            kl_target = action_kl(tensordict.exclude(), self.actor_target, self.actor_expert)
            rewards.append(mse)
            rewards.append(kl_target)

            if self.phase in ("adapt", "finetune"):
                action_adapt = self.actor_adapt(tensordict.exclude(ACTION_KEY))[ACTION_KEY]
                action_target = self.actor_target(tensordict.exclude(ACTION_KEY))[ACTION_KEY]
                sa_adapt = torch.cat([tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY], action_adapt], -1)
                sa_target = torch.cat([tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY], action_target], -1)
                # score = (1. - 0.25 * (self.classifier(sa) - 1.).square()).clamp(0.)
                score_adapt = self.classifier(sa_adapt)
                score_target = self.classifier(sa_target)
                tensordict.set("score_adapt", score_adapt)
                tensordict.set("score_target", score_target)
                if adapt_rewards:
                    rewards.append(score_adapt)
                    rewards.append(score_target)

        rewards = torch.cat(rewards, -1)
        # non-episodic aux rewards
        done = tensordict[DONE_KEY]
        done = torch.cat([done]+ [torch.zeros_like(done)] * (rewards.shape[-1]-1), -1)
        tensordict.set(REWARD_KEY, rewards)
        tensordict.set(DONE_KEY, done)
        return tensordict

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TensorDictModuleBase,
        gae: GAE,
        value_norm: ValueNorm1,
        reward_weights=1.,
        subtract_mean=False
    ):
        with tensordict.view(-1) as td:
            # self.encoder(td)
            # self.encoder(td["next"])
            values = critic(td)["state_value"].reshape(*tensordict.shape, -1)
            next_values = critic(td["next"])["state_value"].reshape(*tensordict.shape, -1)
        
        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = value_norm.denormalize(values)
        next_values = value_norm.denormalize(next_values)

        adv_all, ret = gae(rewards, dones, values, next_values)
        adv = (adv_all * torch.as_tensor(reward_weights, device=tensordict.device)).sum(-1, True)
        value_norm.update(ret)
        ret = value_norm.normalize(ret)

        adv_mean = adv.mean()
        adv_std = adv.std()
        if subtract_mean:
            adv = (adv - adv_mean) / adv_std.clip(1e-7)
        else:
            adv = adv / adv_std.clip(1e-7)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        tensordict.set("same_sign", adv_all[..., [0]].sign() == adv.sign())
        tensordict.set("denormed_values", values)
        return tensordict
    
    def _compute_disc_loss(self, tensordict: TensorDictBase):
        with torch.no_grad():
            # action_expert = tensordict["action_expert"]
            action_expert = self.actor_expert(tensordict)[ACTION_KEY]
            action_adapt = self.actor_adapt(tensordict)[ACTION_KEY]
        sa_expert = torch.cat([tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY], action_expert], -1).detach()
        sa_adapt = torch.cat([tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY], action_adapt], -1).detach()
        return self.classifier.compute_loss(sa_expert, sa_adapt)
    
    def state_dict(self):
        state_dict = super().state_dict()
        for key in list(state_dict.keys()):
            if key.startswith("gae"):
                state_dict.pop(key)
        state_dict["phase"] = self.phase
        state_dict["cfg"] = self.cfg
        return state_dict
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        cfg = state_dict.get("cfg", self.cfg)
        for k, v in cfg.__dict__.items():
            pass
        return super().load_state_dict(state_dict, strict, assign)


import torch.distributions as D
    
def action_kl(
    tensordict: TensorDictBase,
    actor_a: ProbabilisticActor, 
    actor_b: ProbabilisticActor
):
    dist_a = actor_a.get_dist(tensordict)
    dist_b = actor_b.get_dist(tensordict)
    return D.kl_divergence(dist_a, dist_b).unsqueeze(-1)


def elbo(
    tensordict: TensorDictBase,
    encoder: ProbabilisticTensorDictSequential,
    beta: float=1.
):
    dist = encoder.get_dist(tensordict)
    x = tensordict["context_expert"]
    recon = F.mse_loss(dist.rsample(), x, reduction="none").mean(-1)
    # target_dist = IndependentNormal(torch.zeros_like(x), torch.ones_like(x))
    # kl = D.kl_divergence(dist, target_dist)
    return recon # + beta * kl


class ELBO(nn.Module):
    def __init__(self, adaptation_module, beta: float=1.):
        super().__init__()
        self.adaptation_module = adaptation_module
        self.beta = beta
    
    def forward(self, tensordict: TensorDictBase, out: TensorDictBase, mean=True):
        self.adaptation_module(tensordict)
        pred = tensordict["context_adapt"]
        target = tensordict["context_expert"].detach()
        recon = F.mse_loss(pred, target, reduction="none").mean(-1)
        mu, logvar = tensordict["loc"], tensordict["scale"]
        kl = - 0.5 * (1 + logvar - mu.square() - logvar.exp())
        loss = recon + kl
        if mean:
            loss = loss.mean()
        out.set("adaptation_loss", loss)
        return out


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = make_mlp([256])
        self.cls = nn.Sequential(make_mlp([256]), nn.LazyLinear(1))
    
    def forward(self, x):
        return self.cls(self.trunk(x))

    def compute_loss(self, true_input, false_input, wgp: float=0.):
        true_embed = self.trunk(true_input)
        false_embed = self.trunk(false_input)
        true_score = self.cls(true_embed)
        false_score = self.cls(false_embed)
        loss = (
            F.mse_loss(true_score, torch.ones_like(true_score))
            + F.mse_loss(false_score, -torch.ones_like(false_score))
        )
        acc = ((true_score > 0.).sum() + (false_score < 0.).sum()) / (true_score.numel() + false_score.numel())
        if wgp > 0:
            true_embed = true_embed.detach().clone().requires_grad_(True)
            true_score = self.cls(true_embed)
            grad = torch.autograd.grad(
                true_score, true_embed, torch.ones_like(true_score),
                retain_graph=True,
                create_graph=True
            )[0]
            assert grad.shape == true_embed.shape
            grad_norm = grad.norm(dim=-1).square().mean()
            loss += wgp * grad_norm
        return loss, acc


def compute_classifier_loss(
    classifier, 
    true_input: torch.Tensor,
    false_input: torch.Tensor,
    method: str = "ls",
    wgp: float = 0.
):
    if wgp > 0:
        true_input.requires_grad_(True)
    true_score: torch.Tensor = classifier(true_input)
    false_score: torch.Tensor = classifier(false_input)
    if method == "bce":
        loss_func = F.binary_cross_entropy_with_logits
        loss = (
            loss_func(true_score, torch.ones_like(true_score)) 
            + loss_func(false_score, torch.zeros_like(false_score))
        )
    elif method == "ls":
        loss = (
            F.mse_loss(true_score, torch.ones_like(true_score))
            + F.mse_loss(false_score, -torch.ones_like(false_score))
        )
    else:
        raise ValueError(method)
    if wgp > 0.:
        grad = torch.autograd.grad(
            true_score, true_input, torch.ones_like(true_score),
            retain_graph=True,
            create_graph=True
        )[0]
        grad_norm = grad.norm(dim=-1).square().mean()
        loss += wgp * grad_norm
    acc = ((true_score > 0.).sum() + (false_score < 0.).sum()) / (true_score.numel() + false_score.numel())
    return loss, acc


def perturb_(module: nn.Module, alpha: float = 0.1):
    def _perturb(m: nn.Module):
        if isinstance(m, nn.Linear):
            init_weight = nn.init.orthogonal_(m.weight.data.clone(), 0.01)
            init_bias = nn.init.constant_(m.bias.data.clone(), 0.)
            m.weight.data.lerp_(init_weight, alpha)
            m.bias.data.lerp_(init_bias, alpha)
    module.apply(_perturb)


@torch.no_grad()
def find_dormant_neurons(model: nn.Module, input, tau: float = 0.025):
    hooks = {}
    handlers = []
    
    class Hook:
        def __call__(self, module, module_in, module_out):
            self.output_mean = einops.reduce(module_out.abs(), "... d -> d", "mean")

    for name, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            hook = Hook()
            hooks[name] = hook
            handlers.append(m.register_forward_hook(hook))
    
    model(input)

    for name, hook in hooks.items():
        dormant = hook.output_mean < hook.output_mean.mean() * tau
        print(f"{name}: {dormant.sum()}/{dormant.numel()}")

    for handler in handlers:
        handler.remove()
    
    return

