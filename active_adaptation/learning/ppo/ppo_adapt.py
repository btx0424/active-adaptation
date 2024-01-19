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

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, MISSING
from typing import Any, Mapping, Union, Sequence

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE, Duplicate, Actor, make_mlp, make_batch, init_
from .adaptation import Action, Value, ActionValue, MSE
from .ppo_rnn import GRU

from active_adaptation.utils.wandb import parse_path

make_mlp = functools.partial(make_mlp, activation=nn.Mish)

OBS_KEY = "policy" # ("agents", "observation")
OBS_PRIV_KEY = "priv"
OBS_HIST_KEY = "policy_h"
ACTION_KEY = "action" # ("agents", "action")
REWARD_KEY = ("next", "reward") # ("agents", "reward")
# DONE_KEY = ("next", "done")
DONE_KEY = ("next", "terminated")

@dataclass
class PPOConfig:
    name: str = "ppo_rma"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 1e-3
    predict_std: bool = True

    checkpoint_path: Union[str, None] = None
    phase: str = "train"
    condition_mode: str = "cat"

    encoder_mode: str = "shared" # shared, separate, seperate_heads
    adapt_arch: str = "tconv"
    # what the adaptation module learns to predict
    adaptation_key: Any = "context"
    adaptation_loss: str = "mse" # mse, action_kl

    def __post_init__(self):
        assert self.condition_mode.lower() in ("cat", "film")
        assert self.adaptation_key in ("context", OBS_HIST_KEY, "_feature")
        assert self.phase in ("train", "adapt", "finetune")

cs = ConfigStore.instance()
cs.store("rma_train", node=PPOConfig, group="algo")
cs.store("rma_adapt", node=PPOConfig(phase="adapt", checkpoint_path=MISSING), group="algo")
cs.store("rma_adapt_rnn", node=PPOConfig(adapt_arch="rnn", phase="adapt", checkpoint_path=None), group="algo")
cs.store("rma_finetune", node=PPOConfig(phase="finetune", checkpoint_path=MISSING), group="algo")

class TConv(nn.Module):
    def __init__(self, out_dim: int, activation=nn.Mish) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.tconv = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1), activation(),
            nn.LazyConv1d(64, kernel_size=7, stride=2), activation(),
            nn.LazyConv1d(64, kernel_size=5, stride=2), activation(),
        )
        self.mlp = make_mlp([256, out_dim], activation=nn.Mish)
    
    def forward(self, features: torch.Tensor):
        batch_shape = features.shape[:-2]
        features = features.reshape(-1, *features.shape[-2:])
        features_tconv = einops.rearrange(self.tconv(features), "b d t -> b (t d)")
        features = torch.cat([features_tconv, features[:, :, -1]], dim=1)
        features = self.mlp(features)
        return features.reshape(*batch_shape, *features.shape[1:])


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


def make_priv_encoder(mode: str, num_units: Sequence[int]):
    assert mode in ("shared", "separate", "separate_heads")
    if mode == "shared":
        encoder = TensorDictModule(
            nn.Sequential(make_mlp(num_units), Duplicate(2)),
            [OBS_PRIV_KEY], 
            ["context_actor", "context_critic"]
        )
    elif mode == "separate":
        encoder = TensorDictSequential(
            TensorDictModule(
                make_mlp(num_units), [OBS_PRIV_KEY], ["context_actor"]
            ),
            TensorDictModule(
                make_mlp(num_units), [OBS_PRIV_KEY], ["context_critic"]
            )
        )
    else:
        raise NotImplementedError(mode)
    return encoder


def make_adaptation_module(encoder_mode: str, adapt_arch: str, dim: int):
    if adapt_arch == "tconv":
        def make(output_key: str):
            return TensorDictModule(TConv(dim), [OBS_HIST_KEY], [output_key])
    elif adapt_arch == "rnn":
        def make(output_key: str):
            in_keys = [f"_rnn_{output_key}", "is_init", f"{output_key}_hx"]
            out_keys = [output_key, ("next", f"{output_key}_hx")]
            gru = GRU(dim, dim, allow_none=True)
            return TensorDictSequential(
                TensorDictModule(nn.LazyLinear(dim), [OBS_KEY], [f"_rnn_{output_key}"]),
                TensorDictModule(gru, in_keys, out_keys),
            )
    else:
        raise NotImplementedError(adapt_arch)
    
    if encoder_mode == "shared":
        module = TensorDictSequential(
            make("context"),
            TensorDictModule(Duplicate(2), ["context"], ["context_actor", "context_critic"])
        )
    elif encoder_mode == "separate":
        module = TensorDictSequential(
            make("context_actor"),
            make("context_critic")
        )
    else:
        raise NotImplementedError(encoder_mode)
    return module


def make_state_estimator(arch: str, dim: int):
    if arch == "tconv":
        module = nn.Sequential(TConv(dim), nn.LazyLinear(dim))
        module = TensorDictModule(module, [OBS_HIST_KEY], [OBS_PRIV_KEY])
    elif arch == "rnn":
        in_keys = ["_rnn_feature", "is_init", "_hx"]
        out_keys = [OBS_PRIV_KEY, ("next", f"_hx")]
        gru = GRU(dim, dim, allow_none=True)
        return TensorDictSequential(
            TensorDictModule(nn.LazyLinear(dim), [OBS_KEY], ["_rnn_feature"]),
            TensorDictModule(gru, in_keys, out_keys),
        )
    return module


class PPORMAPolicy(TensorDictModuleBase):
    
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
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.adaptation_key = self.cfg.adaptation_key
        self.phase = self.cfg.phase

        if not isinstance(self.adaptation_key, str):
            self.adaptation_key = tuple(self.adaptation_key)
        self.gae = GAE(0.99, 0.95)
        
        self.action_dim = action_spec.shape[-1]

        print(observation_spec)
        observation_priv_dim = observation_spec[OBS_PRIV_KEY].shape[-1]
        # observation_dim = observation_spec[OBS_KEY].shape[-1]

        fake_input = observation_spec.zero()

        self.encoder = make_priv_encoder(
            cfg.encoder_mode, 
            [256, 128]
        ).to(self.device)

        def condition(branch: str, mode: str):
            module = nn.Sequential(make_mlp([512]))
            if mode == "cat":
                return TensorDictSequential(
                    TensorDictModule(module, [OBS_KEY], ["_feature"]),
                    CatTensors(["_feature", f"context_{branch}"], "_feature", del_keys=False)
                )
            elif mode == "film":
                return TensorDictSequential(
                    TensorDictModule(module, [OBS_KEY], ["_feature"]),
                    TensorDictModule(FiLM(256), ["_feature", f"context_{branch}"], ["_feature"])
                )

        actor_module = TensorDictSequential(
            condition("actor", cfg.condition_mode),
            TensorDictModule(
                nn.Sequential(
                    make_mlp([256, 256]), 
                    Actor(self.action_dim, predict_std=cfg.predict_std)
                ), 
                ["_feature"], ["loc", "scale"]
            )
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictSequential(
            condition("critic", cfg.condition_mode),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)), 
                ["_feature"], ["state_value"]
            )
        ).to(self.device)
        
        self.value_norm = ValueNorm1(1).to(self.device)

        if self.adaptation_key == "context":
            self.adaptation_module = make_adaptation_module(
                self.cfg.encoder_mode, self.cfg.adapt_arch, 128
            ).to(self.device)
        elif self.adaptation_key == "raw":
            self.adaptation_module = make_state_estimator(
                self.cfg.adapt_arch, fake_input[OBS_PRIV_KEY].shape[-1]
            ).to(self.device)
        if self.cfg.adaptation_loss == "mse":
            key = "context_actor" if self.adaptation_key == "context" else OBS_PRIV_KEY
            self.adaptation_loss = MSE(
                self.adaptation_module, 
                [key], 
            ).to(self.device)
        elif self.cfg.adaptation_loss == "action_kl":
            self.adaptation_loss = Action(
                self.encoder,
                self.adaptation_module,
                self.actor,
                # closed_kl=True
            ).to(self.device)
        else:
            raise ValueError(self.cfg.adaptation_loss)
        
        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        self.adaptation_module(fake_input)

        checkpoint_path = parse_path(self.cfg.checkpoint_path)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            self.actor.apply(init_)
            self.critic.apply(init_)
            self.encoder.apply(init_)
        
        if self.phase == "adapt":
            self.adaptation_module.apply(init_)
        
        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
                {"params": self.encoder.parameters()},
            ], 
            lr=cfg.lr
        )

        if cfg.adapt_arch == "rnn":
            from torchrl.envs.transforms.transforms import TensorDictPrimer
            from torchrl.data import UnboundedContinuousTensorSpec
            def make_tensordict_primer():
                num_envs = observation_spec.shape[0]
                if cfg.encoder_mode == "separate":
                    return TensorDictPrimer({
                        "context_actor_hx": UnboundedContinuousTensorSpec((num_envs, 128)),
                        "context_critic_hx": UnboundedContinuousTensorSpec((num_envs, 128))
                    })
                else:
                    return TensorDictPrimer({
                        "context_hx": UnboundedContinuousTensorSpec((num_envs, 128))
                    })
            self.make_tensordict_primer = make_tensordict_primer
            self.make_batch = functools.partial(make_batch, seq_len=cfg.train_every)
        else:
            self.make_batch = make_batch

    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, value: str):
        if value == "train":
            self.train_policy = True
            self.train_adapt = False
        elif value == "adapt":
            self.train_policy = False
            self.train_adapt = True
        elif value == "finetune":
            self.train_policy = True
            self.train_adapt = False
        else:
            raise ValueError(value)
        self._phase = value

    def forward(self, tensordict: TensorDict):
        tensordict = self._get_context(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("_feature", "context_actor", "context_critic", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        info = {}
        if self.train_policy:
            with hold_out_net(self.adaptation_module):
                info.update(self._train_policy(tensordict))
        if self.train_adapt:
            with hold_out_net(self.actor), hold_out_net(self.encoder):
                info.update(self._train_adaptation(tensordict))
        return info
    
    def _get_context(self, tensordict: TensorDict):
        if self.phase == "train":
            self.encoder(tensordict)
        elif self.phase in ("adapt", "finetune"):
            if self.adaptation_key == "raw":
                tensordict.rename_key_(OBS_PRIV_KEY, "tmp")
                self.adaptation_module(tensordict)
                self.encoder(tensordict)
                tensordict.exclude(OBS_PRIV_KEY, inplace=True)
                tensordict.rename_key_("tmp", OBS_PRIV_KEY)
            else:
                self.adaptation_module(tensordict)
        return tensordict

    def _train_policy(self, tensordict: TensorDict):
        tensordict = self._compute_advantage(tensordict)
        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = self.make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        self._get_context(tensordict)
        losses = TensorDict({}, [])
        self._policy_loss(tensordict, losses)

        policy_loss = losses["policy_loss"]
        entropy_loss = losses["entropy_loss"]
        entropy = losses["entropy"]

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

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
    
    def _train_adaptation(self, tensordict: TensorDict):
        with torch.no_grad():
            tensordict = self.encoder(tensordict)
        # self._compute_advantage(tensordict)
        infos = []
        for epoch in range(4):
            for batch in self.make_batch(tensordict, 8):
                losses = TensorDict({}, [])
                self.adaptation_loss(batch, losses, mean=True)
                # self._policy_loss(batch, losses)
                loss = sum(losses.values())
                self.adaptation_loss.opt.zero_grad()
                loss.backward()
                self.adaptation_loss.opt.step()
                infos.append(losses)
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    @torch.no_grad()
    def _compute_advantage(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        self._get_context(next_tensordict)
        next_values = self.critic(next_tensordict)["state_value"]
        
        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        return tensordict

    def _policy_loss(self, tensordict: TensorDictBase, out: TensorDictBase):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        out.set("policy_loss", policy_loss)
        out.set("entropy_loss", entropy_loss)
        out.set("entropy", entropy.mean())
        return out