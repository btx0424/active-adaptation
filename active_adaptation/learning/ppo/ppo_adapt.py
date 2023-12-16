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
from .common import GAE, Duplicate, Actor, make_mlp, make_batch
from .adaptation import Action, Value, ActionValue, MSE
from .ppo_rnn import GRU

make_mlp = functools.partial(make_mlp, activation=nn.Mish)

@dataclass
class PPOConfig:
    name: str = "ppo_rma"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    checkpoint_path: Union[str, None] = None
    phase: str = "encoder"
    condition_mode: str = "cat"

    encoder_mode: str = "separate" # shared, separate, seperate_heads
    adapt_mode: str = "tconv"
    # what the adaptation module learns to predict
    adaptation_key: Any = "context"
    adaptation_loss: str = "mse" # mse, action_kl

    def __post_init__(self):
        assert self.condition_mode.lower() in ("cat", "film")
        assert self.adaptation_key in ("context", ("agents", "observation_priv"), "_feature")
        assert self.phase in ("encoder", "adaptation", "joint", "finetune")

cs = ConfigStore.instance()
cs.store("rma_train", node=PPOConfig, group="algo")
cs.store("rma_adapt", node=PPOConfig(phase="adaptation", checkpoint_path=MISSING), group="algo")
cs.store("rma_adapt_rnn", node=PPOConfig(adapt_mode="rnn", phase="adaptation", checkpoint_path=None), group="algo")


class TConv(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.tconv = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1), nn.ELU(),
            nn.LazyConv1d(64, kernel_size=7, stride=2), nn.ELU(),
            nn.LazyConv1d(64, kernel_size=5, stride=2), nn.ELU(),
        )
        self.mlp = make_mlp([256, out_dim], activation=nn.LeakyReLU)
    
    def forward(self, features: torch.Tensor):
        batch_shape = features.shape[:-2]
        features = features.flatten(0, -3) # [*, D, T]
        features_tconv = self.tconv(features).flatten(1)
        features = torch.cat([features_tconv, features[:, :, -1]], dim=1)
        features = self.mlp(features)
        return features.unflatten(0, batch_shape)


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


def make_encoder(mode: str, num_units: Sequence[int]):
    assert mode in ("shared", "separate", "separate_heads")
    if mode == "shared":
        encoder = TensorDictModule(
            nn.Sequential(
                make_mlp(num_units),
                Duplicate(2),
            ),
            [("agents", "observation_priv")], ["context_actor", "context_critic"]
        )
    elif mode == "separate":
        encoder = TensorDictSequential(
            TensorDictModule(
                make_mlp(num_units),
                [("agents", "observation_priv")], ["context_actor"]
            ),
            TensorDictModule(
                make_mlp(num_units),
                [("agents", "observation_priv")], ["context_critic"]
            )
        )
    elif mode == "separate_heads":
        encoder = TensorDictModule(
            nn.Sequential(
                make_mlp(num_units),
                Duplicate(2),
            ),
            [("agents", "observation_priv")], ["context_actor", "context_critic"]
        )
    else:
        raise NotImplementedError(mode)
    return encoder


def make_adaptation_module(encoder_mode: str, adapt_mode: str, dim: int):
    OBS_KEY = ("agents", "observation")
    OBS_HIST_KEY = ("agents", "observation_h")
    if adapt_mode == "tconv":
        def make(output_key: str):
            return TensorDictModule(TConv(dim), [OBS_HIST_KEY], [output_key])
    elif adapt_mode == "rnn":
        def make(output_key: str):
            in_keys = [f"_rnn_{output_key}", "is_init", f"{output_key}_hx"]
            out_keys = [output_key, ("next", f"{output_key}_hx")]
            gru = GRU(dim, dim, skip_conn=None, allow_none=True)
            return TensorDictSequential(
                TensorDictModule(nn.LazyLinear(dim), [OBS_KEY], [f"_rnn_{output_key}"]),
                TensorDictModule(gru, in_keys, out_keys),
            )
    else:
        raise NotImplementedError(adapt_mode)
    
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


def parse_path(path: Union[str, None]):
    if path is None:
        return None
    elif isinstance(path, str):
        if path.startswith("artifact:"):
            import wandb
            import os
            api = wandb.Api()
            artifact = api.artifact(path[9:])
            dir_path = artifact.download()
            checkpoint_path = os.path.join(dir_path, "checkpoint_final.pt")
            return checkpoint_path
        return path


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
        observation_priv_dim = observation_spec[("agents", "observation_priv")].shape[-1]
        observation_dim = observation_spec[("agents", "observation")].shape[-1]

        fake_input = observation_spec.zero()

        self.encoder = make_encoder("separate", [128, 128]).to(self.device)

        if self.cfg.condition_mode == "cat":
            def condition(branch: str):
                return TensorDictSequential(
                    TensorDictModule(
                        nn.Sequential(make_mlp([512])), 
                        [("agents", "observation")], 
                        ["_feature"]
                    ),
                    CatTensors(["_feature", f"context_{branch}"], "_feature", del_keys=False)
                )
        elif self.cfg.condition_mode == "film":
            def condition():
                return TensorDictSequential(
                    TensorDictModule(
                        nn.Sequential(nn.LayerNorm(observation_dim), make_mlp([256, 256])), 
                        [("agents", "observation")], 
                        ["_feature"]
                    ),
                    TensorDictModule(FiLM(128), ["_feature", "context"], ["_feature"])
                )
            raise NotImplementedError(self.cfg.condition_mode)

        actor_module = TensorDictSequential(
            condition("actor"),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256]), Actor(self.action_dim)), 
                ["_feature"], ["loc", "scale"]
            )
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictSequential(
            condition("critic"),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)), 
                ["_feature"], ["state_value"]
            )
        ).to(self.device)
        
        self.value_norm = ValueNorm1(1).to(self.device)

        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        checkpoint_path = parse_path(self.cfg.checkpoint_path)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            
            self.actor.apply(init_)
            self.critic.apply(init_)
            self.encoder.apply(init_)

        if self.phase in ("adaptation", "finetune"):
            self.adaptation_module = make_adaptation_module(
                self.cfg.encoder_mode, self.cfg.adapt_mode, 128
            ).to(self.device)
            self.adaptation_module(fake_input)
            if self.cfg.adaptation_loss == "mse":
                self.adaptation_loss = MSE(
                    self.adaptation_module, 
                    ["context_actor", "context_critic"], 
                ).to(self.device)
            elif self.cfg.adaptation_loss == "action_kl":
                self.adaptation_module(fake_input)
                self.adaptation_loss = Action(
                    self.encoder,
                    self.adaptation_module,
                    self.actor
                ).to(self.device)
            else:
                raise ValueError(self.cfg.adaptation_loss)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        self.actor_opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.encoder.parameters()), lr=5e-4)
        self.critic_opt = torch.optim.Adam(list(self.critic.parameters()) + list(self.encoder.parameters()), lr=5e-4)
        
        if cfg.adapt_mode == "rnn":
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

    def forward(self, tensordict: TensorDict):
        self._get_context(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("_feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        if self.phase == "encoder":
            info = self._train_policy(tensordict)
        elif self.phase == "adaptation":
            info = self._train_adaptation(tensordict)
        elif self.phase == "finetune":
            with hold_out_net(self.encoder):
                info = self._train_policy(tensordict.clone())
            info.update(self._train_adaptation(tensordict.clone()))
        else:
            raise RuntimeError()
        return info
    
    def _get_context(self, tensordict: TensorDict):
        if self.phase == "encoder":
            self.encoder(tensordict)
        elif self.phase in ("adaptation", "finetune"):
            self.adaptation_module(tensordict)
        return tensordict

    def _train_policy(self, tensordict: TensorDict):

        with torch.no_grad():
            N, T = tensordict.shape[:2]
            next_tensordict = tensordict["next"].reshape(-1, 1)
            self._get_context(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"].reshape(N, T)
        
        rewards = tensordict[("next", "agents", "reward")]
        dones = tensordict[("next", "terminated")]
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

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = self.make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        # self.encoder(tensordict)
        self._get_context(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

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
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        # self.encoder_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        # self.encoder_opt.step()
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
        
        info = []
        with hold_out_net(self.actor):
            for epoch in range(4):
                for batch in self.make_batch(tensordict, 8):
                    loss = self.adaptation_loss(batch).mean()
                    self.adaptation_loss.opt.zero_grad()
                    loss.backward()
                    self.adaptation_loss.opt.step()
                    info.append(loss)

        return {"adaptation_loss": torch.stack(info).mean().item()}

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)

    def adaptation_loss_traj(self, traj: TensorDictBase):
        """
        Computes the changes of the adaptation loss in an episode. Not for training.
        """
        
        td_target = self.critic(self.encoder(traj.exclude(self.adaptation_key)))
        td_pred = self.critic(self.adaptation_module(traj.exclude(self.adaptation_key)))
        mse = F.mse_loss(
            td_target.get(self.adaptation_key),
            td_pred.get(self.adaptation_key),
            reduction="none"
        ).mean((1, 2))
        value_error = F.mse_loss(
            td_target.get("state_value"),
            td_pred.get("state_value"),
            reduction="none"
        ).mean((1, 2))
        return {"mse": mse.cpu(), "value_error": value_error.cpu()}

    def __str__(self) -> str:
        return f"PPOAdapt-{self.cfg.phase}"

