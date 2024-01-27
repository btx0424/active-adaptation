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
from torchrl.data import UnboundedContinuousTensorSpec

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, MISSING
from typing import Any, Mapping, Union, Sequence

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .adaptation import Action, Value, ActionValue, MSE
from .ppo_rnn import GRU
from .common import *

from active_adaptation.utils.wandb import parse_path
import copy
import logging

make_mlp = functools.partial(make_mlp, activation=nn.Mish)

@dataclass
class PPOConfig:
    name: str = "ppo_rma"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    predict_std: bool = True

    checkpoint_path: Union[str, None] = None
    phase: str = "train"
    condition_mode: str = "cat"

    adapt_arch: str = "tconv"
    # what the adaptation module learns to predict
    adaptation_key: Any = "context"
    adaptation_loss: str = "mse" # mse, action_kl
    use_separate_critics: bool = True

    def __post_init__(self):
        assert self.condition_mode.lower() in ("cat", "film")
        assert self.adaptation_key in ("context", OBS_HIST_KEY, "_feature")
        assert self.phase in ("train", "adapt", "finetune")

cs = ConfigStore.instance()
cs.store("rma_train", node=PPOConfig, group="algo")
cs.store("rma_adapt", node=PPOConfig(phase="adapt", checkpoint_path=MISSING), group="algo")
cs.store("rma_adapt_rnn", node=PPOConfig(adapt_arch="rnn", phase="adapt", checkpoint_path=None), group="algo")
cs.store("rma_finetune", node=PPOConfig(phase="finetune", checkpoint_path=MISSING), group="algo")
cs.store("rma_finetune_rnn", node=PPOConfig(adapt_arch="rnn", phase="finetune", checkpoint_path=MISSING), group="algo")

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

class GRUModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = make_mlp([128, 128])
        self.gru = GRU(128, dim, allow_none=False)
    
    def forward(self, x, is_init, hx):
        x = self.mlp(x)
        x, hx = self.gru(x, is_init, hx)
        return x, hx

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


def make_adaptation_module(adapt_arch: str, dim: int):
    if adapt_arch == "tconv":
        def make(output_key: str):
            return TensorDictModule(TConv(dim), [OBS_HIST_KEY], [output_key])
    elif adapt_arch == "rnn":
        def make(output_key: str):
            in_keys = [OBS_KEY, "is_init", f"{output_key}_hx"]
            out_keys = [output_key, ("next", f"{output_key}_hx")]
            gru = GRUModule(dim)
            return TensorDictModule(gru, in_keys, out_keys)
    else:
        raise NotImplementedError(adapt_arch)
    module = make("context_adapt")
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


def condition(expert: bool, mode: str):
    module = nn.Sequential(make_mlp([512]))
    in_keys = ["_feature", "context_expert" if expert else "context_adapt"]
    if mode == "cat":
        return TensorDictSequential(
            TensorDictModule(module, [OBS_KEY], ["_feature"]),
            CatTensors(in_keys, "_feature", del_keys=False)
        )
    elif mode == "film":
        return TensorDictSequential(
            TensorDictModule(module, [OBS_KEY], ["_feature"]),
            TensorDictModule(FiLM(256), in_keys, ["_feature"])
        )


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
        observation_dim = observation_spec[OBS_KEY].shape[-1]

        fake_input = observation_spec.zero()

        # create and initialize the modules
        self.encoder = TensorDictModule(
            make_mlp([256, 128]), 
            [OBS_PRIV_KEY], 
            ["context_expert"]
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

        # the critics always observe priviledged information
        self.critic_expert = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "_obs_critic", del_keys=False),
            TensorDictModule(
                nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(1)), 
                ["_obs_critic"], ["state_value"]
            )
        ).to(self.device)
        
        self.value_norm_expert = ValueNorm1(1).to(self.device)

        # self.state_estimator = make_state_estimator(self.cfg.adapt_arch, observation_priv_dim).to(self.device)
        self.classifier = nn.Sequential(make_mlp([256, 128]), nn.LazyLinear(1)).to(self.device)
        # self.projection = nn.Sequential(make_mlp([256, 128]), nn.LazyLinear(64)).to(self.device)

        self.encoder(fake_input)
        self.actor_expert(fake_input)
        self.critic_expert(fake_input)
        self.classifier(torch.cat([fake_input[OBS_KEY], fake_input[ACTION_KEY]], dim=-1))

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor_expert.parameters()},
                {"params": self.critic_expert.parameters()},
                {"params": self.encoder.parameters()},
            ], 
            lr=cfg.lr
        )

        if self.phase in ("adapt", "finetune"):
            self.make_adapt_modules(fake_input)

        checkpoint_path = parse_path(self.cfg.checkpoint_path)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            self.actor_expert.apply(init_)
            self.critic_expert.apply(init_)
            self.encoder.apply(init_)
        
        # initialize the adaptation actor
        if (
            self.phase == "adapt" 
            and not any(key.startswith("actor_adapt") for key in state_dict.keys())
        ):
            logging.info("Initialize the adaptation actor with the expert actor.")
            hard_copy_(self.actor_expert, self.actor_adapt)
        if self.phase in ("adapt", "finetune"):
            hard_copy_(self.actor_expert, self.actor_target)
            self.actor_target.requires_grad_(False)
        
        if cfg.adapt_arch == "rnn":
            def make_tensordict_primer():
                num_envs = observation_spec.shape[0]
                return TensorDictPrimer({
                    "context_adapt_hx": UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)
                })
            self.make_tensordict_primer = make_tensordict_primer
            self.make_batch = functools.partial(make_batch, seq_len=cfg.train_every)
        else:
            self.make_batch = make_batch
        
        self.train_contra = False
        self.exclude_keys = ["_obs_critic", "_feature", "_feature_critic"]

    def make_adapt_modules(self, fake_input: TensorDictBase):
        self.adapt_module = (
            make_adaptation_module(self.cfg.adapt_arch, 128)
            .to(self.device)
        )

        self.actor_adapt: ProbabilisticActor = ProbabilisticActor(
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

        self.critic_adapt = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "_obs_critic", del_keys=False),
            TensorDictModule(make_mlp([512, 256]), ["_obs_critic"], ["_feature_critic"]),
            CatTensors(["context_adapt", "_feature_critic"], "_feature_critic", del_keys=False),
            TensorDictModule(
                nn.Sequential(make_mlp([128]), nn.LazyLinear(2)), 
                ["_feature_critic"], 
                ["state_value"]
            )
        ).to(self.device)

        self.value_norm_adapt = ValueNorm1(2).to(self.device)
        
        if self.cfg.adaptation_loss == "mse":
            self.adaptation_loss = MSE(self.adapt_module).to(self.device)
        elif self.cfg.adaptation_loss == "action_kl":
            self.adaptation_loss = Action(
                self.adapt_module,
                self.actor_expert,
                self.actor_adapt,
                closed_kl=False
            ).to(self.device)
        else:
            raise ValueError(self.cfg.adaptation_loss)
        
        if self.cfg.adapt_arch == "rnn":
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool, device=self.device)
            fake_input["context_adapt_hx"] = torch.zeros(fake_input.shape[0], 128, device=self.device)
        
        self.adapt_module(fake_input)
        self.actor_adapt(fake_input)
        self.actor_target(fake_input)
        self.critic_adapt(fake_input)
        self.critic_adapt.apply(init_)
        
        self.adapt_opt = torch.optim.Adam(
            [
                {"params": self.adapt_module.parameters()},
                {"params": self.actor_adapt.parameters()},
                {"params": self.critic_adapt.parameters()},
                {"params": self.classifier.parameters()},
            ],
            lr=self.cfg.lr,
            fused=True
        )
        self.opt.add_param_group({"params": self.adapt_module.parameters()})
        self.opt.add_param_group({"params": self.actor_adapt.parameters()})
        self.opt.add_param_group({"params": self.critic_adapt.parameters()})
        self.opt.add_param_group({"params": self.classifier.parameters()})

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
        else:
            self.adapt_module(tensordict)
            self.actor_adapt(tensordict)
            # self.adaptation_loss(tensordict, tensordict)
        tensordict.exclude(*self.exclude_keys, inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        info = {}
        tensordict = tensordict.to_tensordict()
        if self.phase == "train":
            info.update(self._train_policy(tensordict))
        elif self.phase == "adapt":
            with (
                hold_out_net(self.encoder),
                hold_out_net(self.actor_expert), 
            ):
                info.update(self._train_adaptation(tensordict))
        elif self.phase == "finetune":
            info.update(self._finetune(tensordict))
        return info
    
    def _get_context(self, tensordict: TensorDict):
        if self.phase == "train":
            self.encoder(tensordict)
        elif self.phase in ("adapt", "finetune"):
            if self.adaptation_key == "raw":
                tensordict.rename_key_(OBS_PRIV_KEY, "tmp")
                self.adapt_module(tensordict)
                self.encoder(tensordict)
                tensordict.exclude(OBS_PRIV_KEY, inplace=True)
                tensordict.rename_key_("tmp", OBS_PRIV_KEY)
            else:
                self.adapt_module(tensordict)
        return tensordict

    def _train_policy(self, tensordict: TensorDict):
        self._compute_advantage(tensordict, self.critic_expert, self.value_norm_expert)
        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = self.make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch, self.critic_expert))
        
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        infos["value_mean"] = tensordict["denormalized_state_value"].mean()
        return {k: v.item() for k, v in infos.items()}

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
            tensordict = self.adapt_module(tensordict).detach()
            actor = self.actor_adapt
        
        losses = TensorDict({}, [])

        policy_loss, entropy_loss, entropy = compute_policy_loss(
            tensordict,
            actor, 
            self.clip_param, 
            self.entropy_coef
        )
        losses["policy_loss"] = policy_loss
        losses["entropy_loss"] = entropy_loss

        value_loss, explained_var = compute_value_loss(
            tensordict, 
            critic, 
            self.clip_param, 
            self.critic_loss_fn
        )
        losses["value_loss"] = value_loss

        loss = sum(losses.values())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 10)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(critic.parameters(), 10)
        self.opt.step()

        infos = losses.to_tensordict()
        infos["actor_grad_norm"] = actor_grad_norm
        infos["critic_grad_norm"] = critic_grad_norm
        infos["explained_var"] = explained_var
        infos["entropy"] = entropy
        return infos
    
    def _update_adaptation(self, tensordict: TensorDict):
        losses = TensorDict({}, [])
        self.adaptation_loss(tensordict, losses, mean=True)
        loss = sum(losses.values())
        self.adapt_opt.zero_grad()
        loss.backward()
        self.adapt_opt.step()
        return losses
    
    def _train_adaptation(self, tensordict: TensorDict):
        # compute values for the adaptation policy
        with torch.no_grad():
            self.actor_expert(self.encoder(tensordict))
            self.adapt_module(tensordict["next"])
            tensordict.set("action_expert", tensordict[ACTION_KEY])

        assert "context_adapt" in tensordict.keys()
        assert "context_adapt" in tensordict["next"].keys()
        
        with torch.no_grad():
            kl = action_kl(tensordict.exclude(), self.actor_target, self.actor_expert)
        tensordict.set(REWARD_KEY, torch.cat([tensordict[REWARD_KEY], kl], -1))
        self._compute_advantage(tensordict, self.critic_adapt, self.value_norm_adapt)

        infos = []
        for epoch in range(4):
            for batch in self.make_batch(tensordict, 8):
                losses = TensorDict({}, [])
                self.adaptation_loss(batch, losses, mean=True)
                value_loss, explained_var = compute_value_loss(
                    batch, 
                    self.critic_adapt, 
                    self.clip_param, 
                    self.critic_loss_fn
                )
                
                action_expert = batch["action_expert"]
                with torch.no_grad():
                    action_adapt = self.actor_adapt(batch)[ACTION_KEY]
                # classifier_loss, classifier_acc, kl = compute_classifier_loss(
                #     self.classifier,
                #     torch.cat([batch[OBS_KEY], action_expert], dim=-1),
                #     torch.cat([batch[OBS_KEY], action_adapt], dim=-1)
                # )

                # losses["classifier_loss"] = classifier_loss
                losses["value_loss"] = value_loss
                loss = sum(losses.values())
                self.adapt_opt.zero_grad()
                loss.backward()
                self.adapt_opt.step()

                losses["explained_var"] = explained_var
                # losses["classifier_acc"] = classifier_acc
                # losses["classifier_kl"] = kl
                infos.append(losses)
        
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        infos["value_mean_0"] = tensordict["denormalized_state_value"][..., 0].mean()
        infos["value_mean_1"] = tensordict["denormalized_state_value"][..., 1].mean()
        return {k: v.item() for k, v in infos.items()}

    def _finetune(self, tensordict: TensorDict):
        with torch.no_grad():
            self.encoder(tensordict)
        if self.cfg.use_separate_critics:
            with torch.no_grad():
                self.adapt_module(tensordict["next"])
                kl = action_kl(tensordict.exclude(), self.actor_target, self.actor_expert)
            critic = self.critic_adapt
            tensordict.set(REWARD_KEY, torch.cat([tensordict[REWARD_KEY], kl], -1))
            self._compute_advantage(tensordict, critic, self.value_norm_adapt, (1., 0.))
        else:
            critic = self.critic_expert
            self._compute_advantage(tensordict, critic, self.value_norm_expert)
        
        # update policy (actor and critic)
        infos_policy = []
        for epoch in range(self.cfg.ppo_epochs):
            for minibatch in self.make_batch(tensordict, self.cfg.num_minibatches):
                infos_policy.append(self._update(minibatch, critic))
        infos_policy: TensorDict = torch.stack(infos_policy).to_tensordict().apply(torch.mean, batch_size=[])

        # update adaptation module
        infos_adapt = []
        for epoch in range(2):
            for minibatch in self.make_batch(tensordict, 8):
                infos_adapt.append(self._update_adaptation(minibatch))
        infos_adapt: TensorDict = torch.stack(infos_adapt).to_tensordict().apply(torch.mean, batch_size=[])

        infos = {**infos_policy, **infos_adapt}

        if self.cfg.use_separate_critics:
            infos["value_mean_0"] = tensordict["denormalized_state_value"][..., 0].mean()
            infos["value_mean_1"] = tensordict["denormalized_state_value"][..., 1].mean()
        else:
            infos["value_mean_0"] = tensordict["denormalized_state_value"].mean()
        return {k: v.item() for k, v in infos.items()}
    
    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict, 
        critic: TensorDictModuleBase,
        value_norm: ValueNorm1,
        reward_weights=1.
    ):
        values = critic(tensordict.view(-1))["state_value"].reshape(*tensordict.shape, -1)
        next_values = critic(tensordict["next"].view(-1))["state_value"].reshape(*tensordict.shape, -1)
        
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
        tensordict.set("denormalized_state_value", values)
        return tensordict


import torch.distributions as D
    
def action_kl(
    tensordict: TensorDictBase,
    actor_a: ProbabilisticActor, 
    actor_b: ProbabilisticActor
):
    dist_a = actor_a.get_dist(tensordict)
    dist_b = actor_b.get_dist(tensordict)
    return D.kl_divergence(dist_a, dist_b).unsqueeze(-1)


def elbo(tensordict: TensorDictBase):
    loc = tensordict["context_adapt"]
    scale = tensordict["context_adapt_std"]
    dist = D.Normal(loc, scale)
    x = tensordict["context_expret"]
    return dist.log_prob(x) + dist.entropy()


def compute_classifier_loss(classifier, true_input, false_input):
    true: torch.Tensor = classifier(true_input)
    false: torch.Tensor = classifier(false_input)
    loss = (
        F.binary_cross_entropy_with_logits(true, torch.ones_like(true))
        + F.binary_cross_entropy_with_logits(false, torch.zeros_like(false))
    )
    acc = ((true > 0.).sum() + (false < 0.).sum()) / (true.numel() + false.numel())
    kl = true.sigmoid().log() - (1.-true).sigmoid().log()
    return loss, acc, kl.mean()

