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
from typing import Any, Mapping, Union, Sequence

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
    name: str = "final"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    recompute_adv: bool = True
    predict_std: bool = True

    checkpoint_path: Union[str, None] = None
    phase: str = "train"
    condition_mode: str = "cat"
    expert_reg: float = 0.0
    norm_context: bool = False

    adapt_arch: str = "rnn"
    # what the adaptation module learns to predict
    adaptation_key: Any = "context"
    adaptation_loss: str = "mse" # mse, action_kl
    use_separate_critics: bool = True

    # coefficients for using adaptation error as exploration bonus or penalty
    exp_reward: float = 0.0
    reg_reward: float = 0.0

    def __post_init__(self):
        assert self.condition_mode.lower() in ("cat", "film")
        assert self.adaptation_key in ("context", OBS_HIST_KEY, "_feature")
        assert self.phase in ("train", "adapt", "finetune")

cs = ConfigStore.instance()
cs.store("final_train", node=PPOConfig, group="algo")


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
    def __init__(self, feature_dim: int, context_dim: int):
        super().__init__()
        self.obs_enc = make_mlp([128])
        self.actor_enc = make_mlp([256])
        self.critic_enc = make_mlp([256])
        self.gru = GRU(128, hidden_size=128, allow_none=False)
        # self.out_feature = nn.LazyLinear(feature_dim)
        self.out_context = nn.LazyLinear(context_dim)
    
    def forward(self, x, is_init, hx):
        obs_enc = self.obs_enc(x)
        actor_enc = self.actor_enc(x)
        critic_enc = self.critic_enc(x)
        x, hx = self.gru(obs_enc, is_init, hx)
        # feature = self.out_feature(obs_enc)
        context = self.out_context(x)
        return actor_enc, critic_enc, context.detach(), hx


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1), nn.Mish(),
            nn.LazyConv2d(16, kernel_size=3, stride=2, padding=1), nn.Mish(),
            nn.Flatten(),
            make_mlp([32])
        )
        self.mlp = make_mlp([256, 96])

    def forward(self, propri, height_scan):
        enc = torch.cat([self.mlp(propri), self.conv(height_scan)], -1)
        return enc


class Policy(TensorDictModuleBase):
    
    # @utils._set_auto_make_functional(False)
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
        
        self.exp_reward = self.cfg.exp_reward
        self.reg_reward = self.cfg.reg_reward

        if not isinstance(self.adaptation_key, str):
            self.adaptation_key = tuple(self.adaptation_key)
        self.gae_expert = GAE((0.995, 0.99, 0.99), 0.95).to(self.device)
        self.gae_adapt = GAE((0.995, 0.99, 0.99), 0.95).to(self.device)
        
        self.action_dim = action_spec.shape[-1]

        print(observation_spec)
        observation_priv_dim = observation_spec[OBS_PRIV_KEY].shape[-1]
        observation_dim = observation_spec[OBS_KEY].shape[-1]

        fake_input = observation_spec.zero()

        # create and initialize the modules
        if "height_scan" in observation_spec.keys():
            self.context_dim = 128
            self.encoder = TensorDictModule(
                Encoder(), [OBS_PRIV_KEY, "height_scan"], ["context_expert"]
            ).to(self.device)
        else:
            self.context_dim = 128
            encoder = make_mlp([256, self.context_dim])
            self.encoder = TensorDictModule(
                encoder,
                [OBS_PRIV_KEY], 
                ["context_expert"]
            ).to(self.device)

        def make_actor(feature_key: str) -> ProbabilisticActor:
            actor = nn.Sequential(make_mlp([256, 128]), Actor(self.action_dim, cfg.predict_std))
            return ProbabilisticActor(
                TensorDictModule(actor, [feature_key], ["loc", "scale"]),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True
            )
        
        def make_critic(feature_key: str):
            critic = nn.Sequential(make_mlp([256, 128]), nn.LazyLinear(3))
            return TensorDictModule(critic, [feature_key], ["state_value"])

        self.actor_expert = make_actor("feature_expert_actor").to(self.device)
        self.actor_target = make_actor("feature_adapt_actor").to(self.device)
        # self.actor_expert = make_actor("feature_expert").to(self.device)
        # self.actor_target = make_actor("feature_adapt").to(self.device)

        # the critics always observe priviledged information
        self.critic_expert = make_critic("feature_expert_critic").to(self.device)
        # self.critic_adapt = make_critic("feature_adapt").to(self.device)

        self.value_norm_expert = ValueNorm1(3).to(self.device)
        self.value_norm_adapt = ValueNorm1(3).to(self.device)
        
        self.adapt_module = TensorDictModule(
            GRUModule(128, self.context_dim),
            [OBS_KEY, "is_init", "hx"],
            ["feature_actor", "feature_critic", "context_adapt", ("next", "hx")]
        ).to(self.device)

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

        fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool, device=self.device)
        fake_input["hx"] = torch.zeros(fake_input.shape[0], 128, device=self.device)
        
        self.encoder(fake_input)
        self.adapt_module(fake_input)
        self._get_features(fake_input)
        self.actor_expert(fake_input)
        self.actor_target(fake_input)
        self.critic_expert(fake_input)
        # self.critic_adapt(fake_input)
        print(fake_input)
        
        self.opt = torch.optim.Adam(
            [
                {"params": self.actor_expert.parameters()},
                {"params": self.critic_expert.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.adapt_module.parameters()}
            ], 
            lr=self.cfg.lr
        )

        if self.phase in ("adapt", "finetune"):
            self.make_adapt_policy(fake_input)

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
            self.phase in ("adapt", "finetune") 
            and not any(key.startswith("actor_adapt") for key in state_dict.keys())
        ):
            logging.info("Initialize the adaptation actor with the expert actor.")
            hard_copy_(self.actor_expert, self.actor_adapt)
        # initialize the target actor
        hard_copy_(self.actor_expert, self.actor_target)
        self.actor_target.requires_grad_(False)
        
        def make_tensordict_primer():
            num_envs = observation_spec.shape[0]
            return TensorDictPrimer({
                "hx": UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)
            })
        self.make_tensordict_primer = make_tensordict_primer
        self.make_batch = functools.partial(make_batch, seq_len=cfg.train_every)
        
        self.exclude_keys = ["context_terrain", "context_state", "feature"]
        self.train_iter = 0

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
                )
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
                nn.Sequential(make_mlp([256, 128]), nn.LazyLinear(4)), 
                ["_feature_critic"], 
                ["state_value"]
            )
        ).to(self.device)
        
        self.actor_adapt(fake_input)
        self.critic_adapt(fake_input)
        self.critic_adapt.apply(init_)
        
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
        self.adapt_module(tensordict)
        self._get_features(tensordict)
        if self.phase == "train":
            self.actor_expert(tensordict)
            if not self.training:
                self.critic_expert(tensordict)
        else:
            self.actor_adapt(tensordict)
            if not self.training:
                self.adaptation_loss(tensordict, tensordict, mean=False)
                self.critic_adapt(tensordict)
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
            with (
                hold_out_net(self.encoder),
                hold_out_net(self.actor_expert),
                hold_out_net(self.actor_adapt)
            ):
                info.update(self._train_adaptation(tensordict))
        elif self.phase == "finetune":
            with hold_out_net(self.encoder):
                info.update(self._finetune(tensordict))
        self.train_iter += 1
        return info
    
    def _get_features(self, tensordict: TensorDict):
        # tensordict["feature_expert"] = torch.cat([tensordict["feature"], tensordict["context_expert"]], -1)
        # tensordict["feature_adapt"] = torch.cat([tensordict["feature"], tensordict["context_adapt"]], -1)
        feature_actor = tensordict.pop("feature_actor")
        feature_critic = tensordict.pop("feature_critic")
        tensordict["feature_expert_actor"] = torch.cat([feature_actor, tensordict["context_expert"]], -1)
        tensordict["feature_expert_critic"] = torch.cat([feature_critic, tensordict["context_expert"]], -1)
        tensordict["feature_adapt_actor"] = torch.cat([feature_actor, tensordict["context_adapt"]], -1)
        return tensordict

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
            for minibatch in self.make_batch(tensordict, self.cfg.num_minibatches):
                infos_policy.append(self._update(minibatch, self.critic_expert))
        infos.update(collect_info(infos_policy))

        # infos_adapt = []
        # for epoch in range(1):
        #     for minibatch in make_batch(tensordict, 8, self.cfg.train_every):
        #         infos_adapt.append(self._update_adaptation(minibatch))
        # infos.update(collect_info(infos_adapt, "adapt/"))

        mean, var = self.value_norm_expert.running_mean_var()
        for name, value_mean in zip(
            ["value_mean", "value_kl", "value_mse"], 
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
        with tensordict.view(-1) as flattened_tensordict:
            self.encoder(flattened_tensordict)
        self.adapt_module(tensordict)
        self._get_features(tensordict)
        actor = self.actor_expert
        
        losses = {}

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

        infos = losses
        infos["actor_grad_norm"] = actor_grad_norm
        infos["critic_grad_norm"] = critic_grad_norm
        infos["explained_var"] = explained_var
        infos["entropy"] = entropy
        return TensorDict(infos, []).detach()
    
    # @functools.partial(torch.compile, mode="reduce-overhead")
    def _update_adaptation(self, tensordict: TensorDict):
        losses = {}
        self.adaptation_loss(tensordict, losses, mean=True)
        # value_loss_adapt, explained_var_adapt = compute_value_loss(
        #     tensordict.detach(), 
        #     self.critic_adapt, 
        #     self.clip_param, 
        #     self.critic_loss_fn
        # )
        # losses["value_loss"] = value_loss_adapt
        loss = sum(losses.values())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return TensorDict(losses, []).detach()
    
    def _train_adaptation(self, tensordict: TensorDict):
        self._compute_rewards(tensordict)
        self._compute_advantage(
            tensordict,
            self.critic_adapt,
            self.gae_adapt,
            self.value_norm_adapt
        )

        infos = []
        for epoch in range(4):
            for batch in self.make_batch(tensordict, 8):
                losses = TensorDict({}, [])
                self.adaptation_loss(batch, losses, mean=True)
                value_loss_adapt, explained_var_adapt = compute_value_loss(
                    batch.detach(), 
                    self.critic_adapt, 
                    self.clip_param, 
                    self.critic_loss_fn
                )
                losses["value_loss"] = value_loss_adapt
                loss = sum(losses.values())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                losses["explained_var"] = explained_var_adapt
                infos.append(losses)
        
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])

        mean, var = self.value_norm_adapt.running_mean_var()
        for i in range(mean.shape[-1]):
            infos[f"value_mean_{i}"] = mean[i]
        return {k: v.item() for k, v in infos.items()}

    def _finetune(self, tensordict: TensorDict):
        self._compute_rewards(tensordict)
        weights = (1., 0., 0., 0.)
        self._compute_advantage(
            tensordict, 
            self.critic_adapt,
            self.gae_adapt,
            self.value_norm_adapt, 
            weights
        )
        
        infos = {}
        # update policy (actor and critic)
        if self.train_iter > 32:
            infos_policy = []
            for epoch in range(self.cfg.ppo_epochs):
                if epoch > 0 and self.cfg.recompute_adv:
                    self._compute_advantage(
                        tensordict, 
                        self.critic_adapt,
                        self.gae_adapt,
                        self.value_norm_adapt, 
                        weights
                    )
                for minibatch in self.make_batch(tensordict, self.cfg.num_minibatches):
                    infos_policy.append(self._update(minibatch, self.critic_adapt))
            infos.update(collect_info(infos_policy))

        # update adaptation module
        infos_adapt = []
        for epoch in range(2):
            for minibatch in self.make_batch(tensordict, 8):
                infos_adapt.append(self._update_adaptation(minibatch))
        infos.update(collect_info(infos_adapt, "adapt/"))

        mean, var = self.value_norm_adapt.running_mean_var()
        for i in range(mean.shape[-1]):
            infos[f"value_mean_{i}"] = mean[i].item()
        return infos
    
    @torch.no_grad()
    def _compute_rewards(self, tensordict: TensorDict):
        """
        Compute auxiliary rewards.
        """

        with tensordict["next"].view(-1) as next_tensordict:
            self.encoder(next_tensordict)
            self.adapt_module(next_tensordict)
            self._get_features(next_tensordict)

            kl_target = action_kl(tensordict.exclude(), self.actor_target, self.actor_expert)
            self.adaptation_loss(tensordict, tensordict, mean=False)
            adapt_error = tensordict["adaptation_loss"]
        
        reward_aug = torch.cat([tensordict[REWARD_KEY], kl_target, adapt_error], -1)
        # non-episodic aux rewards
        done = torch.cat([tensordict[DONE_KEY]]+ [torch.zeros_like(tensordict[DONE_KEY])] * 2, -1)
        tensordict.set(REWARD_KEY, reward_aug)
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
        values = critic(tensordict.view(-1))["state_value"].reshape(*tensordict.shape, -1)
        next_values = critic(tensordict["next"].view(-1))["state_value"].reshape(*tensordict.shape, -1)
        
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

