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
import time

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE, Actor, make_mlp
from .ppo_rnn import GRU

@dataclass
class PPOConfig:
    name: str = "ppo_dual"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    version: str = "v0"
    adaptation_loss: str = "feature_mse"
    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_dual_v0", node=PPOConfig(version="v0"), group="algo")
cs.store("ppo_dual_v1", node=PPOConfig(version="v1"), group="algo")
cs.store("ppo_dual_v2", node=PPOConfig(version="v2"), group="algo")


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


class PPODualPolicy(TensorDictModuleBase):

    OBS_KEY = ("agents", "observation")
    OBS_HIST_KEY = ("agents", "observation_h")
    OBS_PRIV_KEY = ("agents", "observation_priv")

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
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        # observation_dim = observation_spec[self.OBS_KEY].shape[-1]
        # observation_priv_dim = observation_spec[self.OBS_PRIV_KEY].shape[-1]
        
        getattr(self, f"make_models_{cfg.version}")()

        self.actor_critic = TensorDictSequential(self.actor, self.critic)
        self.classifier = nn.Sequential(
            make_mlp([256, 256]),
            nn.LazyLinear(1) 
        ).to(self.device)

        self.encoder(fake_input)
        self.adapt(fake_input)
        self.actor_critic(fake_input)
        
        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
                if isinstance(module, nn.Conv1d):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            
            self.actor.apply(init_)
            self.critic.apply(init_)
            self.encoder.apply(init_)
            self.adapt.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.adapt_opt = torch.optim.Adam(self.adapt.parameters(), lr=5e-4)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        self.classifier_opt = torch.optim.Adam(self.classifier.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(input_shape=1).to(self.device)

        self.adaptation_loss = {
            "feature_mse": self.feature_mse, 
            "action_kl": self.action_kl
        }[cfg.adaptation_loss]

        self.train_adaptation = False
        self.train_classifier = True
        self.adapt_ratio = 0.
    
    def step_schedule(self):
        self.train_adaptation = True
        self.adapt_ratio = 0.05

    def make_models_v0(self):
        self.adapt = TensorDictModule(
            TConv(256),
            [self.OBS_HIST_KEY], 
            ["_feature"]
        ).to(self.device)

        self.encoder = TensorDictSequential(
            CatTensors([self.OBS_KEY, self.OBS_PRIV_KEY], "_obs", del_keys=False),
            TensorDictModule(
                make_mlp([512, 256], nn.Mish),
                ["_obs"], ["_feature"]
            )
        ).to(self.device)

        actor_module = TensorDictModule(
            nn.Sequential(make_mlp([256], nn.Mish), Actor(self.action_dim)),
            ["_feature"], ["loc", "scale"]
        )

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.Sequential(make_mlp([256], nn.Mish), nn.LazyLinear(1)),
            ["_feature"], ["state_value"]
        ).to(self.device)

        self.ADAPT_KEY = "_feature"

    def make_models_v1(self):
        """
        Concat conditioning.
        """
        condition = lambda: CatTensors(["_context", "_feature"], "_feature", del_keys=False)
        
        self.encoder = TensorDictModule(
            make_mlp([256, 256], nn.Mish),
            [self.OBS_PRIV_KEY],
            ["_context"]
        ).to(self.device)

        self.adapt = TensorDictModule(
            TConv(256, nn.Mish),
            [self.OBS_HIST_KEY],
            ["_context"]
        ).to(self.device)

        self.actor = ProbabilisticActor(
            TensorDictSequential(
                TensorDictModule(make_mlp([256, 256], nn.Mish), [self.OBS_KEY], ["_feature"]),
                condition(),
                TensorDictModule(
                    nn.Sequential(make_mlp([256], nn.Mish), Actor(self.action_dim)),
                    ["_feature"], 
                    ["loc", "scale"]
                )
            ),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictSequential(
            TensorDictModule(make_mlp([256, 256], nn.Mish), [self.OBS_KEY], ["_feature"]),
            condition(),
            TensorDictModule(
               nn.Sequential(make_mlp([256], nn.Mish), nn.LazyLinear(1)),
                ["_feature"], 
                ["state_value"]
            )
        ).to(self.device)

        self.ADAPT_KEY = "_context"

    def make_models_v2(self):
        """
        FiLM conditioning.
        """
        self.encoder = TensorDictModule(
            make_mlp([512], nn.Mish),
            [self.OBS_PRIV_KEY],
            ["_context"]
        ).to(self.device)

        self.adapt = TensorDictModule(
            TConv(512, nn.Mish),
            [self.OBS_HIST_KEY],
            ["_context"]
        ).to(self.device)

        class Condition(nn.Module):
            def __init__(self, ):
                super().__init__()
            
            def forward(self, feature: torch.Tensor, context: torch.Tensor):
                weight, bias = context.chunk(2, dim=-1)
                feature = feature * weight + bias
                return feature

        actor = nn.Sequential(make_mlp([256], nn.Mish), Actor(self.action_dim))
        self.actor = ProbabilisticActor(
            TensorDictSequential(
                TensorDictModule(make_mlp([256, 256], nn.Mish), [self.OBS_KEY], ["_feature"]),
                TensorDictModule(Condition(), ["_feature", "_context"], ["_feature"]),
                TensorDictModule(actor, ["_feature"], ["loc", "scale"])
            ), 
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        critic = nn.Sequential(make_mlp([256], nn.Mish), nn.LazyLinear(1))
        self.critic = TensorDictSequential(
            TensorDictModule(make_mlp([256, 256], nn.Mish), [self.OBS_KEY], ["_feature"]),
            TensorDictModule(Condition(), ["_feature", "_context"], ["_feature"]),
            TensorDictModule(critic, ["_feature"], ["state_value"])
        ).to(self.device)

        self.ADAPT_KEY = "_context"

    def __call__(self, tensordict: TensorDict):
        if self.training:
            n_adapt = int(self.adapt_ratio * tensordict.shape[0])
            n_priv = tensordict.shape[0] - n_adapt
            if n_adapt > 0:
                td_priv, td_adapt = tensordict.split([n_priv, n_adapt])
                self.actor_critic(self.encoder(td_priv))
                self.actor_critic(self.adapt(td_adapt))
                for key in (("agents", "action"), "sample_log_prob", "state_value"):
                    tensordict[key] = torch.cat([td_priv[key], td_adapt[key]])
            else:
                self.actor_critic(self.encoder(tensordict))
            tensordict.set(
                "is_adapt", 
                torch.cat([torch.zeros(n_priv, dtype=bool), torch.ones(n_adapt, dtype=bool)])
            )
        else:
            # use adaptation module for testing
            self.adapt(tensordict)
            self.actor_critic(tensordict)
        tensordict.exclude("_feature", "_obs", "loc", "scale", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        start = time.perf_counter()

        next_tensordict = tensordict["next"]
        with torch.no_grad():
            is_adapt = tensordict["is_adapt"]
            self.encoder(next_tensordict)
            if is_adapt.any():
                next_tensordict[self.ADAPT_KEY][is_adapt] = self.adapt(next_tensordict[is_adapt])[self.ADAPT_KEY]
            next_values = self.critic(next_tensordict)["state_value"]
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
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        end = time.perf_counter()
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = {k: torch.mean(v).item() for k, v in infos.items()}
        infos["training_time"] = end - start
        infos["adapt_ratio"] = self.adapt_ratio
        return infos

    def _update(self, tensordict: TensorDict):
        tensordict_priv = self.encoder(tensordict.clone()).exclude("_obs")
        tensordict_adapt = self.adapt(tensordict.clone())
        is_adapt = tensordict["is_adapt"]

        tensordict = torch.where(is_adapt, tensordict_adapt.detach(), tensordict_priv)
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
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        self.encoder.zero_grad(set_to_none=True)
        # self.adapt_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        self.encoder_opt.step()
        # self.adapt_opt.step()
        explained_var = 1 - value_loss_original / b_returns.var()

        info = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }

        if self.train_adaptation:
            loss_adapt = (
                self.adaptation_loss(tensordict_priv, tensordict_adapt)
                # should not match the value of the expert here
                # + F.mse_loss(self.critic(tensordict_adapt)["state_value"], b_returns)
            )
            self.adapt_opt.zero_grad()
            loss_adapt.backward()
            self.adapt_opt.step()
            info["adaptation_loss"] = loss_adapt

        if self.train_classifier:
            with torch.no_grad():
                action_expert = self.actor.build_dist_from_params(
                    self.actor.get_dist_params(tensordict_priv.detach())).sample()
                action_adapt = self.actor.build_dist_from_params(
                    self.actor.get_dist_params(tensordict_adapt.detach())).sample()
            obs = tensordict[self.OBS_PRIV_KEY]
            pred_expert = self.classifier(torch.cat([obs, action_expert], dim=-1))
            pred_adapt = self.classifier(torch.cat([obs, action_adapt], dim=-1))
            classifier_loss = (
                F.binary_cross_entropy_with_logits(pred_expert, torch.ones_like(pred_expert))
                + F.binary_cross_entropy_with_logits(pred_adapt, torch.zeros_like(pred_adapt))
            )
            self.classifier_opt.zero_grad()
            classifier_loss.backward()
            self.classifier_opt.step()
            info["classifier_loss"] = classifier_loss

        return TensorDict(info, [])

    def feature_mse(self, tensordict_priv: TensorDict, tensordict_adapt: TensorDict):
        loss = F.mse_loss(tensordict_adapt[self.ADAPT_KEY], tensordict_priv[self.ADAPT_KEY].detach())
        return loss
    
    def action_kl(self, tensordict_priv: TensorDict, tensordict_adapt: TensorDict):
        dist_target = self.actor.get_dist(tensordict_priv.detach())
        dist_adapt = self.actor.get_dist(tensordict_adapt)
        loss = D.kl_divergence(dist_adapt, dist_target).mean()
        return loss


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]