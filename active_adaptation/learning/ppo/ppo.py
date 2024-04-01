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
from .common import Actor, GAE, make_mlp, make_batch

@dataclass
class PPOConfig:
    name: str = "ppo"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 1e-3
    clip_param: float = 0.2
    recompute_adv: bool = False

    priv_actor: bool = False
    priv_critic: bool = False

    checkpoint_path: Union[str, None] = None
    train_model: bool = False

cs = ConfigStore.instance()
cs.store("ppo", node=PPOConfig, group="algo")
cs.store("ppo_priv", node=PPOConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ppo_priv_critic", node=PPOConfig(priv_critic=True), group="algo")


class PPOPolicy(TensorDictModuleBase):

    OBS_KEY = "policy"
    ACTION_KEY = "action"
    REWARD_KEY = ("next", "reward")
    # DONE_KEY = ("next", "done")
    DONE_KEY = ("next", "terminated")

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

        fake_input = observation_spec.zero()
        observation_dim = observation_spec[self.OBS_KEY].shape[-1]
        
        actor_module=TensorDictModule(
            nn.Sequential(
                # nn.LayerNorm(observation_dim),
                make_mlp([512, 256, 256], nn.Mish), 
                Actor(self.action_dim)
            ),
            [self.OBS_KEY], ["loc", "scale"]
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[self.ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        self.critic = TensorDictModule(
            nn.Sequential(
                # nn.LayerNorm(observation_dim),
                make_mlp([512, 256, 256], nn.Mish), 
                nn.LazyLinear(1)
            ),
            [self.OBS_KEY], ["state_value"]
        ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)
        
        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            
            self.actor.apply(init_)
            self.critic.apply(init_)

        if self.cfg.train_model:
            self.aux_model = AuxModel(observation_spec).to(self.device)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()}
            ],
            lr=cfg.lr
        )
        self.value_norm = ValueNorm1(input_shape=1).to(self.device)
    
    # @torch.compile
    def __call__(self, tensordict: TensorDict):
        tensordict = self.actor(tensordict)
        tensordict = self.critic(tensordict)
        tensordict = tensordict.exclude("loc", "scale", "feature")
        return tensordict

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        infos = []

        for epoch in range(self.cfg.ppo_epochs):
            if epoch == 0 or self.cfg.recompute_adv:
                self._compute_advantage(tensordict)
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos = {k: v.mean().item() for k, v in torch.stack(infos).items()}
        if self.cfg.train_model:
            infos.update(self.aux_model.train_op(tensordict))
        infos["value_mean"] = tensordict["ret"].mean().item()
        return infos

    @torch.no_grad()
    def _compute_advantage(self, tensordict: TensorDict, subtract_mean: bool=False):
        values = self.critic(tensordict)["state_value"]
        next_values = self.critic(tensordict["next"])["state_value"]

        rewards = tensordict[self.REWARD_KEY]
        dones = tensordict[self.DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        
        if subtract_mean:
            adv = (adv - adv_mean) / adv_std.clip(1e-7)
        else:
            adv = adv / adv_std.clip(1e-7)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[self.ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * entropy

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped).mean()

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


class AuxModel(nn.Module):
    def __init__(self, observation_spec: CompositeSpec):
        super().__init__()
        observation_dim = observation_spec["policy"].shape[-1]

        self.beta = 0.
        self.model_state = LatentModel(observation_dim)
        self.model_obs = LatentModel(observation_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_op(self, tensordict: TensorDict):
        infos = []
        for minibatch in make_batch(tensordict, 16):
            infos.append(self._update(minibatch))
            
        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}
    
    def _update(self, tensordict: TensorDict):
        obs_policy = tensordict["policy"]
        obs_priv = tensordict["priv"]
        obs_next = tensordict["next", "policy"]
        action = tensordict["action"]

        loss_obs = self._elbo(
            torch.cat([obs_policy, action], dim=-1), obs_next, self.model_obs
        )

        loss_state = self._elbo(
            torch.cat([obs_policy, obs_priv, action], dim=-1), obs_next, self.model_state
        )
        self.opt.zero_grad()
        (loss_obs + loss_state).backward()
        self.opt.step()

        return TensorDict({"loss_obs": loss_obs, "loss_state": loss_state}, [])
    
    def _elbo(self, x, x_target, model):
        x_hat, mu, logvar = model(x)
        recon = F.mse_loss(x_hat, x_target, reduction="mean").sum(-1)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)
        return (recon + kl * self.beta).mean()


class LatentModel(nn.Module):
    def __init__(self, output_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(latent_dim * 2))
        self.decoder = nn.Sequential(make_mlp([128, 256, 256]), nn.LazyLinear(output_dim))
    
    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        mu, logvar = torch.chunk(latent, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar