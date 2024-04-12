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

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, TensorDictPrimer, ExcludeTransform
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union

from ..utils.valuenorm import ValueNorm1
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

    context_dim: int = 128

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
        self.value_norm_a = ValueNorm1(input_shape=1).to(self.device)
        self.value_norm_b = ValueNorm1(input_shape=1).to(self.device)

        self.observation_spec = observation_spec
        fake_input = observation_spec.zero()

        self.log_alpha = nn.Parameter(torch.tensor(0.))
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=1e-2)
        self.target_kl = 2.0
        
        self.encoder_priv = TensorDictModule(
            make_mlp([self.cfg.context_dim]), [OBS_PRIV_KEY], ["context_expert"]
        ).to(self.device)
        
        self.adapt_module = TensorDictModule(
            GRUModule(self.cfg.context_dim), 
            [OBS_KEY, "is_init", "context_adapt_hx"], 
            ["context_adapt", ("next", "context_adapt_hx")]
        ).to(self.device)
        self.adapt_module_ema = TensorDictModule(
            GRUModule(self.cfg.context_dim),
            [OBS_KEY, "is_init", "context_adapt_hx"],
            ["context_adapt", ("next", "context_adapt_hx")]
        ).to(self.device)
        
        def make_actor(context_key: str) -> ProbabilisticActor:
            actor_module = nn.Sequential(make_mlp([512, 256, 256]), Actor(self.action_dim))
            actor = ProbabilisticActor(
                module=TensorDictSequential(
                    # TensorDictModule(make_mlp([self.context_dim]), [OBS_PRIV_KEY], ["context_expert"]),
                    CatTensors([OBS_KEY, context_key], "actor_feature", del_keys=False),
                    TensorDictModule(actor_module, ["actor_feature"], ["loc", "scale"])
                ),
                # module=TensorDictSequential(
                #     CatTensors([OBS_KEY, OBS_PRIV_KEY], "actor_feature", del_keys=False),
                #     TensorDictModule(make_actor(), ["actor_feature"], ["loc", "scale"])
                # ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True
            ).to(self.device)
            return actor
        
        def make_critic():
            return nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(1))
        
        # expert actor with priviledged information
        self._actor_expert = make_actor("context_expert")
        # expert actor with estimated information used for compute estimation error
        self.__actor_expert = make_actor("context_adapt")
        # target actor with estimated information
        self._actor_adapt = make_actor("context_adapt")

        # expert critic with priviledged information
        self._critic_expert = TensorDictSequential(
            # TensorDictModule(make_mlp([self.context_dim]), [OBS_PRIV_KEY], ["context_expert"]),
            CatTensors([OBS_KEY, "context_expert"], "critic_feature", del_keys=False),
            TensorDictModule(make_critic(), ["critic_feature"], ["state_value"])
        ).to(self.device)
        # self._critic_expert = TensorDictSequential(
        #     CatTensors([OBS_KEY, OBS_PRIV_KEY], "critic_feature", del_keys=False),
        #     TensorDictModule(make_critic(), ["critic_feature"], ["state_value"])
        # ).to(self.device)

        # expert critic with both priviledged and estimated information to be unbiased
        self._critic_adapt = TensorDictSequential(
            CatTensors([OBS_KEY, "context_expert", "context_adapt"], "critic_feature", del_keys=False),
            TensorDictModule(make_critic(), ["critic_feature"], ["state_value"])
        ).to(self.device)

        self.classifer_a = ClassifierA().to(self.device)
        self.classifer_b = ClassifierB().to(self.device)

        self.actor_critic_expert = TensorDictSequential(
            self.encoder_priv,
            self._actor_expert,
            self._critic_expert
        )

        # lazy initialization
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["context_adapt_hx"] = torch.zeros(fake_input.shape[0], 128)
        
        self.encoder_priv(fake_input)
        self.adapt_module(fake_input)
        self.adapt_module_ema(fake_input)
        self._actor_expert(fake_input)
        self._critic_expert(fake_input)
        self._actor_adapt(fake_input)
        self._critic_adapt(fake_input)

        self.classifer_a(fake_input[OBS_KEY], fake_input[OBS_PRIV_KEY], fake_input[ACTION_KEY])
        self.classifer_b(fake_input[OBS_KEY], fake_input[OBS_PRIV_KEY], fake_input[ACTION_KEY])

        self.opt_expert = torch.optim.Adam(
            [
                {"params": self.encoder_priv.parameters()},
                {"params": self._actor_expert.parameters()},
                {"params": self._critic_expert.parameters()},
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
                {"params": self.adapt_module.parameters()},
                {"params": self.classifer_a.parameters()},
                {"params": self.classifer_b.parameters()},
            ],
            lr=cfg.lr
        )
        
        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            
            self.actor_critic_expert.apply(init_)
            self.adapt_module.apply(init_)

            self.classifer_a.apply(init_)
            self.classifer_b.apply(init_)
        
        self.__actor_expert(fake_input)
        self.__actor_expert.requires_grad_(False)
        hard_copy_(self._actor_expert, self.__actor_expert)

        self.adapt_module_ema.requires_grad_(False)
        hard_copy_(self.adapt_module, self.adapt_module_ema)

        self.phase = self.cfg.phase
        self.num_updates = 0
    
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
                ExcludeTransform("actor_feature", "loc", "scale")
            )
        elif self.phase == "adapt" or self.phase == "finetune":
            policy = TensorDictSequential(
                self.adapt_module_ema,
                self._actor_adapt,
                ExcludeTransform("actor_feature", "loc", "scale")
            )
        else:
            raise NotImplementedError
        return policy
    
    def train_op(self, tensordict: TensorDict):
        infos = {}
        match self.phase:
            case "train":
                infos.update(self.train_expert(tensordict.to_tensordict()))
                infos.update(self.train_adaptation(tensordict.to_tensordict()))
            case "adapt":
                infos.update(self.train_adaptation(tensordict.to_tensordict()))
                infos.update(self.train_target(tensordict.to_tensordict(), train_actor=False))
            case "finetune":
                infos.update(self.train_adaptation(tensordict.to_tensordict()))
                infos.update(self.train_target(tensordict.to_tensordict(), train_actor=True))
            case _:
                raise NotImplementedError
        self.num_updates += 1
        return infos
    
    def train_expert(self, tensordict: TensorDict):
        infos = []
        with torch.no_grad():
            # self.encoder_priv(tensordict)
            # self.adapt_module(tensordict)
            self.encoder_priv(tensordict["next"])
            self.adapt_module_ema(tensordict["next"])
        self._compute_advantage(tensordict, self._critic_expert, "adv_expert", "ret_expert", value_norm=self.value_norm_a)
        self._compute_advantage(tensordict, self._critic_adapt, "adv_adapt", "ret_adapt", value_norm=self.value_norm_b)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                self.encoder_priv(minibatch)
                info = self._update_actor_critic(
                    minibatch, self._actor_expert, self._critic_expert, self.opt_expert, "adv_expert", "ret_expert"
                )
                info_adapt = self._update_actor_critic(
                    minibatch.detach(), None, self._critic_adapt, self.opt_target, "adv_adapt", "ret_adapt"
                )
                info.update({f"adapt/{k}": v for k, v in info_adapt.items()})
                infos.append(info)
        
        hard_copy_(self._actor_expert, self.__actor_expert)
        hard_copy_(self._actor_expert, self._actor_adapt)
        
        infos = {k: v.mean() for k, v in torch.stack(infos).items(True, True)}
        if "ret_expert" in tensordict.keys():
            infos["value_mean_expert"] = self.value_norm_a.denormalize(tensordict["ret_expert"]).mean()
        if "ret_adapt" in tensordict.keys():
            infos["value_mean_adapt"] = self.value_norm_b.denormalize(tensordict["ret_adapt"]).mean()
        return {k: v.item() for k, v in sorted(infos.items())}
    
    def train_target(self, tensordict: TensorDict, train_actor: bool):
        infos = []

        with torch.no_grad():
            # self.encoder_priv(tensordict)
            # self.adapt_module_ema(tensordict)
            self.encoder_priv(tensordict["next"])
            self.adapt_module_ema(tensordict["next"])
            action_kl = self._action_kl(tensordict["next"], reduce=False)
            tensordict[REWARD_KEY] = tensordict[REWARD_KEY] - self.log_alpha.exp() * action_kl

        self._compute_advantage(tensordict, self._critic_expert, "adv_expert", "ret_expert", value_norm=self.value_norm_a)
        self._compute_advantage(tensordict, self._critic_adapt, "adv_adapt", "ret_adapt", value_norm=self.value_norm_b)

        alpha_loss = - self.log_alpha * (self._action_kl(tensordict).detach() - self.target_kl)
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        actor = self._actor_adapt if train_actor else None
        adv_key = "adv_expert"
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                with torch.no_grad():
                    minibatch = self.encoder_priv(minibatch).detach()
                    minibatch = self.adapt_module_ema(minibatch).detach()
                info = self._update_actor_critic(
                    minibatch, None, self._critic_expert, self.opt_expert, "adv_expert", "ret_expert")
                info_target = self._update_actor_critic(
                    minibatch, actor, self._critic_adapt, self.opt_target, adv_key, "ret_adapt")
                info.update({f"adapt/{k}": v for k, v in info_target.items()})
                infos.append(info)
        
        infos = {k: v.mean() for k, v in torch.stack(infos).items(True, True)}
        if "ret_expert" in tensordict.keys():
            infos["value_mean_expert"] = self.value_norm_a.denormalize(tensordict["ret_expert"]).mean()
        if "ret_adapt" in tensordict.keys():
            infos["value_mean_adapt"] = self.value_norm_b.denormalize(tensordict["ret_adapt"]).mean()
        infos["alpha"] = self.log_alpha.exp()
        return {k: v.item() for k, v in sorted(infos.items())}
    
    def train_adaptation(self, tensordict: TensorDict):
        infos = []
        with torch.no_grad():
            self.encoder_priv(tensordict)
        
        for epoch in range(2):
            batch = make_batch(tensordict, 8, self.cfg.train_every)
            for minibatch in batch:
                infos.append(self._update_adaptation(minibatch))
        soft_copy_(self.adapt_module, self.adapt_module_ema)
    
        infos = {f"adapt/{k}": v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        with torch.no_grad():
            infos["adapt/action_kl"] = self._action_kl(tensordict).item()
        return infos

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TensorDictModule, 
        adv_key: str="adv",
        ret_key: str="ret",
        value_norm: ValueNorm1=None,
        update_value_norm: bool=True,
    ):
        values = critic(tensordict)["state_value"]
        next_values = critic(tensordict["next"])["state_value"]

        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = value_norm.denormalize(values)
        next_values = value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = normalize(adv, subtract_mean=True)

        if update_value_norm:
            value_norm.update(ret)
        ret = value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update_actor_critic(
        self,
        tensordict: TensorDict,
        actor: ProbabilisticActor,
        critic: TensorDictModule,
        opt: torch.optim.Optimizer,
        adv_key: str="adv",
        ret_key: str="ret"
    ):
        losses = TensorDict({}, [])
        if actor is not None:
            dist = actor.get_dist(tensordict)
            log_probs = dist.log_prob(tensordict[ACTION_KEY])
            entropy = dist.entropy().mean()

            adv = tensordict[adv_key]
            ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
            surr1 = adv * ratio
            surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
            policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
            entropy_loss = - self.entropy_coef * entropy
            losses["policy_loss"] = policy_loss
            losses["entropy_loss"] = entropy_loss
        
        if critic is not None:
            b_returns = tensordict[ret_key]
            values = critic(tensordict)["state_value"]
            value_loss = self.critic_loss_fn(b_returns, values)
            value_loss = (value_loss * (~tensordict["is_init"])).mean()
            losses["value_loss"] = value_loss
            
            explained_var = 1 - F.mse_loss(values.detach(), b_returns) / b_returns.var()
            losses["explained_var"] = explained_var
        
        opt.zero_grad()
        sum(losses.values()).backward()
        # if actor is not None:
        #     actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 10)
        #     losses["actor_grad_norm"] = actor_grad_norm
        # if critic is not None:
        #     critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(critic.parameters(), 10)
        #     losses["critic_grad_norm"] = critic_grad_norm
        for param_group in opt.param_groups:
            nn.utils.clip_grad_norm_(param_group["params"], 10)
        opt.step()
        return losses

    def _update_adaptation(self, tensordict: TensorDictBase):
        losses = TensorDict({}, [])
        losses["adaptation_loss"] = self._feature_mse(tensordict)
        with torch.no_grad():
            action_expert = self._actor_expert(tensordict)[ACTION_KEY]
            action_adapt = self._actor_adapt(tensordict)[ACTION_KEY]
        losses["classifier_a_loss"], acc_a = self._least_square(
            self.classifer_a(tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY], action_expert),
            self.classifer_a(tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY], action_adapt)
        )
        losses["classifier_a_acc"] = acc_a
        # losses["classifier_b_loss"], acc_b = self._cross_entropy(
        #     self.classifer_b(tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY], action_expert),
        #     self.classifer_b(tensordict[OBS_KEY], tensordict[OBS_PRIV_KEY], action_adapt)
        # )
        # losses["classifier_b_acc"] = acc_b
        self.opt_adapt.zero_grad()
        sum(v for k, v in losses.items() if k.endswith("loss")).backward()
        for param_group in self.opt_adapt.param_groups:
            nn.utils.clip_grad_norm_(param_group["params"], 10)
        self.opt_adapt.step()
        return losses

    def _feature_mse(self, tensordict: TensorDict, reduce: bool=True):
        self.adapt_module(tensordict)
        if reduce:
            return F.mse_loss(tensordict["context_adapt"], tensordict["context_expert"])
        else:
            return F.mse_loss(tensordict["context_adapt"], tensordict["context_expert"], reduction="none").mean(-1, True)
    
    def _action_kl(self, tensordict: TensorDict, reduce: bool=True):
        with torch.no_grad():
            dist1 = self._actor_expert.get_dist(tensordict)
        dist2 = self.__actor_expert.get_dist(self.adapt_module(tensordict))
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
    

def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
