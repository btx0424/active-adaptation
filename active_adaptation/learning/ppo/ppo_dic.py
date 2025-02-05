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
import warnings
import functools
import einops
import copy

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuous
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase, 
    TensorDictModule as Mod, 
    TensorDictSequential as Seq
)
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from ..modules.rnn import GRU, set_recurrent_mode, recurrent_mode
from .common import *

torch.set_float32_matmul_precision('high')

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_dic.PPODICPolicy"
    name: str = "ppo_dic"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 4
    lr: float = 5e-4
    clip_param: float = 0.2
    # entropy_coef: float = 0.004
    # entropy_coef: float = 0.002
    entropy_coef_start: float = 0.004
    entropy_coef_end: float = 0.000

    reg_lambda: float = 0.0
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    phase: str = "train"
    short_history: int = 0
    vecnorm: Union[str, None] = None
    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = field(default_factory=lambda: ["command_", OBS_KEY, OBS_PRIV_KEY, "ext", "ext_", "action_buf_", "symmetry"])

cs = ConfigStore.instance()
cs.store("ppo_dic_train", node=PPOConfig(phase="train", vecnorm="train"), group="algo")
cs.store("ppo_dic_adapt", node=PPOConfig(phase="adapt", vecnorm="eval"), group="algo")
cs.store("ppo_dic_finetune", node=PPOConfig(phase="finetune", vecnorm="eval"), group="algo")

class GRU(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        burn_in: bool = False
    ) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.burn_in = burn_in

    def forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        if recurrent_mode():
            N, T = x.shape[:2]
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
        else:
            N = x.shape[0]
            hx = self.gru(x, hx)
            output = self.ln(hx)
            return output, hx


class GRUModule(nn.Module):
    def __init__(self, dim: int, split):
        super().__init__()
        self.split = split
        self.mlp = make_mlp([128, 128])
        self.gru = GRU(128, hidden_size=128)
        self.out = nn.LazyLinear(dim)
    
    def forward(self, x, is_init, hx):
        out1 = self.mlp(x)
        out2, hx = self.gru(out1, is_init, hx)
        out3 = self.out(out2 + out1)
        if self.split is None:
            out = (out3,)
        else:
            out = torch.split(out3, self.split, dim=-1)
        return out + (hx.contiguous(),)


class PolicyUpdateInferenceMod:
    def __init__(
        self, 
        actor: ProbabilisticActor, 
        encoder: Mod=None,
    ) -> None:
        self.actor = actor
        self.encoder = encoder
    
    def __call__(self, tensordict: TensorDictBase, grad_pen: bool=False):
        # TODO@botian: write to tensordict?
        if self.encoder is not None:
            self.encoder(tensordict)
        for k in self.actor.in_keys:
            tensordict[k].requires_grad_(True)
        if grad_pen:
            dist = self.actor.get_dist(tensordict)
            log_probs = dist.log_prob(tensordict[ACTION_KEY])
            entropy = dist.entropy().mean()
            grad = torch.autograd.grad(
                outputs=log_probs,
                inputs=[tensordict[k] for k in self.actor.in_keys],
                grad_outputs=torch.ones_like(log_probs),
                retain_graph=True,
                create_graph=True,
            )
            grad = torch.cat(grad, dim=-1)
        else:
            dist = self.actor.get_dist(tensordict)
            log_probs = dist.log_prob(tensordict[ACTION_KEY])
            entropy = dist.entropy().mean()
            with torch.no_grad():
                grad = torch.autograd.grad(
                    outputs=log_probs,
                    inputs=[tensordict[k] for k in self.actor.in_keys],
                    grad_outputs=torch.ones_like(log_probs),
                    retain_graph=True,
                )
                grad = torch.cat(grad, dim=-1)
        return log_probs, entropy, grad


class ContinuityLoss:
    def __init__(self, module: ModBase,in_keys, out_key):
        self.module = module
        self.in_keys = in_keys
        self.out_key = out_key
    
    def __call__(self, tensordict: TensorDictBase):
        for in_key in self.in_keys:
            tensordict[in_key].requires_grad_(True)
        out = self.module(tensordict)[self.out_key]
        gradient = torch.autograd.grad(
            outputs=out,
            inputs=[tensordict[in_key] for in_key in self.in_keys],
            grad_outputs=torch.ones_like(out),
            retain_graph=True,
            create_graph=True,
        )
        gradient = torch.cat(gradient, dim=-1)
        gradient_penalty = gradient.square().sum(-1).mean()
        return gradient_penalty


class PPODICPolicy(TensorDictModuleBase):
    """
    
    version: 0.1.0, 2024.9.22 @botian
    * cleanup imitation stuff
    * add finetune phase
    * report ext_rec_error instead of ext_rec_loss
    * fix explicit force est

    version: 0.1.1, 2024.10.29 @botian
    * fix loss func reduction following torch update
    * default reg_lambda = 0.0
    * increase rec_lambda to 0.1

    version: 0.1.2, 2024.11.4 @botian
    * reorganized structure
    * fix bug in checkpoint loading
    * tried ActorCov

    """
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
        self.observation_spec = observation_spec
        assert self.cfg.phase in ["train", "adapt", "finetune"]

        self.entropy_coef = self.cfg.get("entropy_coef_start", 0.001)
        self.max_grad_norm = 1.0
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.adapt_loss_fn = nn.MSELoss(reduction="none")
        self.rec_loss = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.ext_dim = observation_spec["ext_"].shape[-1]
        self.gae = GAE(0.99, 0.95)
        self.reg_lambda = 0.0
        self.ext_rec_lambda = 0.1
        self.symmetry_coef = 0.
        self.rec_rew = 0.
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        
        self.encoder_priv = Seq(
            Mod(nn.Sequential(make_mlp([128]), nn.LazyLinear(128)), [OBS_PRIV_KEY], ["priv_feature"]),
            Mod(nn.Sequential(make_mlp([32]), nn.LazyLinear(32)), ["ext"], ["ext_feature"]),
        ).to(self.device)

        if self.cfg.short_history > 0:
            global OBS_KEY 
            OBS_KEY = OBS_KEY + "_h"

        ext_shape = observation_spec["ext_"].shape[-1]
        self.adapt_module =  Mod(
            GRUModule(128 + 32 + ext_shape, split=[128, 32, ext_shape]), 
            [OBS_KEY, "is_init", "adapt_hx"], 
            ["priv_pred", "ext_pred", ("info", "ext_rec"), ("next", "adapt_hx")]
        ).to(self.device)
        
        in_keys = ["command_", OBS_KEY, "priv_feature", "ext_feature"]
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=Seq(
                CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                Mod(make_mlp([512, 256, 256]), ["_actor_inp"], ["_actor_feature"]),
                Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"])
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.dynamics = Seq(
            CatTensors(["_actor_feature", ACTION_KEY], "_dyn_inp", del_keys=False),
            Mod(make_mlp([256, 128]), ["_dyn_inp"], ["_dyn_pred"]),
            Mod(nn.LazyLinear(12), ["_dyn_pred"], ["_dyn_pred"])
        ).to(self.device)

        self.retro = Seq(
            CatTensors(["_actor_feature", ACTION_KEY], "retro_inp", del_keys=False),
            Mod(make_mlp([256, 128]), ["retro_inp"], ["retro_feat"]),
            Mod(nn.LazyLinear(4), ["retro_feat"], [("info", "retro_pred")])
        ).to(self.device)

        in_keys = ["command_", OBS_KEY, "priv_pred", "ext_pred"]
        self.actor_adapt: ProbabilisticActor = ProbabilisticActor(
            module=Seq(
                CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                Mod(make_mlp([512, 256, 256]), ["_actor_inp"], ["_actor_feature"]),
                Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"])
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        _critic = nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(1))
        self.critic = Seq(
            CatTensors(["command_", OBS_KEY, OBS_PRIV_KEY, "ext"], "_critic_input", del_keys=False),
            Mod(_critic, ["_critic_input"], ["state_value"])
        ).to(self.device)

        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["adapt_hx"] = torch.zeros(fake_input.shape[0], 128)
            fake_input["prev_action"] = torch.zeros(fake_input.shape[0], self.action_dim)
            fake_input["prev_loc"] = torch.zeros(fake_input.shape[0], self.action_dim)

        self.encoder_priv(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        self.adapt_module(fake_input)
        self.actor_adapt(fake_input)
        self.dynamics(fake_input)
        self.retro(fake_input)

        self.policy_train_inference = PolicyUpdateInferenceMod(self.actor, self.encoder_priv)
        self.policy_adapt_inference = PolicyUpdateInferenceMod(self.actor_adapt, None)
        
        self.adapt_ema = copy.deepcopy(self.adapt_module)
        self.adapt_ema.requires_grad_(False)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 1) # sqrt(2)
        
        # self.random = nn.Sequential(make_mlp([256, 256], norm=None, activation=nn.LeakyReLU), nn.LazyLinear(32)).to(self.device)
        # self.random(fake_input["symmetry"])
        # self.random.requires_grad_(False)
        # self.random.apply(init_)
        
        # self.random_pred = nn.Sequential(make_mlp([256, 256], norm=None, activation=nn.LeakyReLU), nn.LazyLinear(32)).to(self.device)
        # self.random_pred(fake_input["symmetry"])        
        # self.random_pred.apply(init_)

        # self.opt_rnd = torch.optim.Adam(self.random_pred.parameters())

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
                {"params": self.encoder_priv.parameters()},
                # {"params": self.dynamics.parameters()},
                {"params": self.retro.parameters()},
            ],
            lr=cfg.lr
        )

        self.opt_adapt = torch.optim.Adam(
            [
                {"params": self.adapt_module.parameters()},
            ],
            lr=cfg.lr
        )

        self.opt_finetune = torch.optim.Adam(
            [
                {"params": self.actor_adapt.parameters()},
                {"params": self.critic.parameters()},
                {"params": self.retro.parameters()},
            ],
            lr=cfg.lr
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)
        self.encoder_priv.apply(init_)
        self.adapt_module.apply(init_)
        # self.dynamics.apply(init_)
        self.retro.apply(init_)

        # self.symmetry = nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)).to(self.device)
        # self.symmetry(fake_input["symmetry"])
        # self.symmetry.apply(init_)
        # self.opt_symmetry = torch.optim.Adam(self.symmetry.parameters(), lr=5e-4)
        
        self.num_updates = 0
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        return TensorDictPrimer({
            "adapt_hx": UnboundedContinuous((num_envs, 128), device=self.device),
            # "prev_action": UnboundedContinuous((num_envs, self.action_dim), device=self.device),
            # "prev_loc": UnboundedContinuous((num_envs, self.action_dim), device=self.device),
        }, reset_key="done")

    def get_rollout_policy(self, mode: str="train"):
        modules = []
        
        def mask(x):
            return torch.zeros_like(x)
        
        if self.cfg.phase == "train":
            modules.append(self.encoder_priv)
            # modules.append(Mod(mask, ["ext_feature"], ["ext_feature"]))
            # modules.append(Mod(mask, ["priv_feature"], ["priv_feature"]))
            modules.append(self.actor)
            modules.append(self.adapt_module)
        elif self.cfg.phase == "adapt":
            modules.append(self.adapt_module)
            modules.append(self.actor_adapt)
        elif self.cfg.phase == "finetune":
            modules.append(self.adapt_ema)
            modules.append(self.actor_adapt)
        
        if mode == "deploy":
            # modules.append(self.dynamics)
            modules.append(self.retro)
        
        def foo(tensordict: TensorDict):
            tensordict["next", "prev_action"] = tensordict["action"]
            tensordict["next", "prev_loc"] = tensordict["loc"]
            return tensordict
        # modules.append(foo)
        
        policy = Seq(*modules)
        return policy
    
    def step_schedule(self, progress: float):
        self.reg_lambda = progress * self.cfg.reg_lambda
        self.entropy_coef = self.cfg.entropy_coef_start + (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start) * progress
        self.symmetry_coef = min(progress * 2, 1.)
        self.rec_rew = min(progress * 2, 1.)

    def train_op(self, tensordict: TensorDict):
        info = {}
        if self.cfg.phase == "train":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "adapt":
            info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "finetune":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_adapt(tensordict.copy()))
        self.num_updates += 1
        return info
    
    # @torch.compile
    def train_policy(self, tensordict: TensorDict):    
        infos = []
        # rec_error = self.rec_loss(tensordict["info", "ext_rec"], tensordict["ext_"])
        # tensordict[REWARD_KEY] += -0.1 * self.rec_rew * rec_error.mean(-1, True)
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        policy_inference = self.policy_train_inference if self.cfg.phase == "train" else self.policy_adapt_inference
        opt = self.opt if self.cfg.phase == "train" else self.opt_finetune

        with torch.no_grad():
            both = self.actor.get_dist(tensordict).mean
            wo_ext = self.actor.get_dist(tensordict.replace(ext_feature=torch.zeros_like(tensordict["ext_feature"]))).mean
            wo_priv = self.actor.get_dist(tensordict.replace(priv_feature=torch.zeros_like(tensordict["priv_feature"]))).mean
            
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                info = self._update(minibatch, policy_inference, opt)
                infos.append(TensorDict(info, []))

        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        if self.cfg.phase == "train":
            infos["actor/feature_std"] = tensordict["priv_feature"].std(dim=(0, 1)).mean().item()
        else:
            infos["actor/feature_std"] = tensordict["priv_pred"].std(dim=(0, 1)).mean().item()
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["actor/diff_ext"] = F.l1_loss(both, wo_ext).item()
        infos["actor/diff_priv"] = F.l1_loss(both, wo_priv).item()
        return infos
    
    @set_recurrent_mode(True)
    def train_adapt(self, tensordict: TensorDict):
        infos = []

        with torch.no_grad():
            self.encoder_priv(tensordict)

        for epoch in range(2):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every):
                self.adapt_module(minibatch)
                priv_loss = self.adapt_loss_fn(minibatch["priv_pred"], minibatch["priv_feature"])
                priv_loss = (priv_loss * (~minibatch["is_init"])).mean()
                ext_loss = self.adapt_loss_fn(minibatch["ext_pred"], minibatch["ext_feature"])
                ext_loss = (ext_loss * (~minibatch["is_init"])).mean()
                if self.ext_rec_lambda > 0:
                    ext_rec_error = self.rec_loss(minibatch["info", "ext_rec"], minibatch["ext_"])
                    ext_rec_error = (ext_rec_error * (~minibatch["is_init"])).mean()
                else:
                    ext_rec_error = 0.
                self.opt_adapt.zero_grad()
                (priv_loss + ext_loss + self.ext_rec_lambda * ext_rec_error).backward()
                self.opt_adapt.step()
                infos.append(TensorDict({
                    "adapt/priv_loss": priv_loss,
                    "adapt/ext_loss": ext_loss,
                    "adapt/ext_rec_loss": ext_rec_error,
                }, []))
        
        soft_copy_(self.adapt_module, self.adapt_ema, 0.04)
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        return infos

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: Mod, 
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        with tensordict.view(-1) as tensordict_flat:
            critic(tensordict_flat)
            critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)
        # dones = tensordict["next", "done"]
        # rewards = torch.where(dones, rewards + values * self.gae.gamma, rewards)
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values)
        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    # @torch.compile
    def _update(self, tensordict: TensorDict, policy_inference: PolicyUpdateInferenceMod, opt: torch.optim.Optimizer):
        log_probs, entropy, grad = policy_inference(tensordict, grad_pen=False)

        if self.cfg.phase == "train":
            valid = (tensordict["step_count"] > 1)
        else:
            valid = (tensordict["step_count"] > 5)
        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * valid)
        entropy_loss = - self.entropy_coef * entropy
        
        # self.retro(tensordict)
        # retro_pred = tensordict["info", "retro_pred"]
        # retro_loss = F.l1_loss(retro_pred, tensordict["command_"][:, :retro_pred.shape[-1]])
        # retro_loss = (retro_loss * (~tensordict["is_init"])).mean()
        
        # self.dynamics(tensordict)
        # dyn_loss = F.l1_loss(tensordict["_dyn_pred"], tensordict["next", "state_"][..., :12])
        # a_tm1 = tensordict["action_buf_"][:, :, 0]
        # a_tm2 = tensordict["action_buf_"][:, :, 1]
        # smth1_loss = torch.mean(torch.square(action - a_tm1).sum(-1) * valid) # first order smth
        # smth2_loss = torch.mean(torch.square(action - 2 * a_tm1 + a_tm2).sum(-1) * valid) # second order smth

        gradient_penalty = grad.square().sum(-1).mean()

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()

        if self.cfg.phase == "train" and self.reg_lambda > 0:
            reg_loss = self.adapt_loss_fn(tensordict["priv_feature"], tensordict["priv_pred"])
            reg_loss = self.reg_lambda * (reg_loss * (~tensordict["is_init"])).mean()
        else:
            reg_loss = 0.
            
        loss = policy_loss + entropy_loss + value_loss + reg_loss + 0.002 * gradient_penalty
        
        opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(policy_inference.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        opt.step()
        
        explained_var = 1 - value_loss / b_returns[~tensordict["is_init"]].var()
        info = {
            "actor/policy_loss": policy_loss,
            "actor/entropy": entropy,
            "actor/noise_std": tensordict["scale"].mean(),
            "actor/grad_norm": actor_grad_norm,
            'actor/approx_kl': ((ratio - 1) - log_ratio).mean(),
            "actor/gradient_penalty": gradient_penalty,
            # "actor/smth1_loss": smth1_loss,
            # "actor/smth2_loss": smth2_loss,
            # "actor/dyn_loss": dyn_loss,
            # "actor/retro_loss": retro_loss,
            "adapt/reg_loss": reg_loss,
            "critic/value_loss": value_loss,
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }
        return info

    def _update_symmetry(self, tensordict: TensorDict):
        # left_obs, right_obs = tensordict["symmetry"].unbind(1)
        # rnd_pred = self.random_pred(left_obs)
        # rnd_loss = F.mse_loss(rnd_pred, self.random(left_obs))

        left_obs, right_obs = tensordict["symmetry"].unbind(1)
        left_obs.requires_grad_(True)
        left_score = self.symmetry(left_obs)
        right_score = self.symmetry(right_obs)
        valid = (~tensordict["is_init"]).float()
        loss_left = (left_score - 1).square()
        loss_right = (right_score + 1).square()
        symmetry_loss = torch.mean((loss_left + loss_right) * valid)

        grad = torch.autograd.grad(
            left_score,
            left_obs, 
            torch.ones_like(left_score),
            retain_graph=True,
            create_graph=True
        )[0]
        gradient_penalty = torch.mean(grad.square().sum(dim=-1))
        
        self.opt_symmetry.zero_grad()
        (symmetry_loss + 5 * gradient_penalty).backward()
        self.opt_symmetry.step()

        return {
            "symmetry/loss": symmetry_loss,
            "symmetry/gradient_penalty": gradient_penalty,
            # "symmetry/acc": ((left_score > 0) & (right_score < 0)).float().mean(),
            # "symmetry/loss_rnd": rnd_loss
        }
    
    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        state_dict["last_phase"] = self.cfg.phase
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
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
        if state_dict.get("last_phase", "train") == "train":
            # only copy to initialize the actor once
            hard_copy_(self.actor, self.actor_adapt)
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
