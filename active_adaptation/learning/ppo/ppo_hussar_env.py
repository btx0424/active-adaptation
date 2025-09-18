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

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuous
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential,
    CudaGraphModule
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, Tuple
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *

torch.set_float32_matmul_precision('high')

import active_adaptation
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP
from active_adaptation.utils.torchrl import EnsembleCritic, PerEnvMultiHeadCritic

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_hussar_env.PPOPolicy"
    name: str = "ppo_hussar_env"
    train_every: int = 48
    ppo_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.003
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False
    multi_critic: bool = False
    
    compile: bool = False
    use_ddp: bool = True

    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str] = (OBS_KEY, "height_scan", "grid_map_", "base_height", "base_height_targ")
    per_env_critic: bool = True
    num_env_groups: int = 6
    env_id_key: str = "terrain_types"
    per_env_adv_norm: bool = True
    per_env_value_norm: bool = True

cs = ConfigStore.instance()
cs.store("ppo_hussar_env", node=PPOConfig, group="algo")


class MixedEncoder(nn.Module):
    def __init__(self, mlp_out=256, cnn_out=32, conv3d: bool=False):
        super().__init__()
        self.mlp_out = mlp_out
        self.cnn_out = cnn_out
        self.mlp_encoder = nn.Sequential(
            nn.LazyLinear(256), nn.Mish(), nn.LayerNorm(256), 
            nn.LazyLinear(256)
        )

        if conv3d:
            cnn_cls = nn.LazyConv3d
            data_dim = 4 # [C, X, Z, Y]
        else:
            cnn_cls = nn.LazyConv2d
            data_dim = 3 # [C, X, Y]

        self.cnn_encoder = nn.Sequential(
            FlattenBatch(
                nn.Sequential(
                    cnn_cls(8, kernel_size=3, stride=2, padding=1), 
                    nn.Mish(), # nn.GroupNorm(num_channels=2, num_groups=2),
                    cnn_cls(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=4, num_groups=2),
                    cnn_cls(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=8, num_groups=2), 
                    nn.Flatten(),
                ),
                data_dim=data_dim,
            ),
            nn.LazyLinear(32),
            nn.Mish(),
            nn.LayerNorm(32),
            nn.LazyLinear(256)
        )
        self.out = nn.Sequential(nn.Mish(), nn.LazyLinear(256), nn.Mish())

    def forward(self, mlp_inp, cnn_inp, mask_cnn=None):
        cnn_feature = self.cnn_encoder(cnn_inp.float())
        mlp_feature = self.mlp_encoder(mlp_inp)
        if mask_cnn is not None:
            cnn_feature = cnn_feature * mask_cnn
        feature = mlp_feature + cnn_feature
        return self.out(feature)

class _CriticAlignWrapper(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, td: TensorDict) -> TensorDict:
        td = self.inner(td)
        sv = td.get("state_value", None)
        if sv is None:
            return td

        B = T = None
        if DONE_KEY in td.keys(True, True):
            dn = td.get(DONE_KEY)
            if hasattr(dn, "ndim") and dn.ndim >= 2:
                B, T = dn.shape[:2]
        elif ("next", DONE_KEY) in td.keys(True, True):
            dn = td.get(("next", DONE_KEY))
            if hasattr(dn, "ndim") and dn.ndim >= 2:
                B, T = dn.shape[:2]
        if B is None or T is None or sv.ndim < 2:
            return td 

        if sv.ndim >= 3 and sv.shape[0] == T and sv.shape[1] == B:
            sv = sv.permute(1, 0, *range(2, sv.ndim)).contiguous()
        elif sv.ndim >= 2 and sv.shape[0] == B * T:
            sv = sv.view(B, T, *sv.shape[1:]).contiguous()

        while sv.ndim > 3 and sv.shape[-1] == 1:
            sv = sv.squeeze(-1)
        if sv.ndim > 3:
            sv = sv.flatten(2)
        if sv.ndim == 2:
            sv = sv.unsqueeze(-1) 

        td.set("state_value", sv)
        return td

class GroupValueNorm(nn.Module):
    def __init__(self, num_groups: int, input_shape=1, device=None):
        super().__init__()
        self.num_groups = int(num_groups)
        self.norms = nn.ModuleList([ValueNorm1(input_shape=input_shape) for _ in range(self.num_groups)])
        if device is not None:
            self.to(device)

    def _apply_by_group(self, x: torch.Tensor, group_ids: torch.Tensor, fn):
        x_flat = x.view(-1, *x.shape[-1:])
        gid = group_ids.view(-1).long()
        out = torch.empty_like(x_flat)
        for g in gid.unique():
            m = (gid == g)
            if m.any():
                out[m] = fn(self.norms[int(g)], x_flat[m])
        return out.view_as(x)

    def denormalize(self, x: torch.Tensor, group_ids: torch.Tensor):
        return self._apply_by_group(x, group_ids, lambda n, t: n.denormalize(t))

    def normalize(self, x: torch.Tensor, group_ids: torch.Tensor):
        return self._apply_by_group(x, group_ids, lambda n, t: n.normalize(t))

    @torch.no_grad()
    def update(self, x: torch.Tensor, group_ids: torch.Tensor):
        x_flat = x.view(-1, *x.shape[-1:])
        gid = group_ids.view(-1).long()
        for g in gid.unique():
            m = (gid == g)
            if m.any():
                self.norms[int(g)].update(x_flat[m])


def normalize_by_group(x: torch.Tensor, group_ids: torch.Tensor, subtract_mean: bool = True, eps: float = 1e-8):
    x_flat = x.reshape(-1)
    gid_flat = group_ids.reshape(-1).long()
    out = x_flat.clone()
    for gid in gid_flat.unique():
        m = (gid_flat == gid)
        if m.any():
            g = out[m]
            if subtract_mean:
                g = g - g.mean()
            g = g / g.std().clamp_min(eps)
            out[m] = g
    return out.view_as(x)

class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device,
        env,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        # when multi_critic is False, aggregate (sum and clip) the rewards BEFORE computing the advantage
        self.multi_critic = self.cfg.multi_critic
        self.num_terrains = self.cfg.num_env_groups
        self.num_rewards = reward_spec["reward"].shape[-1]

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        
        if self.cfg.per_env_value_norm:
            self.value_norm = ValueNormFake(input_shape=1).to(self.device)
            self.group_value_norm = None
        else:
            self.value_norm = None
            self.group_value_norm = GroupValueNorm(self.cfg.num_env_groups, input_shape=1, device=self.device)
        
        fake_input = observation_spec.zero()
        if "height_scan" in observation_spec.keys(True, True):
            self.terrain_key = "height_scan"
        else:
            self.terrain_key = "grid_map_"
        conv3d = len(observation_spec[self.terrain_key].shape) == 5 # [N, 1, D, H, W]
        
        self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transforms().to(self.device)
        self.hsc_transform = env.observation_funcs[self.terrain_key].symmetry_transforms().to(self.device)
        self.act_transform = env.action_manager.symmetry_transforms().to(self.device)

        actor_module = TensorDictSequential(
            Mod(MixedEncoder(conv3d=conv3d), [OBS_KEY, self.terrain_key, "mask"], ["_actor_feature"]),
            Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"]),
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        base_critic = TensorDictSequential(
            Mod(MixedEncoder(conv3d=conv3d), [OBS_KEY, self.terrain_key, "mask"], ["_critic_feature"]),
            Mod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
        ).to(self.device)

        if self.cfg.per_env_critic:
            self.critic = PerEnvMultiHeadCritic(
                base_critic=base_critic,
                num_env_groups=self.cfg.num_env_groups,
                env_id_key=self.cfg.env_id_key,
                out_key="state_value"
            ).to(self.device)
            self.critic = _CriticAlignWrapper(self.critic)
        else:
            self.critic = base_critic

        self.actor(fake_input)
        self.critic(fake_input)
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        if not self.cfg.per_env_critic:
            self.critic.apply(init_)

        if self.cfg.multi_critic and (not self.cfg.per_env_critic):
            self.critic = EnsembleCritic(self.critic, num_copies=self.num_rewards, init_=init_)

        if self.cfg.multi_critic and self.cfg.per_env_critic:
            raise NotImplementedError("both multi-env and multi-reward critic has not been implemented!")

        if active_adaptation.is_distributed():
            distr.init_process_group(
                backend="nccl",
                world_size=active_adaptation.get_world_size(),
                rank=active_adaptation.get_local_rank()
            )
            self.world_size = active_adaptation.get_world_size()
            if self.cfg.use_ddp:
                self.actor = DDP(self.actor)
                self.critic = DDP(self.critic, static_graph=True)
            else:
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)
                for param in self.critic.parameters():
                    distr.broadcast(param, src=0)
        
        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )

        if self.cfg.compile and not active_adaptation.is_distributed():
            self.update_batch = torch.compile(self._update_batch)
        else:
            self.update_batch = self._update_batch
    
    # def make_tensordict_primer(self):
    #     num_envs = self.observation_spec.shape[0]
    #     spec = {
    #         "base_height_targ": UnboundedContinuous((num_envs, 1), device=self.device),
    #     }
    #     return TensorDictPrimer(spec, reset_key="done", default_value=0.74)
    
    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(self.actor)
        if self.cfg.compile:
            policy = torch.compile(policy)
        return policy

    def compute_custom_reward(self, tensordict: TensorDict):
        reward = tensordict[REWARD_KEY].sum(-1, True)
        base_height = tensordict["base_height"]
        base_height_targ = tensordict["base_height_targ"]
        base_height_rew = 2. * torch.exp(-(base_height - base_height_targ).square() / 0.25)
        reward += base_height_rew
        tensordict.set(REWARD_KEY, reward)
        return tensordict

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        # self.compute_custom_reward(tensordict)
        
        infos = []
        if self.multi_critic: # False
            # aggregate the rewards AFTER computing the advantage
            self._compute_advantage(tensordict, self.critic, "adv", "ret")
            tensordict["adv"] = normalize(tensordict["adv"].sum(-1, True), subtract_mean=True)
        elif self.cfg.per_env_critic:
            tensordict[REWARD_KEY] = tensordict[REWARD_KEY].sum(-1, True) # no clip here
            self._compute_advantage(tensordict, self.critic, "adv", "ret")
            terrain_types = tensordict.get(self.cfg.env_id_key, None)
            bits = (terrain_types > 0.5).long()
            weights = torch.tensor([4, 2, 1], device=bits.device, dtype=bits.dtype)
            env_ids = (bits * weights).sum(dim=-1).long()
            tensordict["adv"] = normalize_by_group(
                tensordict["adv"].squeeze(-1),
                env_ids,
                subtract_mean=True,
            ).unsqueeze(-1)
        else:
            # aggregate the rewards BEFORE computing the advantage
            tensordict[REWARD_KEY] = tensordict[REWARD_KEY].sum(-1, True).clip(min=0.)
            self._compute_advantage(tensordict, self.critic, "adv", "ret")
            tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(TensorDict(self.update_batch(minibatch), []))
        
        with torch.no_grad(), torch.device(self.device):
            a = self.critic(tensordict.replace(mask=torch.zeros(*tensordict.shape, 1)))
            b = self.critic(tensordict.replace(mask=torch.ones(*tensordict.shape, 1)))
            value_diff = F.mse_loss(a["state_value"], b["state_value"])
            a = self.actor(
                tensordict.replace(mask=torch.zeros(*tensordict.shape, 1)))["loc"]
            b = self.actor(
                tensordict.replace(mask=torch.ones(*tensordict.shape, 1)))["loc"]
            policy_diff = F.mse_loss(a, b)

        out = {}
        for k, v in sorted(torch.stack(infos).items()):
            out[k] = v.detach().mean().item()
        out["critic/value_mean"] = tensordict["ret"].mean().item()
        out["critic/value_std"] = tensordict["ret"].std().item()
        out["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        out["critic/value_diff"] = value_diff.item()
        out["actor/policy_diff"] = policy_diff.item()
        return out

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: Mod, 
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(tensordict_flat)
                critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY]
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        
        terrain_types = tensordict.get(self.cfg.env_id_key, None)
        if (self.cfg.per_env_critic) and (self.group_value_norm is not None) and (terrain_types is not None):
            bits = (terrain_types > 0.5).long()
            weights = torch.tensor([4, 2, 1], device=bits.device, dtype=bits.dtype)
            env_ids = (bits * weights).sum(dim=-1).long()
            values = self.group_value_norm.denormalize(values, env_ids)
            next_values = self.group_value_norm.denormalize(next_values, env_ids)
        else:
            values = self.value_norm.denormalize(values) if self.value_norm is not None else values
            next_values = self.value_norm.denormalize(next_values) if self.value_norm is not None else next_values

        adv, ret = self.gae(rewards, terms, dones, values, next_values)
        if update_value_norm:
            if (self.cfg.per_env_critic) and (self.group_value_norm is not None) and (terrain_types is not None):
                self.group_value_norm.update(ret, env_ids)
                ret = self.group_value_norm.normalize(ret, env_ids)
            else:
                if self.value_norm is not None:
                    self.value_norm.update(ret)
                    ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update_batch(self, tensordict: TensorDict):
        
        bsize = tensordict.shape[0]
        symmetry = tensordict.empty()
        symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
        symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
        symmetry[self.terrain_key] = self.hsc_transform(tensordict[self.terrain_key])
        symmetry["action_log_prob"] = tensordict["action_log_prob"]
        symmetry["is_init"] = tensordict["is_init"]
        symmetry["adv"] = tensordict["adv"]
        symmetry["ret"] = tensordict["ret"]
        symmetry[self.cfg.env_id_key] = tensordict[self.cfg.env_id_key]
        tensordict = torch.cat([tensordict.select(*symmetry.keys(True, True)), symmetry], dim=0)

        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]
        self.actor(tensordict)
        dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
        # dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(action_data)
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        log_ratio = (log_probs - log_probs_data).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]))
        entropy_loss = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        assert values.shape == b_returns.shape
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()
        
        loss = policy_loss + entropy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()

        if active_adaptation.is_distributed() and not self.cfg.use_ddp:
            for param in self.actor.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size
            for param in self.critic.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size
        
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()
        
        with torch.no_grad():
            explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            symmetry_loss = F.mse_loss(dist.mean[bsize:], self.act_transform(dist.mean[:bsize]))
        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/symmetry_loss": symmetry_loss.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, DDP):
                module = module.module
            state_dict[name] = module.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                if isinstance(module, DDP):
                    module = module.module
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)