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
from active_adaptation.utils.torchrl import EnsembleCritic
import spconv.pytorch as spconv

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_hussar_spconv3d.PPOPolicy"
    name: str = "ppo_hussar_spconv3d"
    train_every: int = 48
    ppo_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    desired_kl: Union[float, None] = 0.02
    entropy_coef: float = 0.003
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False
    multi_critic: bool = False

    cnn_history: bool = False
    symmetry: bool = True
    
    compile: bool = False
    use_ddp: bool = True

    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str] = (OBS_KEY, "height_scan", "grid_map_", "base_height", "base_height_targ")

cs = ConfigStore.instance()
cs.store("ppo_hussar_spconv3d", node=PPOConfig, group="algo")


class Spconv3DBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 与 2D CNN 参数/层数对齐：3 层卷积、每层 8 通道、3x3 核，仅在 H/W 下采样
        # 用 3D 核 (1,3,3)、步幅 (1,2,2)、填充 (0,1,1)，避免跨深度维卷积，参数量与 2D 3x3 相当
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(1, 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Mish(),
            spconv.SparseConv3d(8, 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Mish(),
            spconv.SparseConv3d(8, 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Mish(),
            spconv.ToDense()  # -> [B, 8, D, H', W']
        )

    def forward(self, x):
        # 仅支持稠密张量输入；自动处理任意批维，并将 4D 输入补为 5D
        assert not isinstance(x, spconv.SparseConvTensor), "Expect dense tensor, got SparseConvTensor."
        assert x.dim() >= 4, "CNN input must be at least 4D ([..., C, H, W])."
        # 确保为 5D: [..., C, D, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(1)  # [..., 1, D, H, W]（将原通道 D 作为深度维）
        # 扁平化前置批维
        batch_ndims = x.dim() - 5
        if batch_ndims > 0:
            batch_shape = x.shape[:batch_ndims]
            x_5d = x.flatten(0, batch_ndims - 1)
        else:
            batch_shape = None
            x_5d = x
        # from_dense 期望 NDHWC
        x_ndhwc = x_5d.permute(0, 2, 3, 4, 1).contiguous()
        sp_x = spconv.SparseConvTensor.from_dense(x_ndhwc)
        # 若没有任何激活体素（常见于全零假输入初始化阶段），直接返回与 ToDense 后一致尺寸的零向量
        if sp_x.features.shape[0] == 0:
            Bf, _, D_in, H_in, W_in = x_5d.shape
            def ceil_div(a, b): return (a + b - 1) // b
            # 三次 (k=3, s=2, p=1) 在 H/W 维；D 维 (k=1, s=1, p=0) 不变
            H1 = ceil_div(H_in, 2)
            H2 = ceil_div(H1, 2)
            H3 = ceil_div(H2, 2)
            W1 = ceil_div(W_in, 2)
            W2 = ceil_div(W1, 2)
            W3 = ceil_div(W2, 2)
            D3 = D_in
            # 最后一层通道为 8（见 self.net 定义）
            y = torch.zeros(Bf, 8, D3, H3, W3, device=x_5d.device, dtype=x_5d.dtype)
            y = torch.flatten(y, 1)
        else:
            y = self.net(sp_x)  # 稠密输出 [B_flat, C, D, H, W]
            y = torch.flatten(y, 1)  # [B_flat, F]
        if batch_shape is not None:
            y = y.unflatten(0, batch_shape)  # [..., F]
        return y


class MixedEncoder(nn.Module):
    def __init__(self, cnn_history: bool=False, conv3d: bool=False):
        super().__init__()
        self.cnn_history = cnn_history
        self.mlp_encoder = nn.Sequential(
            nn.LazyLinear(256), nn.Mish(), nn.LayerNorm(256), 
            nn.LazyLinear(256)
        )

        if conv3d:
            # 使用稀疏3D卷积骨干（内部自行处理批维展平/还原与 4D->5D 补维）
            self.cnn_encoder = nn.Sequential(
                Spconv3DBackbone(),
                nn.LazyLinear(64),
                nn.Mish(),
                nn.LayerNorm(64),
                nn.LazyLinear(64),
            )
        else:
            # 保持原 2D CNN
            cnn_cls = nn.LazyConv2d
            data_dim = 3  # [C, H, W]
            self.cnn_encoder = nn.Sequential(
                FlattenBatch(
                    nn.Sequential(
                        cnn_cls(8, kernel_size=3, stride=2, padding=1), 
                        nn.Mish(),
                        cnn_cls(8, kernel_size=3, stride=2, padding=1),
                        nn.Mish(),
                        cnn_cls(8, kernel_size=3, stride=2, padding=1),
                        nn.Mish(),
                        nn.Flatten(),
                    ),
                    data_dim=data_dim,
                ),
                nn.LazyLinear(64),
                nn.Mish(),
                nn.LayerNorm(64),
                nn.LazyLinear(64),
            )
        self.out = nn.Sequential(nn.Mish(), nn.LazyLinear(256), nn.Mish())

    def forward(self, mlp_inp, cnn_inp, prev_cnn_feature, mask_cnn=None):
        """
        prev_cnn_feature: [*, 256] from the previous step
        """
        cnn_feature = self.cnn_encoder(cnn_inp.float())
        mlp_feature = self.mlp_encoder(mlp_inp)
        if mask_cnn is not None:
            cnn_feature = cnn_feature * mask_cnn
        if self.cnn_history:
            feature = torch.cat([mlp_feature, cnn_feature, prev_cnn_feature], dim=-1)
            return self.out(feature), cnn_feature
        else:
            feature = torch.cat([mlp_feature, cnn_feature], dim=-1)
            return self.out(feature)


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
        self.cfg = PPOConfig(**cfg)
        self.device = device
        self.observation_spec = observation_spec

        # when multi_critic is False, aggregate (sum and clip) the rewards BEFORE computing the advantage
        self.multi_critic = self.cfg.multi_critic
        self.num_rewards = reward_spec["reward"].shape[-1]

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        
        self.desired_kl = self.cfg.desired_kl
        self.init_lr = self.cfg.lr

        self.value_norm = ValueNormFake(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        fake_input["actor_cnn_feature"] = torch.zeros(fake_input.shape[0], 64, device=self.device)
        fake_input["critic_cnn_feature"] = torch.zeros(fake_input.shape[0], 64, device=self.device)
        
        if "height_scan" in observation_spec.keys(True, True):
            self.terrain_key = "height_scan"
        else:
            self.terrain_key = "grid_map_"
        # 始终走 3D 分支；实际运行时若输入为 4D，会在骨干中自动 unsqueeze
        self.conv3d = True
        self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transforms().to(self.device)
        if OBS_PRIV_KEY in observation_spec.keys(True, True):
            self.priv_transform = env.observation_funcs[OBS_PRIV_KEY].symmetry_transforms().to(self.device)
        self.hsc_transform = env.observation_funcs[self.terrain_key].symmetry_transforms().to(self.device)
        self.act_transform = env.action_manager.symmetry_transforms().to(self.device)
        
        assert not (self.cfg.cnn_history and self.cfg.symmetry), "cnn_history and symmetry cannot be True at the same time"
        if self.cfg.cnn_history:
            actor_in_keys = [OBS_KEY, self.terrain_key, "actor_cnn_feature", "mask"]
            actor_out_keys = ["_actor_feature", ("next", "actor_cnn_feature")]
            critic_in_keys = [OBS_KEY, self.terrain_key, "critic_cnn_feature", "mask"]
            critic_out_keys = ["_critic_feature", ("next", "critic_cnn_feature")]
        else:
            actor_in_keys = [OBS_KEY, self.terrain_key, "mask"]
            actor_out_keys = ["_actor_feature"]
            if OBS_PRIV_KEY in observation_spec.keys(True, True):
                critic_in_keys = ["critic_input", self.terrain_key, "mask"]
            else:
                critic_in_keys = [OBS_KEY, self.terrain_key, "mask"]
            critic_out_keys = ["_critic_feature"]
        
        actor_module = TensorDictSequential(
            Mod(
                MixedEncoder(cnn_history=self.cfg.cnn_history, conv3d=self.conv3d),
                actor_in_keys,
                actor_out_keys
            ),
            Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"]),
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        if OBS_PRIV_KEY in observation_spec.keys(True, True):
            critic_module = TensorDictSequential(
                CatTensors([OBS_KEY, OBS_PRIV_KEY], "critic_input", del_keys=False),
                Mod(
                    MixedEncoder(cnn_history=self.cfg.cnn_history, conv3d=self.conv3d),
                    critic_in_keys,
                    critic_out_keys
                ),
                Mod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
            ).to(self.device)
        else:
            critic_module = TensorDictSequential(
                Mod(
                    MixedEncoder(cnn_history=self.cfg.cnn_history, conv3d=self.conv3d),
                    critic_in_keys,
                    critic_out_keys
                ),
                Mod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
            ).to(self.device)
        self.critic = critic_module

        self.actor(fake_input)
        self.critic(fake_input)
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

        if self.cfg.multi_critic:
            self.critic = EnsembleCritic(self.critic, num_copies=self.num_rewards, init_=init_)

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
        policy = TensorDictSequential(self.actor, self.critic)
        if self.cfg.compile:
            policy = torch.compile(policy)
        return policy
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        spec = CompositeSpec({
            "actor_cnn_feature": UnboundedContinuous((num_envs, 64), device=self.device),
            "critic_cnn_feature": UnboundedContinuous((num_envs, 64), device=self.device),
        }, shape=[num_envs,], device=self.device)
        return TensorDictPrimer(spec, reset_key="done")

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        # self.compute_custom_reward(tensordict)

        infos = []
        if self.multi_critic:
            # aggregate the rewards AFTER computing the advantage
            self._compute_advantage(tensordict, self.critic, "adv", "ret")
            tensordict["adv"] = normalize(tensordict["adv"].sum(-1, True), subtract_mean=True)
        else:
            # aggregate the rewards BEFORE computing the advantage
            tensordict[REWARD_KEY] = tensordict[REWARD_KEY].sum(-1, True).clip(min=0.)
            self._compute_advantage(tensordict, self.critic, "adv", "ret")
            tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(TensorDict(self.update_batch(minibatch), []))
                # if self.desired_kl is not None: # adaptive learning rate
                #     kl = infos[-1]["actor/kl"]
                #     actor_lr = self.opt.param_groups[0]["lr"]
                #     if kl > self.desired_kl * 2.0:
                #         actor_lr = max(1e-5, actor_lr / 1.5)
                #     elif kl < self.desired_kl / 2.0 and kl > 0.0:
                #         actor_lr = min(self.init_lr, actor_lr * 1.1)
                #     self.opt.param_groups[0]["lr"] = actor_lr

        # with torch.no_grad(), torch.device(self.device):
        #     a = self.critic(tensordict.replace(mask=torch.zeros(*tensordict.shape, 1)))
        #     b = self.critic(tensordict.replace(mask=torch.ones(*tensordict.shape, 1)))
        #     value_diff = F.mse_loss(a["state_value"], b["state_value"])
        #     a = self.actor(
        #         tensordict.replace(mask=torch.zeros(*tensordict.shape, 1)))["loc"]
        #     b = self.actor(
        #         tensordict.replace(mask=torch.ones(*tensordict.shape, 1)))["loc"]
        #     policy_diff = F.mse_loss(a, b)

        out = {}
        for k, v in sorted(torch.stack(infos).items()):
            out[k] = v.detach().mean().item()
        # out["actor/kl"] = kl.item()
        out["actor/lr"] = self.opt.param_groups[0]["lr"]
        out["critic/value_mean"] = tensordict["ret"].mean().item()
        out["critic/value_std"] = tensordict["ret"].std().item()
        out["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        # out["critic/value_diff"] = value_diff.item()
        # out["actor/policy_diff"] = policy_diff.item()
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
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values)
        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update_batch(self, tensordict: TensorDict):
        
        bsize = tensordict.shape[0]
        loc_old, scale_old = tensordict.pop("loc"), tensordict.pop("scale")
        if self.cfg.symmetry:
            symmetry = tensordict.empty()
            symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
            if OBS_PRIV_KEY in tensordict.keys(True, True):
                symmetry[OBS_PRIV_KEY] = self.priv_transform(tensordict[OBS_PRIV_KEY])
            symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
            symmetry[self.terrain_key] = self.hsc_transform(tensordict[self.terrain_key])
            symmetry["action_log_prob"] = tensordict["action_log_prob"]
            symmetry["is_init"] = tensordict["is_init"]
            symmetry["adv"] = tensordict["adv"]
            symmetry["ret"] = tensordict["ret"]
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
        
        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
        }
        
        with torch.no_grad():
            info["critic/explained_var"] = 1 - F.mse_loss(values, b_returns) / b_returns.var()
            info["actor/clamp_ratio"] = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            # loc, scale = dist.loc[:bsize], dist.scale[:bsize]
            # kl = IndependentNormal.kl(loc_old, scale_old, loc, scale).mean()
            # info["actor/kl"] = kl
            if self.cfg.symmetry:
                info["actor/symmetry_loss"] = F.mse_loss(dist.mean[bsize:], self.act_transform(dist.mean[:bsize]))
        return info

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