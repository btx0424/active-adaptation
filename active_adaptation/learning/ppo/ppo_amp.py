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
from torchrl.objectives.utils import hold_out_net
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase, 
    TensorDictModule as Mod, 
    TensorDictSequential as Seq
)
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field, MISSING
from typing import Union, List
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from ..modules.rnn import GRU, set_recurrent_mode, recurrent_mode
from .common import *

torch.set_float32_matmul_precision('high')

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_amp.PPOPolicy"
    name: str = "ppo_amp"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    opt: str = "Adam"
    lr: float = 5e-4
    clip_param: float = 0.2
    
    entropy_coef_start: float = 0.002
    entropy_coef_end: float = 0.000
    gradient_penalty: float = 0.002

    reg_lambda: float = 0.0
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    # data_path: str = "/home/btx0424/lab/legged-deploy/go2/logs/12-05_16-43-21.h5py"
    data_path: str = "/home/btx0424/lab/legged-deploy/go2/logs/12-07_22-24-40.h5py"
    phase: str = "train"
    vecnorm: Union[str, None] = None
    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = field(default_factory=lambda: [CMD_KEY, OBS_KEY, OBS_PRIV_KEY])

cs = ConfigStore.instance()
cs.store("ppo_amp_train", node=PPOConfig(phase="train", vecnorm="train"), group="algo")
# cs.store("ppo_orca_adapt", node=PPOConfig(phase="adapt", vecnorm="eval"), group="algo")
cs.store("ppo_amp_finetune", node=PPOConfig(phase="finetune", vecnorm="eval"), group="algo")

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
    
    def __call__(self, tensordict: TensorDictBase):
        # TODO@botian: write to tensordict?
        if self.encoder is not None:
            self.encoder(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()
        return log_probs, entropy


class PPOPolicy(TensorDictModuleBase):
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
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        self.reg_lambda = 0.0
        self.adapt_loss_fn = nn.MSELoss(reduction="none")
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        
        self.value_norm = ValueNormFake(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        obs_keys = list(observation_spec.keys(True, True))
        
        self.encoder_priv = Seq(
            Mod(nn.Sequential(make_mlp([128]), nn.LazyLinear(128)), [OBS_PRIV_KEY], ["priv_feature"]),
        ).to(self.device)

        self.adapt_module =  Mod(
            GRUModule(128, split=None), 
            [OBS_KEY, "is_init", "adapt_hx"], 
            ["priv_pred", ("next", "adapt_hx")]
        ).to(self.device)
        
        in_keys = [CMD_KEY, OBS_KEY, "priv_feature"]
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

        in_keys = [CMD_KEY, OBS_KEY, "priv_pred"]
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
            CatTensors([CMD_KEY, OBS_KEY, OBS_PRIV_KEY], "_critic_input", del_keys=False),
            Mod(_critic, ["_critic_input"], ["state_value"])
        ).to(self.device)

        self.aux = Seq(
            Mod(nn.LazyLinear(25), ["_actor_feature"], ["_implicit_pred"]),
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
        self.aux(fake_input)

        self.policy_train_inference = PolicyUpdateInferenceMod(self.actor, self.encoder_priv)
        self.policy_adapt_inference = PolicyUpdateInferenceMod(self.actor_adapt, None)
        
        self.adapt_ema = copy.deepcopy(self.adapt_module)
        self.adapt_ema.requires_grad_(False)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
                {"params": self.encoder_priv.parameters()},
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
        self.aux.apply(init_)
        self.num_updates = 0

        self.load_data()
        self.noise_dim = 12
        self.gen_s2r = nn.Sequential(make_mlp([128, 128]), nn.LazyLinear(36)).to(self.device)
        self.gen_r2s = nn.Sequential(make_mlp([128, 128]), nn.LazyLinear(36)).to(self.device)
        noise = torch.randn(fake_input.shape[0], self.noise_dim, device=self.device)
        self.gen_s2r(torch.cat([fake_input["amp2_"], noise], dim=-1))
        self.gen_s2r.apply(init_)
        self.gen_r2s(torch.cat([fake_input["amp2_"], noise], dim=-1))
        self.gen_r2s.apply(init_)
        
        jpos_real, imu_real = self.get_amp_batch()
        self.disc_amp = nn.Sequential(make_mlp([512, 256]), nn.LazyLinear(1)).to(self.device)
        self.disc_amp(jpos_real)
        self.disc_amp.apply(init_)
        
        self.disc2 = nn.Sequential(make_mlp([512, 256]), nn.LazyLinear(1)).to(self.device)
        self.disc2(torch.cat([jpos_real, imu_real], dim=-1))
        self.disc2.apply(init_)

        self.opt_disc = torch.optim.Adam(list(self.disc_amp.parameters()) + list(self.disc2.parameters()), lr=1e-4)
        self.opt_gen = torch.optim.Adam(list(self.gen_s2r.parameters()) + list(self.gen_r2s.parameters()), lr=3e-4)
    
    def load_data(self):
        import h5py
        data = h5py.File(self.cfg.data_path, "r")
        cursor = data.attrs["cursor"]
        print(f"Loading data from {self.cfg.data_path} with cursor {cursor}")
        data = TensorDict({
            k: torch.as_tensor(v[100: cursor], device=self.device) 
            for k, v in data.items()
        }, [cursor - 100])
        print(data)
        self.data = data

    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        return TensorDictPrimer({
            "adapt_hx": UnboundedContinuous((num_envs, 128), device=self.device),
            "prev_action": UnboundedContinuous((num_envs, self.action_dim), device=self.device),
            "prev_loc": UnboundedContinuous((num_envs, self.action_dim), device=self.device),
        }, reset_key="done")

    def get_rollout_policy(self, mode: str="train"):
        modules = []
        
        def foo(tensordict: TensorDict):
            imu_sim = tensordict["amp2_"]
            noise = torch.randn(imu_sim.shape[0], self.noise_dim, device=self.device)
            imu_est = self.gen_s2r(torch.cat([imu_sim, noise], dim=-1))
            tensordict["imu_est"] = imu_est
            return tensordict
        
        if self.cfg.phase == "train":
            modules.append(self.encoder_priv)
            modules.append(self.actor)
            modules.append(self.adapt_module)
            if mode == "eval":
                modules.append(foo)
        elif self.cfg.phase == "adapt":
            modules.append(self.adapt_module)
            modules.append(self.actor_adapt)
        elif self.cfg.phase == "finetune":
            modules.append(self.adapt_ema)
            modules.append(self.actor_adapt)
        
        policy = Seq(*modules)
        return policy
    
    def step_schedule(self, progress: float):
        self.reg_lambda = progress * self.cfg.reg_lambda
        self.entropy_coef = self.cfg.entropy_coef_start + (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start) * progress

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
        
        with torch.no_grad():
            amp1_obs = tensordict["amp1_"] # [jpos]
            score_current1 = self.disc_amp(amp1_obs)
            amp_rew = (1 - (score_current1 - 1).square())
            tensordict[REWARD_KEY] += 0.02 * amp_rew

        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        policy_inference = self.policy_train_inference if self.cfg.phase == "train" else self.policy_adapt_inference
        opt = self.opt if self.cfg.phase == "train" else self.opt_finetune

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                info = self._update(minibatch, policy_inference, opt)
                infos.append(TensorDict(info, []))

        infos_symmetry = []
        for i, batch in enumerate(make_batch(tensordict, 4)):
            infos_symmetry.append(TensorDict(self._update_disc(batch), []))
        
        infos = collect_info(infos)
        infos.update(collect_info(infos_symmetry))
        if self.cfg.phase == "train":
            infos["actor/feature_std"] = tensordict["priv_feature"].std(-1).mean().item()
        else:
            infos["actor/feature_std"] = tensordict["priv_pred"].std(-1).mean().item()
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["amp/reward"] = amp_rew.mean().item()
        infos["amp/score"] = score_current1.mean().item()
        return {k: v for k, v in sorted(infos.items())}
    
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
                
                self.opt_adapt.zero_grad()
                (priv_loss).backward()
                self.opt_adapt.step()
                infos.append(TensorDict({
                    "adapt/priv_loss": priv_loss,
                }, []))
        
        soft_copy_(self.adapt_module, self.adapt_ema, 0.05)
        
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
        if self.cfg.phase == "train":
            for key in (CMD_KEY, OBS_KEY):
                tensordict[key].requires_grad_(True)
        else:
            for key in (CMD_KEY, OBS_KEY, "priv_pred"):
                tensordict[key].requires_grad_(True)
        log_probs, entropy = policy_inference(tensordict)

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
        
        # grad = torch.autograd.grad(
        #     log_probs,
        #     [tensordict[key] for key in (CMD_KEY, OBS_KEY, "priv_feature" if self.cfg.phase == "train" else "priv_pred")],
        #     grad_outputs=torch.ones_like(log_probs),
        #     create_graph=True,
        #     retain_graph=True,
        # )
        # gradient_penalty = torch.cat(grad, dim=-1).square().sum(-1).mean()

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()

        if self.cfg.phase == "train" and self.reg_lambda > 0:
            reg_loss = self.adapt_loss_fn(tensordict["priv_feature"], tensordict["priv_pred"])
            reg_loss = self.reg_lambda * (reg_loss * (~tensordict["is_init"])).mean()
        else:
            reg_loss = 0.
            
        loss = policy_loss + entropy_loss + value_loss + reg_loss # + self.cfg.gradient_penalty * gradient_penalty
        
        opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        opt.step()
        
        explained_var = 1 - value_loss / b_returns[~tensordict["is_init"]].var()
        info = {
            "actor/policy_loss": policy_loss,
            "actor/entropy": entropy,
            "actor/noise_std": tensordict["scale"].mean(),
            "actor/grad_norm": actor_grad_norm,
            'actor/approx_kl': ((ratio - 1) - log_ratio).mean(),
            # "actor/gradient_penalty": gradient_penalty,
            "adapt/reg_loss": reg_loss,
            "critic/value_loss": value_loss,
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }
        return info

    def fliplr(self, tensor: torch.Tensor):
        tensor = tensor.reshape(*tensor.shape[:2], 3, 4)[..., [1, 0, 3, 2]]
        tensor[..., 0, :] *= -1 # flip hip joint states
        return tensor.reshape(*tensor.shape[:2], -1)

    def get_amp_batch(self):
        steps = 8
        idx = torch.randint(0, len(self.data) - steps, (1024,), device=self.device)
        target_batch = self.data[idx.unsqueeze(1) + torch.arange(steps, device=self.device)]
        
        jpos = target_batch["jpos"]
        jpos_ = self.fliplr(jpos)
        jpos = torch.cat([jpos.reshape(1024, -1), jpos_.reshape(1024, -1)], dim=0)

        history_step = 1
        gravity = target_batch["gravity_substep"][:, -history_step:]
        gravity_ = gravity * torch.tensor([1., -1., 1.], device=self.device)
        lin_acc = - target_batch["lin_acc_substep"][:, -history_step:] - gravity * 9.81
        lin_acc_ = lin_acc * torch.tensor([1., -1., 1.], device=self.device)
        gyro = target_batch["ang_vel_substep"][:, -history_step:]
        gyro_ = gyro * torch.tensor([-1., 1., -1.], device=self.device)

        imu = torch.cat([
            torch.cat([lin_acc.flatten(1), gravity.flatten(1), gyro.flatten(1)], dim=-1),
            torch.cat([lin_acc_.flatten(1), gravity_.flatten(1), gyro_.flatten(1)], dim=-1),
        ], dim=0)
        return jpos, imu

    def _update_disc(self, tensordict: TensorDict):
        jpos_real, imu_real = self.get_amp_batch()
        jpos_imu_real = torch.cat([jpos_real, imu_real], dim=-1)
        jpos_real.requires_grad_(True)
        jpos_imu_real.requires_grad_(True)
        jpos_sim = tensordict["amp1_"] # [jpos]
        imu_sim = tensordict["amp2_"]

        with hold_out_net(self.gen_s2r), hold_out_net(self.gen_r2s):
            noise = torch.randn(imu_sim.shape[0], self.noise_dim, device=self.device)
            imu_s2r = self.gen_s2r(torch.cat([imu_sim, noise], dim=-1))
            jpos_imu_s2r = torch.cat([jpos_sim, imu_s2r], dim=-1) # [jpos, imu]

        score_real_amp = self.disc_amp(jpos_real)
        score_sim_amp = self.disc_amp(jpos_sim)

        score_real = self.disc2(jpos_imu_real)
        score_sim = self.disc2(jpos_imu_s2r)

        valid = (~tensordict["is_init"]).float()
        
        loss_disc_amp = (score_real_amp - 1).square().mean() + ((score_sim_amp + 1).square() * valid).mean()
        loss_disc_s2r = (score_real - 1).square().mean() + ((score_sim + 1).square() * valid).mean()
        
        grad1 = torch.autograd.grad(
            score_real_amp,
            jpos_real, 
            torch.ones_like(score_real_amp),
            retain_graph=True,
            create_graph=True
        )[0]
        grad2 = torch.autograd.grad(
            score_real,
            jpos_imu_real, 
            torch.ones_like(score_real),
            retain_graph=True,
            create_graph=True
        )[0]
        gradient_penalty1 = torch.mean(grad1.square().sum(dim=-1))
        gradient_penalty2 = torch.mean(grad2.square().sum(dim=-1))
        
        self.opt_disc.zero_grad()
        (loss_disc_amp + loss_disc_s2r + 10 * gradient_penalty1 + 10. * gradient_penalty2).backward()
        self.opt_disc.step()

        with hold_out_net(self.disc2):
            noise = torch.randn(imu_sim.shape[0], self.noise_dim, device=self.device)
            imu_s2r = self.gen_s2r(torch.cat([imu_sim, noise], dim=-1))
            jpos_imu_s2r = torch.cat([jpos_sim, imu_s2r], dim=-1) # [jpos, imu]

            score_sim = self.disc2(jpos_imu_s2r)
            gen_loss = ((score_sim - 1).square() * valid).mean()

            noise = torch.randn(imu_sim.shape[0], self.noise_dim, device=self.device)
            cycle_loss = F.mse_loss(self.gen_r2s(torch.cat([imu_s2r, noise], -1)), imu_sim)
        
        self.opt_gen.zero_grad()
        (gen_loss + cycle_loss).backward()
        grad_norm = nn.utils.clip_grad_norm_(self.gen_s2r.parameters(), 5.0)
        self.opt_gen.step()

        return {
            "amp/loss_gen": gen_loss,
            "amp/gen_grad_norm": grad_norm,
            "amp/loss_disc1": loss_disc_amp,
            "amp/loss_disc12": loss_disc_s2r,
            "amp/gradient_penalty1": gradient_penalty1,
            "amp/gradient_penalty2": gradient_penalty2,
            "amp/cycle_loss": cycle_loss
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
