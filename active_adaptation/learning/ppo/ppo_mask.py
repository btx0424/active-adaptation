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
import copy

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, ExcludeTransform, TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *
from .ppo_adapt import GRUModuleStoch, GRUModule

@dataclass
class PPOConfig:
    name: str = "ppo_mask"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    lr: float = 5e-4
    clip_param: float = 0.2
    recompute_adv: bool = False
    value_norm: bool = False

    context_dim: int = 128
    adapt_module: str = "mse"

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_mask", node=PPOConfig, group="algo")

MASK_KEY = OBS_PRIV_KEY + "_mask"
MASKED_KEY = OBS_PRIV_KEY + "_masked"

class PPOMaskPolicy(TensorDictModuleBase):

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

        self.entropy_coef = 0.001
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.HuberLoss(delta=10, reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.context_dim = self.cfg.context_dim
        self.gae = GAE(0.99, 0.95)
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        # lazy initialization
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["context_adapt_hx"] = torch.zeros(fake_input.shape[0], 128)
        obs_dim = fake_input[OBS_PRIV_KEY].shape[-1]
        
        assert MASK_KEY in fake_input.keys(True, True)

        def _mask_with_embedding():
            return TensorDictModule(
                MaskWithEmbedding(obs_dim),
                [OBS_PRIV_KEY, MASK_KEY], [MASKED_KEY]
            )
        
        self.encoder = TensorDictSequential(
            _mask_with_embedding(),
            TensorDictModule(
                nn.Sequential(make_mlp([self.cfg.context_dim]), nn.LazyLinear(cfg.context_dim)),
                [MASKED_KEY], ["context_expert"]
            )
        ).to(self.device)

        def make_actor(context_key: str) -> ProbabilisticActor:
            actor = ProbabilisticActor(
                module=TensorDictSequential(
                    CatTensors([OBS_KEY, context_key], "actor_input", del_keys=False),
                    TensorDictModule(
                        nn.Sequential(make_mlp([256, 256, 256]), Actor(self.action_dim)),
                        ["actor_input"], ["loc", "scale"]
                    )
                ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True
            ).to(self.device)
            return actor

        # expert actor with privileged information
        self._actor_expert = make_actor("context_expert")
        # expert actor with estimated information used for compute estimation error
        self.__actor_expert = make_actor("context_adapt")
        
        def make_critic():
            return nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(1))
        
        self.critic_priv_a = TensorDictSequential(
            _mask_with_embedding(),
            CatTensors([OBS_KEY, OBS_PRIV_KEY, MASKED_KEY], "policy_priv", del_keys=False),
            TensorDictModule(make_critic(), ["policy_priv"], ["state_value"])
        ).to(self.device)

        self.critic_priv_b = TensorDictSequential(
            CatTensors([OBS_KEY, MASKED_KEY], "policy_priv", del_keys=False),
            TensorDictModule(make_critic(), ["policy_priv"], ["state_value"])
        ).to(self.device)

        def make_adapt_module(mode: str="deter"):
            if mode == "deter":
                module = TensorDictModule(
                    GRUModule(self.context_dim), 
                    [OBS_KEY, "is_init", "context_adapt_hx"], 
                    ["context_adapt", ("next", "context_adapt_hx")]
                ).to(self.device)
            elif mode == "stoch":
                module = TensorDictModule(
                    GRUModuleStoch(self.context_dim),
                    [OBS_KEY, "is_init", "context_adapt_hx"], 
                    ["context_adapt", "context_adapt_std", ("next", "context_adapt_hx")]
                ).to(self.device)
            return module

        self.adapt_module_a = make_adapt_module("stoch")
        self.adapt_module_a(fake_input)

        self.adapt_module_b = make_adapt_module()
        self.adapt_module_b(fake_input)
        self.adapt_modules = {
            "mse": self.adapt_module_a,
            "action_kl": self.adapt_module_b,
        }
        self.adapt_module_ema = copy.deepcopy(self.adapt_modules[self.cfg.adapt_module])
        self.adapt_module_ema(fake_input)
        
        self.encoder(fake_input)
        self._actor_expert(fake_input)
        self.__actor_expert(fake_input)
        self.critic_priv_a(fake_input)
        self.critic_priv_b(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "weight_decay": 0.02},
                {"params": self._actor_expert.parameters()},
                {"params": self.critic_priv_a.parameters()},
                {"params": self.critic_priv_b.parameters()}
            ],
            lr=cfg.lr
        )

        self.opt_adapt: torch.optim.Optimizer = torch.optim.Adam(
            [
                {"params": self.adapt_module_a.parameters(), "name": "adapt_module_a", "max_grad_norm": 10.},
                {"params": self.adapt_module_b.parameters(), "name": "adapt_module_b", "max_grad_norm": 20.},
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
            
            self.encoder.apply(init_)
            self._actor_expert.apply(init_)
            self.critic_priv_a.apply(init_)
            self.critic_priv_b.apply(init_)
            self.adapt_module_a.apply(init_)
            self.adapt_module_b.apply(init_)
        
        self.__actor_expert.requires_grad_(False)
        self.adapt_module_ema.requires_grad_(False)
    
    def get_rollout_policy(self, mode: str="train"):
        if mode == "train":
            policy = TensorDictSequential(
                self.encoder,
                self._actor_expert,
                ExcludeTransform("actor_input", "loc", "scale", MASKED_KEY)
            )
        elif mode == "eval":
            policy = TensorDictSequential(
                self.adapt_module_ema,
                self.__actor_expert,
                ExcludeTransform("actor_input", "loc", "scale", MASKED_KEY)
            )
        return policy

    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        spec = UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)
        return TensorDictPrimer({"context_adapt_hx": spec}, reset_key="done")
    
    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        infos = {}
        infos.update(self.train_expert(tensordict.copy()))
        infos.update(self.train_adaptation(tensordict.copy()))
        return infos

    def train_expert(self, tensordict: TensorDictBase):
        infos = []
        self._compute_advantage(tensordict, self.critic_priv_a, "adv_a", "ret_priv_a")
        self._compute_advantage(tensordict, self.critic_priv_b, "adv_b", "ret_priv_b")

        del tensordict["next"]

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                minibatch["adv_a"] = normalize(minibatch["adv_a"], True)
                minibatch["adv_b"] = normalize(minibatch["adv_b"])
                infos.append(self._update(minibatch))
        
        hard_copy_(self._actor_expert, self.__actor_expert)

        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["value_priv"] = tensordict["ret_priv_a"].mean().item()
        infos["value_obs"] = tensordict["ret_priv_b"].mean().item()
        return infos

    def train_adaptation(self, tensordict: TensorDictBase):
        infos = []
        with torch.no_grad():
            self.encoder(tensordict)
        for epoch in range(2):
            batch = make_batch(tensordict, 8, self.cfg.train_every)
            for minibatch in batch:
                losses = {}
                losses["adapt_module_a_loss"] = self._nll(minibatch.copy(), self.adapt_module_a)
                losses["adapt_module_b_loss"] = (
                    self._action_kl(minibatch.copy(), self.adapt_module_b)
                    + self._feature_mse(minibatch.copy(), self.adapt_module_b)
                )
                self.opt_adapt.zero_grad()
                sum(v for k, v in losses.items() if k.endswith("loss")).backward()
                for param_group in self.opt_adapt.param_groups:
                    grad_norm = nn.utils.clip_grad_norm_(param_group["params"], param_group["max_grad_norm"])
                    losses[param_group["name"] + "_grad_norm"] = grad_norm
                self.opt_adapt.step()
                infos.append(TensorDict(losses, []))
        
        soft_copy_(self.adapt_modules["mse"], self.adapt_module_ema, 0.05)

        infos = {f"adapt/{k}": v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        with torch.no_grad():
            infos["adapt/adapt_module_a_kl"] = self._action_kl(tensordict.copy(), self.adapt_module_a).item()
            infos["adapt/adapt_module_b_kl"] = self._action_kl(tensordict.copy(), self.adapt_module_b).item()
        if "context_adapt_std" in tensordict.keys():
            infos["adapt/estimate_std"] = tensordict["context_adapt_std"].mean().item()
        return infos
    
    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TensorDictModule, 
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        values = critic(tensordict)["state_value"]
        next_values = critic(tensordict["next"])["state_value"]

        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        self.encoder(tensordict)
        dist = self._actor_expert.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv_a"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * entropy

        b_returns_priv = tensordict["ret_priv_a"]
        values_priv = self.critic_priv_a(tensordict)["state_value"]
        value_loss_priv = self.critic_loss_fn(b_returns_priv, values_priv)
        value_loss_priv = (value_loss_priv * (~tensordict["is_init"])).mean()

        b_returns = tensordict["ret_priv_b"]
        values = self.critic_priv_b(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()

        
        loss = policy_loss + entropy_loss + value_loss + value_loss_priv
        self.opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self._actor_expert.parameters(), 10)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic_priv_a.parameters(), 10)
        critic_priv_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic_priv_b.parameters(), 10)
        self.opt.step()
        explained_var_obs = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        explained_var_priv = 1 - F.mse_loss(values_priv, b_returns_priv) / b_returns_priv.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss/obs": value_loss,
            "value_loss/priv": value_loss_priv,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "critic_priv_grad_norm": critic_priv_grad_norm,
            "value_loss/explained_var_obs": explained_var_obs,
            "value_loss/explained_var_priv": explained_var_priv
        }, [])

    def _feature_mse(self, tensordict: TensorDictBase, adapt_module: TensorDictModule, reduce: bool=True):
        adapt_module(tensordict)
        if reduce:
            return F.mse_loss(tensordict["context_adapt"], tensordict["context_expert"])
        else:
            return F.mse_loss(tensordict["context_adapt"], tensordict["context_expert"], reduction="none").mean(-1, True)
        
    def _nll(self, tensordict: TensorDictBase, adapt_module: TensorDictModule, reduce: bool=True):
        adapt_module(tensordict)
        dist = D.Normal(tensordict["context_adapt"], tensordict["context_adapt_std"])
        nll = -dist.log_prob(tensordict["context_expert"]).sum(-1, True)
        if reduce:
            nll = nll.mean()
        return nll

    def _action_kl(self, tensordict: TensorDictBase, adapt_module: TensorDictModule, reduce: bool=True):
        with torch.no_grad():
            dist1 = self._actor_expert.get_dist(tensordict)
        dist2 = self.__actor_expert.get_dist(adapt_module(tensordict))
        kl = D.kl_divergence(dist2, dist1).unsqueeze(-1)
        if reduce:
            kl = kl.mean()
        return kl
    
    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
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
        hard_copy_(self._actor_expert, self.__actor_expert)
        hard_copy_(self.adapt_modules[self.cfg.adapt_module], self.adapt_module_ema)
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
