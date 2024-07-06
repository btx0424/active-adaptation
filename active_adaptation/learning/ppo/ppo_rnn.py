# MIT License
#
# Copyright (c) 2023 Botian Xu
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

from dataclasses import dataclass, replace
from functools import partial
from typing import Union

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import einops

from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs import CatTensors, TensorDictPrimer, ExcludeTransform
from torchrl.modules import ProbabilisticActor

from ..modules.distributions import IndependentNormal
from ..modules.rnn import GRU, LSTM, set_recurrent_mode
from .common import *
from ..utils.valuenorm import ValueNorm1, ValueNormFake


class GRUModule(nn.Module):
    def __init__(
        self, 
        dim: int, 
        learnable_init: bool = False,
        skip_conn: bool = False,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.skip_conn = skip_conn
        self.mlp = make_mlp([256])
        self.gru = GRU(input_size=256, hidden_size=128, learnable_init=learnable_init)
        self.layer_norm = nn.LayerNorm(128) if layer_norm else nn.Identity()
        self.out = make_mlp([256])
    
    def forward(self, obs: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        obs_feature = self.mlp(obs)
        rnn_feature, hx = self.gru(obs_feature, is_init, hx)
        rnn_feature = self.layer_norm(rnn_feature)
        feature = self.out(rnn_feature)
        if self.skip_conn:
            feature = feature + obs_feature
        return feature, hx.contiguous()


@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_rnn.PPORNNPolicy"
    name: str = "ppo_rnn"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 8
    seq_len: int = train_every
    lr: float = 5e-4
    entropy_coef: float = 0.002
    value_norm: bool = False

    rnn: str = "gru"
    hidden_size: int = 128
    learnable_init: bool = True
    skip_conn: bool = True
    aux_target: bool = False

    checkpoint_path: Union[str, None] = None


cs = ConfigStore.instance()
cs.store("ppo_gru", node=PPOConfig, group="algo")
cs.store("ppo_lstm", node=PPOConfig(rnn="lstm"), group="algo")


class PPORNNPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: TensorSpec,
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec

        self.entropy_coef = cfg.entropy_coef
        self.clip_param = 0.1
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        # self.critic_loss_fn = nn.HuberLoss(delta=10, reduction="none")
        self.gae = GAE(0.99, 0.95)
        self.action_dim = action_spec.shape[-1]
        if self.cfg.aux_target:
            aux_target_spec = observation_spec.get("aux_target_", None)
            if aux_target_spec is None:
                raise ValueError("Specify `aux_target_` in observation spec for aux target.")
            self.aux_dim = observation_spec["aux_target_"].shape[-1]
        else:
            self.aux_dim = 0

        if self.cfg.value_norm:
            self.value_norm = ValueNorm1(input_shape=1).to(self.device)
        else:
            self.value_norm = ValueNormFake().to(self.device)

        fake_input = observation_spec.zero()
        
        actor_modules = [
            TensorDictModule(
                GRUModule(128, self.cfg.learnable_init, self.cfg.skip_conn), 
                [OBS_KEY, "is_init", "actor_hx"], 
                ["_actor_in", ("next", "actor_hx")]
            ),
            TensorDictModule(Actor(self.action_dim), ["_actor_in"], ["loc", "scale"]),
        ]
        if self.cfg.aux_target:
            actor_modules.append(TensorDictModule(nn.LazyLinear(self.aux_dim), ["_actor_in"], ["aux_pred"]))
        
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(*actor_modules),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        critic_module = nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(1))
        self.critic = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "policy_priv", del_keys=False),
            TensorDictModule(critic_module, ["policy_priv"], ["state_value"])
        ).to(self.device)

        self._maybe_init_state(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.GRUCell, nn.LSTMCell)):
                nn.init.orthogonal_(module.weight_hh)

        self.actor.apply(init_)
        self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def _maybe_init_state(self, tensordict: TensorDict):
        shape = tensordict.get(OBS_KEY).shape[:-1]
        with torch.device(self.device):
            tensordict["is_init"] = torch.ones(tensordict.shape[0], 1, dtype=bool)
            zeros = torch.zeros(*shape, self.cfg.hidden_size)
            if self.cfg.rnn == "gru":
                keys = ["actor_hx"]
            elif self.cfg.rnn == "lstm":
                keys = ["actor_hx", "actor_cx"]
            for key in keys:
                if key not in tensordict.keys():
                    tensordict.set(key, zeros)
        return tensordict

    def get_rollout_policy(self, mode: str):
        return TensorDictSequential(
            self.actor,
            ExcludeTransform("_actor_in", "loc", "scale")
        )
    
    def make_tensordict_primer(self):
        from torchrl.envs.transforms.transforms import TensorDictPrimer
        num_envs = self.observation_spec.shape[0]
        if self.cfg.rnn == "gru":
            return TensorDictPrimer({
                "actor_hx": UnboundedContinuousTensorSpec((num_envs, 128)),
            })
        else:
            return TensorDictPrimer({
                "actor_hx": UnboundedContinuousTensorSpec((num_envs, 128)),
                "actor_cx": UnboundedContinuousTensorSpec((num_envs, 128)),
            })

    def train_op(self, tensordict: TensorDict):

        with torch.no_grad():
            self.critic(tensordict)
            self.critic(tensordict["next"])

            rewards = tensordict[REWARD_KEY]
            dones = tensordict[DONE_KEY]
            values = tensordict["state_value"]
            next_values = tensordict["next", "state_value"]
            values = self.value_norm.denormalize(values)
            next_values = self.value_norm.denormalize(next_values)

            adv, ret = self.gae(rewards, dones, values, next_values)
            adv = normalize(adv, subtract_mean=True)
            self.value_norm.update(ret)
            ret = self.value_norm.normalize(ret)

            tensordict.set("adv", adv)
            tensordict.set("ret", ret)

        infos = []
        with set_recurrent_mode(True): # re-rollout the trajectory
            for epoch in range(self.cfg.ppo_epochs):
                batch = make_batch(tensordict, self.cfg.num_minibatches, self.cfg.seq_len)
                for minibatch in batch:
                    infos.append(TensorDict(self._update(minibatch), []))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        infos["critic/value_mean"] = tensordict["ret"].mean()
        return {k: v.item() for k, v in sorted(infos.items())}

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = - torch.min(surr1, surr2)
        policy_loss = torch.mean(policy_loss * (~tensordict["is_init"]))
        entropy_loss = -self.entropy_coef * torch.mean(entropy)

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()

        loss = policy_loss + entropy_loss + value_loss
        
        if self.cfg.aux_target:
            aux_loss = F.mse_loss(tensordict["aux_target_"], tensordict["aux_pred"])
            loss += aux_loss
        else:
            aux_loss = torch.tensor(0., device=self.device)

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 2.)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 2.)
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return {
            "actor/policy_loss": policy_loss,
            "actor/entropy": entropy,
            "actor/noise_std": tensordict["scale"].mean(),
            "actor/grad_norm": actor_grad_norm,
            "critic/grad_norm": critic_grad_norm,
            "critic/value_loss": value_loss,
            "critic/explained_var": explained_var,
            "aux/pred_loss": aux_loss,
        }
        


def make_batch(tensordict: TensorDict, num_minibatches: int, seq_len: int = -1):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
