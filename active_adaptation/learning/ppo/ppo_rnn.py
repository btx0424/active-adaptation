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
from torchrl.envs import CatTensors, TensorDictPrimer
from torchrl.modules import ProbabilisticActor

from ..modules.distributions import IndependentNormal
from .common import Actor, GAE, make_mlp
from ..utils.valuenorm import ValueNorm1


class LSTM(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        skip_conn, 
        allow_none=False
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.out = nn.Sequential(nn.LazyLinear(hidden_size), nn.Mish())
        self.skip_conn = skip_conn
        self.allow_none = allow_none

    def forward(
        self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor
    ):
        if x.ndim == 2:

            N = x.shape[0]
            if hx is None and self.allow_none:
                hx = torch.zeros(N, self.lstm.hidden_size, device=x.device)
            if cx is None and self.allow_none:
                cx = torch.zeros(N, self.lstm.hidden_size, device=x.device)
            reset = 1. - is_init.float().reshape(N, 1)
            hx, cx = self.lstm(x, (hx * reset, cx * reset))
            output = self.out(hx)
            return output, hx, cx
        
        elif x.ndim == 3:

            N, T = x.shape[:2]
            if hx is None and self.allow_none:
                hx = torch.zeros(N, self.lstm.hidden_size, device=x.device)
            else:
                hx = hx[:, 0]
            if cx is None and self.allow_none:
                cx = torch.zeros(N, self.lstm.hidden_size, device=x.device)
            else:
                cx = cx[:, 0]
            output = []
            reset = 1. - is_init.float().reshape(N, T, 1)
            for i, x_t, reset_t in zip(range(T), x.unbind(1), reset.unbind(1)):
                hx, cx = self.lstm(x_t, (hx * reset_t, cx * reset_t))
                if i < T // 4: # burn-in
                    hx, cx = hx.detach(), cx.detach()
                output.append(hx)
            output = torch.stack(output, dim=1)
            output = self.out(output)
            return (
                output,
                einops.repeat(hx, "b h -> b t h", t=T),
                einops.repeat(cx, "b h -> b t h", t=T)
            )


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
        self.out = nn.LazyLinear(hidden_size)
        self.allow_none = allow_none
        self.burn_in = burn_in

    def forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        if x.ndim == 2: # single step

            N = x.shape[0]
            if hx is None and self.allow_none:
                hx = torch.zeros(N, self.gru.hidden_size, device=x.device)
            assert (hx[is_init.squeeze()] == 0.).all()
            output = hx = self.gru(x, hx)
            output = self.out(output)
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
            output = self.out(output)
            return output, einops.repeat(hx, "b h -> b t h", t=T)


@dataclass
class PPOConfig:
    name: str = "ppo_rnn"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    seq_len: int = train_every
    lr: float = 5e-4

    # whether to take in priviledged infomation
    priv: bool = False

    rnn: str = "gru"
    skip_conn: Union[str, None] = None
    hidden_size: int = 128

    checkpoint_path: Union[str, None] = None


cs = ConfigStore.instance()
cs.store("ppo_gru", node=PPOConfig, group="algo")
cs.store("ppo_lstm", node=PPOConfig(rnn="lstm"), group="algo")

OBS_KEY = "policy" # OBS_KEY
OBS_PRIV_KEY = "priv"
OBS_HIST_KEY = "history"
ACTION_KEY = "action" # ACTION_KEY
REWARD_KEY = ("next", "reward") # ("agents", "reward")
# DONE_KEY = ("next", "done")
DONE_KEY = ("next", "terminated")

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

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.gae = GAE(0.99, 0.95)
        self.action_dim = action_spec.shape[-1]

        fake_input = observation_spec.zero()

        if cfg.rnn == "gru":

            def make_rnn(branch: str):
                return TensorDictModule(
                    GRU(256, self.cfg.hidden_size, allow_none=False),
                    ["feature", "is_init", f"{branch}_hx"],
                    ["feature", ("next", f"{branch}_hx")],
                )

        elif cfg.rnn == "lstm":

            def make_rnn(branch: str):
                return TensorDictModule(
                    LSTM(256, self.cfg.hidden_size, allow_none=False),
                    ["feature", "is_init", f"{branch}_hx", f"{branch}_cx"],
                    ["feature", ("next", f"{branch}_hx"), ("next", f"{branch}_cx")],
                )

        else:
            raise NotImplementedError(self.cfg.rnn)

        if cfg.priv:
            def make_encoder():
                return TensorDictSequential(
                    TensorDictModule(
                        make_mlp([128, 128]), [OBS_PRIV_KEY], ["context"]
                    ),
                    CatTensors(["feature", "context"], "feature", del_keys=False),
                )

            actor = TensorDictSequential(
                TensorDictModule(
                    make_mlp([256, 256]), [OBS_KEY], ["feature"]
                ),
                make_rnn(branch="actor"),
                make_encoder(),
                TensorDictModule(
                    nn.Sequential(make_mlp([256]), Actor(self.action_dim)),
                    ["feature"], ["loc", "scale"]
                ),
            ).to(self.device)
            self.critic = TensorDictSequential(
                TensorDictModule(
                    make_mlp([256, 256]), [OBS_KEY], ["feature"]
                ),
                make_rnn(branch="critic"),
                make_encoder(),
                TensorDictModule(
                    nn.Sequential(make_mlp([256]), nn.LazyLinear(1)),
                    ["feature"], ["state_value"]
                ),
            ).to(self.device)
        else:
            actor = TensorDictSequential(
                TensorDictModule(
                    make_mlp([512, 256]), [OBS_KEY], ["feature"]
                ),
                make_rnn(branch="actor"),
                TensorDictModule(Actor(self.action_dim), ["feature"], ["loc", "scale"]),
            ).to(self.device)
            self.critic = TensorDictSequential(
                TensorDictModule(
                    make_mlp([512, 256]), [OBS_KEY], ["feature"]
                ),
                make_rnn(branch="critic"),
                TensorDictModule(nn.LazyLinear(1), ["feature"], ["state_value"]),
            ).to(self.device)

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        self._maybe_init_state(fake_input)
        self.actor(fake_input.unsqueeze(1))
        self.critic(fake_input.unsqueeze(1))

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, (nn.GRUCell, nn.LSTMCell)):
                    nn.init.orthogonal_(module.weight_hh)

            self.actor.apply(init_)
            self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.value_norm = ValueNorm1(input_shape=1).to(self.device)

    def _maybe_init_state(self, tensordict: TensorDict):
        shape = tensordict.get(OBS_KEY).shape[:-1]
        if self.cfg.rnn == "gru":
            for key in ("actor_hx", "critic_hx"):
                if key not in tensordict.keys():
                    tensordict.set(
                        key,
                        torch.zeros(*shape, self.cfg.hidden_size, device=self.device),
                    )
        elif self.cfg.rnn == "lstm":
            for key in ("actor_hx", "actor_cx", "critic_hx", "critic_cx"):
                if key not in tensordict.keys():
                    tensordict.set(
                        key,
                        torch.zeros(*shape, self.cfg.hidden_size, device=self.device),
                    )
        return tensordict

    def __call__(self, tensordict: TensorDict):
        # self._maybe_init_state(tensordict)
        tensordict = self.actor(tensordict)
        tensordict = self.critic(tensordict)
        tensordict = tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict
    
    def make_tensordict_primer(self):
        from torchrl.envs.transforms.transforms import TensorDictPrimer

        num_envs = self.observation_spec.shape[0]
        if self.cfg.rnn == "gru":
            return TensorDictPrimer({
                "actor_hx": UnboundedContinuousTensorSpec((num_envs, 128)),
                "critic_hx": UnboundedContinuousTensorSpec((num_envs, 128))
            })
        else:
            return TensorDictPrimer({
                "actor_hx": UnboundedContinuousTensorSpec((num_envs, 128)),
                "actor_cx": UnboundedContinuousTensorSpec((num_envs, 128)),
                "critic_hx": UnboundedContinuousTensorSpec((num_envs, 128)),
                "critic_cx": UnboundedContinuousTensorSpec((num_envs, 128))
            })

    def train_op(self, tensordict: TensorDict):

        with torch.no_grad():
            next_tensordict = tensordict["next"].reshape(-1)
            assert not next_tensordict["is_init"].any()
            next_values = (
                self.critic(next_tensordict)["state_value"]
                .reshape(*tensordict.shape, 1)
            )

        rewards = tensordict[REWARD_KEY]
        dones = tensordict[DONE_KEY]
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
            batch = make_batch(tensordict, self.cfg.num_minibatches, self.cfg.seq_len)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = -self.entropy_coef * torch.mean(entropy)

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
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict(
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "actor_grad_norm": actor_grad_norm,
                "critic_grad_norm": critic_grad_norm,
                "explained_var": explained_var,
            },
            [],
        )


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
