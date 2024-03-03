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
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.modules import ProbabilisticActor


OBS_KEY = "policy" # ("agents", "observation")
OBS_PRIV_KEY = "priv"
OBS_HIST_KEY = "policy_h"
ACTION_KEY = "action" # ("agents", "action")
REWARD_KEY = ("next", "reward") # ("agents", "reward")
# DONE_KEY = ("next", "done")
DONE_KEY = ("next", "terminated")


def make_mlp(num_units, activation=nn.Mish, norm="before", dropout=0.):
    assert norm in ("before", "after", None)
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        if norm == "before":
            layers.append(nn.LayerNorm(n))
            layers.append(activation())
        elif norm == "after":
            layers.append(activation())
            layers.append(nn.LayerNorm(n))
        else:
            layers.append(activation())
        if dropout > 0. :
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


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


class Chunk(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n
    
    def forward(self, x):
        return x.chunk(self.n, dim=-1)

class Duplicate(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n
    
    def forward(self, x):
        return tuple(x for _ in range(self.n))


class Actor(nn.Module):
    def __init__(self, action_dim: int, predict_std: bool=False) -> None:
        super().__init__()
        self.predict_std = predict_std
        if predict_std:
            self.actor_mean = nn.LazyLinear(action_dim * 2)
        else:
            self.actor_mean = nn.LazyLinear(action_dim)
            self.actor_std = nn.Parameter(torch.zeros(action_dim))
        self.scale_mapping = torch.exp
    
    def forward(self, features: torch.Tensor):
        if self.predict_std:
            loc, scale = self.actor_mean(features).chunk(2, dim=-1)
        else:
            loc = self.actor_mean(features)
            scale = self.actor_std.expand_as(loc)
        scale = self.scale_mapping(scale)
        return loc, scale


class GAE(nn.Module):
    def __init__(self, gamma, lmbda, fake_bootstrap=False):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
        self.fake_bootstrap = fake_bootstrap
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            if self.fake_bootstrap:
                next_value_t = torch.where(terminated[:, step], value[:, step], next_value[:, step])
            else:
                next_value_t = next_value[:, step] * not_done[:, step]
            delta = reward[:, step] + self.gamma * next_value_t - value[:, step]
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae)
        returns = advantages + value
        return advantages, returns


def init_(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, 0.01)
        nn.init.constant_(module.bias, 0.)


def compute_policy_loss(
    tensordict: TensorDictBase,
    actor: ProbabilisticActor,
    clip_param: float,
    entropy_coef: float,
    discard_init: bool=True,
):
    dist = actor.get_dist(tensordict)
    log_probs = dist.log_prob(tensordict[ACTION_KEY])
    entropy = dist.entropy()

    adv = tensordict["adv"]
    ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
    surr1 = adv * ratio
    surr2 = adv * ratio.clamp(1. - clip_param, 1. + clip_param)
    policy_loss = torch.min(surr1, surr2)
    if discard_init:
        policy_loss = policy_loss * (~tensordict["is_init"])
    policy_loss = - torch.mean(policy_loss) * dist.event_shape[-1]
    entropy_loss = - entropy_coef * torch.mean(entropy)
    return policy_loss, entropy_loss, entropy.mean()


def compute_value_loss(
    tensordict: TensorDictBase, 
    critic: TensorDictModuleBase,
    clip_param: float,
    critic_loss_fn: nn.Module,
    discard_init: bool=True,
):
    # b_values = tensordict["state_value"]
    b_returns = tensordict["ret"]
    values = critic(tensordict)["state_value"]
    # values_clipped = b_values + (values - b_values).clamp(-clip_param, clip_param)
    # value_loss_clipped = critic_loss_fn(b_returns, values_clipped)
    value_loss_original = critic_loss_fn(b_returns, values)
    # value_loss = torch.max(value_loss_original, value_loss_clipped).mean()

    # mask out first transitions which are generally invalid
    # due to the limiatations of Isaac Sim
    if discard_init:
        value_loss_original = value_loss_original * (~tensordict["is_init"])
    value_loss = value_loss_original.mean()
    explained_var = 1 - value_loss_original.detach() / b_returns.var()

    return value_loss, explained_var


def hard_copy_(source_module: nn.Module, target_module: nn.Module):
    for params_source, params_target in zip(source_module.parameters(), target_module.parameters()):
        params_target.data.copy_(params_source.data)

def soft_copy_(source_module: nn.Module, target_module: nn.Module, tau: float = 0.01):
    for params_source, params_target in zip(source_module.parameters(), target_module.parameters()):
        params_target.data.lerp_(params_source.data, tau)


class L2Norm(nn.Module):
    
    def forward(self, x):
        return x / torch.norm(x, dim=-1, keepdim=True).clamp(1e-7)

class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, dim: int, method="l2"):
        super().__init__()
        self.dim = dim
        if method == "softmax":
            self.f = F.softmax
        elif method == "l2":
            self.f = lambda x: x / x.norm(dim=-1, keepdim=True).clamp(1e-6)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = self.f(x)
        return x.view(*shp)


def collect_info(infos, prefix=""):
    return {prefix+k: v.mean().item() for k, v in torch.stack(infos).items()}

