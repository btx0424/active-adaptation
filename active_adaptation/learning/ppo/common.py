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
from tensordict import TensorDict


def make_mlp(num_units, activation=nn.Mish):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(activation())
        layers.append(nn.LayerNorm(n))
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
