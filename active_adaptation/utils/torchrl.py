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
import time
from torchrl.collectors import SyncDataCollector as _SyncDataCollector
from torchrl.collectors.utils import split_trajectories
from torchrl.envs.utils import _replace_last, step_mdp
from tensordict.tensordict import TensorDictBase

from typing import Iterator

class SyncDataCollector(_SyncDataCollector):

    def rollout(self) -> TensorDictBase:
        start = time.perf_counter()
        _tensordict_out = super().rollout()
        self._fps = _tensordict_out.numel() / (time.perf_counter() - start)
        return _tensordict_out
    
    def iterator(self) -> Iterator[TensorDictBase]:
        """Iterates through the DataCollector.

        Yields: TensorDictBase objects containing (chunks of) trajectories

        """
        
        total_frames = self.total_frames
        i = -1
        self._frames = 0
        while True:
            i += 1
            self._iter = i
            torch.compiler.cudagraph_mark_step_begin()
            tensordict_out = self.rollout()
            self._frames += tensordict_out.numel()
            # if self._frames >= total_frames:
            #     self.env.close()

            if self.split_trajs:
                tensordict_out = split_trajectories(
                    tensordict_out, prefix="collector"
                )
            if self.postproc is not None:
                tensordict_out = self.postproc(tensordict_out)
            if self._exclude_private_keys:
                excluded_keys = [
                    key for key in tensordict_out.keys() if key.startswith("_")
                ]
                tensordict_out = tensordict_out.exclude(
                    *excluded_keys, inplace=True
                )
            if self.return_same_td:
                yield tensordict_out
            else:
                # we must clone the values, as the tensordict is updated in-place.
                # otherwise the following code may break:
                # >>> for i, data in enumerate(collector):
                # >>>      if i == 0:
                # >>>          data0 = data
                # >>>      elif i == 1:
                # >>>          data1 = data
                # >>>      else:
                # >>>          break
                # >>> assert data0["done"] is not data1["done"]
                yield tensordict_out.clone()

            if self._frames >= self.total_frames:
                break


from torchrl.envs.transforms import Transform
from torchrl.data import TensorSpec
from tensordict.utils import NestedKey

import warnings
from typing import Sequence

class StackFrames(Transform):
    """Stacks successive observation frames into a single tensor.

    This transform stacks the history of selected observations over a specified number of steps.
    which can be useful for inferring the state in a partially observable environment. The shape 
    of each stacked observation in the output is extended to `[*shape, steps]`, where `shape` is
    the original shape of the observation. The most recent observation is indexed at `[..., -1]`.

    Note that this transform is stateless. Also see :class:`CatFrames`.

    Args:
        N (int): Number of steps for which the observation history is maintained.
        in_keys (list of NestedKeys, optional): Keys of the observations in the environment's 
            observation spec that need to be recorded. 
        out_keys (list of NestedKeys, optional): Keys under which the recorded observation histories 
            will be stored in the output. Defaults to `f"{in_key}_h"` for each key in `in_keys`.
        padding (str, optional): the padding method. One of ``"same"`` or ``"constant"``.
            Defaults to ``"same"``, ie. the first value is used for padding.
        padding_value (float, optional): the value to use for padding if ``padding="constant"``.
            Defaults to 0.
    
    Examples:
        >>> from torchrl.envs.transforms import TransformedEnv, StackFrames
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(GymEnv("CartPole-v1"), StackFrames(["observation"]))
        >>> td = env.reset()
        >>> print(td["observation_h"].shape)
        torch.Size([4, 16])
    """

    ACCEPTED_PADDING = {"same", "constant", "zeros"}

    def __init__(
        self,
        N: int = 1,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        padding="same",
        padding_value=0,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = [
                f"{key}_h" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_h",)
                for key in in_keys
            ]
        if any(key in in_keys for key in out_keys):
            raise ValueError(
                f"out_keys {out_keys} cannot duplicate with in_keys {in_keys}"
            )
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.N = N
        if padding not in self.ACCEPTED_PADDING:
            raise ValueError(f"padding must be one of {self.ACCEPTED_PADDING}")
        if padding == "zeros":
            warnings.warn(
                "Padding option 'zeros' will be deprecated in v0.4.0. "
                "Please use 'constant' padding with padding_value 0 instead.",
                category=DeprecationWarning,
            )
            padding = "constant"
            padding_value = 0
        self.padding = padding
        self.padding_value = padding_value
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            is_tuple = isinstance(in_key, tuple)
            if in_key in observation_spec.keys(include_nested=is_tuple):
                spec = observation_spec[in_key]
                spec = spec.unsqueeze(-1).expand(*spec.shape, self.N)
                observation_spec[out_key] = spec
        return observation_spec
    
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        state_spec = input_spec["full_state_spec"]
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            spec = self.parent.observation_spec[in_key]
            state_spec[out_key] = spec.unsqueeze(-1).expand(*spec.shape, self.N)
        input_spec["full_state_spec"] = state_spec
        return input_spec

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            current = next_tensordict.get(in_key)
            prev_stacked = tensordict.get(out_key)
            val_next = torch.cat([prev_stacked[..., 1:], current.unsqueeze(-1)], dim=-1)
            next_tensordict.set(out_key, val_next)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        _reset = tensordict.get("_reset", None)
        if _reset is None:
            _reset = torch.ones(tensordict.batch_size, dtype=bool, device=tensordict.device)
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            # get previous observations
            current = tensordict_reset.get(in_key)
            prev_stacked = tensordict.get(out_key, None)
            if prev_stacked is None:
                spec = self.parent.full_observation_spec[in_key]
                stacked = spec.unsqueeze(-1).expand(*spec.shape, self.N).zero()
            else:
                stacked = prev_stacked.clone()
            # handle padding
            val = current[_reset.squeeze()]
            if self.padding == "same":
                padding_val = val.unsqueeze(-1).expand(*val.shape, self.N-1)
            elif self.padding == "constant":
                shape = val.shape + (self.N-1,)
                padding_val == torch.full(shape, self.padding_value, dtype=val.dtype, device=val.device)
            stacked[_reset.squeeze()] = torch.cat([padding_val, val.unsqueeze(-1)], dim=-1)
            tensordict_reset.set(out_key, stacked)
        return tensordict_reset