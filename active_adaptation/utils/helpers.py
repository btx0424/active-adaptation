import torch
from functools import wraps
from typing import Sequence, Dict, Any
from tensordict import TensorDictBase
from prettytable import PrettyTable


def table_print(info: Dict[str, Any]):
    pt = PrettyTable()
    nrow = max(len(v) for v in info.values())
    for k, v in info.items():
        data = [f"{kk}:{vv:.3f}" for kk, vv in v.items()]
        data += [" "] * (nrow - len(data))
        pt.add_column(k, data)
    print(pt)


def batchify(func, broadcast=True):
    @wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = [arg.shape[:-1] for arg in args]
        if broadcast:
            batch_shape = torch.broadcast_shapes(*batch_shapes)
        else:
            batch_shape = set(batch_shapes)
            if len(batch_shape) != 1:
                raise ValueError()
            batch_shape = batch_shape.pop()
        args = [
            arg.expand(*batch_shape, arg.shape[-1]).reshape(-1, arg.shape[-1]) 
            for arg in args
        ]
        ret = func(*args, **kwargs)
        return ret.reshape(*batch_shape, *ret.shape[1:])
    return wrapped

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1

class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    def add(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict = tensordict["next"]
        done = next_tensordict["done"]
        if done.any():
            done = done.squeeze(-1)
            self._episodes += done.sum().item()
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats.extend(
                next_tensordict[done].clone().unbind(0)
            )
        return len(self)
    
    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats).to_tensordict()
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)