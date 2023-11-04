import torch
from functools import wraps


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
        numel = batch_shape.numel()
        args = [
            arg.reshape(-1, arg.shape[-1]).expand(numel, arg.shape[-1]) 
            for arg in args
        ]
        ret = func(*args, **kwargs)
        return ret.reshape(*batch_shape, *ret.shape[1:])
    return wrapped

