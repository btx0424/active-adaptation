import torch
import torch.nn as nn
import einops
import contextlib

from torch.utils._contextlib import _DecoratorContextManager

_RECURRENT_MODE = False

class set_recurrent_mode(_DecoratorContextManager):
    def __init__(self, mode: bool = True):
        super().__init__()
        self.mode = mode
        self.prev = _RECURRENT_MODE
    
    def __enter__(self):
        global _RECURRENT_MODE
        _RECURRENT_MODE = self.mode
    
    def __exit__(self, exc_type, exc_value, traceback):
        global _RECURRENT_MODE
        _RECURRENT_MODE = self.prev


def recurrent_mode():
    return _RECURRENT_MODE


class LSTM(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        burn_in: int = 0, 
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.out = nn.Sequential(nn.LazyLinear(hidden_size), nn.Mish())
        self.burn_in = burn_in

    def forward(
        self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor
    ):  
        if recurrent_mode():
            N, T = x.shape[:2]
            hx = hx[:, 0]
            cx = cx[:, 0]
            output = []
            reset = 1. - is_init.float().reshape(N, T, 1)
            for i, x_t, reset_t in zip(range(T), x.unbind(1), reset.unbind(1)):
                hx, cx = self.lstm(x_t, (hx * reset_t, cx * reset_t))
                output.append(hx)
            output = torch.stack(output, dim=1)
            output = self.out(output)
            return (
                output,
                einops.repeat(hx, "b h -> b t h", t=T),
                einops.repeat(cx, "b h -> b t h", t=T)
            )
        else:
            N = x.shape[0]
            reset = 1. - is_init.float().reshape(N, 1)
            hx, cx = self.lstm(x, (hx * reset, cx * reset))
            output = self.out(hx)
            return output, hx, cx


class GRU(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        burn_in: int = 0,
        learnable_init: bool = False,
    ) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.out = nn.LazyLinear(hidden_size)
        self.learnable_init = learnable_init
        self.burn_in = burn_in

        if self.learnable_init:
            self.init = nn.Parameter(torch.zeros(hidden_size))

    def _maybe_init(self, hx, is_init):
        if self.learnable_init:
            return torch.where(is_init, self.init, hx)
        else:
            return torch.where(is_init, torch.zeros_like(hx), hx)
        
    def forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        if recurrent_mode(): 
            N, T = x.shape[:2]
            is_init = is_init.reshape(N, T, 1)
            hx = self._maybe_init(hx[:, 0], is_init[:, 0])
            output = []
            for i, x_t, init_t in zip(range(T), x.unbind(1), is_init.unbind(1)):
                hx = self._maybe_init(hx, init_t)
                hx = self.gru(x_t, hx)
                output.append(hx)
            output = torch.stack(output, dim=1)
            output = self.out(output)
            return output, einops.repeat(hx, "b h -> b t h", t=T)
        else:
            N = x.shape[0]
            is_init = is_init.reshape(N, 1)
            hx = self._maybe_init(hx, is_init)
            output = hx = self.gru(x, hx)
            output = self.out(output)
            return output, hx