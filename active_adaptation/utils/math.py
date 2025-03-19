# This file contains additional math utilities
# that are not covered by IsaacLab

import torch
import torch.distributions as D
from isaaclab.utils.math import yaw_quat, wrap_to_pi, euler_xyz_from_quat, quat_from_euler_xyz

from .helpers import batchify

# @torch.compile
@batchify
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


# @torch.compile
@batchify
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x

def clamp_along(x: torch.Tensor, axis: torch.Tensor, min: float, max: float):
    projection = (x * axis).sum(dim=-1, keepdim=True)
    return x - projection * axis + projection.clamp(min, max) * axis


@batchify
def yaw_rotate(yaw: torch.Tensor, vec: torch.Tensor):
    yaw_cos = torch.cos(yaw).squeeze(-1)
    yaw_sin = torch.sin(yaw).squeeze(-1)
    return torch.stack(
        [
            yaw_cos * vec[:, 0] - yaw_sin * vec[:, 1],
            yaw_sin * vec[:, 0] + yaw_cos * vec[:, 1],
            vec[:, 2],
        ],
        1,
    )


def quat_from_yaw(yaw: torch.Tensor):
    return torch.cat(
        [
            torch.cos(yaw / 2).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.sin(yaw / 2).unsqueeze(-1),
        ],
        dim=-1,
    )


class MultiUniform(D.Distribution):
    """
    A distribution over the union of multiple disjoint intervals.
    """
    def __init__(self, ranges: torch.Tensor):
        batch_shape = ranges.shape[:-2]
        if not ranges[..., 0].le(ranges[..., 1]).all():
            raise ValueError("Ranges must be non-empty and ordered.")
        super().__init__(batch_shape, validate_args=False)
        self.ranges = ranges
        self.ranges_len = ranges.diff(dim=-1).squeeze(1)
        self.total_len = self.ranges_len.sum(-1)
        self.starts = torch.zeros_like(ranges[..., 0])
        self.starts[..., 1:] = self.ranges_len.cumsum(-1)[..., :-1]

    def sample(self, sample_shape: torch.Size = ()) -> torch.Tensor:
        sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self.batch_shape
        uniform = torch.rand(shape, device=self.ranges.device) * self.total_len
        i = torch.searchsorted(self.starts, uniform) - 1
        return self.ranges[i, 0] + uniform - self.starts[i]

