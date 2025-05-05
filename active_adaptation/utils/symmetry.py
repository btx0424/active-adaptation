import torch
import torch.nn as nn
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class SymmetryTransform(nn.Module):
    def __init__(self, perm: torch.Tensor, signs: torch.Tensor):
        super().__init__()
        self.perm = torch.tensor(perm)
        self.signs = torch.tensor(signs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm] * self.signs


def joint_space_symmetry(asset: Articulation, joint_names: Sequence[str]):
    """
    Return a permutation that transforms a vector of joint positions into its symmetric counterpart.
    """
    if not hasattr(asset.cfg, "joint_symmetry_mapping"):
        raise ValueError("Asset does not have a joint symmetry mapping config.")
    symmetry_mapping = asset.cfg.joint_symmetry_mapping
    ids = []
    ids_inv = []
    signs = []
    for this_joint_name in joint_names:
        sign, other_joint_name = symmetry_mapping[this_joint_name]
        ids.append(asset.joint_names.index(other_joint_name))
        ids_inv.append(asset.joint_names.index(this_joint_name))
        signs.append(sign)
    transform = SymmetryTransform(ids, signs)
    transform_inv = SymmetryTransform(ids_inv, signs)
    return transform, transform_inv


def cartesian_space_symmetry(asset: Articulation, body_names: Sequence[str]):
    """
    Return a permutation that transforms a vector of spatial positions into its symmetric counterpart.
    """
    if not hasattr(asset.cfg, "spatial_symmetry_mapping"):
        raise ValueError("Asset does not have a spatial symmetry mapping config.")
    symmetry_mapping = asset.cfg.spatial_symmetry_mapping

