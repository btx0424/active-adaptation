import torch
import torch.nn as nn
from typing import Sequence, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class SymmetryTransform(nn.Module):
    def __init__(self, perm, signs):
        super().__init__()
        self.perm: torch.Tensor
        self.signs: torch.Tensor
        if not len(perm) == len(signs) > 0:
            raise ValueError("perm and signs must have the same length and be non-empty.")
        
        self.register_buffer("perm", torch.as_tensor(perm))
        self.register_buffer("signs", torch.as_tensor(signs, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm] * self.signs
    
    def repeat(self, n: int) -> "SymmetryTransform":
        return SymmetryTransform.cat([self] * n)

    @staticmethod
    def cat(transforms: Sequence["SymmetryTransform"]) -> "SymmetryTransform":
        if not all(isinstance(t, SymmetryTransform) for t in transforms):
            raise ValueError("All transforms must be SymmetryTransform instances.")
        perm = []
        signs = []
        num = 0
        for t in transforms:
            perm.append(t.perm + num)
            signs.append(t.signs)
            num += t.perm.shape[0]
        return SymmetryTransform(torch.cat(perm), torch.cat(signs))


def joint_space_symmetry(asset: "Articulation", joint_names: Sequence[str]):
    """
    Return a permutation that transforms a vector of joint positions into its 
    left-right symmetric counterpart.
    """
    if getattr(asset.cfg, "joint_symmetry_mapping", None) is None:
        raise ValueError("Asset does not have a joint symmetry mapping config.")
    symmetry_mapping = asset.cfg.joint_symmetry_mapping
    ids = torch.zeros(len(joint_names), dtype=torch.long)
    signs = torch.zeros(len(joint_names), dtype=torch.float32)
    for i, this_joint_name in enumerate(joint_names):
        sign, other_joint_name = symmetry_mapping[this_joint_name]
        ids[i] = joint_names.index(other_joint_name)
        signs[i] = sign
    transform = SymmetryTransform(ids, signs)
    return transform


def cartesian_space_symmetry(asset: "Articulation", body_names: Sequence[str]):
    """
    Return a permutation that transforms a vector of spatial positions into its 
    left-right symmetric counterpart.
    """
    if getattr(asset.cfg, "spatial_symmetry_mapping", None) is None:
        raise ValueError("Asset does not have a spatial symmetry mapping config.")
    symmetry_mapping = asset.cfg.spatial_symmetry_mapping
    ids = []
    signs = []
    for this_body_name in body_names:
        other_body_name = symmetry_mapping[this_body_name]
        ids.append(asset.body_names.index(other_body_name))
        signs.append([1, -1, 1]) # only flip y
    transform = SymmetryTransform(ids, signs)
    return transform
