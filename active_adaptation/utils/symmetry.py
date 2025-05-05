import torch
import torch.nn as nn
from typing import Sequence, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class SymmetryTransform(nn.Module):
    def __init__(self, perm: Optional[torch.Tensor]=None, signs: Optional[torch.Tensor]=None):
        super().__init__()
        self.perm: Optional[torch.Tensor]
        self.signs: Optional[torch.Tensor]
        if perm is not None:
            self.register_buffer("perm", torch.as_tensor(perm))
        else:
            self.perm = slice(None)
        if signs is not None:
            self.register_buffer("signs", torch.as_tensor(signs))
        else:
            self.register_buffer("signs", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm] * self.signs
    
    def repeat(self, n: int) -> "SymmetryTransform":
        return SymmetryTransform.cat([self] * n)

    @staticmethod
    def cat(transforms: Sequence["SymmetryTransform"]) -> "SymmetryTransform":
        perm = []
        signs = []
        num = 0
        for t in transforms:
            perm.append(t.perm + num)
            signs.append(t.signs)
            num += t.perm.shape[0]
        return SymmetryTransform(torch.cat(perm), torch.cat(signs))


def joint_space_symmetry(asset: Articulation, joint_names: Sequence[str]):
    """
    Return a permutation that transforms a vector of joint positions into its 
    left-right symmetric counterpart.
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
    Return a permutation that transforms a vector of spatial positions into its 
    left-right symmetric counterpart.
    """
    if not hasattr(asset.cfg, "spatial_symmetry_mapping"):
        raise ValueError("Asset does not have a spatial symmetry mapping config.")
    symmetry_mapping = asset.cfg.spatial_symmetry_mapping
    ids = []
    ids_inv = []
    signs = []
    for this_body_name in body_names:
        other_body_name = symmetry_mapping[this_body_name]
        ids.append(asset.body_names.index(other_body_name))
        ids_inv.append(asset.body_names.index(this_body_name))
        signs.append([1, -1, 1]) # only flip y
    transform = SymmetryTransform(ids, signs)
    transform_inv = SymmetryTransform(ids_inv, signs)
    return transform, transform_inv
