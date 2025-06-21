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


def mirrored(symmetry_mapping: dict):
    """
    Return a dictionary of mirrored joint names.
    """
    mirrored = {}
    for k, v in symmetry_mapping.items():
        if isinstance(v, tuple): # joint space symmetry
            mirrored[v[1]] = (v[0], k)
        elif isinstance(v, str): # cartesian space symmetry
            mirrored[v] = k
        else:
            raise ValueError(f"Invalid symmetry mapping: ({k}, {v})")
    symmetry_mapping.update(mirrored)
    return symmetry_mapping


def joint_space_symmetry(asset: "Articulation", joint_names: Sequence[str]):
    """
    Return a permutation that transforms a vector of joint positions into its 
    left-right symmetric counterpart.
    """
    if getattr(asset.cfg, "joint_symmetry_mapping", None) is None:
        raise ValueError("Asset does not have a joint symmetry mapping config.")
    symmetry_mapping = asset.cfg.joint_symmetry_mapping
    if not len(symmetry_mapping) == len(asset.joint_names):
        diff = set(asset.joint_names) - set(symmetry_mapping.keys())
        raise ValueError(
            f"Joint symmetry mapping must contain all joint names\n"
            f"\tAll Joints - Specified: {set(asset.joint_names) - set(symmetry_mapping.keys())}\n"
            f"\tSpecified - All Joints: {set(symmetry_mapping.keys()) - set(asset.joint_names)}"
        )
        
    ids = torch.zeros(len(joint_names), dtype=torch.long)
    signs = torch.zeros(len(joint_names), dtype=torch.float32)
    for i, this_joint_name in enumerate(joint_names):
        sign, other_joint_name = symmetry_mapping[this_joint_name]
        ids[i] = joint_names.index(other_joint_name)
        signs[i] = sign
    transform = SymmetryTransform(ids, signs)
    return transform


def cartesian_space_symmetry(asset: "Articulation", body_names: Sequence[str], sign=(1, -1, 1)):
    """
    Return a permutation that transforms a vector of spatial positions into its 
    left-right symmetric counterpart.
    """
    if getattr(asset.cfg, "spatial_symmetry_mapping", None) is None:
        raise ValueError("Asset does not have a spatial symmetry mapping config.")
    symmetry_mapping = asset.cfg.spatial_symmetry_mapping
    if not len(symmetry_mapping) == len(asset.body_names):
        raise ValueError(
            "Spatial symmetry mapping must contain all body names\n"
            f"\tAll Bodies - Specified: {set(asset.body_names) - set(symmetry_mapping.keys())}\n"
            f"\tSpecified - All Bodies: {set(symmetry_mapping.keys()) - set(asset.body_names)}"
        )
        
    ids = torch.zeros(len(body_names), len(sign), dtype=torch.long)
    signs = torch.zeros(len(body_names), len(sign), dtype=torch.float32)
    for i, this_body_name in enumerate(body_names):
        other_body_name = symmetry_mapping[this_body_name]
        ids[i] = body_names.index(other_body_name) * len(sign) + torch.arange(len(sign))
        signs[i] = torch.tensor(sign)
    transform = SymmetryTransform(ids.flatten(), signs.flatten())
    return transform
