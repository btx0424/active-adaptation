from isaaclab.assets import ArticulationCfg as _ArticulationCfg
from isaaclab.utils import configclass

from typing import Mapping

@configclass
class ArticulationCfg(_ArticulationCfg):
    joint_symmetry_mapping: Mapping[str, list[int | tuple[int, str]]] = None
    spatial_symmetry_mapping: Mapping[str, str] = None

