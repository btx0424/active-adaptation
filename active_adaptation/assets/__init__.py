import os
import copy
import torch

from omni.isaac.orbit_assets import (
    ArticulationCfg,
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG,
    UNITREE_GO2_CFG,
    ANYMAL_C_CFG,
)
from .cassie import CASSIE_CFG

from omni.isaac.orbit.sim.utils import bind_physics_material, clone
from omni.isaac.orbit.sim.spawners.from_files.from_files import _spawn_from_usd_file

from omni.isaac.core.materials import PhysicsMaterial
from pxr import PhysxSchema

ASSET_PATH = os.path.dirname(__file__)

@clone
def spawn_with_payload(
    prim_path, 
    cfg, 
    translation, 
    orientation
):
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    import omni.physx.scripts.utils as script_utils
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core import objects
    from pxr import UsdPhysics

    parent_prim = prim_utils.get_prim_at_path(prim_path + "/base")
    path = prim_path + "/payload"
    payload = objects.DynamicCylinder(
        path, radius=0.1, height=0.05, translation=(0., 0., 0.1), mass=1.)

    stage = prim_utils.get_current_stage()
    # joint = script_utils.createJoint(stage, "Prismatic", payload.prim, parent_prim)
    # joint.GetAttribute('physics:axis').Set("X")
    joint = script_utils.createJoint(stage, "Fixed", parent_prim, payload.prim)

    return prim

UNITREE_A1_CFG = copy.deepcopy(UNITREE_A1_CFG)
UNITREE_GO1_CFG = copy.deepcopy(UNITREE_GO1_CFG)
UNITREE_GO2_CFG = copy.deepcopy(UNITREE_GO2_CFG)
