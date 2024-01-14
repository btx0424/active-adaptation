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
    # material = PhysicsMaterial(
    #     prim_path=prim_path + "/" + "body_material",
    #     static_friction=1.,
    #     dynamic_friction=1.,
    #     restitution=0.,
    # )
    # # -- enable patch-friction: yields better results!
    # physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
    # physx_material_api.CreateImprovePatchFrictionAttr().Set(True)
    # for body_name in ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
    #     bind_physics_material(prim_path + "/" + body_name, material.prim_path)
    import omni.physx.scripts.utils as script_utils
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core import objects
    from pxr import UsdPhysics

    parent_prim = prim_utils.get_prim_at_path(prim_path + "/trunk")
    payload_prim = objects.DynamicCuboid(
        prim_path=prim_path + "/payload",
        scale=torch.tensor([0.2, 0.14, 0.14]),
        mass=0.0001,
        translation=torch.tensor([0.0, 0.0, 0.13]),
    ).prim

    stage = prim_utils.get_current_stage()
    script_utils.createJoint(stage, "Fixed", payload_prim, parent_prim)

    return prim

UNITREE_A1_CFG = copy.deepcopy(UNITREE_A1_CFG)
UNITREE_GO1_CFG = copy.deepcopy(UNITREE_GO1_CFG)
UNITREE_GO2_CFG = copy.deepcopy(UNITREE_GO2_CFG)
