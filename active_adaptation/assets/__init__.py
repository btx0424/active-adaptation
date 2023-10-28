import os
from omni.isaac.orbit.assets.config import ArticulationCfg, UNITREE_A1_CFG, ANYMAL_C_CFG
from .cassie import CASSIE_CFG

from omni.isaac.orbit.sim.utils import bind_physics_material, clone
from omni.isaac.core.materials import PhysicsMaterial
from pxr import PhysxSchema

__all__ = ["UNITREE_A1_CFG", "CASSIE_CFG", "ArticulationCfg"]

ASSET_PATH = os.path.dirname(__file__)

spawn_func = UNITREE_A1_CFG.spawn.func.__wrapped__

@clone
def spawn(prim_path, cfg, translation, orientation):
    prim = spawn_func(prim_path, cfg, translation, orientation)
    material = PhysicsMaterial(
        prim_path=prim_path + "/" + "body_material",
        static_friction=1.,
        dynamic_friction=1.,
        restitution=0.,
    )
    # -- enable patch-friction: yields better results!
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
    physx_material_api.CreateImprovePatchFrictionAttr().Set(True)
    for body_name in ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
        bind_physics_material(prim_path + "/" + body_name, material.prim_path)
    return prim


UNITREE_A1_CFG.spawn.func = spawn