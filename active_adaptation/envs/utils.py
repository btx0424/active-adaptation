import torch

import omni.isaac.core.utils.prims as prim_utils
from pxr import PhysxSchema

def attach_payload(parent_path):
    import omni.physx.scripts.utils as script_utils
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core import objects
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.18, 0.16, 0.12]),
        mass=0.0001,
        translation=torch.tensor([0.0, 0.0, 0.1]),
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.0)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.0)

def add_force_sensor(prim_path):
    prim = prim_utils.get_prim_at_path(prim_path)
    api = PhysxSchema.PhysxArticulationForceSensorAPI.Apply(prim)
    return api

