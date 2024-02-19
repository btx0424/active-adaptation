import os
import copy
import torch

from omni.isaac.orbit_assets import (
    ArticulationCfg,
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG,
    UNITREE_GO2_CFG,
    ANYMAL_C_CFG,
    cassie,
    ISAAC_ORBIT_NUCLEUS_DIR
)

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.sim.utils import bind_physics_material, clone
from omni.isaac.orbit.sim.spawners.from_files.from_files import _spawn_from_usd_file
from omni.isaac.orbit.actuators import ImplicitActuatorCfg, DCMotorCfg

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
UNITREE_GO2_CFG.init_state.pos = (0., 0., 0.35)
CASSIE_CFG = copy.deepcopy(cassie.CASSIE_CFG)

UNITREE_GO1M_CFG = copy.deepcopy(UNITREE_A1_CFG)
UNITREE_GO1M_CFG.spawn.usd_path = f"{ASSET_PATH}/widowGo1.usd"
UNITREE_GO1M_CFG.actuators["arm"] = DCMotorCfg(
    joint_names_expr=[".*widow_(waist|shoulder|elbow)"],
    stiffness=80.0,
    velocity_limit=2.0,
    damping=4.0,
    saturation_effort=200
)

H1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=30.0,
            stiffness=100.0,
            damping=3.0,
            friction=0.0,
        ),
    },
)

ROBOTS = {
    "a1": UNITREE_A1_CFG,
    "go1": UNITREE_GO1_CFG,
    "go1m": UNITREE_GO1M_CFG,
    "go2": UNITREE_GO2_CFG,
    "cassie": CASSIE_CFG,
    "h1": H1_CFG
}