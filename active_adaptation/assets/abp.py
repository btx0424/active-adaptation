import os
import copy

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import ArticulationCfg
from omni.isaac.lab.actuators import DCMotorCfg, ImplicitActuatorCfg

ASSET_PATH = os.path.dirname(__file__)

ABP_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Arm",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/abpg.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "joint.*": 0.0,
        },
        joint_vel={
            "joint.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["(joint.*)"],
            effort_limit=200.0,
            velocity_limit=5.0,
            stiffness={
                "joint[1-3]": 40.0,
                "joint[4-6]": 30.0,
            },
            damping={
                "joint[1-3]": 2.0,
                "joint[4-6]": 1.0,
            },
            # stiffness=0.0,
            # damping=20.0,
            friction=0.001,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["end(left|right)"],
            stiffness=2000.0,
            damping=100.0,
            friction=0.001,
        ),
    },
)
