import os
import copy
import torch

import isaaclab.sim as sim_utils

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_rotate_inverse
from isaaclab.sensors import ContactSensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_adaptation.envs.base import EnvBase

from .base import ArticulationCfg


ASSET_PATH = os.path.dirname(__file__)

UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/Go2/go2.usd",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.7,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim={
                ".*_hip_joint": 23.5,
                ".*_thigh_joint": 23.5,
                ".*_calf_joint": 35.5,
            },
            # saturation_effort=35.5,
            velocity_limit_sim=30.0,
            stiffness=25.0,
            damping=0.5,
        ),
    },
    joint_symmetry_mapping = { 
        "FL_hip_joint": [-1, "FR_hip_joint"],
        "FR_hip_joint": [-1, "FL_hip_joint"],
        "RL_hip_joint": [-1, "RR_hip_joint"],
        "RR_hip_joint": [-1, "RL_hip_joint"],
        "FL_thigh_joint": [1, "FR_thigh_joint"],
        "FR_thigh_joint": [1, "FL_thigh_joint"],
        "RL_thigh_joint": [1, "RR_thigh_joint"],
        "RR_thigh_joint": [1, "RL_thigh_joint"],
        "FL_calf_joint": [1, "FR_calf_joint"],
        "FR_calf_joint": [1, "FL_calf_joint"],
        "RL_calf_joint": [1, "RR_calf_joint"],
        "RR_calf_joint": [1, "RL_calf_joint"]
    },
    spatial_symmetry_mapping = {
        "FL_hip": "FR_hip",
        "FR_hip": "FL_hip",
        "RL_hip": "RR_hip",
        "RR_hip": "RL_hip",
        "FL_thigh": "FR_thigh",
        "FR_thigh": "FL_thigh",
        "RL_thigh": "RR_thigh",
        "RR_thigh": "RL_thigh",
        "FL_calf": "FR_calf",
        "FR_calf": "FL_calf",
        "RL_calf": "RR_calf",
        "RR_calf": "RL_calf",
        "FL_foot": "FR_foot",
        "FR_foot": "FL_foot",
        "RL_foot": "RR_foot",
        "RR_foot": "RL_foot",
        "base": "base",
        "Head_upper": "Head_upper",
        "Head_lower": "Head_lower",
    }
)


UNITREE_ALIENGO_CFG = copy.deepcopy(UNITREE_GO2_CFG)
UNITREE_ALIENGO_CFG.spawn.usd_path = f"{ASSET_PATH}/Aliengo/aliengo.usd"
UNITREE_ALIENGO_CFG.init_state.pos = (0.0, 0.0, 0.40)
UNITREE_ALIENGO_CFG.init_state.joint_pos = {
    ".*L_hip_joint": 0.3,
    ".*R_hip_joint": -0.3,
    "F.*_thigh_joint": 1.0,
    "R.*_thigh_joint": 1.1,
    "F.*_calf_joint": -2.0,
    "R.*_calf_joint": -2.1,
}

UNITREE_ALIENGO_CFG.actuators["base_legs"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit={
        ".*_hip_joint": 44.0,
        ".*_thigh_joint": 44.0,
        ".*_calf_joint": 55.0,
    },
    # saturation_effort=60.0,
    velocity_limit=30.0,
    stiffness=60.0,
    damping=2,
    friction=0.0,
)

UNITREE_ALIENGO_A1_CFG = copy.deepcopy(UNITREE_ALIENGO_CFG)
UNITREE_ALIENGO_A1_CFG.init_state.joint_pos = {
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F.*_thigh_joint": 0.6,
    "R.*_thigh_joint": 0.6,
    "F.*_calf_joint": -1.2,
    "R.*_calf_joint": -1.2,
    "arm_joint1": 0.0,
    "arm_joint2": 0.6,
    "arm_joint3": -0.6,
    "arm_joint4": 0.0,
    "arm_joint5": 0.0,
    "arm_joint6": 0.0,
}

UNITREE_ALIENGO_A1_CFG.ee_body_name = "arm_link6"
UNITREE_ALIENGO_A1_CFG.spawn.usd_path = f"{ASSET_PATH}/Aliengo/aliengo_a1.usd"
UNITREE_ALIENGO_A1_CFG.actuators.pop("base_legs")
UNITREE_ALIENGO_A1_CFG.actuators["base_arm"] = ImplicitActuatorCfg(
    joint_names_expr=["arm_joint[1-6]", ".*_(hip|thigh|calf)_joint"],
    effort_limit={
        "arm_joint[1-6]": 200.0,
        ".*_(hip|thigh)_joint": 44.0,
        ".*_(calf)_joint": 55.0,
    },
    velocity_limit={
        "arm_joint[1-6]": 5.0,
        ".*_(hip|thigh)_joint": 30.0,
        ".*_(calf)_joint": 30.0,
    },
    stiffness={
        # "arm_joint[1-3]": 40.0,
        # "arm_joint[4-6]": 30.0,

        "arm_joint1": 40.0,
        "arm_joint2": 47.0,
        "arm_joint3": 42.0,
        "arm_joint[4-6]": 18.0,

        ".*_(hip|thigh|calf)_joint": 60.0,
    },
    damping={
        # "arm_joint[1-3]": 2.0,
        # "arm_joint[4-6]": 1.0,

        "arm_joint1": 1.2,
        "arm_joint2": 1.2,
        "arm_joint3": 1.2,
        "arm_joint[4-6]": 0.7,

        ".*_(hip|thigh|calf)_joint": 2.0,
    },
    friction=0.001,
)
UNITREE_ALIENGO_A1_CFG.actuators["gripper"] = ImplicitActuatorCfg(
    joint_names_expr=["gripper.*"],
    stiffness=2000.0,
    damping=100.0,
    friction=0.001,
)

UNITREE_ALIENGO_A1_FIX_CFG = copy.deepcopy(UNITREE_ALIENGO_A1_CFG)
UNITREE_ALIENGO_A1_FIX_CFG.spawn.articulation_props.fix_root_link = True


UNITREE_B1Z1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/b1/b1_plus_z1.usd",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*L_hip_joint": 0.2,
            ".*R_hip_joint": -0.2,
            "F[L,R]_thigh_joint": 0.6,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.3,
            'arm_joint1': 0.0,
            'arm_joint2': 1.0, # 1.5
            'arm_joint3': -1.8, # -1.5
            'arm_joint4': -0.1, # -0.54
            'arm_joint5': 0.0,
            'arm_joint6': 0.0,
            'jointGripper': 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim=200.0,
            # saturation_effort=35.5,
            velocity_limit_sim=40.0,
            stiffness={
                ".*hip_joint": 100.0,
                ".*thigh_joint": 100.0,
                ".*calf_joint": 100.0,
                "arm_joint.*": 40.0,
            },
            damping={
                ".*hip_joint": 2.0,
                ".*thigh_joint": 2.0,
                ".*calf_joint": 2.0,
                "arm_joint.*": 1.0,
            },
            friction=0.01,
            armature=0.01,
        ),
    },
)