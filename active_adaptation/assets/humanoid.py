import os
import copy
import isaaclab.sim as sim_utils
import torch
from isaaclab_assets import H1_CFG
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import Articulation
from active_adaptation.envs.actuator import HybridActuatorCfg
import active_adaptation.utils.symmetry as symmetry_utils

from .base import ArticulationCfg


ASSET_PATH = os.path.dirname(__file__)

H1_CFG = copy.deepcopy(H1_CFG)
H1_CFG.spawn.usd_path = f"{ASSET_PATH}/H1/h1_minimal.usd"
H1_CFG.actuators = {
    "base_legs": DCMotorCfg(
        joint_names_expr=[".*"],
        effort_limit=300.0,
        saturation_effort=300.0,
        velocity_limit=30.0,
        stiffness={
            ".*hip.*": 200,
            ".*knee.*": 300,
            ".*ankle.*": 40,
            "torso": 300,
            ".*shoulder.*": 100,
            ".*elbow.*": 100
        },
        damping={
            ".*hip.*": 5,
            ".*knee.*": 6,
            ".*ankle.*": 2,
            "torso": 6,
            ".*shoulder.*": 2,
            ".*elbow.*": 2
        },
        friction=0.0,
    )
}

CY1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/ORCA/orca_stable_mesh.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.02,
            angular_damping=0.02,
            max_linear_velocity=50.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=0.5,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=2
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.02,
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            "waist_yaw_joint": 0.0,
            ".*arm_joint[1,3,5,6]": 0.0,
            ".*leg_joint[2,3,5,6]": 0.0,
            "[l,r]arm_joint2": 0.1,
            "[l,r]arm_joint4": 0.1,
            "[l,r]leg_joint1": -0.1,
            "[l,r]leg_joint4": -0.1,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": HybridActuatorCfg(
            # implicit_ratio=0.5,
            # homogeneous_ratio=0.5,
            joint_names_expr=[".*"],
            effort_limit={
                "waist_yaw_joint": 36,
                "[l,r]arm_joint1": 94.,
                "[l,r]arm_joint[2-5]": 36,
                "[l,r]leg_joint[1,4]": 150,
                "[l,r]leg_joint[2,3]": 94,
                "[l,r]leg_joint[5,6]": 36
            },
            # saturation_effort=100.0,
            velocity_limit=30.0,
            stiffness={
                "waist_yaw_joint": 75.,
                "[l,r]arm_joint1": 75.,
                "[l,r]arm_joint2": 50.,
                "[l,r]arm_joint3": 30.,
                "[l,r]arm_joint4": 30.,
                "[l,r]arm_joint5": 15.,
                "[l,r]arm_joint6": 15.,
                "[l,r]leg_joint1": 75.,
                "[l,r]leg_joint2": 50.,
                "[l,r]leg_joint3": 50.,
                "[l,r]leg_joint4": 75.,
                "[l,r]leg_joint5": 15.,
                "[l,r]leg_joint6": 15.,
            },
            damping={
                "waist_yaw_joint": 3.,
                "[l,r]arm_joint1": 6.,
                "[l,r]arm_joint2": 3.,
                "[l,r]arm_joint3": 1.,
                "[l,r]arm_joint4": 1.,
                "[l,r]arm_joint5": 1.,
                "[l,r]arm_joint6": 1.,
                "[l,r]leg_joint1": 6., # 6.
                "[l,r]leg_joint2": 3.,
                "[l,r]leg_joint3": 3.,
                "[l,r]leg_joint4": 6., # 6.
                "[l,r]leg_joint5": 1.,
                "[l,r]leg_joint6": 1.,
            },
            armature=0.01,
            friction=0.01,
        ),
    },
)


G1_27DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/G1/g1_27dof_fakehand/g1_27dof_fakehand.usd",
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
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.28,
            ".*_knee_joint": 0.5,
            ".*_ankle_pitch_joint": -0.23,
            # ".*_elbow_pitch_joint": 0.87,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            ".*wrist_roll_joint": 0.0,
            ".*wrist_pitch_joint": 0.0,
            ".*wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                ".*_hip.*": 88.0,
                ".*_knee.*": 139.0,
                ".*_ankle.*": 50,
                ".*_shoulder.*": 25,
                ".*_elbow.*": 25,
                ".*_wrist.*": 25,
                "waist_yaw_joint": 88,
            },
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 150.0, # unitree_ros
                # "waist_roll_joint": 150.0, # unitree_ros
                ".*ankle_pitch_joint": 20.0,
                ".*ankle_roll_joint": 20.0,
                ".*_shoulder_.*": 40.0,
                ".*_elbow_joint": 40.0,
                ".*wrist_roll_joint": 20.0,
                ".*wrist_pitch_joint": 20.0,
                ".*wrist_yaw_joint": 20.0,
            },
            damping={
                "waist_yaw_joint": 5.0, # unitree_ros
                # "waist_roll_joint": 5.0, # unitree_ros
                ".*_shoulder_.*": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_hip_yaw_joint": 6.0,
                ".*_hip_roll_joint": 6.0,
                ".*_hip_pitch_joint": 6.0,
                ".*_knee_joint": 6.0,
                ".*ankle_pitch_joint": 1.0,
                ".*ankle_roll_joint": 1.0,
                ".*wrist_roll_joint": 1.0,
                ".*wrist_pitch_joint": 1.0,
                ".*wrist_yaw_joint": 1.0,
            },
            armature=0.01,
            friction=0.01,
        ),
    },
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
        "left_wrist_roll_joint": (1, "right_wrist_roll_joint"),
        "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
        "left_wrist_yaw_joint": (1, "right_wrist_yaw_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "pelvis": "pelvis",
        "torso_link": "torso_link",
        "torso_com_link": "torso_com_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_roll_link": "waist_roll_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
        "left_wrist_roll_link": "right_wrist_roll_link",
        "left_wrist_pitch_link": "right_wrist_pitch_link",
        "left_wrist_yaw_link": "right_wrist_yaw_link",
        "left_rubber_hand": "right_rubber_hand",
    })
)

G1_23DOF_CFG = ArticulationCfg( # no wrist pitch and yaw
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ASSET_PATH}/G1/g1_23dof_fakehand/g1_23dof_fakehand.usd",
        usd_path=f"{ASSET_PATH}/G1/g1_23dof/g1_23dof.usd",
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
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.28,
            ".*_knee_joint": 0.5,
            ".*_ankle_pitch_joint": -0.23,
            # ".*_elbow_pitch_joint": 0.87,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            ".*wrist_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 150.0, # unitree_ros
                # "waist_roll_joint": 150.0, # unitree_ros
                ".*ankle_pitch_joint": 20.0,
                ".*ankle_roll_joint": 20.0,
                ".*_shoulder_.*": 40.0,
                ".*_elbow_joint": 40.0,
                ".*wrist_roll_joint": 20.0,
            },
            damping={
                "waist_yaw_joint": 5.0, # unitree_ros
                # "waist_roll_joint": 5.0, # unitree_ros
                ".*_shoulder_.*": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_hip_yaw_joint": 6.0,
                ".*_hip_roll_joint": 6.0,
                ".*_hip_pitch_joint": 6.0,
                ".*_knee_joint": 6.0,
                ".*ankle_pitch_joint": 1.0,
                ".*ankle_roll_joint": 1.0,
                ".*wrist_roll_joint": 1.0,
            },
            armature=0.01,
            friction=0.01,
        ),
    },
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
        "left_wrist_roll_joint": (1, "right_wrist_roll_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "pelvis": "pelvis",
        "torso_link": "torso_link",
        "torso_com_link": "torso_com_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_roll_link": "waist_roll_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
        "left_wrist_roll_link": "right_wrist_roll_link",
        "left_rubber_hand": "right_rubber_hand",
    })
)


G1_WAIST_UNLOCKED_CFG = ArticulationCfg( # no wrist pitch and yaw
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/G1/g1_waist_unlocked/g1_waist_unlocked.usd",
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
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.28,
            ".*_knee_joint": 0.5,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 150.0, # unitree_ros
                "waist_roll_joint": 150.0, # unitree_ros
                "waist_pitch_joint": 150.0, # unitree_ros
                ".*ankle_pitch_joint": 20.0,
                ".*ankle_roll_joint": 20.0,
                ".*_shoulder_.*": 40.0,
                ".*_elbow_joint": 40.0,
            },
            damping={
                "waist_yaw_joint": 5.0, # unitree_ros
                "waist_roll_joint": 5.0, # unitree_ros
                "waist_pitch_joint": 5.0, # unitree_ros
                ".*_shoulder_.*": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_hip_yaw_joint": 6.0,
                ".*_hip_roll_joint": 6.0,
                ".*_hip_pitch_joint": 6.0,
                ".*_knee_joint": 6.0,
                ".*ankle_pitch_joint": 1.0,
                ".*ankle_roll_joint": 1.0,
            },
            armature=0.01,
            friction=0.01,
        ),
    },
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "waist_roll_joint": (-1, "waist_roll_joint"),
        "waist_pitch_joint": (1, "waist_pitch_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "pelvis": "pelvis",
        "torso_link": "torso_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_roll_link": "waist_roll_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
    })
)


H2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/h1_2_handless/h2_handless.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=0.5,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.15,  # -16 degrees
            ".*_knee_joint": 0.5,  # 45 degrees
            ".*_ankle_pitch_joint": -0.35,  # -30 degrees
            ".*_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.28,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.52,
            ".*_wrist_.*_joint": 0.0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                ".*_hip_.*": 300.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.,
                ".*_ankle_.*": 100.0,
                ".*_shoulder_.*": 300.,
                ".*_elbow_joint": 300.0,
            },
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 300.0,
                "torso_joint": 150.0,
                ".*_shoulder_.*": 40.,
                ".*_elbow_joint": 40.0,
                ".*_ankle_.*": 20.0,
                ".*_wrist_.*_joint": 20.,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
                ".*_ankle_.*": 2.0,
                ".*_shoulder_.*": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_.*_joint": 2.0,
            },
            armature=0.01,
            friction=0.01,
        ),
    },
)