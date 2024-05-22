import os

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit_assets import ArticulationCfg
from omni.isaac.orbit.actuators import DCMotorCfg

ASSET_PATH = os.path.dirname(__file__)

H1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ASSET_PATH}/h1_isaacgym.usd",
        usd_path=f"{ASSET_PATH}/h1_isaacgym_white.usd",
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
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            '.*knee_joint': 0.8,
            '.*ankle_joint': -0.4,
            '.*hip_pitch_joint': -0.4,
            'left_hip_yaw_joint' : 0. ,   
            'left_hip_roll_joint' : 0,               
            'right_hip_yaw_joint' : 0., 
            'right_hip_roll_joint' : 0, 
            'torso_joint' : 0., 
            'left_shoulder_pitch_joint' : 0., 
            'left_shoulder_roll_joint' : 0, 
            'left_shoulder_yaw_joint' : 0.,
            'left_elbow_joint'  : 0.0,
            'right_shoulder_pitch_joint' : 0.,
            'right_shoulder_roll_joint' : 0.0,
            'right_shoulder_yaw_joint' : 0.,
            'right_elbow_joint' : 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=300.0,
            saturation_effort=300.0,
            velocity_limit=30.0,
            stiffness={
                ".*hip.*": 200,
                ".*knee.*": 300,
                ".*ankle.*": 40,
                "torso_joint": 300,
                ".*shoulder.*": 100,
                ".*elbow.*": 100
            },
            damping={
                ".*hip.*": 5,
                ".*knee.*": 6,
                ".*ankle.*": 2,
                "torso_joint": 6,
                ".*shoulder.*": 2,
                ".*elbow.*": 2
            },
            friction=0.0,
        ),
    },
)

CY1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/cy1.usd",
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
        pos=(0.0, 0.0, 0.93),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit={
                "waist_yaw_joint": 36,
                "[l,r]arm_joint1": 94.,
                "[l,r]arm_joint[2-5]": 36,
                "[l,r]leg_joint[1,4]": 150,
                "[l,r]leg_joint[2,3]": 94,
                "[l,r]leg_joint[5,6]": 36
            },
            saturation_effort=100.0,
            velocity_limit=30.0,
            stiffness={
                "waist_yaw_joint": 75.,
                "[l,r]arm_joint1": 75.,
                "[l,r]arm_joint2": 50.,
                "[l,r]arm_joint3": 30.,
                "[l,r]arm_joint4": 30.,
                "[l,r]arm_joint5": 15.,
                "[l,r]leg_joint1": 75.,
                "[l,r]leg_joint2": 50.,
                "[l,r]leg_joint3": 50.,
                "[l,r]leg_joint4": 75.,
                "[l,r]leg_joint5": 50.,
                "[l,r]leg_joint6": 50.,
            },
            damping={
                "waist_yaw_joint": 6.,
                "[l,r]arm_joint1": 6.,
                "[l,r]arm_joint2": 3.,
                "[l,r]arm_joint3": 0.5,
                "[l,r]arm_joint4": 1.,
                "[l,r]arm_joint5": 1.,
                "[l,r]leg_joint1": 6.,
                "[l,r]leg_joint2": 3.,
                "[l,r]leg_joint3": 3.,
                "[l,r]leg_joint4": 6.,
                "[l,r]leg_joint5": 3.,
                "[l,r]leg_joint6": 3.,
            },
            friction=0.0,
        ),
    },
)