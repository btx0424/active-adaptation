from omni.isaac.orbit.assets import ArticulationCfg
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import IdealPDActuatorCfg
import os

ASSET_PATH = os.path.dirname(os.path.realpath(__file__))

CASSIE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/cassie.usd",
        activate_contact_sensors=False,
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02, rest_offset=0.
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.),
        joint_pos={
            'hip_abduction_left': 0.1,
            'hip_rotation_left': 0.,
            'hip_flexion_left': 1.,
            'thigh_joint_left': -1.8,
            'ankle_joint_left': 1.57,
            'toe_joint_left': -1.57,

            'hip_abduction_right': -0.1,
            'hip_rotation_right': 0.,
            'hip_flexion_right': 1.,
            'thigh_joint_right': -1.8,
            'ankle_joint_right': 1.57,
            'toe_joint_right': -1.57
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hip_abduction": IdealPDActuatorCfg(
            joint_names_expr=["hip_abduction.*"],
            stiffness=100.0,
            damping=3.0,
            effort_limit=100.,
            velocity_limit=21.0,
        ),
        "hip_rotation": IdealPDActuatorCfg(
            joint_names_expr=["hip_rotation.*"],
            stiffness=100.0,
            damping=3.0,
            effort_limit=100.,
            velocity_limit=21.0,
        ),
        "hip_flexion": IdealPDActuatorCfg(
            joint_names_expr=["hip_flexion.*"],
            stiffness=200.0,
            damping=6.0,
            effort_limit=100.,
            velocity_limit=21.0,
        ),
        "thigh_joint": IdealPDActuatorCfg(
            joint_names_expr=["thigh_joint.*"],
            stiffness=200.0,
            damping=6.0,
            effort_limit=100.,
            velocity_limit=21.0,
        ),
        "ankle_joint": IdealPDActuatorCfg(
            joint_names_expr=["ankle_joint.*"],
            stiffness=200.0,
            damping=6.0,
            effort_limit=100.,
            velocity_limit=21.0,
        ),
        "toe_joint": IdealPDActuatorCfg(
            joint_names_expr=["toe_joint.*"],
            stiffness=40.0,
            damping=1.0,
            effort_limit=100.,
            velocity_limit=21.0,
        ),
    },
)