import os
import isaaclab.sim as sim_utils

from isaaclab_assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg

ASSET_PATH = os.path.dirname(__file__)

SIRIUS_WHEEL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/sirius_wheel_mid/sirius_wheel_mid.usd",
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
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*_HAA": 0.,
            "[L,R]F_HFE":  0.4,
            "[L,R]H_HFE": -0.4,
            "[L,R]F_KFE": -1.2,
            "[L,R]H_KFE":  1.2,
            "wheel_.*": 0.
        },
        joint_vel={".*": 0.}
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                ".*_HAA": 40.,
                ".*_HFE": 40.,
                ".*_KFE": 80.,
                "wheel_.*": 40.
            },
            velocity_limit=40.,
            velocity_limit_sim=40.,
            # saturation_effort=100.0,
            stiffness={
                ".*_HAA": 40.,
                ".*_HFE": 40.,
                ".*_KFE": 40.,
                "wheel_.*": 0.
            },
            damping={
                ".*_HAA": 1.,
                ".*_HFE": 1.,
                ".*_HFE": 1.,
                "wheel_.*": 20.
            }
        )
    }
)

