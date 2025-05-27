import torch
import os
import copy

import isaaclab.sim as sim_utils
from isaaclab_assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_rotate_inverse

ASSET_PATH = os.path.dirname(__file__)

class Manipulator(Articulation):
    def _create_buffers(self):
        super()._create_buffers()

        self.ee_body_id = self.find_bodies(self.cfg.ee_body_name)[0][0]
        self.ee_pos_w = self.data.body_pos_w[:, self.ee_body_id].clone()
        self.ee_pos_b = torch.zeros_like(self.ee_pos_w)
        self._ee_pos_w_buffer = torch.zeros(self.num_instances, 4, 3, device=self.device)
        self.ee_lin_vel_w = torch.zeros(self.num_instances, 3, device=self.device)

    def update(self, dt: float):
        super().update(dt)
        self.ee_pos_w[:] = self.data.body_pos_w[:, self.ee_body_id]
        self.ee_pos_b = quat_rotate_inverse(
            self.data.root_quat_w,
            self.ee_pos_w - self.data.root_pos_w
        )
        self._ee_pos_w_buffer = self._ee_pos_w_buffer.roll(1, dims=1)
        self._ee_pos_w_buffer[:, 0] = self.ee_pos_w
        self.ee_lin_vel_w[:] = torch.mean(-self._ee_pos_w_buffer.diff(dim=1) / dt, dim=1)

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

A1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Arm",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/Aliengo/a1.usd",
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
            "arm_joint1": 0.0,
            "arm_joint2": 1.0,
            "arm_joint3": -1.0,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
        },
        joint_vel={
            "arm_joint.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[1-6]"],
            effort_limit=200.0,
            velocity_limit=5.0,
            stiffness={
                "arm_joint[1-3]": 80.0,
                "arm_joint[4-6]": 30.0,
            },
            damping={
                "arm_joint[1-3]": 2.0,
                "arm_joint[4-6]": 1.0,
            },
            friction=0.001,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper.*"],
            stiffness=2000.0,
            damping=100.0,
            friction=0.001,
        ),
    },
)

A1_CFG.class_type = Manipulator
A1_CFG.ee_body_name = "arm_link6"
