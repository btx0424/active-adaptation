import os
import copy
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import UNITREE_GO2_CFG, UNITREE_A1_CFG, ArticulationCfg
from omni.isaac.lab.actuators import DCMotorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import quat_rotate_inverse


class Quadruped(Articulation):
    pass


class QuadrupedManipulator(Articulation):
    def _create_buffers(self):
        super()._create_buffers()

        self.ee_body_id = self.find_bodies(self.cfg.ee_body_name)[0][0]
        self.ee_pos_w = self.data.body_pos_w[:, self.ee_body_id]
        self.ee_pos_b = torch.zeros_like(self.ee_pos_w)
        self._ee_pos_w_buffer = torch.zeros(self.num_instances, 4, 3, device=self.device)
        self.ee_lin_vel_w = torch.zeros(self.num_instances, 3, device=self.device)

    def update(self, dt: float):
        super().update(dt)
        self.ee_pos_b = quat_rotate_inverse(
            self.data.root_quat_w,
            self.ee_pos_w - self.data.root_pos_w
        )
        self._ee_pos_w_buffer = self._ee_pos_w_buffer.roll(1, dims=1)
        self._ee_pos_w_buffer[:, 0] = self.ee_pos_w
        self.ee_lin_vel_w[:] = torch.mean(self._ee_pos_w_buffer.diff(dim=1) / dt, dim=1)


ASSET_PATH = os.path.dirname(__file__)

UNITREE_GO2_CFG = copy.deepcopy(UNITREE_GO2_CFG)
UNITREE_GO2_CFG.spawn.usd_path = f"{ASSET_PATH}/Go2/go2.usd"
UNITREE_GO2_CFG.init_state.pos = (0., 0., 0.35)
UNITREE_GO2_CFG.init_state.joint_pos["F[L,R]_thigh_joint"] = 0.78
UNITREE_GO2_CFG.init_state.joint_pos["R[L,R]_thigh_joint"] = 0.78
UNITREE_GO2_CFG.actuators["base_legs"].effort_limit = {
    "(?!.*_calf_joint).*": 23.5,
    ".*_calf_joint": 35.5,
}
UNITREE_GO2_CFG.actuators["base_legs"].saturation_effort = 35.5

UNITREE_GO2M_CFG = copy.deepcopy(UNITREE_GO2_CFG)
UNITREE_GO2M_CFG.spawn.usd_path = f"{ASSET_PATH}/go2m.usd"
UNITREE_GO2M_CFG.actuators["arm"] = DCMotorCfg(
    joint_names_expr=["joint.*"],
    effort_limit=200.,
    saturation_effort=200.,
    velocity_limit=5.0,
    # stiffness=30.0,
    # damping=1.0,
    # friction=0.0,
    stiffness={
        "joint[1-3]": 20.0,
        "joint[4-6]": 15.0,
    },
    damping={
        "joint[1-3]": 1.0,
        "joint[4-6]": 0.5,
    },
    friction=0.001,
)
UNITREE_GO2M_CFG.init_state.joint_pos["joint[1,2]"] = 0.3

UNITREE_GO2ABP_CFG = copy.deepcopy(UNITREE_GO2_CFG)
UNITREE_GO2ABP_CFG.spawn.usd_path = f"{ASSET_PATH}/Go2/go2abpg.usd"
UNITREE_GO2ABP_CFG.actuators["arm"] = DCMotorCfg(
    joint_names_expr=["(joint.*)"],
    effort_limit=200.,
    saturation_effort=200.,
    velocity_limit=5.0,
    stiffness={
        "joint[1-3]": 20.0,
        "joint[4-6]": 15.0,
    },
    damping={
        "joint[1-3]": 1.0,
        "joint[4-6]": 0.5,
    },
    friction=0.001,
)
UNITREE_GO2ABP_CFG.actuators["gripper"] = ImplicitActuatorCfg(
    joint_names_expr=["end(left|right)"],
    effort_limit=200.,
    stiffness=100.0,
    damping=0.5,
    friction=0.001,
)
UNITREE_GO2ABP_CFG.init_state.joint_pos["joint2"] = -0.3
UNITREE_GO2ABP_CFG.init_state.joint_pos["joint3"] = 0.3
# UNITREE_GO2ABP_CFG.init_state.joint_pos["endleft"] = 0.02
# UNITREE_GO2ABP_CFG.init_state.joint_pos["endright"] = -0.02


UNITREE_GO2ARX_CFG = copy.deepcopy(UNITREE_GO2_CFG)
UNITREE_GO2ARX_CFG.spawn.usd_path = f"{ASSET_PATH}/Go2/go2arxg.usd"
UNITREE_GO2ARX_CFG.actuators["arm"] = DCMotorCfg(
    joint_names_expr=["(joint.*)"],
    effort_limit=200.,
    saturation_effort=200.,
    velocity_limit=5.0,
    stiffness={
        "joint[1-3]": 20.0,
        "joint[4-6]": 15.0,
    },
    damping={
        "joint[1-3]": 1.0,
        "joint[4-6]": 0.5,
    },
    friction=0.001,
)
# UNITREE_GO2ABP_CFG.init_state.joint_pos["joint2"] = -0.3
# UNITREE_GO2ABP_CFG.init_state.joint_pos["joint3"] = 0.3
UNITREE_ALIENGO_CFG = copy.deepcopy(UNITREE_GO2_CFG)
UNITREE_ALIENGO_CFG.spawn.usd_path = f"{ASSET_PATH}/Aliengo/aliengo.usd"
UNITREE_ALIENGO_CFG.init_state.pos = (0., 0., 0.40)
UNITREE_ALIENGO_CFG.init_state.joint_pos = {
    ".*hip_joint": 0,
    ".*thigh_joint": 0.8,
    ".*calf_joint": -1.5,
}
UNITREE_ALIENGO_CFG.actuators["base_legs"] = DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit={
        ".*_hip_joint": 44.0,
        ".*_thigh_joint": 44.0,
        ".*_calf_joint": 55.0,
    },
    saturation_effort=60.,
    velocity_limit=30.0,
    stiffness=40.0,
    damping=2,
    friction=0.0,
)

UNITREE_ALIENGO_A1_CFG = copy.deepcopy(UNITREE_ALIENGO_CFG)
UNITREE_ALIENGO_A1_CFG.class_type = QuadrupedManipulator
UNITREE_ALIENGO_A1_CFG.ee_body_name = "arm_link6"
UNITREE_ALIENGO_A1_CFG.spawn.usd_path = f"{ASSET_PATH}/Aliengo/aliengo_a1.usd"
UNITREE_ALIENGO_A1_CFG.actuators["arm"] = ImplicitActuatorCfg(
    joint_names_expr=["arm_joint[1-6]"],
    effort_limit=200.0,
    velocity_limit=5.0,
    stiffness={
        "arm_joint[1-3]": 40.0,
        "arm_joint[4-6]": 30.0,
    },
    damping={
        "arm_joint[1-3]": 2.0,
        "arm_joint[4-6]": 1.0,
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

CYBERDOG_CFG = copy.deepcopy(UNITREE_A1_CFG)
CYBERDOG_CFG.spawn.usd_path = f"{ASSET_PATH}/cyberdog2_v3.usd"
CYBERDOG_CFG.actuators["base_legs"].stiffness = 20.
CYBERDOG_CFG.actuators["base_legs"].damping = 0.5
CYBERDOG_CFG.actuators["base_legs"].effort_limit = 12.
CYBERDOG_CFG.actuators["base_legs"].saturation_effort = 12.
CYBERDOG_CFG.actuators["base_legs"].friction = 0.02
CYBERDOG_CFG.init_state.pos = (0., 0., 0.33)
CYBERDOG_CFG.init_state.joint_pos = {
    ".*_hip_joint": 0.0,
    ".*thigh_joint": 0.78,
    ".*calf_joint": -1.22,
}
CYBERDOG_CFG.spawn.collision_props=sim_utils.CollisionPropertiesCfg(
    contact_offset=0.05,
    rest_offset=0.0,
)

UNITREE_GO1M_CFG: ArticulationCfg = UNITREE_A1_CFG.replace()
UNITREE_GO1M_CFG.spawn.usd_path = f"{ASSET_PATH}/widowGo1.usd"
UNITREE_GO1M_CFG.actuators["arm"] = DCMotorCfg(
    joint_names_expr=[".*widow_(waist|shoulder|elbow)"],
    stiffness=30.0,
    velocity_limit=2.0,
    damping=1.0,
    saturation_effort=200
)