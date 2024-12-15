import os
import copy
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import UNITREE_GO2_CFG, UNITREE_A1_CFG, ArticulationCfg
from omni.isaac.lab.actuators import DCMotorCfg, ImplicitActuatorCfg, ImplicitActuator
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import quat_rotate_inverse
from omni.isaac.lab.sensors import ContactSensor
from active_adaptation.envs.base import EnvBase


class Quadruped(Articulation):

    _env: EnvBase

    def _create_buffers(self):
        super()._create_buffers()
        self.feet_ids, self.feet_names = self.find_bodies(".*_foot")
        self.feet_ids = torch.tensor(self.feet_ids, device=self.device)

        # oscillators
        self.phi = torch.zeros(self.num_instances, 4, device=self.device)
        self.phi_dot = torch.zeros(self.num_instances, 4, device=self.device)

        self.decimation = self._env.cfg.decimation

        self.contact_sensor: ContactSensor = self._env.scene.sensors.get(
            "contact_forces", None
        )
        if self.contact_sensor is not None:
            shape = (self.num_instances, len(self.feet_ids))
            self._feet_contact_ids = None
            # self.in_contact = torch.zeros(*shape, dtype=bool, device=self.device)
            # self.impact = torch.zeros(*shape, dtype=bool, device=self.device)
            # self.detach = torch.zeros(*shape, dtype=bool, device=self.device)
            # self.has_impact = torch.zeros(*shape, dtype=bool, device=self.device)
            # self.impact_point_w = torch.zeros(*shape, 3, device=self.device)
            # self.detach_point_w = torch.zeros(*shape, 3, device=self.device)

            self.grf_substep = torch.zeros(
                self.num_instances, self._env.cfg.decimation, 4, device=self.device
            )

    def post_step(self, substep: int):
        if substep < self.decimation:
            if self.contact_sensor.is_initialized:
                if self._feet_contact_ids is None:
                    self._feet_contact_ids = self.contact_sensor.find_bodies(".*_foot")[
                        0
                    ]
                contact_force = self.contact_sensor.data.net_forces_w[
                    :, self._feet_contact_ids
                ]
                self.grf_substep[:, substep] = contact_force.norm(dim=-1)
        else:
            root_quat_w = self.data.root_quat_w
            root_pos_w = self.data.root_pos_w

            self.feet_pos_w = self.data.body_pos_w[:, self.feet_ids]
            self.feet_pos_b = quat_rotate_inverse(
                root_quat_w.unsqueeze(1), self.feet_pos_w - root_pos_w.unsqueeze(1)
            )
            self.feet_lin_vel_w = self.data.body_lin_vel_w[:, self.feet_ids]

        # if self.contact_sensor.is_initialized and self._feet_contact_ids is None:
        #     self._feet_contact_ids = self.contact_sensor.find_bodies(".*_foot")[0]

        #     in_contact = (contact_force.norm(dim=-1) > 0.01).any(dim=1)
        #     self.impact = (~self.in_contact) & in_contact
        #     self.detach = self.in_contact & (~in_contact)
        #     self.in_contact = in_contact
        #     self.has_impact.logical_or_(self.impact)
        #     self.impact_point_w[self.impact] = self.feet_pos_w[self.impact]
        #     self.detach_point_w[self.detach] = self.feet_pos_w[self.detach]


class QuadrupedManipulator(Articulation):
    def _create_buffers(self):
        super()._create_buffers()

        self.ee_body_id = self.find_bodies(self.cfg.ee_body_name)[0][0]
        self.ee_pos_w = self.data.body_pos_w[:, self.ee_body_id].clone()
        self.ee_pos_b = torch.zeros_like(self.ee_pos_w)
        self._ee_pos_w_buffer = torch.zeros(
            self.num_instances, 4, 3, device=self.device
        )
        self.ee_lin_vel_w = torch.zeros(self.num_instances, 3, device=self.device)

    def update(self, dt: float):
        super().update(dt)
        self.ee_pos_w[:] = self.data.body_pos_w[:, self.ee_body_id]
        self.ee_pos_b = quat_rotate_inverse(
            self.data.root_quat_w, self.ee_pos_w - self.data.root_pos_w
        )
        self._ee_pos_w_buffer = self._ee_pos_w_buffer.roll(1, dims=1)
        self._ee_pos_w_buffer[:, 0] = self.ee_pos_w
        self.ee_lin_vel_w[:] = torch.mean(
            -self._ee_pos_w_buffer.diff(dim=1) / dt, dim=1
        )


ASSET_PATH = os.path.dirname(__file__)

UNITREE_GO2_CFG = ArticulationCfg(
    class_type=Quadruped,
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
            enabled_self_collisions=False,
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
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={
                ".*_hip_joint": 23.5,
                ".*_thigh_joint": 23.5,
                ".*_calf_joint": 35.5,
            },
            saturation_effort=35.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
        ),
    },
)

UNITREE_GO2M_CFG = copy.deepcopy(UNITREE_GO2_CFG)
UNITREE_GO2M_CFG.spawn.usd_path = f"{ASSET_PATH}/go2m.usd"
UNITREE_GO2M_CFG.actuators["arm"] = DCMotorCfg(
    joint_names_expr=["joint.*"],
    effort_limit=200.0,
    saturation_effort=200.0,
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
    effort_limit=200.0,
    saturation_effort=200.0,
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
    effort_limit=200.0,
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
    effort_limit=200.0,
    saturation_effort=200.0,
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
UNITREE_ALIENGO_CFG.init_state.pos = (0.0, 0.0, 0.35)
UNITREE_ALIENGO_CFG.init_state.joint_pos = {
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    ".*thigh_joint": 0.8,
    ".*calf_joint": -1.5,
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
CYBERDOG_CFG.actuators["base_legs"].stiffness = 20.0
CYBERDOG_CFG.actuators["base_legs"].damping = 0.5
CYBERDOG_CFG.actuators["base_legs"].effort_limit = 12.0
CYBERDOG_CFG.actuators["base_legs"].saturation_effort = 12.0
CYBERDOG_CFG.actuators["base_legs"].friction = 0.02
CYBERDOG_CFG.init_state.pos = (0.0, 0.0, 0.33)
CYBERDOG_CFG.init_state.joint_pos = {
    ".*_hip_joint": 0.0,
    ".*thigh_joint": 0.78,
    ".*calf_joint": -1.22,
}
CYBERDOG_CFG.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
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
    saturation_effort=200,
)
