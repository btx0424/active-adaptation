import os
import copy
import isaaclab.sim as sim_utils
import torch
from isaaclab_assets import ArticulationCfg, H1_CFG
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import Articulation
from active_adaptation.envs.actuator import HybridActuatorCfg


class Humanoid(Articulation):
    def _create_buffers(self):
        super()._create_buffers()

        # oscillators
        self.phi = torch.zeros(self.num_instances, 2, device=self.device)
        self.phi_dot = torch.zeros(self.num_instances, 2, device=self.device)

        if hasattr(self.cfg, "hand_body_name"):
            self.hand_body_ids = self.find_bodies(self.cfg.hand_body_name)[0]
            self.hand_pos_w = self.data.body_pos_w[:, self.hand_body_ids]

        if hasattr(self.cfg, "foot_body_name"):
            self.foot_body_ids = self.find_bodies(self.cfg.foot_body_name)[0]
            self.foot_pos_w = self.data.body_pos_w[:, self.foot_body_ids]
    
    def update(self, dt: float):
        super().update(dt)
    
        if hasattr(self.cfg, "hand_body_name"):
            self.hand_pos_w[:] = self.data.body_pos_w[:, self.hand_body_ids]
        if hasattr(self.cfg, "foot_body_name"):
            self.foot_pos_w[:] = self.data.body_pos_w[:, self.foot_body_ids]


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
    class_type=Humanoid,
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

