import torch

from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg, DCMotorCfg
import omni.isaac.lab.sim as sim_utils


class DoorArticulation(Articulation):

        def _initialize_impl(self):
            super()._initialize_impl()
        
        def _create_buffers(self):
            super()._create_buffers()
            
            self.handle_body_id = self.find_bodies("Handle")[0][0]
            self.handle_joint_id = self.find_joints("handle_joint")[0][0]
            self.door_joint_id = self.find_joints("door_joint")[0][0]

            self.type = torch.zeros(self.num_instances, dtype=torch.int, device=self.device)
            self.locked = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
            self.unlock_pos = torch.zeros(self.num_instances, device=self.device)
            self.stiffness = torch.zeros(self.num_instances, device=self.device)
            self.damping = torch.zeros(self.num_instances, device=self.device)
            
            self.default_jpos = torch.zeros_like(self.data.joint_pos[0])
            self.default_jvel = torch.zeros_like(self.data.joint_vel[0])

        def reset(self, env_ids: torch.Tensor):
            super().reset(env_ids)
            self.unlock_pos[env_ids] = (torch.pi / 6)
            self.stiffness[env_ids] = 100.
            self.write_joint_state_to_sim(self.default_jpos, self.default_jvel, env_ids=env_ids)

        def update(self, dt: float):
            super().update(dt)
            self.locked[:] = self.data.joint_pos[:, self.handle_joint_id] > self.unlock_pos
            self.actuators["door_joints"].stiffness[:, self.door_joint_id] = torch.where(self.locked, self.stiffness, 0.)


DOOR_CFG = ArticulationCfg(
    class_type=DoorArticulation,
    prim_path="{ENV_REGEX_NS}/Door",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/btx0424/isaac_lab/active-adaptation/active_adaptation/assets/Doors/DoorC_Flattened.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            enabled_self_collisions=False
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.0),
    ),
    actuators={
        "door_joints": DCMotorCfg(
            joint_names_expr=".*",
            stiffness=0.5, 
            damping=0.02,
            friction=0.01,
            saturation_effort=20,
            velocity_limit=5,
        )
    },
)