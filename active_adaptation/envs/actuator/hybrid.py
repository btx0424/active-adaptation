import torch

from isaaclab.assets import Articulation
from isaaclab.actuators import ImplicitActuatorCfg, ImplicitActuator
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class HybridActuator(ImplicitActuator):
    
    articulation: Articulation
    cfg: "HybridActuatorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.implicit = torch.zeros(self._num_envs, self.num_joints, dtype=torch.bool, device=self._device)
        self.default_stiffness = self.stiffness[0].clone()
        self.default_damping = self.damping[0].clone()
    
    def reset(self, env_ids: torch.Tensor):
        implicit = torch.where(
            torch.rand(len(env_ids), 1, device=self._device) < self.cfg.homogeneous_ratio,
            torch.rand(len(env_ids), 1, device=self._device),
            torch.rand(len(env_ids), self.num_joints, device=self._device)
        )  < self.cfg.implicit_ratio

        stiffness = self.default_stiffness.expand(len(env_ids), -1)
        damping = self.default_damping.expand(len(env_ids), -1)
        self.implicit[env_ids] = implicit

        # # these values are kept non-zero for computing the applied torque
        # self.stiffness[env_ids] = stiffness * (~implicit.unsqueeze(1))
        # self.damping[env_ids] = damping * (~implicit.unsqueeze(1))
        self.articulation.write_joint_stiffness_to_sim(stiffness * implicit, self.joint_indices, env_ids)
        self.articulation.write_joint_damping_to_sim(damping * implicit, self.joint_indices, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort * (~self.implicit)
        return control_action


@configclass
class HybridActuatorCfg(ImplicitActuatorCfg):
    implicit_ratio: float = 0.5
    homogeneous_ratio: float = 0.5
    class_type = HybridActuator

