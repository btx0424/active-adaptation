from math import inf
import torch

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.utils.math import yaw_quat
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify

from .locomotion import Reward

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

class feet_distance(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=...):
        super().__init__(env, weight, enabled, clip_range)


class knee_distance(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=...):
        super().__init__(env, weight, enabled, clip_range)


class feet_swing(Reward):
    def __init__(self, env, feet_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id = self.asset.find_bodies(feet_names)[0]
        self.phase: torch.Tensor = self.asset.data.phase
        self.command_manager = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        feet_linvel = self.asset.data.body_lin_vel_w[:, self.feet_id]
        swing_vel = torch.zeros_like(feet_linvel)
        swing_vel[:] = self.command_manager._command_linvel.unsqueeze(1)
        phase_sin = self.phase.sin()
        swing_vel[:, 0] *= (phase_sin > +0.1).unsqueeze(1)
        swing_vel[:, 1] *= (phase_sin < -0.1).unsqueeze(1)
        swing_vel = quat_rotate(yaw_quat(self.asset.data.root_quat_w).unsqueeze(1), swing_vel)
        reward = torch.exp(- 2 * (feet_linvel - swing_vel).abs().sum(-1)).sum(1, True)
        return 0.5 * reward.reshape(self.num_envs, 1)
    
    def debug_draw(self):
        feet_pos = self.asset.data.body_pos_w[:, self.feet_id]
        swing_vel = torch.zeros_like(feet_pos)
        swing_vel[:] = self.command_manager._command_linvel.unsqueeze(1)
        swing_vel[:, 0] *= (self.phase.sin() > +0.1).unsqueeze(1)
        swing_vel[:, 1] *= (self.phase.sin() < -0.1).unsqueeze(1)
        swing_vel = quat_rotate(yaw_quat(self.asset.data.root_quat_w).unsqueeze(1), swing_vel)
        self.env.debug_draw.vector(feet_pos.reshape(-1, 3), swing_vel.reshape(-1, 3))

