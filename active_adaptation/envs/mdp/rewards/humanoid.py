from math import inf
import torch

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.utils.math import yaw_quat
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify

from .locomotion import Reward, normalize

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(-1, True)

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
        self.a = 2.

        self.feet_vel_buf = torch.zeros(self.num_envs, 2, 3, 4, device=self.device)
    
    def update(self):
        self.feet_vel_buf[..., 1:] = self.feet_vel_buf[..., :-1]
        self.feet_vel_buf[..., 0] = self.asset.data.body_lin_vel_w[:, self.feet_id]

    def reset(self, env_ids):
        self.feet_vel_buf[env_ids] = 0.

    def compute(self) -> torch.Tensor:
        feet_linvel = self.feet_vel_buf.mean(-1)
        swing_vel = torch.zeros_like(feet_linvel)
        swing_vel[:] = self.command_manager._command_linvel.unsqueeze(1)
        phase_sin = self.phase.sin()
        swing_vel[:, 0] *= (phase_sin > +0.15).float().unsqueeze(1)
        swing_vel[:, 1] *= (phase_sin < -0.15).float().unsqueeze(1)
        swing_vel = quat_rotate(yaw_quat(self.asset.data.root_quat_w).unsqueeze(1), swing_vel)
        # reward = torch.exp(- 2 * (feet_linvel - swing_vel).abs().sum(-1)).sum(1, True)
        reward = self.a * (normalize(swing_vel) * feet_linvel).sum(-1)
        reward = torch.where(reward>0, reward.log1p().clamp(max=self.a), reward).sum(1, True)
        return reward.reshape(self.num_envs, 1)
    
    def debug_draw(self):
        feet_pos = self.asset.data.body_pos_w[:, self.feet_id]
        feet_linvel = self.feet_vel_buf.mean(-1)
        swing_vel = torch.zeros_like(feet_pos)
        swing_vel[:] = self.command_manager._command_linvel.unsqueeze(1)
        swing_vel[:, 0] *= (self.phase.sin() > +0.15).float().unsqueeze(1)
        swing_vel[:, 1] *= (self.phase.sin() < -0.15).float().unsqueeze(1)
        swing_vel = quat_rotate(yaw_quat(self.asset.data.root_quat_w).unsqueeze(1), swing_vel)
        self.env.debug_draw.vector(feet_pos.reshape(-1, 3), swing_vel.reshape(-1, 3))
        self.env.debug_draw.vector(feet_pos.reshape(-1, 3), feet_linvel.reshape(-1, 3), color=(1., 0., 0.2, 1.))


class feet_orientation(Reward):

    def __init__(self, env, feet_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id = self.asset.find_bodies(feet_names)[0]
        self.heading_feet = torch.tensor([[[0., 1., 0.]]], device=self.device)
        self.heading_root = torch.tensor([[[1., 0., 0.]]], device=self.device)
    
    def compute(self) -> torch.Tensor:
        quat_feet = yaw_quat(self.asset.data.body_quat_w[:, self.feet_id])
        quat_root = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        reward = dot(
            quat_rotate(quat_feet, self.heading_feet), 
            quat_rotate(quat_root, self.heading_root)
        )
        return reward.mean(1)

    def debug_draw(self):
        feet_pos = self.asset.data.body_pos_w[:, self.feet_id]
        quat_feet = yaw_quat(self.asset.data.body_quat_w[:, self.feet_id])
        self.env.debug_draw.vector(
            feet_pos.reshape(-1, 3),
            quat_rotate(quat_feet, self.heading_feet).reshape(-1, 3),
            color=(1., 1., 0., 1.)
        )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w, 
            quat_rotate(self.asset.data.root_quat_w, self.heading_root),
            color=(1., 1., 0., 1.)
        )


class joint_pos_default(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
    
    def compute(self) -> torch.Tensor:
        dev = self.asset.data.joint_pos - self.default_joint_pos
        return - dev.square().mean(1, True)

