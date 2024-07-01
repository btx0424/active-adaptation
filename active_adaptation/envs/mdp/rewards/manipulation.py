import torch

from omni.isaac.lab.assets import Articulation
from active_adaptation.utils.math import quat_rotate
from omni.isaac.lab.utils.math import yaw_quat
from .locomotion import Reward
from ..commands import CommandEEPose

class ee_pose_tracking(Reward):
    def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
        self.ee_id =self.ee_id[0]
        self.command_manager: CommandEEPose = self.env.command_manager
        with torch.device(self.device):
            self.ee_forward_w = torch.zeros(self.num_envs, 3)
            self.ee_upward_w = torch.zeros(self.num_envs, 3)

    def update(self):
        ee_quat_w = self.asset.data.body_quat_w[:, self.ee_id]
        forward = torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, -1)
        upward = torch.tensor([0., 0., 1.], device=self.device).expand(self.num_envs, -1)
        self.ee_forward_w[:] = quat_rotate(ee_quat_w, forward)
        self.ee_upward_w[:] = quat_rotate(ee_quat_w, upward)

    def compute(self) -> torch.Tensor:
        ee_pos = self.asset.data.body_pos_w[:, self.ee_id]
        # r_linvel = torch.exp(-self.asset.data.body_lin_vel_w.square().sum(1, True) / 0.25)
        pos_error = (ee_pos - self.command_manager.command_ee_pos_w).square().sum(1, True)
        r_pos = torch.exp(- pos_error / 0.25)
        r_forward = (self.ee_forward_w * self.command_manager.command_ee_forward_w).sum(1, True)
        r_upward = (self.ee_upward_w * self.command_manager.command_ee_upward_w).sum(1, True)
        return r_pos + 0.5 * (r_forward + r_upward)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.ee_id],
            self.ee_forward_w * 0.2,
            color=(1., 0.1, 0.1, 1.)
        )
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.ee_id],
            self.ee_upward_w * 0.2,
            color=(1., 0.1, 0.1, 1.)
        )

class ee_vel_tracking(Reward):
    def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
        self.ee_id =self.ee_id[0]

    def compute(self) -> torch.Tensor:
        ee_vel = self.asset.data.body_lin_vel_w[:, self.ee_id]
        return torch.exp(-ee_vel.norm(dim=-1, keepdim=True) / 0.25)