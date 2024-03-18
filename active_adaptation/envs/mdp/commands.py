import torch
import math

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
import omni.isaac.orbit.utils.math as math_utils

COMMAND_CFG = UniformVelocityCommandCfg(
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.02,
    rel_heading_envs=1.0,
    heading_command=True,
    debug_vis=False,
    ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    ),
)


class Command1:
    """
    Generate commands of liner velocity in body frame, angular velocity, and base height.
    """
    command_dim: int=4 # linvel_xy, angvel_z, base_height

    def __init__(
        self,
        env,
        speed_range=(0.5, 2.0),
        angvel_range=(-1.0, 1.0),
        base_height_range=(0.2, 0.4),
        stand_prob=0.1
    ):
        self.env = env
        self.robot: Articulation = env.scene["robot"]
        self.device = env.device
        self.speed_range = speed_range
        self.base_height_range = base_height_range
        self.angvel_range = angvel_range
        self.sand_prob = stand_prob

        with torch.device(env.device):
            self._target_yaw = torch.zeros(env.num_envs)
            self._target_base_height = torch.zeros(env.num_envs, 1)
            self._command_stand = torch.zeros(env.num_envs, 1)
            self._command_linvel = torch.zeros(env.num_envs, 3)
            self._command_angvel_yaw = torch.zeros(env.num_envs)
            self._command_heading = torch.zeros(env.num_envs, 3)
            self._command_speed = torch.zeros(env.num_envs, 1)
            
            self.command = torch.zeros(env.num_envs, self.command_dim)
            self.command_prev = torch.zeros(env.num_envs, self.command_dim)
        self.is_standing_env = self._command_stand

    def reset(self, env_ids: torch.Tensor):
        self.sample_commands(env_ids)
        self.command_prev[env_ids] = self.command[env_ids]

    def update(self, resample: torch.Tensor=None):
        if resample is not None and len(resample) > 0:
            self.sample_commands(resample)
        
        yaw_diff = self._target_yaw - self.robot.data.heading_w
        self._command_angvel_yaw[:] = math_utils.wrap_to_pi(yaw_diff).clamp(*self.angvel_range)

        self.command_prev[:] = self.command
        self.command[:, :2] = self._command_linvel[:, :2]
        self.command[:, 2] = self._command_angvel_yaw

    def sample_commands(self, env_ids: torch.Tensor):
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        stand = torch.rand(len(env_ids), device=self.device) < self.sand_prob
        speed = torch.zeros(len(env_ids), device=self.device).uniform_(*self.speed_range)
        speed = speed * (~stand).float()
        
        self._command_stand[env_ids] = stand.float().unsqueeze(1)
        self._command_speed[env_ids] = speed.unsqueeze(1)
        self._command_linvel[env_ids, 0] = speed * a.cos()
        self._command_linvel[env_ids, 1] = speed * a.sin()
        
        yaw = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        self._target_yaw[env_ids] = yaw
        self._command_heading[env_ids, 0] = yaw.cos()
        self._command_heading[env_ids, 1] = yaw.sin()
        
        self._target_base_height[env_ids] = sample_uniform(
            env_ids.shape, *self.base_height_range, self.env.device
        ).unsqueeze(1)
        self.command[:, 3:4] = self._target_base_height


def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low