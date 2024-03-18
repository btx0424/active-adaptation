import torch
import math

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
import omni.isaac.orbit.utils.math as math_utils
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

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

class Command:
    def __init__(self, env) -> None:
        self.env = env
    
    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device
    
    def debug_draw(self):
        pass


class Command1(Command):
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
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
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


class CommandPos(Command):

    def __init__(
        self, 
        env, 
        speed_range=(0.7, 1.4),
        offset_range=(2.0, 4.0)
    ):
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.speed_range = speed_range
        self.offset_range = offset_range
        with torch.device(self.device):
            self.target_speed = torch.zeros(self.num_envs, 1)
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.command = torch.zeros(self.num_envs, 4)
            self._command_linvel = torch.zeros(self.num_envs, 3)
            self._command_heading = torch.zeros(self.num_envs, 3)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
    
    def reset(self, env_ids):
        self.sample_commands(env_ids)

    def update(self, resample: torch.Tensor=None):
        if resample is not None and len(resample) > 0:
            self.sample_commands(resample)
        pos_diff_xy = (self.target_pos_w - self.robot.data.root_pos_w)[:, :2]
        distance_xy = pos_diff_xy.norm(dim=-1, keepdim=True)
        
        command_speed = torch.minimum(distance_xy, self.target_speed)
        self.is_standing_env[:] = (command_speed < 0.2)

        direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        direction_w[:, :2] = pos_diff_xy / distance_xy.clamp(1e-6)
        direction_b = quat_rotate_inverse(
            self.robot.data.root_quat_w, direction_w
        )
        self._command_linvel[:, :2] = direction_b[:, :2] * command_speed
        self._command_heading[:, :2] = torch.where(
            self.is_standing_env,
            torch.tensor([1., 0.], device=self.device),
            direction_b[:, :2]
        )

        self.command[:, :2] = self._command_linvel[:, :2]
        self.command[:, 2:4] = self._command_heading[:, :2]
    
    def sample_commands(self, env_ids: torch.Tensor):
        robot_pos_w = self.robot.data.root_pos_w[env_ids, :2]
        d = sample_uniform(env_ids.shape, *self.offset_range, device=self.device)
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2.
        offset = torch.stack([a.cos(), a.sin()], 1) * d.unsqueeze(1)
        self.target_pos_w[env_ids, :2] = robot_pos_w + offset
        self.target_speed[env_ids, 0] = sample_uniform(env_ids.shape, *self.speed_range, self.device)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w,
            self.target_pos_w - self.robot.data.root_pos_w,
            color=(1., .6, .4, 1.),
        )
        self.env.debug_draw.point(self.target_pos_w, size=20)


def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low