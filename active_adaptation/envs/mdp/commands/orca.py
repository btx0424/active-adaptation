from math import pi
import torch
import torch.distributions as D
import math
import warp as wp
from typing import Sequence, TYPE_CHECKING

from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, MultiUniform
from active_adaptation.utils.helpers import batchify
from isaaclab.utils.math import quat_apply_yaw, yaw_quat
from tensordict import TensorDict
from .base import Command


class OrcaLocoCommand(Command):
    def __init__(
        self, 
        env, 
        linvel_x_range=(-1.0, 1.0),
        linvel_y_range=(-1.0, 1.0),
        angvel_range=(-1, 1),
        yaw_stiffness_range=(0.5, 0.6),
        use_stiffness_ratio: float = 0.5,
        resample_interval: int = 300, 
        resample_prob: float = 0.75, 
        stand_prob=0.2,
        target_yaw_range=(0, torch.pi * 2),
        adaptive: bool = False,
        teleop: bool = False,
    ):
        super().__init__(env, teleop=teleop)
        self.robot: Articulation = env.scene["robot"]
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.angvel_range = angvel_range
        self.use_stiffness_ratio = use_stiffness_ratio
        self.yaw_stiffness_range = yaw_stiffness_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.stand_prob = stand_prob
        self.adaptive = adaptive

        with torch.device(self.device):
            if all(isinstance(r, Sequence) for r in target_yaw_range):
                self.target_yaw_dist = MultiUniform(torch.tensor(target_yaw_range))
            else:
                self.target_yaw_dist = D.Uniform(*torch.tensor(target_yaw_range))

            self.command = torch.zeros(self.num_envs, 4)
            self.target_yaw = torch.zeros(self.num_envs)
            self.yaw_stiffness = torch.zeros(self.num_envs)
            self.use_stiffness = torch.zeros(self.num_envs, dtype=bool)
            self.fixed_yaw_speed = torch.zeros(self.num_envs)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            self.command_speed = torch.zeros(self.num_envs, 1)
            self._target_direction = torch.zeros(self.num_envs, 3)
            self._target_linvel = torch.zeros(self.num_envs, 3)
            self.command_linvel = torch.zeros(self.num_envs, 3)
            self.command_linvel_w = torch.zeros(self.num_envs, 3)
            self.command_angvel = torch.zeros(self.num_envs)

            self.aux_input = torch.zeros(self.num_envs, 1)

            self._cum_error = torch.zeros(self.num_envs, 2)
            self._cum_linvel_error = self._cum_error[:, 0].unsqueeze(1)
            self._cum_angvel_error = self._cum_error[:, 1].unsqueeze(1)
            
        self._decay = 0.999
        self._sum_error = torch.tensor(0.0, device=self.device)
        self._count = torch.tensor(0.0, device=self.device)
        self._avg_error = torch.tensor(0.0, device=self.device)

        if self.teleop:
            self.key_mappings_pos = {
                "W": torch.tensor([self.linvel_x_range[1], 0., 0.], device=self.device),
                "S": torch.tensor([self.linvel_x_range[0], 0., 0.], device=self.device),
                "A": torch.tensor([0., self.linvel_y_range[1], 0.], device=self.device),
                "D": torch.tensor([0., self.linvel_y_range[0], 0.], device=self.device),
            }
        
    def reset(self, env_ids, reward_stats = None):
        self.command[env_ids] = 0.
        self._target_linvel[env_ids] = 0.
        self.target_yaw[env_ids] = self.asset.data.heading_w[env_ids]
        
        self._cum_linvel_error[env_ids] = 0.
        self._cum_angvel_error[env_ids] = 0.
        self.env.extra["stats/avg_error"] = self._avg_error.item()
    
    def update(self):
        root_linvel_w = self.robot.data.root_lin_vel_w
        root_angvel_w = self.robot.data.root_ang_vel_w
        root_heading_w = self.robot.data.heading_w

        if self.teleop:
            command_linvel_target = torch.tensor([0., 0., 0.], device=self.device)
            for key, vec in self.key_mappings_pos.items():
                if self.key_pressed[key]:
                    command_linvel_target.add_(vec)
            if not self.key_pressed["LEFT_SHIFT"]:
                command_linvel_target *= 0.6
            self.command_linvel.lerp_(command_linvel_target, 0.5)
        else:
            interval_reached = (self.env.episode_length_buf - 40) % self.resample_interval == 0
            resample_vel = interval_reached # & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
            resample_vel = resample_vel.nonzero().squeeze(-1)
            if len(resample_vel) > 0:
                self.sample_vel_command(resample_vel)
            resample_yaw = interval_reached # & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
            resample_yaw = resample_yaw.nonzero().squeeze(-1)
            if len(resample_yaw) > 0:
                self.sample_yaw_command(resample_yaw)

        yaw_diff = self.target_yaw - root_heading_w
        command_yaw_speed = torch.clamp(
            self.yaw_stiffness * math_utils.wrap_to_pi(yaw_diff), 
            min=self.angvel_range[0],
            max=self.angvel_range[1]
        )
        self.command_angvel[:] = torch.where(self.use_stiffness, command_yaw_speed, self.fixed_yaw_speed)

        # this is used for terminating episodes where the robot is inactive due to whatever reason
        linvel_error = (self.robot.data.root_lin_vel_w[:, :2] - self.command_linvel_w[:, :2]).norm(dim=-1, keepdim=True)
        angvel_error = (self.command_angvel - self.robot.data.root_ang_vel_w[:, 2]).abs().unsqueeze(1)

        self._sum_error.add_(linvel_error.sum()).mul_(self._decay)
        self._count.add_(self.num_envs).mul_(self._decay)
        self._avg_error.copy_(self._sum_error / self._count)

        target_linvel = self._target_linvel

        self._cum_linvel_error.mul_(0.98).add_(linvel_error * self.env.step_dt)
        self._cum_angvel_error.mul_(0.98).add_(angvel_error * self.env.step_dt)
        self.command_linvel[:] = self.command_linvel + clamp_norm((target_linvel - self.command_linvel) * 0.1, max=0.1)

        self.command_linvel_w[:] = quat_apply_yaw(self.robot.data.root_quat_w, self.command_linvel)
        self.command[:, :2] = self.command_linvel[:, :2]
        self.command[:, 2] = self.command_angvel
        self.command[:, 3] = math_utils.wrap_to_pi(yaw_diff)
        
        # self.command[:, :2] = torch.tensor([1.0, 0.], device=self.device)
        self.is_standing_env[:, 0] = (
            (self.command_linvel.norm(dim=-1) < 0.1)
            & (self.command_angvel < 0.1)
        )

    
    def sample_vel_command(self, env_ids: torch.Tensor):
        linvel = torch.zeros(len(env_ids), 2, device=self.device)
        linvel[:, 0].uniform_(*self.linvel_x_range)
        linvel[:, 0] = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.2, 
            linvel[:, 0].abs(), linvel[:, 0]
        )
        linvel[:, 1].uniform_(*self.linvel_y_range)
        speed = linvel.norm(dim=-1, keepdim=True)
        direction = linvel / speed.clamp(1e-6)
        stand = (speed < 0.3) | (torch.rand(len(env_ids), 1, device=self.device) < self.stand_prob)
        speed = speed * (~stand)

        self.command_speed[env_ids] = speed
        self._target_direction[env_ids, :2] = direction
        self._target_linvel[env_ids, :2] = direction * speed

    def sample_yaw_command(self, env_ids: torch.Tensor):
        self.target_yaw[env_ids] = self.target_yaw_dist.sample(env_ids.shape)
        self.yaw_stiffness[env_ids] = sample_uniform(env_ids.shape, *self.yaw_stiffness_range, self.device)
        self.use_stiffness[env_ids] = torch.rand(len(env_ids), device=self.device) < self.use_stiffness_ratio
        self.fixed_yaw_speed[env_ids] = sample_uniform(env_ids.shape, *self.angvel_range, self.device) 

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            self.command_linvel_w,
            color=(1., 1., 1., 1.)
        )
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            torch.stack([self.target_yaw.cos(), self.target_yaw.sin(), torch.zeros_like(self.target_yaw)], 1),
            color=(.2, .2, 1., 1.)
        )
        zeros = torch.zeros(self.num_envs, 1, device=self.device)
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            torch.stack([zeros, zeros, self._cum_linvel_error], 1),
            color=(.2, 1., .2, 1.)
        )
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            torch.stack([zeros, zeros, self._cum_angvel_error], 1),
            color=(1., .2, .2, 2.)
        )



def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x


def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low