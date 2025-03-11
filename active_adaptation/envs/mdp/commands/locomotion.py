from math import pi
import torch
import torch.nn.functional as F
import torch.distributions as D
import math
import warp as wp
from typing import Sequence, TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from active_adaptation.utils.math import (
    quat_rotate, 
    quat_rotate_inverse,
    clamp_norm,
    yaw_quat,
    wrap_to_pi,
    MultiUniform
)

from .base import Command
from ..observations import _initialize_warp_meshes, raycast_mesh


class Command1(Command):
    """
    Generate commands of liner velocity in body frame, angular velocity, and base height.
    """

    command_dim: int = 4

    def __init__(
        self,
        env,
        speed_range=(0.5, 2.0),
        angvel_range=(-1.0, 1.0),
        base_height_range=(0.2, 0.4),
        resample_interval: int = 300,
        resample_prob: float = 0.75,
        stand_prob=0.2,
    ):
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.height_scanner = env.scene.sensors.get("height_scanner", None)
        self.speed_range = speed_range
        self.base_height_range = base_height_range
        self.angvel_range = angvel_range

        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.stand_prob = stand_prob

        with torch.device(env.device):
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.target_yaw = torch.zeros(self.num_envs, 1)
            self.ref_yaw = torch.zeros(self.num_envs, 1)
            self.yaw_stiffness = torch.zeros(self.num_envs, 1)

            self.command = torch.zeros(self.num_envs, self.command_dim)
            self.command_linvel_w = torch.zeros(self.num_envs, 3)
            self.command_linvel_b = torch.zeros(self.num_envs, 3)
            self.command_linvel = self.command_linvel_b
            self.command_angvel = torch.zeros(self.num_envs, 1)
            self.max_speed = torch.zeros(self.num_envs, 1)
            self.command_speed = torch.zeros(self.num_envs, 1)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            self._cum_error = torch.zeros(self.num_envs, 2)
            self._cum_linvel_error = self._cum_error[:, 0].unsqueeze(1)
            self._cum_angvel_error = self._cum_error[:, 1].unsqueeze(1)

    def reset(self, env_ids: torch.Tensor):
        self.target_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids]
        self.target_yaw[env_ids] = self.asset.data.heading_w[env_ids].unsqueeze(1)
        self.max_speed[env_ids] = 0.6 + torch.rand(len(env_ids), 1, device=self.device) * 0.8

    def update(self):
        yaw = self.asset.data.heading_w.unsqueeze(1)
        pos = self.asset.data.root_pos_w
        yaw_diff = self.target_yaw - yaw
        self.command_angvel[:] = self.yaw_stiffness * wrap_to_pi(yaw_diff)
        self.command_angvel.clamp_(-2., 2.)

        max_speed = (self.max_speed - self.command_angvel.abs()).clamp(0.)
        self.command_linvel_w[:] = quat_rotate(self.asset.data.root_quat_w, self.command_linvel_b)
        self.command[:, :2] = self.command_linvel_b[:, :2]
        self.command[:, 2:3] = wrap_to_pi(self.target_yaw - yaw)
        self.command[:, 3:4] = self.command_angvel
        self.command_speed = self.command_linvel_w.norm(dim=-1, keepdim=True)
        
        resample = ((self.env.episode_length_buf - 40) % self.resample_interval == 0)
        if resample.any():
            self._resample(resample.nonzero().squeeze(-1))
        self.ref_yaw.add_(self.command_angvel * self.env.step_dt)
    
    # def _from_world(self, max_speed):
    #     pos_diff = self.target_pos_w - self.asset.data.root_pos_w
    #     pos_diff[:, 2] = 0.0
    #     command_linvel_w = torch.zeros(self.num_envs, 3, device=self.device)
    #     command_linvel_w[:, :2] = clamp_norm(pos_diff[:, :2], max=max_speed)
    #     command_linvel_b = quat_rotate_inverse(self.asset.data.root_quat_w, self.command_linvel_w)
    #     return command_linvel_w, command_linvel_b
    
    # def _from_body(self, max_speed):
    #     command_linvel_w = quat_rotate(self.asset.data.root_quat_w, self.command_linvel_b)
    #     return command_linvel_w, self.command_linvel_b

    def _resample(self, env_ids):
        self.target_pos_w[env_ids] = self.target_pos_w[env_ids] + torch.tensor([8., 0., 0.], device=self.device)
        yaw = self.asset.data.heading_w[env_ids].unsqueeze(1)
        command_linvel_b = torch.rand(len(env_ids), 3, device=self.device)
        command_linvel_b[:, 0].uniform_(0.4, 1.2)
        command_linvel_b[:, 1].uniform_(-0.2, 0.2).mul_(command_linvel_b[:, 0])
        command_linvel_b = torch.where(yaw.abs() < 0.2, command_linvel_b, command_linvel_b * 0.5)
        self.command_linvel_b[env_ids] = command_linvel_b

        self.target_yaw[env_ids] = torch.rand(len(env_ids), 1, device=self.device) * torch.pi / 3 - torch.pi / 6
        self.ref_yaw[env_ids] = self.asset.data.heading_w[env_ids].unsqueeze(1)
        self.yaw_stiffness[env_ids] = 1. + torch.rand(len(env_ids), 1, device=self.device)
    
    def debug_draw(self):
        ref_vec = torch.cat([self.ref_yaw.cos(), self.ref_yaw.sin(), torch.zeros_like(self.ref_yaw)], 1)
        yaw = self.asset.data.heading_w.unsqueeze(1)
        vec = torch.cat([yaw.cos(), yaw.sin(), torch.zeros_like(yaw)], 1).squeeze([])
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            # self.target_pos_w - self.asset.data.root_pos_w,
            # self.command_linvel_w,
            # ref_vec,
            vec,
            color=(0.5, 1.0, 0.5, 1.0),
        )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            # self.target_pos_w - self.asset.data.root_pos_w,
            self.command_linvel_w,
            # ref_vec,
            color=(1.0, 0.5, 0.5, 1.0),
        )


class Command3(Command):
    def __init__(self, env):
        super().__init__(env)
        
        with torch.device(self.device):
            self.offsets_b = torch.tensor([[0.25, 0., 0.], [-0.15, 0., 0.]])
            self.des_pos_w = torch.zeros(self.num_envs, 3)
            self.des_yaw_w = torch.zeros(self.num_envs, 1)
            self.des_vel_w = torch.zeros(self.num_envs, 3)
            # self.des_key_pos_w = torch.zeros(self.num_envs, 2, 3)
            self.key_pos_w = torch.zeros(self.num_envs, 2, 3)
            self.key_vel_w = torch.zeros(self.num_envs, 2, 3)

            self.command = torch.zeros(self.num_envs, 10)
            self._cum_error = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
        
        self.mesh = _initialize_warp_meshes("/World/ground", "cuda")

    def get_height_at(self, pos_w):
        ray_starts = pos_w + torch.tensor([0., 0., 10.], device=self.device)
        ray_directions = torch.tensor([0., 0., -1.], device=self.device).expand_as(ray_starts)
        ray_hit_w = raycast_mesh(ray_starts, ray_directions, self.mesh)[0]
        return ray_hit_w[..., 2]

    def reset(self, env_ids):
        self.des_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids]
        self.des_yaw_w[env_ids] = 0.
        des_vel_w = torch.zeros(len(env_ids), 3, device=self.device)
        des_vel_w[:, 0].uniform_(0.3, 1.3)
        des_vel_w[:, 1].uniform_(-0.2, 0.2)
        self.des_vel_w[env_ids] = des_vel_w
        self._cum_error[env_ids] = 0.
        self.key_vel_w[env_ids] = 0.
    
    @property
    def des_key_pos_w(self):
        des_key_pos_w = self.des_pos_w.unsqueeze(1) + self.offsets_b
        des_key_pos_w[:, :, 2] = self.get_height_at(des_key_pos_w) + 0.35
        return des_key_pos_w
    
    def update(self):
        quat = yaw_quat(self.asset.data.root_quat_w)
        key_pos_w = (
            self.asset.data.root_pos_w.unsqueeze(1) +
            quat_rotate(self.asset.data.root_quat_w.unsqueeze(1), self.offsets_b.unsqueeze(0))
        )
        self.key_vel_w = (key_pos_w - self.key_pos_w) / self.env.step_dt
        self.key_pos_w = key_pos_w

        diff = self.des_key_pos_w - self.key_pos_w
        self._cum_error = diff.norm(dim=-1, keepdim=True).mean(1)
        self.command[:, 0:6] = quat_rotate_inverse(quat.unsqueeze(1), diff).reshape(self.num_envs, 6)
        self.command[:, 6:9] = quat_rotate_inverse(quat, self.des_vel_w)
        self.command[:, 9:10] = wrap_to_pi(self.des_yaw_w - self.asset.data.heading_w.unsqueeze(1))
        
        self.des_pos_w += self.des_vel_w * self.env.step_dt
        self.des_pos_w[:, 2] = self.get_height_at(self.des_pos_w) + 0.35
    
    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.des_pos_w - self.asset.data.root_pos_w,
            color=(1.0, 0.5, 0.5, 1.0),
        )
        down = torch.tensor([0., 0., -1.], device=self.device)
        self.env.debug_draw.vector(
            self.key_pos_w.reshape(-1, 3),
            self.des_key_pos_w.reshape(-1, 3) - self.key_pos_w.reshape(-1, 3),
            color=(0.5, 1.0, 0.5, 1.0),
        )


class Command2(Command):

    def __init__(
        self,
        env,
        linvel_x_range=(-1.0, 1.0),
        linvel_y_range=(-1.0, 1.0),
        angvel_range=(-1, 1),
        yaw_stiffness_range=(0.5, 0.6),
        use_stiffness_ratio: float = 0.5,
        aux_input_range=(0.2, 0.4),
        resample_interval: int = 300,
        resample_prob: float = 0.75,
        stand_prob=0.2,
        target_yaw_range=(0, torch.pi * 2),
        adaptive: bool = False,
        body_name: str = None,
        teleop: bool = False,
    ):
        super().__init__(env, teleop=teleop)
        self.robot: Articulation = env.scene["robot"]
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.angvel_range = angvel_range
        self.use_stiffness_ratio = use_stiffness_ratio
        self.yaw_stiffness_range = yaw_stiffness_range
        self.aux_input_range = aux_input_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.stand_prob = stand_prob
        self.adaptive = adaptive

        if self.adaptive:
            self.ground_mesh = _initialize_warp_meshes("/World/ground", "cuda")

        if body_name is not None:
            self.body_id = self.asset.find_bodies(body_name)[0][0]
            self.FWDVEC = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(
                self.num_envs, 3
            )
        else:
            self.body_id = None

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
            self.next_command_linvel = torch.zeros(self.num_envs, 3)
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
                "W": torch.tensor(
                    [self.linvel_x_range[1], 0.0, 0.0], device=self.device
                ),
                "S": torch.tensor(
                    [self.linvel_x_range[0], 0.0, 0.0], device=self.device
                ),
                "A": torch.tensor(
                    [0.0, self.linvel_y_range[1], 0.0], device=self.device
                ),
                "D": torch.tensor(
                    [0.0, self.linvel_y_range[0], 0.0], device=self.device
                ),
            }

    def reset(self, env_ids, reward_stats=None):
        self.command[env_ids] = 0.0
        self.next_command_linvel[env_ids] = 0.0
        self.command_linvel[env_ids] = 0.0

        self.target_yaw[env_ids] = self.asset.data.heading_w[env_ids]

        self._cum_linvel_error[env_ids] = 0.0
        self._cum_angvel_error[env_ids] = 0.0
        self.is_standing_env[env_ids] = True
        self.env.extra["stats/avg_error"] = self._avg_error.item()

    def update(self):
        if self.body_id is not None:
            self.body_quat_w = self.asset.data.body_quat_w[:, self.body_id]
            forward_w = quat_rotate(self.body_quat_w, self.FWDVEC)
            self.body_heading_w = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        else:
            self.body_heading_w = self.asset.data.heading_w
        if self.teleop:
            command_linvel_target = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            for key, vec in self.key_mappings_pos.items():
                if self.key_pressed[key]:
                    command_linvel_target.add_(vec)
            if not self.key_pressed["LEFT_SHIFT"]:
                command_linvel_target *= 0.6
            self.command_linvel.lerp_(command_linvel_target, 0.5)
        else:
            interval_reached = (
                self.env.episode_length_buf - 20
            ) % self.resample_interval == 0
            resample_vel = interval_reached & (
                (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
                | self.is_standing_env.squeeze(1)
            )
            resample_yaw = interval_reached & (
                (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
                | self.is_standing_env.squeeze(1)
            )
            self.sample_vel_command(resample_vel.nonzero().squeeze(-1))
            self.sample_yaw_command(resample_yaw.nonzero().squeeze(-1))

        self.target_yaw[~self.use_stiffness] = self.body_heading_w[~self.use_stiffness]
        yaw_diff = self.target_yaw - self.body_heading_w
        command_yaw_speed = torch.clamp(
            self.yaw_stiffness * wrap_to_pi(yaw_diff),
            min=self.angvel_range[0],
            max=self.angvel_range[1],
        )
        self.command_angvel[:] = torch.where(
            self.use_stiffness, command_yaw_speed, self.fixed_yaw_speed
        )

        # this is used for terminating episodes where the robot is inactive due to whatever reason
        linvel_error = (
            self.robot.data.root_lin_vel_w[:, :2] - self.command_linvel_w[:, :2]
        ).norm(dim=-1, keepdim=True)
        angvel_error = (
            (self.command_angvel - self.robot.data.root_ang_vel_w[:, 2])
            .abs()
            .unsqueeze(1)
        )

        self._sum_error.add_(linvel_error.sum()).mul_(self._decay)
        self._count.add_(self.num_envs).mul_(self._decay)
        self._avg_error.copy_(self._sum_error / self._count)

        if self.adaptive:
            self.ray_start_w = self.robot.data.root_pos_w + torch.tensor(
                [0.0, 0.0, -0.2], device=self.device
            )
            ray_direction = quat_rotate(
                self.asset.data.root_quat_w, self._target_direction
            )
            ray_direction[:, 2] = 0.0
            self.ray_hit_w = raycast_mesh(
                self.ray_start_w, ray_direction, max_dist=2, mesh=self.ground_mesh
            )[0]
            distance_to_obstacle = (
                (self.ray_hit_w - self.ray_start_w)
                .norm(dim=-1, keepdim=True)
                .nan_to_num(2.0)
            )
            self.close_to_obstacle = distance_to_obstacle < 0.75
            fast = self.command_speed > 1.4
            target_linvel = torch.where(
                (self.close_to_obstacle & fast),
                self._target_linvel / 2,
                self._target_linvel,
            )
        else:
            target_linvel = self._target_linvel

        self._cum_linvel_error.mul_(0.98).add_(linvel_error * self.env.step_dt)
        self._cum_angvel_error.mul_(0.98).add_(angvel_error * self.env.step_dt)

        max_command_speed = (2.5 - self.command_angvel.abs()).clamp(0.0).unsqueeze(1)
        self.command_linvel.lerp_(self.next_command_linvel, 0.1)
        self.command_linvel = clamp_norm(self.command_linvel, max=max_command_speed)
        self.command_speed = self.command_linvel.norm(dim=-1, keepdim=True)

        self.command_linvel_w[:] = quat_rotate(
            yaw_quat(self.robot.data.root_quat_w), self.command_linvel
        )
        self.command[:, :2] = self.command_linvel[:, :2]
        self.command[:, 2] = self.command_angvel
        self.command[:, 3] = self.aux_input.squeeze(1)
        # self.command[:, :2] = torch.tensor([1.0, 0.], device=self.device)
        self.is_standing_env[:, 0] = (
            (self.command_linvel.norm(dim=-1) < 0.1) & (self.command_angvel < 0.1)
        )

    def sample_vel_command(self, env_ids: torch.Tensor):
        next_command_linvel = torch.zeros(len(env_ids), 3, device=self.device)
        next_command_linvel[:, 0].uniform_(*self.linvel_x_range)
        next_command_linvel[:, 1].uniform_(*self.linvel_y_range)

        speed = next_command_linvel.norm(dim=-1, keepdim=True)
        r = torch.rand(len(env_ids), 1, device=self.device) < self.stand_prob
        valid = ~((speed < 0.10) | r)
        self.next_command_linvel[env_ids] = next_command_linvel * valid
        self.aux_input[env_ids] = sample_uniform(
            env_ids.shape, *self.aux_input_range, self.device
        ).unsqueeze(1)

    def sample_yaw_command(self, env_ids: torch.Tensor):
        r = torch.rand(len(env_ids), device=self.device) < self.stand_prob
        self.target_yaw[env_ids] = torch.where(
            r,
            self.asset.data.heading_w[env_ids],
            self.target_yaw_dist.sample(env_ids.shape)
        )
        self.yaw_stiffness[env_ids] = sample_uniform(
            env_ids.shape, *self.yaw_stiffness_range, self.device
        )
        self.use_stiffness[env_ids] = (
            torch.rand(len(env_ids), device=self.device) < self.use_stiffness_ratio
        )
        self.fixed_yaw_speed[env_ids] = sample_uniform(
            env_ids.shape, *self.angvel_range, self.device
        )

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w
            + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            self.command_linvel_w,
            color=(1.0, 1.0, 1.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w
            + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            torch.stack(
                [
                    self.target_yaw.cos(),
                    self.target_yaw.sin(),
                    torch.zeros_like(self.target_yaw),
                ],
                1,
            ),
            color=(0.2, 0.2, 1.0, 1.0),
        )
        zeros = torch.zeros(self.num_envs, 1, device=self.device)
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w
            + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            torch.stack([zeros, zeros, self._cum_linvel_error], 1),
            color=(0.2, 1.0, 0.2, 1.0),
        )
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w
            + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            torch.stack([zeros, zeros, self._cum_angvel_error], 1),
            color=(1.0, 0.2, 0.2, 2.0),
        )
        if self.adaptive:
            self.env.debug_draw.vector(
                self.ray_start_w[self.close_to_obstacle.squeeze(1)],
                (self.ray_hit_w - self.ray_start_w)[self.close_to_obstacle.squeeze(1)],
                # self._target_direction,
                color=(1.0, 1.0, 0.0, 1.0),
            )

    def fliplr(self, command: torch.Tensor) -> torch.Tensor:
        # flip y and yaw velocity
        return command * torch.tensor([1.0, -1.0, -1.0, 1.0], device=self.device)



def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low


def sample_quat_yaw(size, yaw_range=(0, torch.pi * 2), device: torch.device = "cpu"):
    yaw = torch.rand(size, device=device).uniform_(*yaw_range)
    quat = torch.cat(
        [
            torch.cos(yaw / 2).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.sin(yaw / 2).unsqueeze(-1),
        ],
        dim=-1,
    )
    return quat


def quat_to_yaw(quat: torch.Tensor):
    q_w, q_x, q_y, q_z = quat.unbind(-1)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return yaw % (2 * torch.pi)


