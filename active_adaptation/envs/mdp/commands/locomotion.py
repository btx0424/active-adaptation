import torch
import torch.distributions as D
import math
from typing import Sequence

from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.utils.math as math_utils
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, MultiUniform
from omni.isaac.lab.utils.math import quat_apply_yaw


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
            self.target_yaw = torch.zeros(env.num_envs)
            self._target_base_height = torch.zeros(env.num_envs, 1)
            self._target_heading = torch.zeros(env.num_envs, 3)

            self._command_direction = torch.zeros(env.num_envs, 3)
            self._command_speed = torch.zeros(env.num_envs, 1)
            self._command_linvel = torch.zeros(env.num_envs, 3)

            self._command_stand = torch.zeros(env.num_envs, 1, dtype=bool)
            self.command_angvel_yaw = torch.zeros(env.num_envs)
            
            self.command = torch.zeros(env.num_envs, self.command_dim)
        self.is_standing_env = self._command_stand
        self._command_heading = self._target_heading

    def reset(self, env_ids: torch.Tensor):
        self.sample_vel_command(env_ids)
        self.sample_yaw_command(env_ids)

    def update(self):
        interval_reached = (self.env.episode_length_buf + 1) % self.resample_interval == 0
        resample_vel = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        resample_yaw = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        self.sample_vel_command(resample_vel.nonzero().squeeze(-1))
        self.sample_yaw_command(resample_yaw.nonzero().squeeze(-1))
        
        yaw_diff = self.target_yaw - self.robot.data.heading_w
        self.command_angvel_yaw[:] = math_utils.wrap_to_pi(yaw_diff).clamp(*self.angvel_range)

        command_speed = self._command_speed
        if self.height_scanner is not None:
            height_scan_z: torch.Tensor = self.height_scanner.data.ray_hits_w[:, :, [2]]
            near_stairs = height_scan_z.max(1)[0] - height_scan_z.min(1)[0] > 0.2
            assert near_stairs.shape == command_speed.shape
            command_speed = torch.where(
                near_stairs,
                command_speed.clamp(max=1.0),
                command_speed
            )
        self.is_standing_env[:] = torch.logical_and(
            self.command_angvel_yaw.unsqueeze(1).abs() < 0.1,
            self._command_speed < 0.2
        )
        self._command_linvel[:, :2] = command_speed * self._command_direction[:, :2]
        
        self.command[:, :2] = self._command_linvel[:, :2]
        self.command[:, 2] = self.command_angvel_yaw

    def sample_vel_command(self, env_ids: torch.Tensor):
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        stand = torch.rand(len(env_ids), device=self.device) < self.stand_prob
        speed = torch.zeros(len(env_ids), device=self.device).uniform_(*self.speed_range)
        speed = speed * (~stand).float()
        
        self._command_speed[env_ids] = speed.unsqueeze(1)
        self._command_direction[env_ids, 0] = a.cos()
        self._command_direction[env_ids, 1] = a.sin() * 0.6
    
    def sample_yaw_command(self, env_ids: torch.Tensor):
        yaw = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        self.target_yaw[env_ids] = yaw
        self._target_heading[env_ids, 0] = yaw.cos()
        self._target_heading[env_ids, 1] = yaw.sin()
        
        self._target_base_height[env_ids] = sample_uniform(
            env_ids.shape, *self.base_height_range, self.env.device
        ).unsqueeze(1)
        self.command[:, 3:4] = self._target_base_height


class Command2(Command):
    def __init__(
        self, 
        env, 
        linvel_x_range=(-1.0, 1.0),
        linvel_y_range=(-1.0, 1.0),
        angvel_range=(-1, 1),
        yaw_stiffness_range=(0.5, 0.6),
        use_stiffness_ratio: float = 0.5,
        base_height_range=(0.2, 0.4), 
        resample_interval: int = 300, 
        resample_prob: float = 0.75, 
        stand_prob=0.2,
        target_yaw_range=(-torch.pi, torch.pi),
    ):
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.angvel_range = angvel_range
        self.use_stiffness_ratio = use_stiffness_ratio
        self.yaw_stiffness_range = yaw_stiffness_range
        self.base_height_range = base_height_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.stand_prob = stand_prob

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

            self._command_speed = torch.zeros(self.num_envs, 1)
            self._target_direction = torch.zeros(self.num_envs, 3)
            self._target_linvel = torch.zeros(self.num_envs, 3)
            self._command_linvel = torch.zeros(self.num_envs, 3)
            self.command_angvel = torch.zeros(self.num_envs)

            self._target_base_height = torch.zeros(self.num_envs, 1)

            self._cum_error = torch.zeros(self.num_envs, 2)
            self._cum_linvel_error = self._cum_error[:, 0].unsqueeze(1)
            self._cum_angvel_error = self._cum_error[:, 1].unsqueeze(1)
        
    def reset(self, env_ids):
        self.command[env_ids] = 0.
        self.sample_vel_command(env_ids)
        self.sample_yaw_command(env_ids)
        self._cum_linvel_error[env_ids] = 0.
        self._cum_angvel_error[env_ids] = 0.
    
    def update(self):
        interval_reached = (self.env.episode_length_buf + 1) % self.resample_interval == 0
        resample_vel = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        resample_yaw = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        self.sample_vel_command(resample_vel.nonzero().squeeze(-1))
        self.sample_yaw_command(resample_yaw.nonzero().squeeze(-1))

        self.target_yaw[~self.use_stiffness] = self.robot.data.heading_w[~self.use_stiffness]
        yaw_diff = self.target_yaw - self.robot.data.heading_w
        command_yaw_speed = torch.clamp(
            self.yaw_stiffness * math_utils.wrap_to_pi(yaw_diff), 
            min=self.angvel_range[0],
            max=self.angvel_range[1]
        )
        self.command_angvel[:] = torch.where(self.use_stiffness, command_yaw_speed, self.fixed_yaw_speed)

        # this is used for terminating episodes where the robot is inactive due to whatever reason
        linvel_error = (self.robot.data.root_lin_vel_b[:, :2] - self.command[:, :2]).square().sum(-1, True)
        angvel_error = (self.command_angvel - self.robot.data.root_ang_vel_w[:, 2]).square().unsqueeze(1)
        
        self._cum_linvel_error[:] = self._cum_linvel_error * 0.98 + linvel_error * self.env.step_dt
        self._cum_angvel_error[:] = self._cum_angvel_error * 0.98 + angvel_error * self.env.step_dt
        self._command_linvel[:] = self._command_linvel + clamp_norm((self._target_linvel - self._command_linvel) * 0.1, max=0.1)

        self.command[:, :2] = self._command_linvel[:, :2]
        self.command[:, 2] = self.command_angvel
        self.command[:, 3] = 0 # self._distance_to_cover.squeeze(1)
        # self.command[:, :2] = torch.tensor([1.0, 0.], device=self.device)
    
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

        self._command_speed[env_ids] = speed
        self._target_direction[env_ids, :2] = direction
        self._target_linvel[env_ids, :2] = direction * speed
        self.is_standing_env[env_ids] = stand

    def sample_yaw_command(self, env_ids: torch.Tensor):
        self.target_yaw[env_ids] = self.target_yaw_dist.sample(env_ids.shape)
        self.yaw_stiffness[env_ids] = sample_uniform(env_ids.shape, *self.yaw_stiffness_range, self.device)
        self.use_stiffness[env_ids] = torch.rand(len(env_ids), device=self.device) < self.use_stiffness_ratio
        self.fixed_yaw_speed[env_ids] = sample_uniform(env_ids.shape, *self.angvel_range, self.device) 

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            quat_apply_yaw(self.robot.data.root_quat_w, self._command_linvel),
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


class CommandPosTracking(Command):

    def __init__(self, env) -> None:
        super().__init__(env)
        self.asset: Articulation = env.scene["robot"]
        with torch.device(self.device):
            self._command_linvel = torch.zeros(self.num_envs, 3)
            self._target_pos = torch.zeros(self.num_envs, 3)
            self._origins = torch.zeros(self.num_envs, 3)
            
            self.command = torch.zeros(self.num_envs, 6)
            self.d = torch.zeros(self.num_envs)
            self.omega = torch.zeros(self.num_envs)
            self.r = torch.zeros(self.num_envs)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.asset.data._tracking_error = torch.zeros(self.num_envs, 1)

    def update(self):
        t = self.omega * self.env.episode_length_buf * self.env.step_dt
        
        root_pos_w = self.asset.data.root_pos_w.clone()
        root_pos_w[:, 2] = 0.

        self._target_pos[: , :2] = self.trajeval(t)[:, :2]
        diff = self._target_pos - root_pos_w
        
        self.asset.data._tracking_error = diff.square().sum(-1, True)

        self.command[:, 0:2] = self.tobodyframe(diff)[:, :2]
        self.command[:, 2:4] = self.tobodyframe(self.trajeval(t + 0.1) - root_pos_w)[:, :2]
        self.command[:, 4:6] = self.tobodyframe(self.trajeval(t + 0.2) - root_pos_w)[:, :2]
    
    def trajeval(self, t: torch.Tensor):
        return (
            torch.stack([
                t.sin(), (t.cos() - 1) * self.d, torch.zeros_like(t)
            ], 1) * self.r.unsqueeze(1)
            + self._origins
        )
    
    def tobodyframe(self, vec):
        return quat_rotate(self.asset.data.root_quat_w, vec)

    def reset(self, env_ids: torch.Tensor):
        with torch.device(self.device):
            radius = torch.rand(len(env_ids), device=self.device) * 2.0 + 1.0
            self.r[env_ids] = radius
            speed = torch.rand(len(env_ids), device=self.device) * 0.5 + 0.6
            self.omega[env_ids] = speed / radius * torch.randn(len(env_ids)).sign()
            self.d[env_ids] = torch.randn(len(env_ids), device=self.device).sign()
            
        self._origins[env_ids, :2] = self.asset.data.root_pos_w[env_ids, :2]
    
    def debug_draw(self):
        robot_pos = self.asset.data.root_pos_w
        self.env.debug_draw.vector(
            robot_pos,
            self._target_pos - robot_pos
        )


class CommandPos(Command):

    def __init__(
        self, 
        env, 
        speed_range=(0.7, 1.4),
        offset_range=(2.0, 4.0),
        angvel_range=(-1.0, 1.0),
        resample_interval: int = 300,
        resample_prob: float = 0.75,
    ):
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.height_scanner = env.scene.sensors.get("height_scanner", None)
        self.speed_range = speed_range
        self.offset_range = offset_range
        self.angvel_range = angvel_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 4)

            self.target_yaw = torch.zeros(self.num_envs)
            self._target_speed = torch.zeros(self.num_envs, 1)
            self._target_pos_w = torch.zeros(self.num_envs, 3)

            self._command_speed = torch.zeros(self.num_envs, 1)
            self._command_linvel_b = torch.zeros(self.num_envs, 3)
            self._command_linvel_w = torch.zeros(self.num_envs, 3)
            self._command_linvel = self._command_linvel_b

            self.command_angvel = torch.zeros(self.num_envs)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self._cum_error = torch.zeros(self.num_envs, 1)
    
    def reset(self, env_ids):
        self.command[env_ids] = 0.
        self.sample_vel_command(env_ids)
        self.sample_yaw_command(env_ids)

    def update(self):
        interval_reached = (self.env.episode_length_buf + 1) % self.resample_interval == 0
        resample_vel = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        resample_yaw = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        self.sample_vel_command(resample_vel.nonzero().squeeze(-1))
        self.sample_yaw_command(resample_yaw.nonzero().squeeze(-1))

        pos_diff_xy = (self._target_pos_w - self.robot.data.root_pos_w)[:, :2]
        distance_xy = pos_diff_xy.norm(dim=-1, keepdim=True)
        direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        direction_w[:, :2] = pos_diff_xy / distance_xy.clamp(1e-6)
        direction_b = quat_rotate_inverse(self.robot.data.root_quat_w, direction_w)
        direction_b[:, 2] = 0.
        
        self._command_speed[:] = torch.minimum(distance_xy, self._target_speed)
        self.is_standing_env[:] = (self._command_speed < 0.2)
        self._command_linvel_w[:] = direction_w * self._command_speed
        self._command_linvel_b[:] = direction_b * self._command_speed

        # current_speed = (self.robot.data.root_lin_vel_w * direction_w).sum(-1, True)
        # command_speed = current_speed + (command_speed - current_speed).clamp_max(1.)

        yaw_diff = self.target_yaw - self.robot.data.heading_w
        self.command_angvel[:] = torch.clamp(
            0.6 * math_utils.wrap_to_pi(yaw_diff), 
            min=self.angvel_range[0],
            max=self.angvel_range[1]
        )

        self.command[:, :2] = self._command_linvel_b[:, :2]
        self.command[:, 2] = self.command_angvel
    
    def sample_vel_command(self, env_ids: torch.Tensor):
        robot_pos_w = self.robot.data.root_pos_w[env_ids, :2]
        d = sample_uniform(env_ids.shape, *self.offset_range, device=self.device)
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2.
        offset = torch.stack([a.cos(), a.sin()], 1) * d.unsqueeze(1)
        self._target_pos_w[env_ids, :2] = robot_pos_w + offset
        self._target_speed[env_ids, 0] = sample_uniform(env_ids.shape, *self.speed_range, self.device)

    def sample_yaw_command(self, env_ids: torch.Tensor):
        yaw = torch.rand(len(env_ids), device=self.device) * torch.pi * 2 - torch.pi
        self.target_yaw[env_ids] = yaw

    def debug_draw(self):
        start = self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device)
        # self.env.debug_draw.vector(
        #     start,
        #     self._command_linvel_w,
        #     color=(1., 1., 1., 1.),
        # )
        self.env.debug_draw.vector(
            start,
            quat_rotate(self.robot.data.root_quat_w, self._command_linvel_b),
            color=(1., .4, .4, 1.),
        )
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            torch.stack([self.target_yaw.cos(), self.target_yaw.sin(), torch.zeros_like(self.target_yaw)], 1),
            color=(.2, .2, 1., 1.)
        )
        self.env.debug_draw.point(self._target_pos_w, size=20)





def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def sample_quat_yaw(size, device: torch.device = "cpu"):
    yaw = torch.rand(size, device=device) * 2 * torch.pi
    # in (w x y z)
    quat = torch.cat([
        torch.cos(yaw / 2).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.sin(yaw / 2).unsqueeze(-1),
    ], dim=-1)
    return quat

def clamp_norm(x: torch.Tensor, min: float=0, max: float=torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    return x / x_norm * x_norm.clamp(min, max)

