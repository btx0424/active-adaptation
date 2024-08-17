from math import pi
import torch
import torch.distributions as D
import math
from typing import Sequence, TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.utils.math as math_utils
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, MultiUniform
from active_adaptation.utils.helpers import batchify
from omni.isaac.lab.utils.math import quat_apply_yaw, yaw_quat
from tensordict import TensorDict

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env

class Command:
    def __init__(self, env) -> None:
        self.env: Env = env
        self.asset: Articulation = env.scene["robot"]
        self.init_root_state = self.asset.data.default_root_state.clone()
        self.init_joint_pos = self.asset.data.default_joint_pos.clone()
        self.init_joint_vel = self.asset.data.default_joint_vel.clone()

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device
    
    def step(self, substep: int):
        pass

    def update(self):
        pass
    
    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        init_root_state = self.init_root_state[env_ids]
        if self.env.scene.terrain.cfg.terrain_type == "plane":
            origins = self.env.scene.env_origins[env_ids]
        else:
            origins = self.env.scene.env_origins[torch.randint(0, self.env.scene.num_envs, (len(env_ids),), device=self.device)]
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = sample_quat_yaw(len(env_ids), device=self.device)
        return init_root_state


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
            self._integrated_yaw = torch.zeros(env.num_envs, 3)

            self._command_direction = torch.zeros(env.num_envs, 3)
            self.command_speed = torch.zeros(env.num_envs, 1)
            self.command_linvel = torch.zeros(env.num_envs, 3)

            self._command_stand = torch.zeros(env.num_envs, 1, dtype=bool)
            self.command_angvel_yaw = torch.zeros(env.num_envs)
            
            self.command = torch.zeros(env.num_envs, self.command_dim)
        self.is_standing_env = self._command_stand
        self._command_heading = self._integrated_yaw

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

        command_speed = self.command_speed
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
            self.command_speed < 0.2
        )
        self.command_linvel[:, :2] = command_speed * self._command_direction[:, :2]
        
        self.command[:, :2] = self.command_linvel[:, :2]
        self.command[:, 2] = self.command_angvel_yaw

    def sample_vel_command(self, env_ids: torch.Tensor):
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        stand = torch.rand(len(env_ids), device=self.device) < self.stand_prob
        speed = torch.zeros(len(env_ids), device=self.device).uniform_(*self.speed_range)
        speed = speed * (~stand).float()
        
        self.command_speed[env_ids] = speed.unsqueeze(1)
        self._command_direction[env_ids, 0] = a.cos()
        self._command_direction[env_ids, 1] = a.sin() * 0.6
    
    def sample_yaw_command(self, env_ids: torch.Tensor):
        yaw = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        self.target_yaw[env_ids] = yaw
        self._integrated_yaw[env_ids, 0] = yaw.cos()
        self._integrated_yaw[env_ids, 1] = yaw.sin()
        
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
        aux_input_range=(0.2, 0.4), 
        resample_interval: int = 300, 
        resample_prob: float = 0.75, 
        stand_prob=0.2,
        target_yaw_range=(0, torch.pi * 2),
        adaptive: bool = False
    ):
        super().__init__(env)
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
        
    def reset(self, env_ids, reward_stats = None):
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
        linvel_error = (self.robot.data.root_lin_vel_b[:, :2] - self.command[:, :2]).norm(dim=-1, keepdim=True)
        angvel_error = (self.command_angvel - self.robot.data.root_ang_vel_w[:, 2]).abs().unsqueeze(1)
        
        if self.adaptive:
            self.ray_start_w = self.robot.data.root_pos_w + torch.tensor([0., 0., -0.2], device=self.device)
            ray_direction = quat_rotate(self.asset.data.root_quat_w, self._target_direction)
            ray_direction[:, 2] = 0.
            self.ray_hit_w = raycast_mesh(self.ray_start_w, ray_direction, max_dist=2, mesh=self.ground_mesh)[0]
            distance_to_obstacle = (self.ray_hit_w - self.ray_start_w).norm(dim=-1, keepdim=True).nan_to_num(2.0)
            self.close_to_obstacle = distance_to_obstacle < 0.75
            fast = self.command_speed > 1.4
            target_linvel = torch.where((self.close_to_obstacle & fast), self._target_linvel / 2, self._target_linvel)
        else:
            target_linvel = self._target_linvel

        self._cum_linvel_error.mul_(0.98).add_(linvel_error * self.env.step_dt)
        self._cum_angvel_error.mul_(0.98).add_(angvel_error * self.env.step_dt)
        self.command_linvel[:] = self.command_linvel + clamp_norm((target_linvel - self.command_linvel) * 0.1, max=0.1)

        self.command_linvel_w[:] = quat_apply_yaw(self.robot.data.root_quat_w, self.command_linvel)
        self.command[:, :2] = self.command_linvel[:, :2]
        self.command[:, 2] = self.command_angvel
        self.command[:, 3] = self.aux_input.squeeze(1)
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

        self.command_speed[env_ids] = speed
        self._target_direction[env_ids, :2] = direction
        self._target_linvel[env_ids, :2] = direction * speed
        self.is_standing_env[env_ids] = stand

        self.aux_input[env_ids] = sample_uniform(env_ids.shape, *self.aux_input_range, self.device).unsqueeze(1)

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
        if self.adaptive:
            self.env.debug_draw.vector(
                self.ray_start_w[self.close_to_obstacle.squeeze(1)],
                (self.ray_hit_w - self.ray_start_w)[self.close_to_obstacle.squeeze(1)],
                # self._target_direction,
                color=(1., 1., 0., 1.)
            )


from ..observations import _initialize_warp_meshes, raycast_mesh

class InterpCommand(Command):
    
    decimation: int = 2
    default_height: float = 0.35

    def __init__(
        self, 
        env,
        linvel_x_range=(-1.0, 1.0),
        angvel_range=(-2.0, 2.0),
    ):
        super().__init__(env)
        
        self.linvel_x_range = linvel_x_range
        self.angvel_range = angvel_range
        self.asset: Articulation = self.env.scene["robot"]
        self.ground_mesh = _initialize_warp_meshes("/World/ground", "cuda")

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 4)
            self.target_yaw = torch.zeros(self.num_envs)
            self.target_linvel = torch.zeros(self.num_envs, 3)
            self.command_angvel = torch.zeros(self.num_envs)
            self.command_linvel = torch.zeros(self.num_envs, 3)
            self.command_speed = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self._cum_error = torch.zeros(self.num_envs, 2)

            # path integration
            T = (self.env.max_episode_length // self.decimation) + 1
            self.integrated_pos_w = torch.zeros(self.num_envs, 3)
            self.integrated_pos_w_history = torch.zeros(self.num_envs, T, 3)
            self.integrated_yaw = torch.zeros(self.num_envs, 1)
            self.integrated_yaw_history = torch.zeros(self.num_envs, T, 1)
            self.integrated_step_mask = torch.zeros(self.num_envs, T, dtype=bool)

            self.pos_w_history = torch.zeros(self.num_envs, 3, T)
            self.yaw_history = torch.zeros(self.num_envs, T)
            
            # ray casting for getting height at a specific xy position
            self.ray_starts = torch.tensor([
                [0., 0., 10.], 
                [0.1, 0.1, 10.],
                [0.1, -.1, 10.],
                [-.1, -.1, 10.],
                [-.1, 0.1, 10.],
            ])
            self.ray_directions = torch.tensor([0., 0., -1.])

            # interpolation
            self.interp_pos = torch.zeros(self.num_envs, 3, T)
            self.interp_yaw = torch.zeros(self.num_envs, T)
            self.ep_count = torch.zeros(self.num_envs)
            self.use_interp = torch.zeros(self.num_envs, dtype=bool)
            self.last_init_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        
        self.DT = self.env.step_dt * self.decimation
        self.step_count = 0
        self.alpha = 0.5

        self.ep_library = []
        
    def reset(self, env_ids):
        self.command[env_ids] = 0.
        self.command_speed[env_ids] = 0.
        self.command_linvel[env_ids] = 0.

        self.sample_vel_command(env_ids)
        self.sample_yaw_command(env_ids)

        # store trajectory history
        
        self.ep_library.append(TensorDict({
            "pos_w_history": self.pos_w_history[env_ids],
            "yaw_history": self.yaw_history[env_ids],
        }, [len(env_ids), self.pos_w_history.shape]))
        
        self.interp_pos[env_ids] = self.pos_w_history[env_ids].lerp(self.integrated_pos_w_history[env_ids], self.alpha)
        yaw_diff = self.integrated_yaw_history[env_ids] - self.yaw_history[env_ids]
        self.interp_yaw[env_ids] = (self.yaw_history[env_ids] + self.alpha * math_utils.wrap_to_pi(yaw_diff))

        self.use_interp[env_ids] = (self.ep_count[env_ids] > 0)
        # reset path integration
        self.integrated_step_mask[env_ids] = False
        root_pos_w = self.asset.data.root_pos_w[env_ids]
        self.integrated_pos_w[env_ids] = root_pos_w
        self.integrated_pos_w_history[env_ids] = root_pos_w.unsqueeze(2)
        self.pos_w_history[env_ids] = root_pos_w.unsqueeze(2)

        root_yaw = self.asset.data.heading_w[env_ids]
        self.integrated_yaw[env_ids] = root_yaw
        self.integrated_yaw_history[env_ids] = root_yaw.unsqueeze(1)
        self.yaw_history[env_ids] = self.asset.data.heading_w[env_ids].unsqueeze(1)

        self.ep_count[env_ids] += 1
        
    def update(self):
        self.command_angvel[:] = torch.clamp(
            0.6 * math_utils.wrap_to_pi(self.target_yaw - self.asset.data.heading_w), 
            *self.angvel_range
        )
        self.command_linvel.add_(clamp_norm((self.target_linvel - self.command_linvel) * 0.1, max=0.1))
        self.command_speed[:] = self.command_linvel.norm(dim=-1, keepdim=True)

        n = torch.arange(self.num_envs, device=self.device)
        # t = self.env.episode_length_buf // self.decimation
        # tp1 = (t + 1).clamp_max(self.interp_pos.shape[2] - 1)
        # p_curr = self.interp_pos[n, :, t]
        # p_next = self.interp_pos[n, :, tp1]
        # switch to second order
        # v = quat_rotate_inverse(self.asset.data.root_quat_w, (p_next - p_curr) / self.DT)
        # w = math_utils.wrap_to_pi(self.interp_yaw[n, tp1] - self.interp_yaw[n, t]) / self.DT

        self.command[:, :2] = self.command_linvel[:, :2]
        self.command[:, 2] = self.command_angvel
        self.command[:, 3] = 0

        # self.command[:, :2] = torch.where(self.use_interp.unsqueeze(1), v[:, :2], self.command[:, :2])
        # self.command[:, 2] = torch.where(self.use_interp, w, self.command[:, 2])

        # path integration
        command_linvel_w = quat_rotate(quat_from_yaw(self.integrated_yaw), self.command_linvel)
        self.integrated_pos_w.add_(command_linvel_w * self.env.step_dt)
        self.integrated_pos_w[:, :2].lerp_(self.asset.data.root_pos_w[:, :2], 0.01)
        self.integrated_pos_w[:, 2] = self._height(self.integrated_pos_w)
        command_angvel = torch.clamp(
            0.6 * math_utils.wrap_to_pi(self.target_yaw - self.integrated_yaw), 
            *self.angvel_range
        )
        self.integrated_yaw.add_(command_angvel * self.env.step_dt)

        self._cum_error[:, 0] = (self.integrated_pos_w - self.asset.data.root_pos_w).norm(dim=-1)

        if self.step_count % self.decimation == 0:
            i = (self.env.episode_length_buf // self.decimation)
            n = torch.arange(self.num_envs, device=self.device)
            
            self.integrated_step_mask[n, i] = True

            self.integrated_pos_w_history[n, :, i] = self.integrated_pos_w
            self.pos_w_history[n, :, i] = self.asset.data.root_pos_w
            
            self.integrated_yaw_history[n, i] = self.integrated_yaw
            self.yaw_history[n, i] = self.asset.data.heading_w
        
        self.step_count += 1

    # def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
    #     if self.step_count < 1:
    #         init_root_state = super().sample_init(env_ids)
    #         self.last_init_root_state[env_ids] = init_root_state
    #         return init_root_state
    #     else:
    #         return self.last_init_root_state[env_ids]

    def sample_vel_command(self, env_ids: torch.Tensor):
        linvel = torch.zeros(len(env_ids), 2, device=self.device)
        linvel[:, 0].uniform_(*self.linvel_x_range)
        linvel[:, 1].uniform_(-0.1, 0.1)
        speed = linvel.norm(dim=-1, keepdim=True)
        self.is_standing_env[env_ids] = speed < 0.2
        self.target_linvel[env_ids, :2] = torch.where(speed < 0.2, linvel * 0., linvel)

    def sample_yaw_command(self, env_ids: torch.Tensor):
        self.target_yaw[env_ids] = 0.

    def _height(self, pos_w: torch.Tensor):
        ray_starts = self.ray_starts + pos_w.unsqueeze(1)
        ray_directions = self.ray_directions.expand_as(ray_starts).clone()
        ray_hits_w = raycast_mesh(ray_starts, ray_directions, max_dist=100, mesh=self.ground_mesh)[0]
        assert not ray_hits_w.isnan().any()
        height = ray_hits_w[:, :, 2].mean(1)
        return height + self.default_height
    
    def debug_draw(self):
        t = self.env.episode_length_buf // self.decimation
        root_pos_w = self.asset.data.root_pos_w
        for i in range(self.num_envs):
            if self.use_interp[i]:
                target_pos = self.interp_pos[i, :, :t[i].item()]
                self.env.debug_draw.plot(target_pos.T, color=(0., 1., 1., 1.))
                interp_yaw = self.interp_yaw[i, :t[i].item()]
                vec = torch.stack([torch.cos(interp_yaw), torch.sin(interp_yaw), torch.zeros_like(interp_yaw)], 1)
                self.env.debug_draw.vector(target_pos.T, vec, color=(0., 1., 1., .5))
            else:
                target_pos = self.integrated_pos_w_history[i, :, :t[i].item()]
                self.env.debug_draw.plot(target_pos.T, color=(0., 1., 1., 1.))
                integrated_yaw = self.integrated_yaw_history[i, :t[i].item()]
                vec = torch.stack([torch.cos(integrated_yaw), torch.sin(integrated_yaw), torch.zeros_like(integrated_yaw)], 1)
                self.env.debug_draw.vector(target_pos.T, vec, color=(0., 1., 1., .5))

            pos = self.pos_w_history[i, :, :t[i].item()]
            self.env.debug_draw.plot(pos.T, color=(1., 1., 0., 1.))
            
            # yaw = self.yaw_history[i, :t[i].item()]
            # interp_pos = pos.lerp(target_pos, self.alpha)
            # interp_yaw = yaw + self.alpha * math_utils.wrap_to_pi(integrated_yaw - yaw)
            # vec = torch.stack([torch.cos(interp_yaw), torch.sin(interp_yaw), torch.zeros_like(interp_yaw)], 1)
            # self.env.debug_draw.plot(interp_pos.T, color=(1., 0., 0., 1.))
            # self.env.debug_draw.vector(interp_pos.T, vec, color=(1., 0., 0., .5))
        
        self.env.debug_draw.vector(
            root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            quat_apply_yaw(self.asset.data.root_quat_w, self.command[:, :3]),
        )


class CommandPosVel(Command2):

    def __init__(
        self, 
        env, 
        linvel_x_range=(-1, 1), 
        linvel_y_range=(-1, 1), 
        angvel_range=(-1, 1), 
        yaw_stiffness_range=(0.5, 0.6), 
        use_stiffness_ratio: float = 0.5, 
        base_height_range=(0.2, 0.4), 
        resample_interval: int = 300, 
        resample_prob: float = 0.75, 
        stand_prob=0.2, 
        target_yaw_range=(0, torch.pi * 2)
    ):
        super().__init__(
            env, 
            linvel_x_range, 
            linvel_y_range, 
            angvel_range, 
            yaw_stiffness_range, 
            use_stiffness_ratio, 
            base_height_range, 
            resample_interval, 
            resample_prob, 
            stand_prob, 
            target_yaw_range
        )
        self.ground_mesh = _initialize_warp_meshes("/World/ground", "cuda")
        with torch.device(self.device):
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.ray_starts = torch.tensor([
                [0., 0., 10.], 
                [0.1, 0.1, 10.],
                [0.1, -.1, 10.],
                [-.1, -.1, 10.],
                [-.1, 0.1, 10.],
            ])
            self.ray_directions = torch.tensor([0., 0., -1.])
        self.default_height = 0.35

    def _height(self, pos_w: torch.Tensor):
        ray_starts = self.ray_starts + pos_w.unsqueeze(1)
        ray_directions = self.ray_directions.expand_as(ray_starts).clone()
        ray_hits_w = raycast_mesh(ray_starts, ray_directions, max_dist=100, mesh=self.ground_mesh)[0]
        assert not ray_hits_w.isnan().any()
        height = ray_hits_w[:, :, 2].mean(1)
        return height + self.default_height
    
    def reset(self, env_ids, reward_stats=None):
        super().reset(env_ids, reward_stats)
        self.target_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids]

    def update(self):
        super().update()
        command_linvel_w = quat_rotate(self.asset.data.root_quat_w, self.command_linvel)
        self.target_pos_w.add_(command_linvel_w * self.env.step_dt)
        self.target_pos_w[:, 2] = self._height(self.target_pos_w)
        self.target_pos_w.lerp_(self.asset.data.root_pos_w, 0.01)
    
    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.target_pos_w - self.asset.data.root_pos_w,
            color=(1., 1., 1., 1.)
        )


class Impedance(Command):
    
    future: int = 3
    
    def __init__(
        self, 
        env,
        linvel_x_range=(-1.0, 1.0),
        linvel_y_range=(-1.0, 1.0),
        angvel_range=(-2.0, 2.0),
        yaw_stiffness_range=(0.5, 0.5),
        virtual_mass_range=(0.5, 1.0),
        compliant_ratio: float = 0.2,
    ) -> None:
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.angvel_range = angvel_range
        self.yaw_stiffness_range = yaw_stiffness_range
        self.virtual_mass_range = virtual_mass_range
        self.resample_prob = 0.01
        self.compliant_ratio = compliant_ratio # kp=0 for compliant mode

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 6)
            self.command_hidden = torch.zeros(self.num_envs, 7)
            
            self.command_linvel = torch.zeros(self.num_envs, 3)
            self.command_speed = torch.zeros(self.num_envs, 1)

            self.command_pos_w = torch.zeros(self.num_envs, 3)
            self.command_linvel_w = torch.zeros(self.num_envs, 3)

            # integration
            self.desired_linacc_w = torch.zeros(self.num_envs, self.future, 3)
            self.desired_linvel_w = torch.zeros(self.num_envs, self.future, 3)
            self.desired_pos_w = torch.zeros(self.num_envs, self.future, 3)
            
            self.desired_yawacc_w = torch.zeros(self.num_envs, self.future, 1)
            self.desired_yawvel_w = torch.zeros(self.num_envs, self.future, 1)
            self.desired_yaw_w = torch.zeros(self.num_envs, self.future, 1)
            
            self.command_angvel = torch.zeros(self.num_envs)
            self.command_setpos_w = torch.zeros(self.num_envs, 3)
            self.command_setrpy_w = torch.zeros(self.num_envs, 3)

            self.kp = torch.zeros(self.num_envs, 1)
            self.kd = torch.zeros(self.num_envs, 1)

            self.default_mass = self.asset.root_physx_view.get_masses().sum(1, True).to(self.device)
            self.default_inertia = self.asset.root_physx_view.get_inertias()[:, 0, [0, 4, 8]].to(self.device)

            self.virtual_mass = torch.zeros(self.num_envs, 1)
            self.virtual_inertia = torch.zeros(self.num_envs, 3)

            self.force_ext_w = torch.zeros(self.num_envs, 3)
            self.torque_ext_w = torch.zeros(self.num_envs, 3)
            self.force_offset_b = torch.zeros(self.num_envs, 3)

            self._cum_error = torch.zeros(self.num_envs, 3)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.xy = torch.tensor([1., 1., 0.], device=self.device)
    
    def reset(self, env_ids: torch.Tensor):
        self._sample_command(env_ids)
        self._cum_error[env_ids] = 0.
    
    def step(self, substep: int):
        forces_b = self.asset._external_force_b.clone()
        forces_b[:, 0] += quat_rotate_inverse(self.asset.data.root_quat_w, self.force_ext_w)
        torques_b = self.asset._external_torque_b.clone()
        torques_b[:, 0] += self.force_offset_b.cross(forces_b[:, 0], dim=-1)
        self.asset.set_external_force_and_torque(forces_b, torques_b)

    def _integrate(self):
        desired_acc_w = (
            self.kp.unsqueeze(1) * (self.command_setpos_w.unsqueeze(1) - self.desired_pos_w) 
            + self.kd.unsqueeze(1) * (0. - self.desired_linvel_w)
            + (self.force_ext_w / self.virtual_mass).unsqueeze(1)
        ) # [n, t, 3]
        self.desired_linacc_w[:] = desired_acc_w * self.xy
        self.desired_linvel_w.add_(self.desired_linacc_w * self.env.physics_dt)
        self.desired_pos_w.add_(self.desired_linvel_w * self.env.physics_dt)

        force_offset_w = yaw_rotate(self.desired_yaw_w, self.force_offset_b.unsqueeze(1))
        torque = torch.cross(force_offset_w, self.force_ext_w.unsqueeze(1), dim=-1)
        
        yaw_diff = self.command_setrpy_w[:, 2:3].unsqueeze(1) - self.desired_yaw_w
        desired_yaw_acc_w = (
            self.kp.unsqueeze(1) * math_utils.wrap_to_pi(yaw_diff)
            + self.kd.unsqueeze(1) * (0. - self.desired_yawvel_w)
            + (torque / self.virtual_inertia.unsqueeze(1))[..., 2:3]
        )
        self.desired_yawacc_w[:] = desired_yaw_acc_w
        self.desired_yawvel_w.add_(desired_yaw_acc_w * self.env.physics_dt)
        self.desired_yaw_w.add_(desired_yaw_acc_w * self.env.physics_dt)

    def update(self):
        self.desired_linacc_w.roll(1, dims=1)
        self.desired_linvel_w.roll(1, dims=1)
        self.desired_pos_w.roll(1, dims=1)
        self.desired_yawacc_w.roll(1, dims=1)
        self.desired_yawvel_w.roll(1, dims=1)
        self.desired_yaw_w.roll(1, dims=1)
        
        self.desired_linvel_w[:, 0] = self.asset.data.root_lin_vel_w
        self.desired_pos_w[:, 0] = self.asset.data.root_pos_w
        self.desired_yawvel_w[:, 0] = self.asset.data.root_ang_vel_w[:, 2:3]
        self.desired_yaw_w[:, 0] = self.asset.data.heading_w.unsqueeze(1)
        
        for _ in range(4):
            self._integrate()

        self.command_linvel_w[:] = self.desired_linvel_w.mean(1)
        self.command_angvel[:] = self.desired_yawvel_w.mean(1).squeeze(-1)
        self.command_pos_w[:] = self.desired_pos_w.mean(1)

        linvel_error = (self.command_linvel_w - self.asset.data.root_lin_vel_w).norm(dim=-1)
        pos_error = (self.command_pos_w - self.asset.data.root_pos_w).norm(dim=-1)
        angvel_error = (self.command_angvel - self.asset.data.root_ang_vel_w[:, 2]).abs()
        self._cum_error[:, 0].add_(linvel_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 1].add_(pos_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 2].add_(angvel_error * self.env.step_dt).mul_(0.99)

        command_pos_b = quat_rotate_inverse(self.asset.data.root_quat_w, self.command_pos_w - self.asset.data.root_pos_w)
        command_setpos_b = quat_rotate_inverse(self.asset.data.root_quat_w, self.command_setpos_w - self.asset.data.root_pos_w)

        self.command_linvel[:] = quat_rotate_inverse(self.asset.data.root_quat_w, self.command_linvel_w)
        self.command_speed[:] = self.command_linvel.norm(dim=-1, keepdim=True)
        
        self.command[:, :2] = command_setpos_b[:, :2]
        self.command[:, 2] = math_utils.wrap_to_pi(self.command_setrpy_w[:, 2] - self.asset.data.heading_w)
        self.command[:, 3:4] = self.kp
        self.command[:, 4:5] = self.kd
        self.command[:, 5:6] = self.virtual_mass

        self.command_hidden[:, 0:3] = command_pos_b
        self.command_hidden[:, 3:6] = self.command_linvel
        self.command_hidden[:, 6] = self.command_angvel

        _ = torch.rand(self.num_envs, device=self.device) < self.resample_prob
        self._sample_command(_.nonzero().squeeze(-1))
        _ = torch.rand(self.num_envs, device=self.device) < self.resample_prob
        self._sample_force(_.nonzero().squeeze(-1))
    
    def _sample_command(self, env_ids: torch.Tensor):
        command_setpoint_w = torch.zeros(len(env_ids), 3, device=self.device)
        command_setpoint_w[:, 0].uniform_(2., 3.)
        command_setpoint_w[:, 1].uniform_(-1, 1)
        command_setpoint_w.add_(self.asset.data.root_pos_w[env_ids])
        
        kp = torch.empty(len(env_ids), 1, device=self.device).uniform_(2.0, 3.0)
        kd = 2.0 * kp.sqrt()    # to make the system critically damped
        compliant = torch.rand(len(env_ids), device=self.device) < self.compliant_ratio
        kp *= (~compliant).unsqueeze(1)

        self.command_setpos_w[env_ids] = command_setpoint_w
        self.kp[env_ids] = kp
        self.kd[env_ids] = kd

        self.desired_linacc_w[env_ids] = 0.
        self.desired_linvel_w[env_ids] = self.asset.data.root_lin_vel_w[env_ids].unsqueeze(1)
        self.desired_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids].unsqueeze(1)

        self.command_setrpy_w[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * torch.pi * 2

        virtual_mass = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.virtual_mass_range)
        self.virtual_mass[env_ids] = self.default_mass[env_ids] * virtual_mass
        self.virtual_inertia[env_ids] = self.default_inertia[env_ids] * virtual_mass

    def _sample_force(self, env_ids: torch.Tensor):
        force_ext_w = torch.zeros(len(env_ids), 3, device=self.device)
        force_ext_w[:, 0].uniform_(-40, 40)
        force_ext_w[:, 1].uniform_(-40, 40)
        force_ext_w[:, 2].uniform_(-10, 10)
        force_ext_w = clamp_norm(force_ext_w, max=self.virtual_mass[env_ids] * 8.)
        self.force_ext_w[env_ids] = force_ext_w * (torch.rand(len(env_ids), 1, device=self.device) < 0.5)

        force_offset_b = torch.zeros(len(env_ids), 3, device=self.device)
        force_offset_b[:, 0].uniform_(-0.3, 0.3)
        force_offset_b[:, 1].uniform_(-0.2, 0.2)
        self.force_offset_b[env_ids] = force_offset_b

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            self.command_linvel_w,
            color=(1., 1., 1., 1.)
        )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.command_setpos_w - self.asset.data.root_pos_w,
            color=(1., 0., 0., 1.)
        )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            yaw_rotate(self.command_setrpy_w[:, 2:3], torch.tensor([[1., 0., 0.]], device=self.device)),
            color=(1., 0., 1., 1.)
        )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + yaw_rotate(self.asset.data.heading_w.unsqueeze(1), self.force_offset_b),
            self.force_ext_w / self.virtual_mass,
            color=(0., 1., 0., 1.),
            size=2.0
        )
        # self.env.debug_draw.vector(
        #     self.asset.data.root_pos_w,
        #     quat_rotate(self.asset.data.root_quat_w, self.asset._external_torque_b[:, 0]),
        #     color=(1., 1., 1., 1.)
        # )


def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def sample_quat_yaw(size, yaw_range = (0, torch.pi * 2), device: torch.device = "cpu"):
    yaw = torch.rand(size, device=device).uniform_(*yaw_range)
    quat = torch.cat([
        torch.cos(yaw / 2).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.sin(yaw / 2).unsqueeze(-1),
    ], dim=-1)
    return quat


def quat_from_yaw(yaw: torch.Tensor):
    return torch.cat([
        torch.cos(yaw / 2).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.sin(yaw / 2).unsqueeze(-1),
    ], dim=-1)

@batchify
def yaw_rotate(yaw: torch.Tensor, vec: torch.Tensor):
    yaw_cos = torch.cos(yaw).squeeze(-1)
    yaw_sin = torch.sin(yaw).squeeze(-1)
    return torch.stack([
        yaw_cos * vec[:, 0] - yaw_sin * vec[:, 1],
        yaw_sin * vec[:, 0] + yaw_cos * vec[:, 1],
        vec[:, 2]
    ], 1)


def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x

