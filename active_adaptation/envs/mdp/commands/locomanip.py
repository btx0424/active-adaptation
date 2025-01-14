import torch
import einops
import warp as wp

from torchrl.data import UnboundedContinuousTensorSpec
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat, quat_from_euler_xyz, quat_mul, quat_inv, euler_xyz_from_quat, normalize
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify
from .locomotion import Command, sample_quat_yaw, sample_uniform, clamp_norm
from tensordict import TensorDict
from .generate_command_traj import generate_random_trajectories

from typing import Dict, Optional
import wandb

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wraps input angles (in radians) to the range [-pi, pi].

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles in the range [-pi, pi].
    """
    angles = angles.clone()
    angles = angles % (2 * torch.pi)
    angles -= 2 * torch.pi * (angles > torch.pi)
    return angles

@batchify
def yaw_rotate(yaw: torch.Tensor, vec: torch.Tensor):
    yaw_cos = torch.cos(yaw).squeeze(-1)
    yaw_sin = torch.sin(yaw).squeeze(-1)
    return torch.stack(
        [
            yaw_cos * vec[:, 0] - yaw_sin * vec[:, 1],
            yaw_sin * vec[:, 0] + yaw_cos * vec[:, 1],
            vec[:, 2],
        ],
        1,
    )
def slerp_angle(angle1, angle2, t):
    """Spherical linear interpolation for angles."""
    delta = torch.remainder(angle2 - angle1 + torch.pi, 2 * torch.pi) - torch.pi
    return angle1 + t * delta

def slerp_angle_with_limit(angle1, angle2, t, limit):
    """Spherical linear interpolation for angles with consideration of joint limits."""
    # clip angle1 and angle2 to the range of limit
    lower_limit, upper_limit = limit
    angle1 = angle1.clamp(lower_limit, upper_limit)
    angle2 = angle2.clamp(lower_limit, upper_limit)

    return angle1 + t * (angle2 - angle1)

import pickle
import numpy as np
from tqdm import tqdm

def pitch_yaw_to_vec(pitch, yaw):
    """Convert pitch and yaw to 3D unit vectors."""
    return np.concatenate([
        np.cos(yaw) * np.cos(pitch),
        np.sin(yaw) * np.cos(pitch),
        -np.sin(pitch)
    ], axis=-1)

def vec_to_pitch_yaw(vec):
    """Convert 3D unit vectors to pitch and yaw."""
    pitch = -np.arcsin(vec[..., 2])
    yaw = np.arctan2(vec[..., 1], vec[..., 0])
    return pitch, yaw

def slerp_vec(v1, v2, t):
    """Spherical linear interpolation for vectors."""
    dot = np.sum(v1 * v2, axis=-1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    mask = sin_theta > 1e-6
    t1 = np.where(mask, np.sin((1 - t) * theta) / sin_theta, 1 - t)
    t2 = np.where(mask, np.sin(t * theta) / sin_theta, t)

    result = v1 * t1 + v2 * t2
    return result / np.linalg.norm(result, axis=-1, keepdims=True)

def slerp_angle(a1, a2, t):
    """Spherical linear interpolation for angles."""
    return a1 + t * (a2 - a1)

def slerp_angle_with_limit(a1, a2, t, angle_range):
    """Spherical linear interpolation for angles with range limit."""
    a1 = np.clip(a1, *angle_range)
    a2 = np.clip(a2, *angle_range)
    return a1 + t * (a2 - a1)

def interpolate_position(last_pos, next_pos, t, yaw_range):
    last_r, last_pitch, last_yaw = np.split(last_pos, 3, axis=-1)
    next_r, next_pitch, next_yaw = np.split(next_pos, 3, axis=-1)

    current_r = last_r + t * (next_r - last_r)
    current_pitch = slerp_angle(last_pitch, next_pitch, t)
    current_yaw = slerp_angle_with_limit(last_yaw, next_yaw, t, yaw_range)
    
    return current_r, current_pitch, current_yaw

def interpolate_orientation(last_ori, next_ori, t):
    last_roll, last_ori_pitch, last_ori_yaw = np.split(last_ori, 3, axis=-1)
    next_roll, next_ori_pitch, next_ori_yaw = np.split(next_ori, 3, axis=-1)

    current_roll = last_roll + t * (next_roll - last_roll)
    last_ori_dir = pitch_yaw_to_vec(last_ori_pitch, last_ori_yaw)
    next_ori_dir = pitch_yaw_to_vec(next_ori_pitch, next_ori_yaw)
    current_ori_dir = slerp_vec(last_ori_dir, next_ori_dir, t[:, np.newaxis])
    current_ori_pitch, current_ori_yaw = vec_to_pitch_yaw(current_ori_dir)

    return current_roll, current_ori_pitch, current_ori_yaw

if __name__ == "__main__":
    sampling_rate = 50
    parsed_plan = []
    yaw_range = [-2.5, 2.5]
    pitch_range = [-1.5, 0.5]
    radius_range = [0.4, 0.8]
    ee_lin_vel_bounds = np.array([0.05, 2.0])
    episode_len = 300
    num_waypoints = 30
    num_trajectories = 40

    for _ in tqdm(range(num_trajectories)):
        last_pos_waypoint = np.column_stack((
            np.random.uniform(*radius_range),
            np.random.uniform(*pitch_range),
            np.random.uniform(*yaw_range)
        ))
        last_ori_waypoint = np.column_stack((
            0.0,
            last_pos_waypoint[0, 1] + np.random.uniform(-np.pi/4, np.pi/4),
            last_pos_waypoint[0, 2] + np.random.uniform(-np.pi/4, np.pi/4)
        ))

        ee_pos = []
        ee_forward = []
        
        while sum([pos.shape[0] for pos in ee_pos]) < episode_len:
            # generate a new waypoint
            new_pos_waypoint = np.column_stack((
                np.random.uniform(*radius_range),
                np.random.uniform(*pitch_range),
                np.random.uniform(*yaw_range)
            ))
            new_ori_waypoint = np.column_stack((
                0.0,
                new_pos_waypoint[0, 1] + np.random.uniform(-np.pi/4, np.pi/4),
                new_pos_waypoint[0, 2] + np.random.uniform(-np.pi/4, np.pi/4)
            ))
            
            distance = np.linalg.norm(new_pos_waypoint - last_pos_waypoint)
                
            time = distance / np.random.uniform(*ee_lin_vel_bounds)
            try:
                n_steps = int(time * sampling_rate)
            except ZeroDivisionError:
                breakpoint()
                continue
            
            if n_steps < 20:
                continue

            t = np.linspace(1 / n_steps, 1, n_steps)[:, None]
            
            r, pitch, yaw = interpolate_position(last_pos_waypoint, new_pos_waypoint, t, yaw_range)
            pos = np.concatenate((
                r * np.cos(yaw) * np.cos(pitch),
                r * np.sin(yaw) * np.cos(pitch),
                -r * np.sin(pitch)
            ), axis=-1)
            ee_pos.append(pos)
            
            _, ori_pitch, ori_yaw = interpolate_orientation(last_ori_waypoint, new_ori_waypoint, t)
            forward = pitch_yaw_to_vec(ori_pitch, ori_yaw)
            ee_forward.append(forward)
            
            last_pos_waypoint = new_pos_waypoint
            last_ori_waypoint = new_ori_waypoint
        
        ee_pos = np.concatenate(ee_pos)[:episode_len]
        ee_forward = np.concatenate(ee_forward)[:episode_len]
        t = np.linspace(0, episode_len / sampling_rate, episode_len + 1)

        parsed_plan.append({
            "t": t,
            "pos": ee_pos,
            "forward": ee_forward,
        })

    pickle.dump(parsed_plan, open("command_traj.pkl", "wb"))
def rotation_angle_from_matrix(r_matrix):
    trace = torch.diagonal(r_matrix, dim1=-2, dim2=-1).sum(dim=-1)
    trace.clamp_(-1. + 1e-8, 3. - 1e-8)
    rotation_magnitude = torch.acos((trace - 1.) / 2.)
    assert not torch.isnan(rotation_magnitude).any()
    assert torch.all(rotation_magnitude >= 0.)
    assert torch.all(rotation_magnitude <= torch.pi)
    return rotation_magnitude


class CommandEEPose(Command):
    def __init__(
        self, 
        env,
        ee_name: str,
        yaw_range: tuple = (-torch.pi, torch.pi),
        pitch_range: tuple = (-torch.pi / 3, 0.),
        angvel_range=(-1., 1.),
    ) -> None:
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
        assert len(self.ee_id) == 1
        self.ee_id = self.ee_id[0]

        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.angvel_range = angvel_range
        self.resample_interval = 300
        self.resample_prob = 0.5

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 2 + 1 + 3 + 3)
            self.target_yaw = torch.zeros(self.num_envs)
            self.command_linvel = torch.zeros(self.num_envs, 3)
            self.command_speed = torch.zeros(self.num_envs, 1)
            self.command_angvel = torch.zeros(self.num_envs)

            self.command_ee_pos_b = torch.zeros(self.num_envs, 3)
            self.command_ee_quat_b = torch.zeros(self.num_envs, 4)
            self.command_ee_quat_w = torch.zeros(self.num_envs, 4)
            self.command_ee_forward_w = torch.zeros(self.num_envs, 3)
            self.command_ee_upward_w = torch.zeros(self.num_envs, 3)

            self.fwd_vec = torch.tensor([1., 0., 0.]).expand(self.num_envs, -1)
            self.up_vec = torch.tensor([0., 0., 1.]).expand(self.num_envs, -1)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

    def reset(self, env_ids: torch.Tensor):
        self.command[env_ids] = 0.
        self.sample_ee(env_ids)
        self.sample_yaw(env_ids)
    
    def update(self):
        interval_reached = (self.env.episode_length_buf + 1) % self.resample_interval == 0
        resample_ee = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        self.sample_ee(resample_ee.nonzero().squeeze(-1))

        root_quat_w = self.asset.data.root_quat_w
        self.command_ee_quat_w[:] = quat_mul(root_quat_w, self.command_ee_quat_b)
        self.command_ee_forward_w[:] = quat_rotate(self.command_ee_quat_w, self.fwd_vec)
        self.command_ee_upward_w[:] = quat_rotate(self.command_ee_quat_w, self.up_vec)

        ee_forward_b = quat_rotate_inverse(self.asset.data.root_quat_w, self.command_ee_forward_w)
        
        self.is_standing_env[:] = True

        yaw_diff = self.target_yaw - self.asset.data.heading_w
        self.command_angvel[:] = torch.clamp(
            0.6 * wrap_to_pi(yaw_diff), 
            min=self.angvel_range[0],
            max=self.angvel_range[1]
        )

        self.command[:, 2] = self.command_angvel
        self.command[:, 3:6] = self.command_ee_pos_b
        self.command[:, 6:9] = ee_forward_b

    def sample_ee(self, env_ids: torch.Tensor):        
        yaw = torch.zeros(env_ids.shape, device=self.device).uniform_(*self.yaw_range)
        pitch = torch.zeros(env_ids.shape, device=self.device).uniform_(*self.pitch_range)
        radius = torch.zeros(env_ids.shape, device=self.device).uniform_(0.4, 0.8)
        self.command_ee_pos_b[env_ids] = torch.stack([
            radius * torch.cos(yaw) * torch.cos(pitch),
            radius * torch.sin(yaw) * torch.cos(pitch),
            radius * -torch.sin(pitch),
        ], dim=-1)

        ee_roll = torch.zeros_like(yaw)
        ee_yaw = yaw + torch.empty_like(yaw).uniform_(-torch.pi/4, torch.pi/4)
        ee_pitch = pitch + torch.empty_like(pitch).uniform_(-torch.pi/4, torch.pi/4)
        self.command_ee_quat_b[env_ids] = quat_from_euler_xyz(ee_roll, ee_pitch, ee_yaw)

    def sample_yaw(self, env_ids: torch.Tensor):
        self.target_yaw[env_ids] = torch.rand(env_ids.shape, device=self.device) * 2 * torch.pi

    def debug_draw(self):
        ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id]
        ee_pos_target = quat_rotate(
            yaw_quat(self.asset.data.root_quat_w),
            self.command_ee_pos_b
        ) + self.asset.data.root_pos_w

        self.env.debug_draw.vector(
            ee_pos_w,
            self.command_ee_forward_w * 0.2,
        )
        self.env.debug_draw.vector(
            ee_pos_w,
            self.command_ee_upward_w * 0.2,
        )
        self.env.debug_draw.vector(
            ee_pos_w,
            ee_pos_target - ee_pos_w,
            color=(0., 0.8, 0., 1.)
        )

# track a locomotion command and a ee pose command, the locomotion command and ee pose command is sampled in the body frame and fixed during the resample interval
class CommandEEPose_Loco(Command):
    def __init__(
        self, 
        env,
        ee_name: str,
        # lin vel in x, y, ang vel
        lin_vel_x_range=(-1.0, 1.4),
        lin_vel_y_range=(-0.5, 0.5),
        ang_vel_range: tuple = (-1., 1.),
        # ee pose
        yaw_range: tuple = (-torch.pi, torch.pi),
        pitch_range: tuple = (-torch.pi / 3, 0.),
        radius_range: tuple =(0.4, 0.8),
        command_only_yaw_quat: bool = False, # this is false for old experiments
    ) -> None:
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
        assert len(self.ee_id) == 1
        self.ee_id = self.ee_id[0]

        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_range = ang_vel_range
        
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.radius_range = radius_range

        self.command_only_yaw_quat = command_only_yaw_quat
        
        self.resample_interval = 300
        self.resample_prob = 0.5

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 2 + 1 + 3 + 4)

            # Locomotion
            self.command_lin_vel = torch.zeros(self.num_envs, 3)
            self._command_speed = torch.zeros(self.num_envs, 1)
            self.command_ang_vel = torch.zeros(self.num_envs)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            # Manipulation
            ## to give command; to compute world frame targets
            self.command_ee_pos_b = torch.zeros(self.num_envs, 3)
            self.command_ee_quat_b = torch.zeros(self.num_envs, 4)
            
            ## world frame targets for reward computation
            self.command_ee_pos_w = torch.zeros(self.num_envs, 3)
            self.command_ee_quat_w = torch.zeros(self.num_envs, 4)
            self.fwd_vec = torch.tensor([1., 0., 0.]).expand(self.num_envs, -1)
            self.up_vec = torch.tensor([0., 0., 1.]).expand(self.num_envs, -1)
            self.command_ee_forward_w = torch.zeros(self.num_envs, 3)
            self.command_ee_upward_w = torch.zeros(self.num_envs, 3)

            self.ee_pos_b = self.asset.data.ee_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
            self.command_ee_pos_b_yaw = self.asset.data.command_ee_pos_b_yaw = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor, reward_stats: TensorDict = None):
        self.command[env_ids] = 0.
        self.sample_ee(env_ids)
        self.sample_loco(env_ids)
    
    def update(self):
        root_quat_w = self.asset.data.root_quat_w
        root_quat_w_yaw = yaw_quat(root_quat_w)
        self.ee_pos_b[:] = quat_rotate_inverse(
            root_quat_w_yaw, 
            self.asset.data.body_pos_w[:, self.ee_id] - self.asset.data.root_pos_w
        )
        self.command_ee_pos_b_yaw[:] = torch.atan2(self.command_ee_pos_b[:, 1], self.command_ee_pos_b[:, 0])
        
        interval_reached = (self.env.episode_length_buf + 1) % self.resample_interval == 0
        resample_ids = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        self.sample_ee(resample_ids.nonzero().squeeze(-1))
        self.sample_loco(resample_ids.nonzero().squeeze(-1))

        quat = root_quat_w_yaw if self.command_only_yaw_quat else root_quat_w
        self.command_ee_pos_w[:] = quat_rotate(
            quat,
            self.command_ee_pos_b,
        ) + self.asset.data.root_pos_w

        self.command_ee_quat_w[:] = quat_mul(quat, self.command_ee_quat_b)
        self.command_ee_forward_w[:] = quat_rotate(self.command_ee_quat_w, self.fwd_vec)
        self.command_ee_upward_w[:] = quat_rotate(self.command_ee_quat_w, self.up_vec)
        
        self.command[:, :2] = self.command_lin_vel[:, :2]
        self.command[:, 2] = self.command_ang_vel
        self.command[:, 3:6] = self.command_ee_pos_b.view(self.num_envs, -1)
        self.command[:, 6:] = self.command_ee_quat_b.view(self.num_envs, -1)

    def sample_ee(self, env_ids: torch.Tensor):        
        yaw = torch.zeros(env_ids.shape, device=self.device).uniform_(*self.yaw_range)
        pitch = torch.zeros(env_ids.shape, device=self.device).uniform_(*self.pitch_range)
        radius = torch.zeros(env_ids.shape, device=self.device).uniform_(*self.radius_range)
        self.command_ee_pos_b[env_ids] = torch.stack([
            radius * torch.cos(yaw) * torch.cos(pitch),
            radius * torch.sin(yaw) * torch.cos(pitch),
            radius * -torch.sin(pitch),
        ], dim=-1)

        ee_roll = torch.zeros_like(yaw)
        ee_yaw = yaw + torch.empty_like(yaw).uniform_(-torch.pi/4, torch.pi/4)
        ee_pitch = pitch + torch.empty_like(pitch).uniform_(-torch.pi/4, torch.pi/4)
        self.command_ee_quat_b[env_ids] = quat_from_euler_xyz(ee_roll, ee_pitch, ee_yaw)

    def sample_loco(self, env_ids: torch.Tensor):
        # sample speed and direction
        linvel = torch.zeros(len(env_ids), 2, device=self.device)
        linvel[:, 0].uniform_(*self.lin_vel_x_range)
        linvel[:, 1].uniform_(*self.lin_vel_y_range)
        speed = linvel.norm(dim=-1, keepdim=True)
        stand = speed < 0.3
        speed = speed * (~stand)
        self.command_lin_vel[env_ids, :2] = linvel
        self._command_speed[env_ids] = speed
        self.is_standing_env[env_ids] = stand

        # sample angvel
        self.command_ang_vel[env_ids] = torch.empty(env_ids.shape, device=self.device).uniform_(*self.ang_vel_range)

    def debug_draw(self):
        ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id]
        ee_pos_target = quat_rotate(
            yaw_quat(self.asset.data.root_quat_w),
            self.command_ee_pos_b,
        ) + self.asset.data.root_pos_w

        self.env.debug_draw.vector(
            ee_pos_w,
            self.command_ee_forward_w * 0.2,
        )
        self.env.debug_draw.vector(
            ee_pos_w,
            self.command_ee_upward_w * 0.2,
        )
        self.env.debug_draw.vector(
            ee_pos_w,
            ee_pos_target - ee_pos_w,
            color=(0., 0.8, 0., 1.)
        )
        # draw command lin_vel in purple, real lin vel in blue
        root_quat_w = self.asset.data.root_quat_w
        command_lin_vel_w = quat_rotate(root_quat_w, self.command_lin_vel)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.1], device=self.device),
            command_lin_vel_w,
            color=(1., 0., 1., 1.)
        )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.1], device=self.device),
            self.asset.data.root_lin_vel_w,
            color=(0., 0., 1., 1.)
        )

class CommandEEPose_Cont(Command):
    """For each environment, sample a target to reach in the next 300 steps.
    Set the command as the linear interpolation between the current and target pose, command updated every 10 steps.
    When to sample a new target:
    1. At the beginning of the episode (reset)
    2. Every 300 steps
    3. If the robot fails to track the command within some error `threshold` for `tolerance` * 10 steps
    """
    def __init__(
        self, 
        env,
        ee_name: str,
        ee_base_name: str = 'arm_link00',
        # lin vel in x, y, ang vel
        lin_vel_x_range: tuple =(-1.0, 1.4),
        lin_vel_y_range: tuple =(-0.5, 0.5),
        ang_vel_range: tuple = (-1., 1.),
        # ee pose
        yaw_range: tuple = (-torch.pi, torch.pi),
        pitch_range: tuple = (-torch.pi / 3, 0.),
        radius_range: tuple =(0.4, 0.8),
        # update/resample
        update_interval: int = 10,
        resample_interval: int =300,
        threshold: float = 0.5,
        tolerance: int = 4,
        future_targets: int = 4,
        imagined_root: bool = True,
    ) -> None:
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        ee_ids, ee_names = self.asset.find_bodies(ee_name)
        assert len(ee_ids) == 1
        self.ee_id = ee_ids[0]
        ee_base_ids, ee_base_names = self.asset.find_bodies(ee_base_name)
        assert len(ee_base_ids) == 1
        self.ee_base_id = ee_base_ids[0]

        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_range = ang_vel_range
        
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.radius_range = radius_range
        
        self.update_interval = update_interval
        resample_interval = ((resample_interval + update_interval - 1) // update_interval) * update_interval
        self.resample_interval = resample_interval
        self.updates_per_resample = resample_interval // update_interval
        self.threshold = threshold
        self.tolerance = tolerance
        
        self.imagined_root = imagined_root

        self.last_env_reset_ids = None

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 2 + 1 + 3 * future_targets + 4 * future_targets)

            # Locomotion
            self.command_lin_vel = torch.zeros(self.num_envs, 3)
            self._command_speed = torch.zeros(self.num_envs, 1)
            self.command_ang_vel = torch.zeros(self.num_envs)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            # Manipulation
            ## for interpolation, give command and to compute world frame targets
            self.command_ee_pos_b = torch.zeros(self.num_envs, future_targets, 3)
            self.command_ee_quat_b = torch.zeros(self.num_envs, future_targets, 4)
            
            ## world frame targets for reward computation
            self.command_ee_pos_w_ = torch.zeros(self.num_envs, future_targets, 3)
            self.command_ee_pos_w = self.command_ee_pos_w_[:, 0]

            self.command_ee_quat_w = torch.zeros(self.num_envs, 4)
            self.fwd_vec = torch.tensor([1., 0., 0.]).expand(self.num_envs, -1)
            self.up_vec = torch.tensor([0., 0., 1.]).expand(self.num_envs, -1)
            self.command_ee_forward_w = torch.zeros(self.num_envs, 3)
            self.command_ee_upward_w = torch.zeros(self.num_envs, 3)

            # maintain world frame heading and root pos if command is perfectly tracked
            if self.imagined_root:
                self.command_heading_yaw_w = torch.zeros(self.num_envs)
                self.command_root_pos_w = torch.zeros(self.num_envs, 3)

                # get the offset of arm base link relative to root in base frame
                arm_base_offset_w = self.asset.data.body_pos_w[:, self.ee_base_id] - self.asset.data.root_pos_w
                root_quat_w = self.asset.data.root_quat_w
                self.arm_base_offset_b = quat_rotate_inverse(root_quat_w, arm_base_offset_w)

            # ee pose interpolation
            self.last_target_pos_radius_pitch_yaw = torch.zeros(self.num_envs, 3)
            self.last_target_ori_roll_pitch_yaw = torch.zeros(self.num_envs, 3)
            self.next_target_pos_radius_pitch_yaw = torch.zeros(self.num_envs, 3)
            self.next_target_ori_roll_pitch_yaw = torch.zeros(self.num_envs, 3)
            
            self.steps_since_last_sample = torch.zeros(self.num_envs, dtype=torch.long)
            self.steps = torch.arange(1, future_targets + 1, device=self.device).expand(self.num_envs, -1) # [num_envs, future_targets]
            self.future_targets = future_targets

            self.failed_track_counts = torch.zeros(self.num_envs, dtype=torch.long)
            
            self._cum_error = torch.zeros(self.num_envs, 2)
            self._cum_ee_pos_error = self._cum_error[:, 0]
            self._cum_ee_ori_error = self._cum_error[:, 1]
            
            self.ee_pos_b = self.asset.data.ee_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
            self.command_ee_pos_b_yaw = self.asset.data.command_ee_pos_b_yaw = torch.zeros(self.num_envs, device=self.device)
        
        self.debug_draw_dict = {}
        self.debug_draw_count = 0

    def reset(self, env_ids: torch.Tensor, reward_stats: TensorDict = None):
        self.command[env_ids] = 0.
        self._cum_error[env_ids] = 0.
        self.last_env_reset_ids = env_ids
        self.debug_draw_dict = {}
        self.debug_draw_count = 0

    """For each environment, sample a target to reach in the next 300 steps.
    Set the command as the linear interpolation between the current and target pose, command updated every 10 steps.
    When to sample a new target:
    1. At the beginning of the episode (reset)
    2. Every 300 steps
    3. If the robot fails to track the command within some error `threshold` for `tolerance` * 10 steps
    """
    def update(self):
        quat_yaw = yaw_quat(self.asset.data.root_quat_w)
        root_pos_w = self.asset.data.root_pos_w
        self.ee_pos_b[:] = quat_rotate_inverse(
            quat_yaw, 
            self.asset.data.body_pos_w[:, self.ee_id] - root_pos_w
        )
        self.command_ee_pos_b_yaw[:] = torch.atan2(self.command_ee_pos_b[:, 0, 1], self.command_ee_pos_b[:, 0, 0])

        if self.last_env_reset_ids is not None and len(self.last_env_reset_ids) > 0:
            self.sample_ee(self.last_env_reset_ids)
            self.sample_loco(self.last_env_reset_ids)
            if self.imagined_root:
                # TODO: check if should put this in reset
                self.command_heading_yaw_w[self.last_env_reset_ids] = self.asset.data.heading_w[self.last_env_reset_ids]
                self.command_root_pos_w[self.last_env_reset_ids] = root_pos_w[self.last_env_reset_ids]
            self.last_env_reset_ids = None

        # update cum error
        ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id]
        ee_pos_error = (ee_pos_w - self.command_ee_pos_w).norm(dim=-1)
        
        ee_quat_w = self.asset.data.body_quat_w[:, self.ee_id]
        ee_forward_w = quat_rotate(ee_quat_w, self.fwd_vec)
        ee_upward_w = quat_rotate(ee_quat_w, self.up_vec)
        ee_ori_error = 1 - (ee_forward_w * self.command_ee_forward_w).sum(dim=-1) / 2 - (ee_upward_w * self.command_ee_upward_w).sum(dim=-1) / 2

        valid = (self.env.episode_length_buf > 2).float()
        ee_pos_error.mul_(valid)
        ee_ori_error.mul_(valid)
        self._cum_ee_pos_error.add_(ee_pos_error * self.env.step_dt).mul_(0.99)
        self._cum_ee_ori_error.add_(ee_ori_error * self.env.step_dt).mul_(0.99)
            
        # update failed track counts, do not update if just resampled
        update_mask = torch.logical_and((self.steps_since_last_sample % self.update_interval) == 0, self.steps_since_last_sample != 0)
        update_ids = update_mask.nonzero().squeeze(-1)

        if self.tolerance > 0 and len(update_ids) > 0:
            # Update failed track counts
            failed_track = (ee_pos_error[update_ids] > self.threshold) | (ee_ori_error[update_ids] > self.threshold)
            self.failed_track_counts[update_ids[failed_track]] += 1
            self.failed_track_counts[update_ids[~failed_track]] = 0
        
        # resample if interval reached or failed to track
        interval_reached = self.steps_since_last_sample == self.resample_interval
        resample_mask = interval_reached | (self.failed_track_counts > self.tolerance)
        self.sample_ee(resample_mask.nonzero().squeeze(-1))
        self.sample_loco(interval_reached.nonzero().squeeze(-1))
        if self.imagined_root:
            self.command_heading_yaw_w[resample_mask] = self.asset.data.heading_w[resample_mask]
            self.command_root_pos_w[resample_mask] = root_pos_w[resample_mask]
        
        # update command to be the linear interpolation between last and current target
        command_alpha = (self.steps_since_last_sample.unsqueeze(-1) // self.update_interval + self.steps * 4) / self.updates_per_resample
        command_alpha.clamp_(0., 1.)
        # TODO: at the end of this sample, maybe need to inform about the next sample target?
        # command_alpha: [num_envs, future_steps]

        ## interpolate target position
        current_r, current_pitch, current_yaw = interpolate_position(
            einops.repeat(self.last_target_pos_radius_pitch_yaw, 'n d -> n t d', t=self.future_targets),
            einops.repeat(self.next_target_pos_radius_pitch_yaw, 'n d -> n t d', t=self.future_targets),
            command_alpha,
            self.yaw_range
        )
        self.command_ee_pos_b[:] = torch.stack([
            current_r * torch.cos(current_yaw) * torch.cos(current_pitch),
            current_r * torch.sin(current_yaw) * torch.cos(current_pitch),
            current_r * -torch.sin(current_pitch),
        ], dim=-1)
        if torch.isnan(self.command_ee_pos_b).any():
            breakpoint()

        ## interpolate target orientation
        current_roll, current_ori_pitch, current_ori_yaw = interpolate_orientation(
            self.last_target_ori_roll_pitch_yaw.unsqueeze(1).expand(-1, self.future_targets, -1),
            self.next_target_ori_roll_pitch_yaw.unsqueeze(1).expand(-1, self.future_targets, -1),
            command_alpha
        )
        self.command_ee_quat_b[:] = quat_from_euler_xyz(current_roll, current_ori_pitch, current_ori_yaw)
        if torch.isnan(self.command_ee_quat_b).any():
            breakpoint()

        # set command
        self.command[:, :2] = self.command_lin_vel[:, :2]
        self.command[:, 2] = self.command_ang_vel
        self.command[:, 3:3+3*self.future_targets] = self.command_ee_pos_b.view(self.num_envs, -1)
        self.command[:, 3+3*self.future_targets:] = self.command_ee_quat_b.view(self.num_envs, -1)
        if torch.isnan(self.command).any():
            breakpoint()

        if self.imagined_root:
            # update command heading and root pos
            self.command_heading_yaw_w += self.command_ang_vel * self.env.step_dt
            command_quat_yaw_w = torch.cat([
                torch.cos(self.command_heading_yaw_w / 2).unsqueeze(-1),
                torch.zeros_like(self.command_heading_yaw_w).unsqueeze(-1),
                torch.zeros_like(self.command_heading_yaw_w).unsqueeze(-1),
                torch.sin(self.command_heading_yaw_w / 2).unsqueeze(-1),
            ], dim=-1)
            command_lin_vel_w = quat_rotate(command_quat_yaw_w, self.command_lin_vel)
            self.command_root_pos_w += command_lin_vel_w * self.env.step_dt

            quat_yaw = command_quat_yaw_w
            arm_base_pos_w = self.command_root_pos_w + quat_rotate(quat_yaw, self.arm_base_offset_b)
        else:
            arm_base_pos_w = self.asset.data.body_pos_w[:, self.ee_base_id]
            
        # compute commanded ee position and orientation in world frame
        self.command_ee_pos_w_[:] = quat_rotate(
            quat_yaw.unsqueeze(1),
            self.command_ee_pos_b
        ) + arm_base_pos_w.unsqueeze(1)

        self.command_ee_quat_w[:] = quat_mul(quat_yaw, self.command_ee_quat_b[:, 0, :])
        self.command_ee_forward_w[:] = quat_rotate(self.command_ee_quat_w, self.fwd_vec)
        self.command_ee_upward_w[:] = quat_rotate(self.command_ee_quat_w, self.up_vec)

        self.steps_since_last_sample += 1
        
    def sample_ee(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        # sample target position and orientation
        target_pos_yaw = torch.zeros(env_ids.shape, device=self.device).uniform_(*self.yaw_range)
        target_pos_pitch = torch.zeros(env_ids.shape, device=self.device).uniform_(*self.pitch_range)
        target_pos_radius = torch.zeros(env_ids.shape, device=self.device).uniform_(*self.radius_range)
        self.next_target_pos_radius_pitch_yaw[env_ids] = torch.stack([target_pos_radius, target_pos_pitch, target_pos_yaw], dim=-1)

        target_ori_roll = torch.zeros_like(target_pos_yaw)
        target_ori_yaw = target_pos_yaw + torch.empty_like(target_pos_yaw).uniform_(-torch.pi/4, torch.pi/4)
        target_ori_pitch = target_pos_pitch + torch.empty_like(target_pos_pitch).uniform_(-torch.pi/4, torch.pi/4)
        self.next_target_ori_roll_pitch_yaw[env_ids] = torch.stack([target_ori_roll, target_ori_pitch, target_ori_yaw], dim=-1)
        if torch.isnan(self.next_target_ori_roll_pitch_yaw).any():
            breakpoint()

        # get current ee position and orientation as last target
        # this is errorneous if env has just been reset
        ee_pos_b = self.ee_pos_b[env_ids]
        curr_pos_radius = (ee_pos_b ** 2).sum(dim=-1).sqrt()
        curr_pos_pitch = -torch.asin(ee_pos_b[:, 2] / curr_pos_radius)
        curr_pos_yaw = torch.atan2(ee_pos_b[:, 1], ee_pos_b[:, 0])
        self.last_target_pos_radius_pitch_yaw[env_ids] = torch.stack([curr_pos_radius, curr_pos_pitch, curr_pos_yaw], dim=-1)
        self.command_ee_pos_w_[env_ids] = self.asset.data.body_pos_w[env_ids, self.ee_id].unsqueeze(1)

        ee_quat_w = self.asset.data.body_quat_w[env_ids, self.ee_id]
        root_quat_w = self.asset.data.root_quat_w[env_ids]
        ee_quat_b = quat_mul(quat_inv(yaw_quat(root_quat_w)), ee_quat_w)
        curr_ori_roll, curr_ori_pitch, curr_ori_yaw = euler_xyz_from_quat(ee_quat_b)
        self.last_target_ori_roll_pitch_yaw[env_ids] = torch.stack([wrap_to_pi(curr_ori_roll), wrap_to_pi(curr_ori_pitch), wrap_to_pi(curr_ori_yaw)], dim=-1)
        self.command_ee_quat_w[env_ids] = ee_quat_w
        self.command_ee_forward_w[env_ids] = quat_rotate(ee_quat_w, self.fwd_vec[env_ids])
        self.command_ee_upward_w[env_ids] = quat_rotate(ee_quat_w, self.up_vec[env_ids])
        if torch.isnan(self.last_target_ori_roll_pitch_yaw).any():
            breakpoint()
        
        # reset buffers
        self.failed_track_counts[env_ids] = 0
        self.steps_since_last_sample[env_ids] = 0
    
    def sample_loco(self, env_ids: torch.Tensor):
        # sample speed and direction
        linvel = torch.zeros(len(env_ids), 2, device=self.device)
        linvel[:, 0].uniform_(*self.lin_vel_x_range)
        linvel[:, 1].uniform_(*self.lin_vel_y_range)
        speed = linvel.norm(dim=-1, keepdim=True)
        stand = speed < 0.3
        speed = speed * (~stand)
        self.command_lin_vel[env_ids, :2] = linvel
        self._command_speed[env_ids] = speed
        self.is_standing_env[env_ids] = stand

        # sample angvel
        self.command_ang_vel[env_ids] = torch.empty(env_ids.shape, device=self.device).uniform_(*self.ang_vel_range)
        still = self.command_ang_vel[env_ids] < 0.1
        self.command_ang_vel[env_ids[still]] = 0.


    def debug_draw(self):
        # append to self.debug_draw_dict
        # which will contain the keys: commmand_ee_pos_w, command_ee_forward_w, command_ee_upward_w
        # if len(self.debug_draw_dict) == 0:
        #     self.debug_draw_dict = {
        #         "command_ee_pos_w": torch.empty(self.num_envs, 0, 3, device=self.device),
        #         "command_ee_forward_w": torch.empty(self.num_envs, 0, 3, device=self.device),
        #         "command_ee_upward_w": torch.empty(self.num_envs, 0, 3, device=self.device)
        #     }
        
        # if self.debug_draw_count % 2 == 1:
        # # if True:
        #     self.debug_draw_dict["command_ee_pos_w"] = torch.cat([self.debug_draw_dict["command_ee_pos_w"], self.command_ee_pos_w.unsqueeze(1)], dim=1)
        #     self.debug_draw_dict["command_ee_forward_w"] = torch.cat([self.debug_draw_dict["command_ee_forward_w"], self.command_ee_forward_w.unsqueeze(1)], dim=1)
        #     self.debug_draw_dict["command_ee_upward_w"] = torch.cat([self.debug_draw_dict["command_ee_upward_w"], self.command_ee_upward_w.unsqueeze(1)], dim=1)
        
        self.debug_draw_count += 1
        
        # for i in range(self.num_envs):
        #     self.env.debug_draw.plot(
        #         self.debug_draw_dict["command_ee_pos_w"][i],
        #         color=(0., 0.8, 0., 1.)
        #     )
        # self.env.debug_draw.vector(
        #     self.debug_draw_dict["command_ee_pos_w"].view(-1, 3),
        #     self.debug_draw_dict["command_ee_forward_w"].view(-1, 3) * 0.2,
        #     color=(1., 0., 0., 1.)
        # )
        # self.env.debug_draw.vector(
        #     self.debug_draw_dict["command_ee_pos_w"].view(-1, 3),
        #     self.debug_draw_dict["command_ee_upward_w"].view(-1, 3) * 0.2,
        #     color=(1., 0., 0., 1.)
        # )
        ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id].unsqueeze(1)
        ee_pos_diff = self.command_ee_pos_w_ - ee_pos_w
        self.env.debug_draw.vector(
            ee_pos_w.expand_as(ee_pos_diff).reshape(-1, 3),
            ee_pos_diff.reshape(-1, 3),
            color=(0., 0.8, 0., 1.)
        )
        
        ee_ori_fwd = quat_rotate(self.command_ee_quat_w, self.fwd_vec)
        self.env.debug_draw.vector(
            ee_pos_w.squeeze(1),
            ee_ori_fwd.reshape(-1, 3) * 0.2,
            color=(1., 0., 0., 1.)
        )
        
        # zeros = torch.zeros(self.num_envs, device=self.device)
        # self.env.debug_draw.vector(
        #     self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
        #     torch.stack([zeros, zeros, self._cum_ee_pos_error], dim=-1),
        #     color=(0.2, 1.0, 0.2, 1)
        # )
        # self.env.debug_draw.vector(
        #     self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
        #     torch.stack([zeros, zeros, self._cum_ee_ori_error], dim=-1),
        #     color=(1.0, 0.2, 0.2, 1)
        # )
        # ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id]
        # ee_pos_target = quat_rotate(
        #     yaw_quat(self.asset.data.root_quat_w),
        #     self.command_ee_pos_b
        # ) + self.asset.data.root_pos_w

        # # self.env.debug_draw.vector(
        # #     ee_pos_w,
        # #     self.command_ee_forward_w * 0.2,
        # # )
        # # self.env.debug_draw.vector(
        # #     ee_pos_w,
        # #     self.command_ee_upward_w * 0.2,
        # # )
        # self.env.debug_draw.vector(
        #     ee_pos_w,
        #     ee_pos_target - ee_pos_w,
        #     color=(0., 0.8, 0., 1.)
        # )

class CommandEEPose_UMI(Command):
    """For each environment, sample a target to reach in the next 300 steps.
    Set the command as the linear interpolation between the current and target pose, command updated every 10 steps.
    When to sample a new target:
    1. At the beginning of the episode (reset)
    2. Every 300 steps
    3. If the robot fails to track the command within some error `threshold` for `tolerance` * 10 steps
    """
    def __init__(
        self, 
        env,
        ee_name: str,
        ee_base_name: str = 'arm_link00',
        # umi
        pos_obs_scale: float = 10,
        orn_obs_scale: float = 1.5,
        pos_err_sigma: float = 0.5,
        orn_err_sigma: float = 1.5,
        pos_sigma_curriculum: Optional[Dict[float, float]] = None,  # maps from error to sigma
        orn_sigma_curriculum: Optional[Dict[float, float]] = None,  # maps from error to sigma
        smoothing_dt_multiplier: float = 4.0,
        episode_length: int = 300,
        target_times: list = [0.02, 0.04, 0.06, 1.0],
        ee_lin_vel_range: tuple = (0.05, 1.0),
        # lin vel in x, y, ang vel
        lin_vel_x_range: tuple = (-1.0, 1.4),
        lin_vel_y_range: tuple = (-0.5, 0.5),
        ang_vel_range: tuple = (-1., 1.),
        # ee pose
        yaw_range: tuple = (-torch.pi, torch.pi),
        pitch_range: tuple = (-torch.pi / 3, 0.),
        radius_range: tuple = (0.4, 0.8),
        arm_command_prob: float = 1.0,
        fwd_vec = (1.0, 0.0, 0.0), # for special assets
        # force randomization
        force_resample_prob: float = 0.02,
        force_application_prob: float = 0.5,
        const_force_scale: float = 4.0,
        linear_drag_coeff_range: tuple = (0.5, 2.0),
        angular_drag_coeff_range: tuple = (0.1, 1.0),
        spring_stiffness_range: tuple = (4.0, 8.0),
    ) -> None:
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        ee_ids, ee_names = self.asset.find_bodies(ee_name)
        assert len(ee_ids) == 1
        self.ee_id = ee_ids[0]
        ee_base_ids, ee_base_names = self.asset.find_bodies(ee_base_name)
        assert len(ee_base_ids) == 1
        self.ee_base_id = ee_base_ids[0]

        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_range = ang_vel_range

        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.radius_range = radius_range
        self.ee_lin_vel_range = ee_lin_vel_range

        self.episode_length = episode_length
        self.future_targets = future_targets = len(target_times)

        self.pos_obs_scale = pos_obs_scale
        self.orn_obs_scale = orn_obs_scale
        self.smoothing_dt_multiplier = smoothing_dt_multiplier
        self.arm_command_prob = arm_command_prob

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 2 + 1 + 3 * future_targets + 3 * future_targets)

            # Locomotion
            self.command_lin_vel = torch.zeros(self.num_envs, 3)
            self._command_speed = torch.zeros(self.num_envs, 1)
            self.command_ang_vel = torch.zeros(self.num_envs)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            # Manipulation
            self.has_arm_command = torch.zeros(self.num_envs, 1, dtype=bool)
            self.target_steps = (torch.tensor(target_times) / self.env.step_dt).round().long()
            self.command_ee_pos_b_traj = torch.zeros(self.num_envs, episode_length + 1, 3)
            self.command_ee_fwd_b_traj = torch.zeros(self.num_envs, episode_length + 1, 3)

            self.command_ee_pos_b = torch.zeros(self.num_envs, future_targets, 3)
            self.command_ee_fwd_b = torch.zeros(self.num_envs, future_targets, 3)

            ## world frame targets for visualization
            self._command_ee_pos_w = torch.zeros(self.num_envs, future_targets, 3)
            self._command_ee_fwd_w = torch.zeros(self.num_envs, 3)
            self._fwd_vec = torch.tensor(fwd_vec).expand(self.num_envs, -1)

            ## current target for error computation
            self.ee_pos_error = torch.zeros(self.num_envs, 1)
            self.ee_orn_error = torch.zeros(self.num_envs, 1)
            self.ee_pos_rew = torch.zeros(self.num_envs, 1)
            self.ee_orn_rew = torch.zeros(self.num_envs, 1)
            self.ee_orn_dot = torch.zeros(self.num_envs, 1)

            self.past_pos_error = torch.zeros(self.num_envs, 1)
            self.past_orn_error = torch.zeros(self.num_envs, 1)

            # running stats
            self.pos_error_sum = torch.tensor(0.)
            self.orn_error_sum = torch.tensor(0.)
            self.count = torch.tensor(0.)
            self.pos_error_avg = torch.tensor(0.)
            self.orn_error_avg = torch.tensor(0.)
            self.stats_decay = 0.99

            # asset states
            self.ee_pos_b = self.asset.data.ee_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
            self.command_ee_pos_b_yaw = self.asset.data.command_ee_pos_b_yaw = torch.zeros(self.num_envs)

            # external wrench on EE
            self.force_resample_prob = force_resample_prob
            self.force_resample_min_interval = 20
            self.force_application_prob = force_application_prob

            self.ee_force_b = torch.zeros(self.num_envs, 3)
            self.apply_force = torch.zeros(self.num_envs, 1, dtype=bool)
            self.force_application_time = torch.zeros(self.num_envs, 1)
            self.torque_b = torch.zeros(self.num_envs, 3)

            # const force
            self.const_force_scale = const_force_scale
            self.const_force = torch.zeros(self.num_envs, 3)

            # drag force
            self.linear_drag_coeff_range = linear_drag_coeff_range
            self.angular_drag_coeff_range = angular_drag_coeff_range
            self.linear_drag_coeff = torch.zeros(self.num_envs, 1)
            self.angular_drag_coeff = torch.zeros(self.num_envs, 1)

            # spring force
            self.spring_stiffness_range = spring_stiffness_range
            self.spring_setpoint_b = torch.zeros(self.num_envs, 3)
            self.spring_setpoint_w = torch.zeros(self.num_envs, 3)
            self.spring_stiffness = torch.zeros(self.num_envs, 1)

            self.force_type_mask = torch.zeros(self.num_envs, 3, dtype=int) # 0: const, 1: drag, 2: spring
            self.drag_force = torch.zeros(self.num_envs, 3)
            self.spring_force = torch.zeros(self.num_envs, 3)

        # stats for curriculum
        self.pos_err_sigma = pos_err_sigma
        self.orn_err_sigma = orn_err_sigma
        self.pos_sigma_curriculum_level = 0
        self.orn_sigma_curriculum_level = 0
        self.pos_sigma_curriculum = pos_sigma_curriculum
        self.orn_sigma_curriculum = orn_sigma_curriculum
        if self.pos_sigma_curriculum is not None:
            # make sure the curriculum is sorted
            self.pos_sigma_curriculum = dict(
                map(
                    lambda x: (float(x[0]), float(x[1])),
                    sorted(
                        self.pos_sigma_curriculum.items(),
                        key=lambda x: x[0],
                        reverse=True,
                    ),
                )
            )
            self.pos_err_sigma = list(self.pos_sigma_curriculum.values())[
                self.pos_sigma_curriculum_level
            ]
            self.past_pos_error.fill_(list(self.pos_sigma_curriculum.keys())[
                self.pos_sigma_curriculum_level
            ])
        if self.orn_sigma_curriculum is not None:
            # make sure the curriculum is sorted
            self.orn_sigma_curriculum = dict(
                map(
                    lambda x: (float(x[0]), float(x[1])),
                    sorted(
                        self.orn_sigma_curriculum.items(),
                        key=lambda x: x[0],
                        reverse=True,
                    ),
                )
            )
            self.orn_err_sigma = list(self.orn_sigma_curriculum.values())[
                self.orn_sigma_curriculum_level
            ]
            self.past_orn_error.fill_(list(self.orn_sigma_curriculum.keys())[
                self.orn_sigma_curriculum_level
            ])
        self.debug_draw_dict = {}
        self.debug_draw_count = 0

        self.env.reward_spec["stats", "ee_pos_error_avg"] = UnboundedContinuousTensorSpec([self.num_envs, 1], device=self.device)
        self.env.reward_spec["stats", "ee_orn_error_avg"] = UnboundedContinuousTensorSpec([self.num_envs, 1], device=self.device)

    def _body2world(self, pos_b: torch.Tensor):
        bshape = pos_b.shape[:-1]
        quat = yaw_quat(self.asset.data.root_quat_w)
        root_pos_w = self.asset.data.root_pos_w
        if len(bshape) > 1:
            quat = quat.unsqueeze(1).expand(bshape + (4,))
            root_pos_w = root_pos_w.unsqueeze(1).expand(bshape + (3,))
        return quat_rotate(quat, pos_b) + root_pos_w

    def _world2body(self, pos_w: torch.Tensor):
        bshape = pos_w.shape[:-1]
        quat = yaw_quat(self.asset.data.root_quat_w)
        root_pos_w = self.asset.data.root_pos_w
        if len(bshape) > 1:
            quat = quat.unsqueeze(1).expand(bshape + (4,))
            root_pos_w = root_pos_w.unsqueeze(1).expand(bshape + (3,))
        return quat_rotate_inverse(quat, pos_w - root_pos_w)

    def step(self, substep: int):
        forces_b = self.asset._external_force_b.clone()
        torques_b = self.asset._external_torque_b.clone()

        ee_quat = self.asset.data.body_quat_w[:, self.ee_id]
        ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id]
        ee_vel_w = self.asset.data.body_lin_vel_w[:, self.ee_id]

        self.drag_force[:] = quat_rotate_inverse(ee_quat, - ee_vel_w * self.linear_drag_coeff)
        self.spring_setpoint_w[:] = self._body2world(self.spring_setpoint_b)
        self.spring_force[:] = quat_rotate_inverse(ee_quat, self.spring_stiffness * (self.spring_setpoint_w - ee_pos_w))
        const_force = quat_rotate_inverse(ee_quat, self.const_force)

        self.ee_force_b.zero_()
        self.ee_force_b.add_(const_force * self.force_type_mask[:, 0].unsqueeze(-1))
        self.ee_force_b.add_(self.drag_force * self.force_type_mask[:, 1].unsqueeze(-1))
        self.ee_force_b.add_(self.spring_force * self.force_type_mask[:, 2].unsqueeze(-1))
        self.ee_force_b = torch.where(self.apply_force, self.ee_force_b / self.force_type_mask.sum(1, True), 0.)

        forces_b[:, self.ee_id] += self.ee_force_b

        self.asset.set_external_force_and_torque(forces_b, torques_b)

    def reset(self, env_ids: torch.Tensor):
        # TODO: check reset logic
        self.sample_loco(env_ids=env_ids)

        # sample a entire trajectory
        # TODO: check sampling execution time
        # TODO: now takes around 10 seconds, need to parallel
        # import time
        # st = time.time()
        # print(f"Sampling trajectories for {env_ids}...")
        ee_pos_trajs, ee_forward_trajs = generate_random_trajectories(
            len(env_ids), 
            self.yaw_range, 
            self.pitch_range, 
            self.radius_range, 
            self.ee_lin_vel_range, 
            self.episode_length + 1, 
            self.env.step_dt, 
            self.device
        )
        # torch.cuda.synchronize()
        # print("Sampling time:", time.time() - st)
        self.command_ee_pos_b_traj[env_ids] = ee_pos_trajs
        self.command_ee_fwd_b_traj[env_ids] = ee_forward_trajs

        # update curriculum
        if self.pos_sigma_curriculum is not None:
            avg_pos_err = self.past_pos_error.mean().item()
            # find the first threshold that is greater than the average error
            old_pos_error_sigma = self.pos_err_sigma
            for level, (threshold, sigma) in enumerate(
                self.pos_sigma_curriculum.items()
            ):
                if avg_pos_err < threshold:
                    self.pos_err_sigma = sigma
                    self.pos_sigma_curriculum_level = level
            if old_pos_error_sigma != self.pos_err_sigma:
                print(f"avg pos error: {avg_pos_err}, new sigma level {self.pos_sigma_curriculum_level} with sigma {self.pos_err_sigma}")
        if self.orn_sigma_curriculum is not None:
            avg_orn_err = self.past_orn_error.mean().item()
            # find the first threshold that is greater than the average error
            for level, (threshold, sigma) in enumerate(
                self.orn_sigma_curriculum.items()
            ):
                if avg_orn_err < threshold:
                    self.orn_err_sigma = sigma
                    self.orn_sigma_curriculum_level = level

        if wandb.run is not None:
            wandb.log({"pos_err_sigma": self.pos_err_sigma, "pos_err_sigma_level": self.pos_sigma_curriculum_level, "orn_err_sigma": self.orn_err_sigma, "orn_err_sigma_level": self.orn_sigma_curriculum_level}, commit=False)

        self.command[env_ids] = 0.
        self.debug_draw_dict = {}
        self.debug_draw_count = 0
        self.has_arm_command[env_ids] = torch.rand(len(env_ids), 1, device=self.device) < self.arm_command_prob

    def get_targets_at_steps(self):
        # get the index of the closest time
        target_indices = self.env.episode_length_buf.unsqueeze(1) + self.target_steps.unsqueeze(0)
        target_indices = torch.clamp(target_indices, 0, self.episode_length - 1)
        command_ee_pos_b = torch.gather(self.command_ee_pos_b_traj, 1, target_indices.unsqueeze(-1).expand(-1, -1, 3))
        command_ee_fwd_b = torch.gather(self.command_ee_fwd_b_traj, 1, target_indices.unsqueeze(-1).expand(-1, -1, 3))
        return command_ee_pos_b, command_ee_fwd_b

    def update(self):
        self._maybe_sample_force()
        # update asset states
        root_quat_yaw = yaw_quat(self.asset.data.root_quat_w)
        arm_base_pos_w = self.asset.data.body_pos_w[:, self.ee_base_id]
        self.ee_pos_b[:] = quat_rotate_inverse(
            root_quat_yaw, 
            self.asset.data.body_pos_w[:, self.ee_id] - arm_base_pos_w
        )
        self.command_ee_pos_b_yaw[:] = torch.atan2(self.command_ee_pos_b[:, 0, 1], self.command_ee_pos_b[:, 0, 0])

        indices = self.env.episode_length_buf.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)
        assert torch.all(indices <= self.episode_length)
        current_command_ee_pos_b = self.command_ee_pos_b_traj.gather(1, indices).squeeze(1)
        current_command_ee_pos_w = quat_rotate(
            root_quat_yaw,
            current_command_ee_pos_b
        ) + arm_base_pos_w
        current_command_ee_fwd_b = self.command_ee_fwd_b_traj.gather(1, indices).squeeze(1)
        current_command_ee_fwd_w = quat_rotate(root_quat_yaw, current_command_ee_fwd_b)

        # update pos and ori cum error
        ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id]
        self.ee_pos_error[:] = (ee_pos_w - current_command_ee_pos_w).norm(dim=-1, keepdim=True)
        self.ee_pos_rew[:] = torch.exp(-self.ee_pos_error / self.pos_err_sigma)

        ee_quat_w = self.asset.data.body_quat_w[:, self.ee_id]
        ee_fwd_w = quat_rotate(ee_quat_w, self._fwd_vec)
        self.ee_orn_dot[:] = (ee_fwd_w * current_command_ee_fwd_w).sum(dim=-1, keepdim=True)
        self.ee_orn_error[:] = 1 - self.ee_orn_dot # 1 - cos(theta)
        self.ee_orn_rew[:] = torch.exp(-self.ee_orn_error / self.orn_err_sigma)

        # moving average of the error
        valid = (self.env.episode_length_buf > 2).float().unsqueeze(-1)
        smoothing = self.env.step_dt * self.smoothing_dt_multiplier
        self.past_pos_error.mul_(1 - smoothing * valid).add_(smoothing * self.ee_pos_error * valid)
        self.past_orn_error.mul_(1 - smoothing * valid).add_(smoothing * self.ee_orn_error * valid)

        self.pos_error_sum.mul_(self.stats_decay).add_((self.ee_pos_error * valid).sum())
        self.orn_error_sum.mul_(self.stats_decay).add_((self.ee_orn_error * valid).sum())
        self.count.mul_(self.stats_decay).add_(self.num_envs)
        self.pos_error_avg = self.pos_error_sum / self.count
        self.orn_error_avg = self.orn_error_sum / self.count
        self.env.stats["ee_pos_error_avg"].fill_(self.pos_error_avg)
        self.env.stats["ee_orn_error_avg"].fill_(self.orn_error_avg)

        # get new command ee pos from the trajectory
        command_ee_pos_b, command_ee_fwd_b = self.get_targets_at_steps()
        self.command_ee_pos_b[:] = command_ee_pos_b
        self.command_ee_fwd_b[:] = command_ee_fwd_b

        # set new command
        self.command[:, :2] = self.command_lin_vel[:, :2]
        self.command[:, 2] = self.command_ang_vel
        self.command[:, 3:3+3*self.future_targets] = self.command_ee_pos_b.view(self.num_envs, -1) * self.pos_obs_scale
        self.command[:, 3+3*self.future_targets:] = self.command_ee_fwd_b.view(self.num_envs, -1) * self.orn_obs_scale

        if self.arm_command_prob < 1.0:
            self.command[(~self.has_arm_command).nonzero().squeeze(-1), 3:] = 0.

        # compute commanded ee position and orientation in world frame
        self._command_ee_pos_w[:] = quat_rotate(
            root_quat_yaw.unsqueeze(1),
            self.command_ee_pos_b
        ) + arm_base_pos_w.unsqueeze(1)

        self._command_ee_fwd_w[:] = quat_rotate(root_quat_yaw, self.command_ee_fwd_b[:, 0, :])

    def sample_loco(self, env_ids: torch.Tensor):
        # sample speed and direction
        linvel = torch.zeros(len(env_ids), 2, device=self.device)
        linvel[:, 0].uniform_(*self.lin_vel_x_range)
        linvel[:, 1].uniform_(*self.lin_vel_y_range)
        speed = linvel.norm(dim=-1, keepdim=True)
        stand = speed < 0.3
        speed = speed * (~stand)
        self.command_lin_vel[env_ids, :2] = linvel * (~stand)
        self._command_speed[env_ids] = speed
        self.is_standing_env[env_ids] = stand

        # sample angvel
        self.command_ang_vel[env_ids] = torch.empty(env_ids.shape, device=self.device).uniform_(*self.ang_vel_range)
        still = self.command_ang_vel[env_ids] < 0.1
        self.command_ang_vel[env_ids[still]] = 0.

    def _maybe_sample_force(self):
        resample_mask = (
            (torch.rand(self.num_envs, 1, device=self.device) < self.force_resample_prob)
            & (self.force_application_time > self.force_resample_min_interval)
        )
        force_type_mask = torch.rand_like(self.force_type_mask, dtype=float) < 0.5
        apply_force = (
            (torch.rand_like(self.apply_force, dtype=float) < self.force_application_prob)
            & (force_type_mask.any(1, True))
        )

        const_force = torch.randn_like(self.const_force).clip(-3., 3.) * self.const_force_scale
        linear_drag_coeff = torch.empty_like(self.linear_drag_coeff).uniform_(*self.linear_drag_coeff_range)
        angular_drag_coeff = torch.empty_like(self.angular_drag_coeff).uniform_(*self.angular_drag_coeff_range)
        spring_stiffness = torch.empty_like(self.spring_stiffness).uniform_(*self.spring_stiffness_range)
        spring_setpoint_b = self.ee_pos_b + torch.randn_like(self.ee_pos_b).clip(-3., 3.) * 0.4

        self.apply_force = torch.where(resample_mask, apply_force, self.apply_force)
        self.force_application_time = torch.where(resample_mask, 0., self.force_application_time + 1.)
        self.force_type_mask = torch.where(resample_mask, force_type_mask, self.force_type_mask)
        self.const_force = torch.where(resample_mask, const_force, self.const_force)
        self.linear_drag_coeff = torch.where(resample_mask, linear_drag_coeff, self.linear_drag_coeff)
        self.angular_drag_coeff = torch.where(resample_mask, angular_drag_coeff, self.angular_drag_coeff)
        self.spring_stiffness = torch.where(resample_mask, spring_stiffness, self.spring_stiffness)
        self.spring_setpoint_b = torch.where(resample_mask, spring_setpoint_b, self.spring_setpoint_b)

    def debug_draw(self):        
        self.debug_draw_count += 1

        ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id].unsqueeze(1)
        ee_pos_diff = self._command_ee_pos_w - ee_pos_w
        self.env.debug_draw.vector(
            ee_pos_w.expand_as(ee_pos_diff).reshape(-1, 3),
            ee_pos_diff.reshape(-1, 3),
            color=(0., 0.8, 0., 1.)
        )

        self.env.debug_draw.vector(
            ee_pos_w.squeeze(1),
            self._command_ee_fwd_w.reshape(-1, 3) * 0.2,
            color=(1., 0., 0., 1.)
        )
        ee_quat = self.asset.data.body_quat_w[:, self.ee_id]
        ee_fwd = quat_rotate(ee_quat, self._fwd_vec)
        self.env.debug_draw.vector(
            ee_pos_w.squeeze(1),
            ee_fwd.reshape(-1, 3) * 0.2,
            color=(1., 1., 0.1, 1.)
        )

        self.env.debug_draw.vector(
            ee_pos_w.squeeze(1),
            quat_rotate(ee_quat, self.ee_force_b) / 9.81,
            color=(1., 0., 1., 1.)
        )

class BaseEEImpedance(Command):
    """Model the Base And EEF as a two body spring-damper system. The base is connected to a setpoint in world frame and the EEF is connected to a setpoint in body frame. We model forces applied  onto these two bodies as external forces. We model xy for base and xyz for ee.
    We also model a rotational spring-damper system and the corresponding torques in z direction for the base yaw.
    
    
    to check base xy policy: sample yaw to be the root heading
    1. stiff mode: large base kp, sample setpoint in the root pos, use force to check if it can resist the force
    2. stiff mode: sample setpoint around the root pos, check if it can return to setpoint
    3. compliant mode: 0 base kp, use force to check if it can follow the force

    If continuous_waking=False, a position setpoint will not move until resampled

    """

    def __init__(
        self,
        env,
        ee_name: str = "arm_link06",
        ee_base_name: str = "arm_link00",
        # setpoint sample range
        base_setpoint_radius_range: tuple = (2.0, 3.0),
        # TODO: add range for ee
        
        # impedance paramaters range
        kp_base_range: tuple = (0.5, 5.0),
        kp_yaw_range: tuple = (0.5, 5.0),
        kp_ee_range: tuple = (100.0, 150.0),
        damping_ratio_range: tuple = (0.7, 1.5),
        default_mass_base: float = 4.0,
        default_mass_ee: float = 0.5,
        default_inertia_z: float = 1.0,
        virtual_mass_range: tuple = (0.5, 1.5),

        # command mode ratio
        compliant_xy_ratio: float = 0.2,
        compliant_yaw_ratio: float = 0.2,
        compliant_ee_ratio: float = 0.2,
        continuous_walking_ratio: float = 0.25,

        # misc
        temporal_smoothing: int = 8,

        # force randomization
        force_type_probs = (0.4, 0.6, 0.0),
        impulse_force_momentum_scale = (5.0, 5.0, 1.0),
        impulse_force_duration_range = (0.1, 0.5),
        constant_force_scale = (50, 50, 10),
        constant_force_duration_range = (1, 4),
        force_offset_scale = (0.3, 0.2, 0.1),

        # eef force
        eef_force_ratio = 0.2,
        eef_force_scale = (15, 15, 15),
        eef_force_duration_range = (1, 4),
    ) -> None:
        super().__init__(env)
        self.base_body_id = self.asset.find_bodies("base")[0][0]
        self.ee_body_id = self.asset.find_bodies(ee_name)[0][0]
        self.ee_base_body_id = self.asset.find_bodies(ee_base_name)[0][0]
        self.body_ids = [self.base_body_id, self.ee_body_id]

        self.base_setpoint_radius_range = base_setpoint_radius_range

        self.kp_base_range = kp_base_range
        self.kp_yaw_range = kp_yaw_range
        self.kp_ee_range = kp_ee_range
        self.damping_ratio_range = damping_ratio_range
        self.default_mass_base = default_mass_base
        self.default_mass_ee = default_mass_ee
        self.default_inertia_z = default_inertia_z
        self.virtual_mass_range = virtual_mass_range

        self.compliant_xy_ratio = compliant_xy_ratio
        self.compliant_yaw_ratio = compliant_yaw_ratio
        self.compliant_ee_ratio = compliant_ee_ratio

        self.resample_force_prob = 0.005
        self.resample_command_interval = 300
        self.no_command_steps = 20
        self.decimation = int(self.env.step_dt / self.env.physics_dt)
        self.temporal_smoothing = temporal_smoothing

        from active_adaptation.assets.quadruped import QuadrupedManipulator

        self.asset: QuadrupedManipulator

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 18)
            self.command_hidden = torch.zeros(self.num_envs, 12)

            # integration
            self.force_spring_base_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_linacc_base_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_linvel_base_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_pos_base_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)

            self.force_spring_ee_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_lin_acc_ee_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_linvel_ee_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_pos_ee_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)

            self.torque_spring_yaw_w = torch.zeros(self.num_envs, self.temporal_smoothing, 1)
            self.desired_yawacc_w = torch.zeros(self.num_envs, self.temporal_smoothing, 1)
            self.desired_angvel_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_yawvel_w = self.desired_angvel_w[:, :, 2:3]
            self.desired_yaw_w = torch.zeros(self.num_envs, self.temporal_smoothing, 1)

            self.smoothing_weight = torch.full((1, self.temporal_smoothing), 0.9).cumprod(1)
            self.smoothing_weight = self.smoothing_weight.flip(dims=(1,)) / self.smoothing_weight.sum()

            # command setpoints in world/body frame
            self.command_setpoint_pos_base_w = torch.zeros(self.num_envs, 3)
            self.command_setpoint_pos_base_diff_b = torch.zeros(self.num_envs, 3)

            self.command_setpoint_pos_ee_b = torch.zeros(self.num_envs, 3)
            self.command_setpoint_pos_ee_diff_b = torch.zeros(self.num_envs, 3)

            self.command_setpoint_yaw_w = torch.zeros(self.num_envs, 1)
            self.command_setpoint_yaw_diff = torch.zeros(self.num_envs, 1)

            # hidden command (privileged information) be provided to tell the agent the desired behavior at the **next time step**
            self.command_pos_base_w = torch.zeros(self.num_envs, 3)
            self.command_pos_base_diff_b = torch.zeros(self.num_envs, 3)
            self.command_linvel_base_w = torch.zeros(self.num_envs, 3)
            self.command_linvel_base_b = torch.zeros(self.num_envs, 3)

            self.command_pos_ee_w = torch.zeros(self.num_envs, 3)
            self.command_pos_ee_diff_b = torch.zeros(self.num_envs, 3)
            self.command_linvel_ee_w = torch.zeros(self.num_envs, 3)
            self.command_linvel_ee_b = torch.zeros(self.num_envs, 3)

            self.command_yaw_w = torch.zeros(self.num_envs, 1)
            self.command_yaw_diff = torch.zeros(self.num_envs, 1)
            self.command_angvel_w = torch.zeros(self.num_envs, 3)
            self.command_yaw_vel = self.command_angvel_w[:, 2:3]

            # for reward computation (legacy)
            self.command_linvel = self.command_linvel_base_b
            self.command_speed = torch.zeros(self.num_envs, 1)
            self.command_angvel = self.command_yaw_vel[:, 0]

            self.command_pos_ee_b = torch.zeros(self.num_envs, 3)
            self.linvel_ee_b = torch.zeros(self.num_envs, 3)
            self.pos_ee_b = torch.zeros(self.num_envs, 3)

            # spring-damper parameters
            self.kp_base = torch.zeros(self.num_envs, 1)
            self.kd_base = torch.zeros(self.num_envs, 1)
            self.kp_ee = torch.zeros(self.num_envs, 1)
            self.kd_ee = torch.zeros(self.num_envs, 1)
            self.kp_yaw = torch.zeros(self.num_envs, 1)
            self.kd_yaw = torch.zeros(self.num_envs, 1)
            self.compliant_base = torch.zeros(self.num_envs, 1, dtype=bool)
            self.compliant_ee = torch.zeros(self.num_envs, 1, dtype=bool)
            self.compliant_yaw = torch.zeros(self.num_envs, 1, dtype=bool)

            self.default_mass_base = default_mass_base
            self.default_mass_ee = default_mass_ee
            self.default_inertia_z = default_inertia_z

            self.virtual_mass_base = torch.ones(self.num_envs, 1) * self.default_mass_base
            self.virtual_mass_ee = torch.ones(self.num_envs, 1) * self.default_mass_ee
            self.virtual_inertia_z = torch.ones(self.num_envs, 1) * self.default_inertia_z

            self.force_ext_base_w = torch.zeros(self.num_envs, 3)
            self.force_ext_ee_w = torch.zeros(self.num_envs, 3)
            self.force_base_offset_b = torch.zeros(self.num_envs, 3)

            self._cum_error = torch.zeros(self.num_envs, 6)
            self._cum_count = torch.zeros(self.num_envs, 1)

            self.continuous_walking_ratio = continuous_walking_ratio
            self.continuous_walking = torch.zeros(self.num_envs, 1, dtype=bool)
            self.continuous_walking_vel = torch.zeros(self.num_envs, 3)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.is_arm_activated = torch.ones(self.num_envs, 1, dtype=bool)

            """
            Force Types:
            0: None
            1: Constant
            2: Impulse
            """
            # force parameters
            self.impulse_force_momentum_scale = torch.tensor(impulse_force_momentum_scale) * self.default_mass_base
            self.impulse_force_duration_range = impulse_force_duration_range

            self.constant_force_scale = torch.tensor(constant_force_scale)
            self.constant_force_duration_range = constant_force_duration_range

            self.force_offset_scale = torch.tensor(force_offset_scale)

            self.force_type_probs = torch.tensor(force_type_probs)
            self.force_type = torch.zeros(self.num_envs, 1, dtype=torch.int64)

            self.constant_force_struct = torch.zeros(self.num_envs, 3 + 1 + 1) # force, duration, time
            self.constant_force_magnitude = self.constant_force_struct[:, :3]
            self.constant_force_duration = self.constant_force_struct[:, -2].unsqueeze(1)
            self.constant_force_time = self.constant_force_struct[:, -1].unsqueeze(1)

            self.impulse_force_struct = torch.zeros(self.num_envs, 3 + 1 + 1) # force, duration, time
            self.impulse_force_magnitude = self.impulse_force_struct[:, :3]
            self.impulse_force_duration = self.impulse_force_struct[:, -2].unsqueeze(1)
            self.impulse_force_duration.fill_(0.1) # non-zero placeholder
            self.impulse_force_time = self.impulse_force_struct[:, -1].unsqueeze(1)

            self.eef_force_ratio = eef_force_ratio
            self.eef_force_scale = torch.tensor(eef_force_scale)
            self.eef_force_duration_range = eef_force_duration_range

            self.eef_force_struct = torch.zeros(self.num_envs, 3 + 1 + 1) # force, duration, time
            self.eef_force_magnitude = self.eef_force_struct[:, :3]
            self.eef_force_duration = self.eef_force_struct[:, -2].unsqueeze(1)
            self.eef_force_time = self.eef_force_struct[:, -1].unsqueeze(1)

        self.cnt = 0

        if self.teleop:
            self.key_mappings_pos = {
                "W": torch.tensor([1., 0., 0.], device=self.device),
                "S": torch.tensor([-1., 0., 0.], device=self.device),
                "A": torch.tensor([0., 1., 0.], device=self.device),
                "D": torch.tensor([0., -1., 0.], device=self.device),
            }
            self.key_mappings_rpy = {
                "Q": torch.tensor([0., 0., +torch.pi], device=self.device),
                "E": torch.tensor([0., 0., -torch.pi], device=self.device),
            }

    def _sample_command(self, env_ids: torch.Tensor):

        continuous_walking_vel = torch.zeros(len(env_ids), 3, device=self.device)
        continuous_walking_vel[:, 0].uniform_(0.4, 1.0)
        self.continuous_walking[env_ids, 0] = torch.rand(len(env_ids), device=self.device) < self.continuous_walking_ratio
        self.continuous_walking_vel[env_ids] = continuous_walking_vel

        command_setpoint_pos_base_w = sample_disk(
            size=len(env_ids), 
            radius_range=self.base_setpoint_radius_range,
            device= self.device
        )
        self.command_setpoint_pos_base_w[env_ids] = command_setpoint_pos_base_w + self.asset.data.root_pos_w[env_ids]

        setpoint_ee_r = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.3, 0.8)
        setpoint_ee_pitch = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.1 * torch.pi, 0.49 * torch.pi)
        setpoint_ee_yaw = torch.empty(len(env_ids), 1, device=self.device).uniform_(-torch.pi / 2, torch.pi / 2)
        command_setpoint_pos_ee_b = torch.concat([
            setpoint_ee_r * torch.cos(setpoint_ee_pitch) * torch.cos(setpoint_ee_yaw),
            setpoint_ee_r * torch.cos(setpoint_ee_pitch) * torch.sin(setpoint_ee_yaw),
            setpoint_ee_r * torch.sin(setpoint_ee_pitch),
        ], dim=-1)
        self.command_setpoint_pos_ee_b[env_ids] = command_setpoint_pos_ee_b

        command_setpoint_yaw_w = torch.empty(len(env_ids), 1, device=self.device).uniform_(-torch.pi / 2, torch.pi / 2)
        command_setpoint_yaw_w += self.asset.data.heading_w[env_ids].unsqueeze(1)
        self.command_setpoint_yaw_w[env_ids] = command_setpoint_yaw_w

        kp_base = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.kp_base_range
        )
        kd_base = (
            2.0
            * kp_base.sqrt()
            * torch.empty(len(env_ids), 1, device=self.device).uniform_(
                *self.damping_ratio_range
            )
        )
        compliant_base = (
            torch.rand(len(env_ids), 1, device=self.device) < self.compliant_xy_ratio
        )
        self.kp_base[env_ids] = kp_base * (~compliant_base)
        self.kd_base[env_ids] = kd_base
        self.compliant_base[env_ids] = compliant_base

        kp_ee = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.kp_ee_range
        )
        kd_ee = (
            2.0
            * kp_ee.sqrt()
            * torch.empty(len(env_ids), 1, device=self.device).uniform_(
                *self.damping_ratio_range
            )
        )
        compliant_ee = (
            torch.rand(len(env_ids), 1, device=self.device) < self.compliant_ee_ratio
        )
        self.kp_ee[env_ids] = kp_ee * (~compliant_ee)
        self.kd_ee[env_ids] = kd_ee
        self.compliant_ee[env_ids] = compliant_ee

        kp_yaw = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.kp_yaw_range
        )
        kd_yaw = (
            2.0
            * kp_yaw.sqrt()
            * torch.empty(len(env_ids), 1, device=self.device).uniform_(
                *self.damping_ratio_range
            )
        )
        compliant_yaw = (
            torch.rand(len(env_ids), 1, device=self.device) < self.compliant_yaw_ratio
        )
        self.kp_yaw[env_ids] = kp_yaw * (~compliant_yaw)
        self.kd_yaw[env_ids] = kd_yaw
        self.compliant_yaw[env_ids] = compliant_yaw

        self.virtual_mass_base[env_ids] = self.default_mass_base * torch.empty(
            len(env_ids), 1, device=self.device
        ).uniform_(*self.virtual_mass_range)
        self.virtual_mass_ee[env_ids] = self.default_mass_ee * torch.empty(
            len(env_ids), 1, device=self.device
        ).uniform_(*self.virtual_mass_range)
        self.virtual_inertia_z[env_ids] = self.default_inertia_z * torch.empty(
            len(env_ids), 1, device=self.device
        ).uniform_(*self.virtual_mass_range)

    def reset(self, env_ids: torch.Tensor):
        """body related quantities are updated in the next step, can not use them."""
        self._cum_error[env_ids] = 0.0
        self._cum_count[env_ids] = 0

        self.command[env_ids] = 0.0
        self.command_hidden[env_ids] = 0.0

        self.force_ext_base_w[env_ids] = 0.0
        self.force_ext_ee_w[env_ids] = 0.0
        self.force_base_offset_b[env_ids] = 0.0

    def step(self, substep: int):
        forces_ext_base_b = quat_rotate_inverse(
            self.asset.data.body_quat_w[:, self.base_body_id],
            self.force_ext_base_w,
        )
        forces_base_b = self.asset._external_force_b[:, [self.base_body_id]].clone()
        forces_base_b += forces_ext_base_b.unsqueeze(1)
        torques_base_b = self.asset._external_torque_b[:, [self.base_body_id]].clone()
        torques_base_b += torch.cross(
            self.force_base_offset_b, forces_ext_base_b, dim=-1
        ).unsqueeze(1)
        self.asset.set_external_force_and_torque(
            forces_base_b, torques_base_b, self.base_body_id
        )

        forces_ext_ee_b = quat_rotate_inverse(
            self.asset.data.body_quat_w[:, self.ee_body_id],
            self.force_ext_ee_w,
        )
        forces_ee_b = self.asset._external_force_b[:, [self.ee_body_id]].clone()
        forces_ee_b += forces_ext_ee_b.unsqueeze(1)
        torques_ee_b = self.asset._external_torque_b[:, [self.ee_body_id]].clone()
        self.asset.set_external_force_and_torque(
            forces_ee_b, torques_ee_b, self.ee_body_id
        )

    def _update_buffers(self):
        """update pos_ee_b and linvel_ee_b.
        
        do not call from reset as body_pos_w is not updated yet."""
        pos_ee_w = (
            self.asset.data.body_pos_w[:, self.ee_body_id] - self.asset.data.root_pos_w
        )
        root_ang_vel_w_only_yaw = self.asset.data.root_ang_vel_w.clone()
        root_ang_vel_w_only_yaw[:, :2] = 0.0
        coriolis_vel_ee_w = self.asset.data.root_lin_vel_w + torch.cross(
            root_ang_vel_w_only_yaw,
            pos_ee_w,
            dim=-1,
        )
        coriolis_vel_ee_w[:, 2] = 0.0

        self.pos_ee_b[:] = yaw_rotate(-self.asset.data.heading_w[:, None], pos_ee_w)
        print(self.pos_ee_b)
        self.linvel_ee_b[:] = yaw_rotate(
            -self.asset.data.heading_w[:, None],
            self.asset.data.body_lin_vel_w[:, self.ee_body_id] - coriolis_vel_ee_w,
        )

    def _compute_error(self):
        linvel_base_error = (
            self.command_linvel_base_w - self.asset.data.root_lin_vel_w
        ).norm(dim=-1)
        pos_base_error = (self.command_pos_base_w - self.asset.data.root_pos_w).norm(
            dim=-1
        )

        linvel_ee_error = (self.command_linvel_ee_b - self.linvel_ee_b).norm(dim=-1)
        pos_ee_error = (self.command_pos_ee_b - self.pos_ee_b).norm(dim=-1)

        angvel_error = (
            self.command_yaw_vel.squeeze() - self.asset.data.root_ang_vel_w[:, 2]
        ).abs()
        yaw_error = wrap_to_pi(
            self.command_yaw_w.squeeze() - self.asset.data.heading_w
        ).abs()

        self._cum_error[:, 0].add_(linvel_base_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 1].add_(pos_base_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 2].add_(linvel_ee_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 3].add_(pos_ee_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 4].add_(angvel_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 5].add_(yaw_error * self.env.step_dt).mul_(0.99)
        self._cum_count.add_(1).mul_(0.99)

    def _integrate(self, dt: float):
        kp_base = self.kp_base.unsqueeze(1)
        kd_base = self.kd_base.unsqueeze(1)
        base_pos_diff = (
            self.command_setpoint_pos_base_w.unsqueeze(1) - self.desired_pos_base_w
        )
        base_vel_diff = 0.0 - self.desired_linvel_base_w
        self.force_spring_base_w[:] = kp_base * base_pos_diff + kd_base * base_vel_diff

        ee_setpoint_to_base_w = yaw_rotate(
            self.desired_yaw_w, self.command_setpoint_pos_ee_b[:, None, :]
        )
        ee_setpoint_pos_w = self.desired_pos_base_w + ee_setpoint_to_base_w
        des_coriolis_vel_ee_w = self.desired_linvel_base_w + torch.cross(
            self.desired_angvel_w, self.desired_pos_ee_w - self.desired_pos_base_w, dim=-1
        )
        kp_ee = self.kp_ee.unsqueeze(1)
        kd_ee = self.kd_ee.unsqueeze(1)
        ee_pos_diff = ee_setpoint_pos_w - self.desired_pos_ee_w
        ee_vel_diff = des_coriolis_vel_ee_w - self.desired_linvel_ee_w
        self.force_spring_ee_w[:] = kp_ee * ee_pos_diff + kd_ee * ee_vel_diff

        desired_linacc_base_w = (
            self.force_spring_base_w
            - self.force_spring_ee_w
            + self.force_ext_base_w.unsqueeze(1)
        ) / self.virtual_mass_base.unsqueeze(1)
        desired_linacc_base_w[:, :, 2] = 0.0
        desired_lin_acc_ee_w = (
            self.force_spring_ee_w
            + self.force_ext_ee_w.unsqueeze(1)
        ) / self.virtual_mass_ee.unsqueeze(1) 

        desired_linacc_base_w[desired_linacc_base_w.norm(dim=-1) < 0.8] = 0.0
        desired_lin_acc_ee_w[desired_lin_acc_ee_w.norm(dim=-1) < 0.8] = 0.0

        self.desired_linacc_base_w[:, :, :2] = desired_linacc_base_w[:, :, :2]
        self.desired_linvel_base_w.add_(self.desired_linacc_base_w * dt)
        self.desired_pos_base_w.add_(self.desired_linvel_base_w * dt)

        self.desired_lin_acc_ee_w[:] = clamp_norm(desired_lin_acc_ee_w, max=10.0)
        self.desired_linvel_ee_w.add_(self.desired_lin_acc_ee_w * dt)
        self.desired_linvel_ee_w[:] = clamp_norm(self.desired_linvel_ee_w, max=0.5)
        self.desired_pos_ee_w.add_(self.desired_linvel_ee_w * dt)
        desired_pos_ee_b = yaw_rotate(-self.desired_yaw_w, self.desired_pos_ee_w - self.desired_pos_base_w)
        desired_pos_ee_b = clamp_norm(desired_pos_ee_b, max=0.9)
        self.desired_pos_ee_w[:] = yaw_rotate(self.desired_yaw_w, desired_pos_ee_b) + self.desired_pos_base_w

        kp_yaw = self.kp_yaw.unsqueeze(1)
        kd_yaw = self.kd_yaw.unsqueeze(1)
        yaw_diff = wrap_to_pi(self.command_setpoint_yaw_w.unsqueeze(1) - self.desired_yaw_w)
        yaw_vel_diff = 0.0 - self.desired_yawvel_w
        self.torque_spring_yaw_w[:] = kp_yaw * yaw_diff + kd_yaw * yaw_vel_diff

        force_ext_offset_w = yaw_rotate(
            self.desired_yaw_w, self.force_base_offset_b[:, None, :]
        )  # [n, t, 3]
        torque_ext_z = torch.cross(
            force_ext_offset_w, self.force_ext_base_w[:, None, :], dim=-1
        )[:, :, 2:3]
        torque_int_z = torch.cross(
            ee_setpoint_to_base_w,
            self.force_spring_ee_w,
            dim=-1,
        )[:, :, 2:3]

        desired_yaw_acc_w = (
            self.torque_spring_yaw_w
            + torque_int_z
            + torque_ext_z
        ) / self.virtual_inertia_z.unsqueeze(1)
        desired_yaw_acc_w[desired_yaw_acc_w.abs() < 0.5] = 0.0

        self.desired_yawacc_w[:] = desired_yaw_acc_w
        self.desired_yawvel_w.add_(self.desired_yawacc_w * dt)
        self.desired_yaw_w.add_(self.desired_yawvel_w * dt)

    def _update_command(self):
        # smooth desired command
        self.command_pos_base_w[:]    = self._smooth(self.desired_pos_base_w)
        self.command_linvel_base_w[:] = self._smooth(self.desired_linvel_base_w)
        self.command_pos_ee_w[:]      = self._smooth(self.desired_pos_ee_w)
        self.command_linvel_ee_w[:]   = self._smooth(self.desired_linvel_ee_w)
        self.command_yaw_w[:]         = self._smooth(self.desired_yaw_w)
        self.command_yaw_vel[:]       = self._smooth(self.desired_yawvel_w)

        assert torch.all(self.command_linvel_base_w[:, 2] == 0.0)

        # setpoints to diff in body frame
        self.command_setpoint_pos_base_diff_b[:] = yaw_rotate(
            -self.asset.data.heading_w[:, None],
            self.command_setpoint_pos_base_w - self.asset.data.root_pos_w,
        )
        self.command_setpoint_pos_ee_diff_b[:] = (
            self.command_setpoint_pos_ee_b - self.pos_ee_b
        )
        self.command_setpoint_yaw_diff[:] = wrap_to_pi(
            self.command_setpoint_yaw_w - self.asset.data.heading_w[:, None]
        )

        # desired command to pos_diff/vel in body frame
        self.command_pos_base_diff_b[:] = yaw_rotate(
            -self.asset.data.heading_w[:, None],
            self.command_pos_base_w - self.asset.data.root_pos_w,
        )
        self.command_linvel_base_b[:] = yaw_rotate(
            -self.asset.data.heading_w[:, None], self.command_linvel_base_w
        )
        self.command_speed[:] = self.command_linvel_base_w.norm(dim=-1, keepdim=True)

        des_coriolis_vel_ee_w = self.command_linvel_base_w + torch.cross(
            self.command_angvel_w,
            self.command_pos_ee_w - self.command_pos_base_w, 
            dim=-1,
        )
        self.command_pos_ee_b[:] = yaw_rotate(
            -self.command_yaw_w,
            self.command_pos_ee_w - self.command_pos_base_w,
        )
        self.command_linvel_ee_b[:] = yaw_rotate(
            -self.command_yaw_w,
            self.command_linvel_ee_w - des_coriolis_vel_ee_w,
        )
        self.command_pos_ee_diff_b[:] = self.command_pos_ee_b - self.pos_ee_b

        self.command_yaw_diff[:] = wrap_to_pi(
            self.command_yaw_w - self.asset.data.heading_w[:, None]
        )

        # populate command tensor
        self.command[:, 0:2] = self.command_setpoint_pos_base_diff_b[:, :2] * (
            ~self.compliant_base
        )
        self.command[:, 2:3] = self.command_setpoint_yaw_diff * (~self.compliant_yaw)
        self.command[:, 3:6] = self.command_setpoint_pos_ee_diff_b * (
            ~self.compliant_ee
        )
        self.command[:, 6:8] = self.kp_base * self.command_setpoint_pos_base_diff_b[:, :2]
        self.command[:, 8:9] = self.kp_yaw * self.command_setpoint_yaw_diff
        self.command[:, 9:12] = self.kp_ee * self.command_setpoint_pos_ee_diff_b
        self.command[:, 12:13] = self.kd_base
        self.command[:, 13:14] = self.kd_yaw
        self.command[:, 14:15] = self.kd_ee
        self.command[:, 15:16] = self.virtual_mass_base
        self.command[:, 16:17] = self.virtual_inertia_z
        self.command[:, 17:18] = self.virtual_mass_ee

        self.command_hidden[:, 0:2] = self.command_pos_base_diff_b[:, :2]
        self.command_hidden[:, 2:3] = self.command_yaw_diff
        self.command_hidden[:, 3:6] = self.command_pos_ee_diff_b
        self.command_hidden[:, 6:8] = self.command_linvel_base_b[:, :2]
        self.command_hidden[:, 8:9] = self.command_yaw_vel
        self.command_hidden[:, 9:12] = self.command_linvel_ee_b

    def update(self):
        self._update_buffers()
        self._compute_error()
        self._cum_error[self.env.episode_length_buf <= self.no_command_steps] = 0.0
        self._cum_count[self.env.episode_length_buf <= self.no_command_steps] = 0

        # resample command and force
        if self.teleop:
            pass
        else:
            sample_command = ((self.env.episode_length_buf - self.no_command_steps) % self.resample_command_interval) == 0
            sample_command = sample_command.nonzero().squeeze(-1)
            if len(sample_command):
                self._sample_command(sample_command)

            self.command_setpoint_pos_base_w[:] = torch.where(
                self.continuous_walking & ~self.compliant_base,
                self.kd_base / self.kp_base * self.continuous_walking_vel + self.asset.data.root_pos_w,
                self.command_setpoint_pos_base_w,
            )

        sample_force = (torch.rand(self.num_envs, device=self.device) < self.resample_force_prob) & (self.env.episode_length_buf > self.no_command_steps)
        force_type = torch.multinomial(self.force_type_probs, self.num_envs, replacement=True)
        # print(sample_force)
        # print(self.env.episode_length_buf > self.no_command_steps)
        # print()

        wp.launch(
            maybe_sample_force,
            dim=self.num_envs,
            inputs=[
                self.cnt,
                self.constant_force_scale,
                self.constant_force_duration_range,
                self.impulse_force_momentum_scale,
                self.impulse_force_duration_range,
                self.force_offset_scale,
                wp.from_torch(sample_force, dtype=wp.bool),
                wp.from_torch(force_type.int(), dtype=wp.int32),
                wp.from_torch(self.constant_force_struct, dtype=vec5f),
                wp.from_torch(self.impulse_force_struct, dtype=vec5f),
                wp.from_torch(self.force_base_offset_b, dtype=wp.vec3),
            ],
            device=str(self.device)
        )

        constant_force = self.constant_force_magnitude * (self.constant_force_time < self.constant_force_duration)
        impulse_force_t = (self.impulse_force_time / self.impulse_force_duration).clamp(0., 1.)
        impulse_force = torch.where(
            impulse_force_t < 0.5,
            impulse_force_t * self.impulse_force_magnitude * 2,
            (1 - impulse_force_t) * self.impulse_force_magnitude * 2
        )
        self.force_ext_base_w[:] = constant_force + impulse_force
        self.constant_force_time.add_(self.env.step_dt)
        self.impulse_force_time.add_(self.env.step_dt)

        # sample eef force
        sample_ee_force = (torch.rand(self.num_envs, device=self.device) < self.eef_force_ratio * self.resample_force_prob) & (self.env.episode_length_buf > self.no_command_steps) & (self.eef_force_time >= self.eef_force_duration).squeeze(-1)
        eef_sample_ids = torch.nonzero(sample_ee_force).squeeze(-1)
        if len(eef_sample_ids):
            eef_force_duration = sample_uniform((len(eef_sample_ids), 1), self.eef_force_duration_range[0], self.eef_force_duration_range[1], device=self.device)
            eef_force_magnitude = sample_uniform((len(eef_sample_ids), 3), 0, self.eef_force_scale, device=self.device)
            self.eef_force_magnitude[eef_sample_ids] = eef_force_magnitude
            self.eef_force_duration[eef_sample_ids] = eef_force_duration
            self.eef_force_time[eef_sample_ids] = 0.0
        
        eef_constant_force = self.eef_force_magnitude * (self.eef_force_time < self.eef_force_duration)
        self.force_ext_ee_w[:] = eef_constant_force
        self.eef_force_time.add_(self.env.step_dt)
        
        self.desired_linvel_base_w[:] = self.desired_linvel_base_w.roll(1, dims=1)
        self.desired_pos_base_w[:] = self.desired_pos_base_w.roll(1, dims=1)

        self.desired_linvel_ee_w[:] = self.desired_linvel_ee_w.roll(1, dims=1)
        self.desired_pos_ee_w[:] = self.desired_pos_ee_w.roll(1, dims=1)

        self.desired_yawvel_w[:] = self.desired_yawvel_w.roll(1, dims=1)
        self.desired_yaw_w[:] = self.desired_yaw_w.roll(1, dims=1)

        self.desired_linvel_base_w[:, 0, :2] = self.asset.data.root_lin_vel_w[:, :2]
        self.desired_pos_base_w[:, 0] = self.asset.data.root_pos_w

        self.desired_linvel_ee_w[:, 0] = self.asset.ee_lin_vel_w
        self.desired_pos_ee_w[:, 0] = self.asset.ee_pos_w

        self.desired_yawvel_w[:, 0] = self.asset.data.root_ang_vel_w[:, 2:3]
        self.desired_yaw_w[:, 0] = self.asset.data.heading_w.unsqueeze(1)

        self._integrate(self.env.step_dt)

        yaw_diff = self.desired_yaw_w - self.desired_yaw_w[:, 0:1]
        self.desired_yaw_w[:] = self.desired_yaw_w[:, 0:1] + wrap_to_pi(yaw_diff)

        self._update_command()

    def _smooth(self, q: torch.Tensor):
        return (q * self.smoothing_weight.unsqueeze(-1)).sum(1)

    """
    --------------------------------------------------------------------------------
    |   setpoint_pos_ee_b --> desired_ee_pos_w --|--> desired_ee_pos_b --> reward  |
    | + desired_base_.*                          |  + real_base_pos                |
    ------------- physics integration ------------------ reward computation --------
    
    we visualize the first part to verify the correctness of physic model
    visualize the second part to verify the correctness of reward computation, to check whether the policy is learning the correct behavior
    
    """
    def _debug_draw_base(self):
        # setpoint pos for base (red/blue for stiff/compliant)
        self.env.debug_draw.point(
            self.command_setpoint_pos_base_w[~self.compliant_base.squeeze(-1)], color=(1.0, 0.0, 0.0, 1.0), size=40.0
        )
        self.env.debug_draw.point(
            self.command_setpoint_pos_base_w[self.compliant_base.squeeze(-1)], color=(0.0, 0.0, 1.0, 1.0), size=40.0
        )
        # desired pos and linvel for base (green)
        self.env.debug_draw.point(
            self.command_pos_base_w, color=(0.0, 1.0, 0.0, 1.0), size=40.0
        )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            self.command_linvel_base_w,
            color=(0.0, 1.0, 0.0, 1.0),
            size=2.0,
        )
        # real pos for base (yellow)
        self.env.debug_draw.point(
            self.asset.data.root_pos_w, color=(1.0, 1.0, 0.0, 1.0), size=40.0
        )
        # draw vector from desired pos to real pos (red/blue for stiff/compliant)
        self.env.debug_draw.vector(
            self.command_pos_base_w[~self.compliant_base.squeeze(-1)],
            self.command_setpoint_pos_base_w[~self.compliant_base.squeeze(-1)] - self.command_pos_base_w[~self.compliant_base.squeeze(-1)],
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.command_pos_base_w[self.compliant_base.squeeze(-1)],
            self.command_setpoint_pos_base_w[self.compliant_base.squeeze(-1)] - self.command_pos_base_w[self.compliant_base.squeeze(-1)],
            color=(0.0, 0.0, 1.0, 1.0),
        )

    def _debug_draw_yaw(self):
        # setpoint yaw (red/blue for stiff/compliant)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w[~self.compliant_yaw.squeeze(-1)],
            yaw_rotate(
                self.command_setpoint_yaw_w[~self.compliant_yaw.squeeze(-1)],
                torch.tensor([1.0, 0.0, 0.0], device=self.device),
            ),
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w[self.compliant_yaw.squeeze(-1)],
            yaw_rotate(
                self.command_setpoint_yaw_w[self.compliant_yaw.squeeze(-1)],
                torch.tensor([1.0, 0.0, 0.0], device=self.device),
            ),
            color=(0.0, 0.0, 1.0, 1.0),
        )
        # desired yaw (green)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            yaw_rotate(
                self.command_yaw_w,
                torch.tensor([1.0, 0.0, 0.0], device=self.device),
            ),
            color=(0.0, 1.0, 0.0, 1.0),
        )
        # real yaw (yellow)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            yaw_rotate(
                self.asset.data.heading_w.unsqueeze(1),
                torch.tensor([1.0, 0.0, 0.0], device=self.device),
            ),
            color=(1.0, 1.0, 0.0, 1.0),
        )

    def _debug_draw_desired_yaw(self):
        yaw_headings = yaw_rotate(
            self.desired_yaw_w,
            torch.tensor([1.0, 0.0, 0.0], device=self.device)[None, None, :],
        ) # [n, h, 3]
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w.unsqueeze(1).repeat(1, self.temporal_smoothing, 1).view(-1, 3),
            yaw_headings.view(-1, 3),
            color=(0.0, 1.0, 0.0, 1.0),
            size=2.0,
        )

    def _debug_draw_ee(self):
        # setpoint pos for ee (red/blue for stiff/compliant)
        setpoint_ee_w = self.asset.data.root_pos_w + yaw_rotate(
            self.asset.data.heading_w.unsqueeze(1), self.command_setpoint_pos_ee_b
        )
        self.env.debug_draw.point(
            setpoint_ee_w[~self.compliant_ee.squeeze(-1)], color=(1.0, 0.0, 0.0, 1.0), size=20.0
        )
        self.env.debug_draw.point(
            setpoint_ee_w[self.compliant_ee.squeeze(-1)], color=(0.0, 0.0, 1.0, 1.0), size=20.0
        )
        # imaginary setpoint pos for ee (red/blue for stiff/compliant)
        command_setpoint_ee_w = self.command_pos_base_w + yaw_rotate(
            self.command_yaw_w, self.command_setpoint_pos_ee_b
        )
        self.env.debug_draw.point(
            command_setpoint_ee_w[~self.compliant_ee.squeeze(-1)], color=(1.0, 0.0, 0.0, 0.5), size=30.0
        )
        self.env.debug_draw.point(
            command_setpoint_ee_w[self.compliant_ee.squeeze(-1)], color=(0.0, 0.0, 1.0, 0.5), size=30.0
        )
        # desired pos and linvel for ee (green)
        command_pos_ee_w_rew = self.asset.data.root_pos_w + yaw_rotate(
            self.asset.data.heading_w[:, None], self.command_pos_ee_b
        )
        self.env.debug_draw.point(
            command_pos_ee_w_rew, color=(0.0, 1.0, 0.0, 1.0), size=20.0
        )
        self.env.debug_draw.vector(
            self.asset.ee_pos_w,
            self.command_linvel_ee_w,
            color=(0.0, 1.0, 0.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.asset.ee_pos_w,
            self.asset.ee_lin_vel_w,
            color=(1.0, 0.0, 1.0, 1.0),
        )
        # real pos for ee (yellow)
        self.env.debug_draw.point(
            self.asset.data.body_pos_w[:, self.ee_body_id], color=(1.0, 1.0, 0.0, 1.0), size=20.0
        )

    def _debug_draw_desired_ee(self):
        # desired pos and linvel for ee (green)
        self.env.debug_draw.point(
            self.desired_pos_ee_w.reshape(-1, 3), color=(0.0, 1.0, 0.0, 1.0), size=20.0
        )

    # def _debug_draw_desired_to_setpoint(self):
    #     # command pos for base, ee and direction for yaw (green)
    #     self.env.debug_draw.point(
    #         self.command_pos_base_w, color=(0.0, 1.0, 0.0, 1.0), size=40.0
    #     )
    #     self.env.debug_draw.point(
    #         self.command_pos_ee_w, color=(0.0, 1.0, 0.0, 1.0), size=20.0
    #     )
    #     self.env.debug_draw.vector(
    #         self.asset.data.root_pos_w,
    #         yaw_rotate(
    #             self.command_yaw_w,
    #             torch.tensor([1.0, 0.0, 0.0], device=self.device),
    #         ),
    #         color=(0.0, 1.0, 0.0, 1.0),
    #     )
    #     # command lin vel for base and ee (white)
    #     self.env.debug_draw.vector(
    #         self.asset.data.root_pos_w +
    #         torch.tensor([0.0, 0.0, 0.2], device=self.device),
    #         self.command_linvel_base_w,
    #         color=(1.0, 1.0, 1.0, 1.0),
    #         size=2.0,
    #     )
    #     self.env.debug_draw.vector(
    #         self.command_pos_ee_w,
    #         self.command_linvel_ee_w,
    #         color=(1.0, 1.0, 1.0, 1.0),
    #     )

    #     command_setpoint_ee_w = self.command_pos_base_w + yaw_rotate(
    #         self.command_yaw_w, self.command_setpoint_pos_ee_b
    #     )
    #     # from desired to setpoint for base, ee (green)
    #     self.env.debug_draw.vector(
    #         self.command_pos_base_w,
    #         self.command_setpoint_pos_base_w - self.command_pos_base_w,
    #         color=(0.0, 1.0, 0.0, 1.0),
    #     )
    #     self.env.debug_draw.vector(
    #         self.command_pos_ee_w,
    #         command_setpoint_ee_w - self.command_pos_ee_w,
    #         color=(0.0, 1.0, 0.0, 1.0),
    #     )

    #     # draw setpoints (red if not compliant, blue if compliant)
    #     self.env.debug_draw.point(
    #         self.command_setpoint_pos_base_w[~self.compliant_base.squeeze(-1)], color=(1.0, 0.0, 0.0, 0.5), size=40.0
    #     )
    #     self.env.debug_draw.point(
    #         self.command_setpoint_pos_base_w[self.compliant_base.squeeze(-1)], color=(0.0, 0.0, 1.0, 0.5), size=40.0
    #     )
    #     self.env.debug_draw.point(
    #         command_setpoint_ee_w[~self.compliant_ee.squeeze(-1)], color=(1.0, 0.0, 0.0, 0.5), size=20.0
    #     )
    #     self.env.debug_draw.point(
    #         command_setpoint_ee_w[self.compliant_ee.squeeze(-1)], color=(0.0, 0.0, 1.0, 0.5), size=20.0
    #     )
    #     self.env.debug_draw.vector(
    #         self.asset.data.root_pos_w[~self.compliant_yaw.squeeze(-1)],
    #         yaw_rotate(
    #             self.command_setpoint_yaw_w[~self.compliant_yaw.squeeze(-1)],
    #             torch.tensor([1.0, 0.0, 0.0], device=self.device),
    #         ),
    #         color=(1.0, 0.0, 0.0, 0.5),
    #     )
    #     self.env.debug_draw.vector(
    #         self.asset.data.root_pos_w[self.compliant_yaw.squeeze(-1)],
    #         yaw_rotate(
    #             self.command_setpoint_yaw_w[self.compliant_yaw.squeeze(-1)],
    #             torch.tensor([0.0, 0.0, 1.0], device=self.device),
    #         ),
    #         color=(0.0, 0.0, 1.0, 0.5),
    #     )

    # def _debug_draw_real_to_desired(self):
    #     # real position for base and ee and real heading vector (yellow)
    #     self.env.debug_draw.point(
    #         self.asset.data.root_pos_w, color=(1.0, 1.0, 0.0, 1.0), size=40.0
    #     )
    #     self.env.debug_draw.point(
    #         self.asset.data.body_pos_w[:, self.ee_body_id],
    #         color=(1.0, 1.0, 0.0, 1.0),
    #         size=20.0,
    #     )
    #     self.env.debug_draw.vector(
    #         self.asset.data.root_pos_w,
    #         yaw_rotate(
    #             self.asset.data.heading_w[:, None],
    #             torch.tensor([1.0, 0.0, 0.0], device=self.device),
    #         ),
    #         color=(1.0, 1.0, 0.0, 1.0),
    #     )

    #     command_pos_ee_w_rew = self.asset.data.root_pos_w + yaw_rotate(
    #         self.asset.data.heading_w[:, None], self.command_pos_ee_b
    #     )
    #     # desire pos rew for ee (green)
    #     self.env.debug_draw.point(
    #         command_pos_ee_w_rew, color=(0.0, 1.0, 0.0, 1.0), size=20.0
    #     )
    #     # from real to desired for base, ee (yellow)
    #     self.env.debug_draw.vector(
    #         self.asset.data.root_pos_w,
    #         self.command_pos_base_w - self.asset.data.root_pos_w,
    #         color=(1.0, 1.0, 0.0, 1.0),
    #     )
    #     self.env.debug_draw.vector(
    #         self.asset.data.body_pos_w[:, self.ee_body_id],
    #         command_pos_ee_w_rew - self.asset.data.body_pos_w[:, self.ee_body_id],
    #         color=(1.0, 1.0, 0.0, 1.0),
    #     )

    def _debug_draw_forces(self):
        # force on base (orange)
        force_acc_base = self.force_ext_base_w / self.virtual_mass_base
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w
            + yaw_rotate(self.asset.data.heading_w[:, None], self.force_base_offset_b),
            force_acc_base,
            color=(1.0, 0.8, 0.0, 1.0),
            size=4.0,
        )
        # force on ee (orange)
        force_acc_ee = self.force_ext_ee_w / self.virtual_mass_ee
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.ee_body_id],
            force_acc_ee,
            color=(1.0, 0.8, 0.0, 1.0),
            size=4.0,
        )

    def debug_draw(self):
        self._debug_draw_ee()
        # self._debug_draw_base()
        # self._debug_draw_yaw()
        self._debug_draw_forces()


class BaseEEImpedanceMixed(Command):
    """Model the Base And EEF as a two body spring-damper system. The base is connected to a setpoint in world frame and the EEF is connected to a setpoint in body frame. We model forces applied  onto these two bodies as external forces. We model xy for base and xyz for ee.
    We also model a rotational spring-damper system and the corresponding torques in z direction for the base yaw.
    
    
    to check base xy policy: sample yaw to be the root heading
    1. stiff mode: large base kp, sample setpoint in the root pos, use force to check if it can resist the force
    2. stiff mode: sample setpoint around the root pos, check if it can return to setpoint
    3. compliant mode: 0 base kp, use force to check if it can follow the force
    """

    def __init__(
        self,
        env,
        ee_name: str = "arm_link06",
        ee_base_name: str = "arm_link00",
        base_setpoint_radius_range: tuple = (2.0, 3.0),
        base_setpoint_radius_range_arm_activated: tuple = (1.0, 2.0),
        yaw_setpoint_range: tuple = (-torch.pi / 2, torch.pi / 2),
        yaw_setpoint_range_arm_activated: tuple = (-torch.pi / 4, torch.pi / 4),
        kp_base_range: tuple = (2.0, 12.0),
        kp_yaw_range: tuple = (2.0, 12.0),
        kp_ee_range: tuple = (100.0, 150.0),
        damping_ratio_range: tuple = (0.7, 1.5),
        default_mass_base: float = 25.0,
        default_mass_ee: float = 1.0,
        default_inertia_z: float = 3.0,
        virtual_mass_range: tuple = (0.5, 1.5),
        compliant_ratio: float = 0.2,
        force_type_probs = (0.2, 0.4, 0.4),
        constant_force_scale = (50, 50, 10),
        constant_force_duration_range = (1, 4),
        impulse_force_velocity_scale = (2.0, 2.0, 0.5),
        impulse_force_duration_range = (0.05, 0.2),
        temporal_smoothing: int = 3,
        command_acc: bool = False,
        arm_activated_prob: float = 0.5,
        arm_activated_base_vel_max: float = 0.5,
        arm_activated_base_vel_min: float = 0.1,
    ) -> None:
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.base_body_id = self.asset.find_bodies("base")[0][0]
        self.ee_body_id = self.asset.find_bodies(ee_name)[0][0]
        self.ee_base_body_id = self.asset.find_bodies(ee_base_name)[0][0]
        self.body_ids = [self.base_body_id, self.ee_body_id]

        self.base_setpoint_radius_range = base_setpoint_radius_range
        self.base_setpoint_radius_range_arm_activated = base_setpoint_radius_range_arm_activated
        self.yaw_setpoint_range = yaw_setpoint_range
        self.yaw_setpoint_range_arm_activated = yaw_setpoint_range_arm_activated

        self.kp_base_range = kp_base_range
        self.kp_yaw_range = kp_yaw_range
        self.kp_ee_range = kp_ee_range
        self.damping_ratio_range = damping_ratio_range
        self.default_mass_base = default_mass_base
        self.default_mass_ee = default_mass_ee
        self.default_inertia_z = default_inertia_z
        self.virtual_mass_range = virtual_mass_range

        self.compliant_ratio = compliant_ratio

        self.resample_prob = 0.005
        self.temporal_smoothing = temporal_smoothing
        self.command_acc = command_acc
        self.arm_activated_prob = arm_activated_prob
        self.arm_activated_base_vel_max = arm_activated_base_vel_max
        self.arm_activated_base_vel_min = arm_activated_base_vel_min

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 23)
            self.command_hidden = torch.zeros(self.num_envs, 12)
            self.need_reset_mask = torch.zeros(self.num_envs, 1, dtype=bool)

            # integration
            self.acc_spring_base_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_linacc_base_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_linvel_base_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_pos_base_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)

            self.acc_spring_ee_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_lin_acc_ee_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_linvel_ee_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_pos_ee_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)

            self.acc_spring_yaw_w = torch.zeros(self.num_envs, self.temporal_smoothing, 1)
            self.desired_yawacc_w = torch.zeros(self.num_envs, self.temporal_smoothing, 1)
            self.desired_angvel_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self.desired_yawvel_w = self.desired_angvel_w[:, :, 2:3]
            self.desired_yaw_w = torch.zeros(self.num_envs, self.temporal_smoothing, 1)

            self.smoothing_weight = torch.full((1, self.temporal_smoothing, 1), 0.9).cumprod(1)
            self.smoothing_weight /= self.smoothing_weight.sum()

            # command setpoints in world/body frame
            self.command_setpoint_pos_base_w = torch.zeros(self.num_envs, 3)
            self.command_setpoint_pos_base_diff_b = torch.zeros(self.num_envs, 3)

            self.command_setpoint_pos_ee_b = torch.zeros(self.num_envs, 3)
            self.command_setpoint_pos_ee_diff_b = torch.zeros(self.num_envs, 3)

            self.command_setpoint_yaw_w = torch.zeros(self.num_envs, 1)
            self.command_setpoint_yaw_diff = torch.zeros(self.num_envs, 1)

            # hidden command (privileged information) be provided to tell the agent the desired behavior at the **next time step**
            self.command_pos_base_w = torch.zeros(self.num_envs, 3)
            self.command_pos_base_diff_b = torch.zeros(self.num_envs, 3)
            self.command_linvel_base_w = torch.zeros(self.num_envs, 3)
            self.command_linvel_base_b = torch.zeros(self.num_envs, 3)

            self.command_pos_ee_w = torch.zeros(self.num_envs, 3)
            self.command_pos_ee_diff_b = torch.zeros(self.num_envs, 3)
            self.command_linvel_ee_w = torch.zeros(self.num_envs, 3)
            self.command_linvel_ee_b = torch.zeros(self.num_envs, 3)

            self.command_yaw_w = torch.zeros(self.num_envs, 1)
            self.command_yaw_diff = torch.zeros(self.num_envs, 1)
            self.command_angvel_w = torch.zeros(self.num_envs, 3)
            self.command_yaw_vel = self.command_angvel_w[:, 2:3]

            # for reward computation (legacy)
            self.command_linvel = self.command_linvel_base_b
            self.command_speed = torch.zeros(self.num_envs, 1)
            self.command_angvel = self.command_yaw_vel[:, 0]

            self.command_pos_ee_b = torch.zeros(self.num_envs, 3)
            self.linvel_ee_b = torch.zeros(self.num_envs, 3)
            self.pos_ee_b = torch.zeros(self.num_envs, 3)

            # spring-damper parameters
            self.kp_base = torch.zeros(self.num_envs, 3)
            self.kd_base = torch.zeros(self.num_envs, 3)
            self.kp_ee = torch.zeros(self.num_envs, 3)
            self.kd_ee = torch.zeros(self.num_envs, 3)
            self.kp_yaw = torch.zeros(self.num_envs, 1)
            self.kd_yaw = torch.zeros(self.num_envs, 1)
            self.compliant_base = torch.zeros(self.num_envs, 1, dtype=bool)
            self.compliant_ee = torch.zeros(self.num_envs, 1, dtype=bool)
            self.compliant_yaw = torch.zeros(self.num_envs, 1, dtype=bool)

            self.default_mass_base = default_mass_base
            self.default_mass_ee = default_mass_ee
            self.default_inertia_z = default_inertia_z

            self.virtual_mass_base = torch.zeros(self.num_envs, 1)
            self.virtual_mass_ee = torch.zeros(self.num_envs, 1)
            self.virtual_inertia_z = torch.zeros(self.num_envs, 1)

            self.force_ext_base_w = torch.zeros(self.num_envs, 3)
            self.force_ext_ee_w = torch.zeros(self.num_envs, 3)
            self.force_base_offset_b = torch.zeros(self.num_envs, 3)

            self.force_type_probs = torch.tensor(force_type_probs)
            self.impulse_force_velocity_scale = torch.tensor(impulse_force_velocity_scale)
            self.impulse_force_duration_range = impulse_force_duration_range
            self.constant_force_scale = torch.tensor(constant_force_scale)
            self.constant_force_duration_range = constant_force_duration_range
        
            self.constant_force_base_struct = torch.zeros(self.num_envs, 3 + 1 + 1)
            self.impulse_force_base_struct = torch.zeros(self.num_envs, 3 + 1 + 1)
            self.constant_force_base = self.constant_force_base_struct[:, :3]
            self.constant_force_base_time = self.constant_force_base_struct[:, 3:4]
            self.constant_force_duration = self.constant_force_base_struct[:, 4:5]
            self.impulse_force_base = self.impulse_force_base_struct[:, :3]
            self.impulse_force_base_time = self.impulse_force_base_struct[:, 3:4]
            self.impulse_force_duration_base = self.impulse_force_base_struct[:, 4:5]
            self.impulse_force_duration_base.fill_(0.1)

            self.constant_force_ee_struct = torch.zeros(self.num_envs, 3 + 1 + 1) # force, duration, time
            self.impulse_force_ee_struct = torch.zeros(self.num_envs, 3 + 1 + 1)
            self.constant_force_ee = self.constant_force_ee_struct[:, :3]
            self.constant_force_ee_time = self.constant_force_ee_struct[:, 3:4]
            self.constant_force_ee_duration = self.constant_force_ee_struct[:, 4:5]
            self.impulse_force_ee = self.impulse_force_ee_struct[:, :3]
            self.impulse_force_ee_time = self.impulse_force_ee_struct[:, 3:4]
            self.impulse_force_ee_duration = self.impulse_force_ee_struct[:, 4:5]
            self.impulse_force_ee_duration.fill_(0.1)
            
            self._cum_error = torch.zeros(self.num_envs, 6)
            self._cum_count = torch.zeros(self.num_envs, 1)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.xy = torch.tensor([1.0, 1.0, 0.0])
            
            self.is_arm_activated = torch.zeros(self.num_envs, 1, dtype=bool)

        self.cnt = 0

    def _sample_command(self, env_ids: torch.Tensor):
        # sample arm activation: if not activated, the arm setpoint should be set to near rest pose, no forces on arm, no reward on arm
        self.is_arm_activated[env_ids] = torch.rand(len(env_ids), 1, device=self.device) < self.arm_activated_prob
        
        setpoint_pos_base_radius = torch.empty(len(env_ids), 1, device=self.device)
        activated_mask = self.is_arm_activated[env_ids].squeeze(-1)
        if activated_mask.sum() > 0:
            setpoint_pos_base_radius[activated_mask] = torch.empty(activated_mask.sum(), 1, device=self.device).uniform_(*self.base_setpoint_radius_range_arm_activated)
        
        non_activated_mask = ~activated_mask
        if non_activated_mask.sum() > 0:
            setpoint_pos_base_radius[non_activated_mask] = torch.empty(non_activated_mask.sum(), 1, device=self.device).uniform_(*self.base_setpoint_radius_range)

        setpoint_pos_base_yaw = torch.empty(len(env_ids), 1, device=self.device).uniform_(-torch.pi, torch.pi)
        command_setpoint_pos_base_w = torch.cat(
            [
                setpoint_pos_base_radius * torch.cos(setpoint_pos_base_yaw),
                setpoint_pos_base_radius * torch.sin(setpoint_pos_base_yaw),
                torch.zeros(len(env_ids), 1, device=self.device),
            ],
            dim=1,
        )
        self.command_setpoint_pos_base_w[env_ids] = command_setpoint_pos_base_w
        self.command_setpoint_pos_base_w[env_ids] += self.asset.data.root_pos_w[env_ids]    
        
        command_setpoint_pos_ee_b = torch.empty(len(env_ids), 3, device=self.device)
        command_setpoint_pos_ee_b[:, 0].uniform_(0.2, 0.6)
        command_setpoint_pos_ee_b[:, 1].uniform_(-0.2, 0.2)
        command_setpoint_pos_ee_b[:, 2].uniform_(0.3, 0.7)
        if non_activated_mask.sum() > 0:
            command_setpoint_pos_ee_b[non_activated_mask, 0] = 0.16
            command_setpoint_pos_ee_b[non_activated_mask, 1] = 0.0
            command_setpoint_pos_ee_b[non_activated_mask, 2] = 0.3
        self.command_setpoint_pos_ee_b[env_ids] = command_setpoint_pos_ee_b

        setpoint_yaw_w = torch.empty(len(env_ids), 1, device=self.device)
        if activated_mask.sum() > 0:
            setpoint_yaw_w[activated_mask] = torch.empty(activated_mask.sum(), 1, device=self.device).uniform_(*self.yaw_setpoint_range_arm_activated)
        if non_activated_mask.sum() > 0:
            setpoint_yaw_w[non_activated_mask] = torch.empty(non_activated_mask.sum(), 1, device=self.device).uniform_(*self.yaw_setpoint_range)
        setpoint_yaw_w += self.asset.data.heading_w[env_ids].unsqueeze(1)
        self.command_setpoint_yaw_w[env_ids] = setpoint_yaw_w

        empty = torch.empty(len(env_ids), 1, device=self.device)

        kp_base = empty.uniform_(*self.kp_base_range).clone()
        kd_base = 2.0 * kp_base.sqrt() * empty.uniform_(*self.damping_ratio_range)

        compliant_base = torch.rand(len(env_ids), 1, device=self.device) < self.compliant_ratio
        self.kp_base[env_ids] = kp_base * (~compliant_base)
        self.kd_base[env_ids] = kd_base
        self.compliant_base[env_ids] = compliant_base

        kp_ee = empty.uniform_(*self.kp_ee_range).clone()
        kd_ee = 2.0 * kp_ee.sqrt() * empty.uniform_(*self.damping_ratio_range)
        
        compliant_ee = torch.rand(len(env_ids), 1, device=self.device) < self.compliant_ratio
        self.kp_ee[env_ids] = kp_ee * (~compliant_ee)
        self.kd_ee[env_ids] = kd_ee
        self.compliant_ee[env_ids] = compliant_ee

        kp_yaw = empty.uniform_(*self.kp_yaw_range).clone()
        kd_yaw = 2.0 * kp_yaw.sqrt() * empty.uniform_(*self.damping_ratio_range)
        
        compliant_yaw = torch.rand(len(env_ids), 1, device=self.device) < self.compliant_ratio
        self.kp_yaw[env_ids] = kp_yaw * (~compliant_yaw)
        self.kd_yaw[env_ids] = kd_yaw
        self.compliant_yaw[env_ids] = compliant_yaw

        self.virtual_mass_base[env_ids] = self.default_mass_base * empty.uniform_(*self.virtual_mass_range)
        self.virtual_mass_ee[env_ids] = self.default_mass_ee * empty.uniform_(*self.virtual_mass_range)
        self.virtual_inertia_z[env_ids] = self.default_inertia_z * empty.uniform_(*self.virtual_mass_range)

    def reset(self, env_ids: torch.Tensor):
        """body related quantities are updated in the next step, can not use them."""
        self._sample_command(env_ids)

        self._cum_error[env_ids] = 0.0
        
        self.command[env_ids] = 0.0
        self.command_hidden[env_ids] = 0.0

        self.need_reset_mask[env_ids] = True
        # because we can not use the body related quantities in reset, we need to reset the desired body pos and linvel in command.update

    def step(self, substep: int):
        forces_ext_b = self.asset._external_force_b.clone()
        torques_ext_b = self.asset._external_torque_b.clone()
        
        forces_ext_base_b = quat_rotate_inverse(
            self.asset.data.body_quat_w[:, self.base_body_id],
            self.force_ext_base_w,
        )
        forces_ext_b[:, self.base_body_id] += forces_ext_base_b
        torques_ext_b[:, self.base_body_id] += torch.cross(self.force_base_offset_b, forces_ext_base_b, dim=-1)

        forces_ext_ee_b = quat_rotate_inverse(
            self.asset.data.body_quat_w[:, self.ee_body_id],
            self.force_ext_ee_w,
        )
        forces_ext_b[:, self.ee_body_id] += forces_ext_ee_b
        self.asset.set_external_force_and_torque(forces_ext_b, torques_ext_b)

    def _update_buffers(self):
        """update pos_ee_b and linvel_ee_b.
        
        do not call from reset as body_pos_w is not updated yet."""
        pos_ee_w = (
            self.asset.data.body_pos_w[:, self.ee_body_id] - self.asset.data.root_pos_w
        )
        root_ang_vel_w_only_yaw = self.asset.data.root_ang_vel_w.clone()
        root_ang_vel_w_only_yaw[:, :2] = 0.0
        coriolis_vel_ee_w = self.asset.data.root_lin_vel_w * self.xy + torch.cross(
            root_ang_vel_w_only_yaw,
            pos_ee_w,
            dim=-1,
        )

        self.pos_ee_b[:] = yaw_rotate(-self.asset.data.heading_w[:, None], pos_ee_w)
        self.linvel_ee_b[:] = yaw_rotate(
            -self.asset.data.heading_w[:, None],
            self.asset.data.body_lin_vel_w[:, self.ee_body_id] - coriolis_vel_ee_w,
        )

    def _compute_error(self):
        linvel_base_error = (
            self.command_linvel_base_w - self.asset.data.root_lin_vel_w
        ).norm(dim=-1)
        pos_base_error = (self.command_pos_base_w - self.asset.data.root_pos_w).norm(
            dim=-1
        )

        linvel_ee_error = (self.command_linvel_ee_b - self.linvel_ee_b).norm(dim=-1)
        pos_ee_error = (self.command_pos_ee_b - self.pos_ee_b).norm(dim=-1)

        angvel_error = (
            self.command_yaw_vel.squeeze() - self.asset.data.root_ang_vel_w[:, 2]
        ).abs()
        yaw_error = wrap_to_pi(
            self.command_yaw_w.squeeze() - self.asset.data.heading_w
        ).abs()

        self._cum_error[:, 0].add_(linvel_base_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 1].add_(pos_base_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 2].add_(linvel_ee_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 3].add_(pos_ee_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 4].add_(angvel_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 5].add_(yaw_error * self.env.step_dt).mul_(0.99)
        self._cum_count.add_(1).mul_(0.99)

    def _integrate(self):
        kp_base = self.kp_base.unsqueeze(1)
        kd_base = self.kd_base.unsqueeze(1)
        base_pos_diff = (
            self.command_setpoint_pos_base_w.unsqueeze(1) - self.desired_pos_base_w
        )
        base_vel_diff = 0.0 - self.desired_linvel_base_w
        self.acc_spring_base_w[:] = kp_base * base_pos_diff + kd_base * base_vel_diff

        ee_setpoint_to_base_w = yaw_rotate(
            self.desired_yaw_w, self.command_setpoint_pos_ee_b[:, None, :]
        )
        ee_setpoint_pos_w = self.desired_pos_base_w + ee_setpoint_to_base_w
        des_coriolis_vel_ee_w = self.desired_linvel_base_w + torch.cross(
            self.desired_angvel_w, self.desired_pos_ee_w - self.desired_pos_base_w, dim=-1
        )
        kp_ee = self.kp_ee.unsqueeze(1)
        kd_ee = self.kd_ee.unsqueeze(1)
        ee_pos_diff = ee_setpoint_pos_w - self.desired_pos_ee_w
        ee_vel_diff = des_coriolis_vel_ee_w - self.desired_linvel_ee_w
        self.acc_spring_ee_w[:] = kp_ee * ee_pos_diff + kd_ee * ee_vel_diff

        desired_linacc_base_w = (
            self.acc_spring_base_w
            - self.acc_spring_ee_w
            * (self.virtual_mass_ee / self.virtual_mass_base).unsqueeze(1)
            + (self.force_ext_base_w / self.virtual_mass_base)[:, None, :]
        )
        desired_lin_acc_ee_w = (
            self.acc_spring_ee_w
            + (self.force_ext_ee_w / self.virtual_mass_ee)[:, None, :]
        )

        self.desired_linacc_base_w[:] = desired_linacc_base_w * self.xy
        self.desired_linvel_base_w.add_(
            self.desired_linacc_base_w * self.env.physics_dt
        )
        if not torch.all(self.desired_linvel_base_w[:, :, 2] == 0.0):
            breakpoint()
        self.desired_pos_base_w.add_(self.desired_linvel_base_w * self.env.physics_dt)

        self.desired_lin_acc_ee_w[:] = desired_lin_acc_ee_w
        self.desired_linvel_ee_w.add_(self.desired_lin_acc_ee_w * self.env.physics_dt)
        self.desired_pos_ee_w.add_(self.desired_linvel_ee_w * self.env.physics_dt)
        if torch.isnan(self.desired_pos_ee_w).any():
            breakpoint()

        kp_yaw = self.kp_yaw.unsqueeze(1)
        kd_yaw = self.kd_yaw.unsqueeze(1)
        yaw_diff = wrap_to_pi(self.command_setpoint_yaw_w.unsqueeze(1) - self.desired_yaw_w)
        yaw_vel_diff = 0.0 - self.desired_yawvel_w
        self.acc_spring_yaw_w[:] = kp_yaw * yaw_diff + kd_yaw * yaw_vel_diff

        force_ext_offset_w = yaw_rotate(
            self.desired_yaw_w, self.force_base_offset_b[:, None, :]
        )  # [n, t, 3]
        torque_ext_z = torch.cross(
            force_ext_offset_w, self.force_ext_base_w[:, None, :], dim=-1
        )[:, :, 2:3]
        torque_int_z = torch.cross(
            ee_setpoint_to_base_w,
            -self.virtual_mass_ee[:, None, :] * self.acc_spring_ee_w,
            dim=-1,
        )[:, :, 2:3]
        torque_z = torque_ext_z + torque_int_z

        desired_yaw_acc_w = self.acc_spring_yaw_w + (
            torque_z / self.virtual_inertia_z[:, None, :]
        )

        self.desired_yawacc_w[:] = desired_yaw_acc_w
        self.desired_yawvel_w.add_(self.desired_yawacc_w * self.env.physics_dt)
        self.desired_yaw_w.add_(self.desired_yawvel_w * self.env.physics_dt)

    def _update_command(self):
        # smooth desired command
        self.command_pos_base_w[:] = (self.desired_pos_base_w * self.smoothing_weight).sum(1)
        self.command_linvel_base_w[:] = (self.desired_linvel_base_w * self.smoothing_weight).sum(1)
        self.command_pos_ee_w[:] = (self.desired_pos_ee_w * self.smoothing_weight).sum(1)
        self.command_linvel_ee_w[:] = (self.desired_linvel_ee_w * self.smoothing_weight).sum(1)
        self.command_yaw_w[:] = (self.desired_yaw_w * self.smoothing_weight).sum(1)
        self.command_yaw_vel[:] = (self.desired_yawvel_w * self.smoothing_weight).sum(1)

        assert torch.all(self.command_linvel_base_w[:, 2] == 0.0)

        # setpoints to diff in body frame
        self.command_setpoint_pos_base_diff_b[:] = yaw_rotate(
            -self.asset.data.heading_w[:, None],
            self.command_setpoint_pos_base_w - self.asset.data.root_pos_w,
        )
        self.command_setpoint_pos_ee_diff_b[:] = (
            self.command_setpoint_pos_ee_b - self.pos_ee_b
        )
        self.command_setpoint_yaw_diff[:] = wrap_to_pi(
            self.command_setpoint_yaw_w - self.asset.data.heading_w[:, None]
        )

        # desired command to pos_diff/vel in body frame
        self.command_pos_base_diff_b[:] = yaw_rotate(
            -self.asset.data.heading_w[:, None],
            self.command_pos_base_w - self.asset.data.root_pos_w,
        )
        self.command_linvel_base_b[:] = yaw_rotate(
            -self.asset.data.heading_w[:, None], self.command_linvel_base_w
        )
        self.command_speed[:] = self.command_linvel_base_w.norm(dim=-1, keepdim=True)

        des_coriolis_vel_ee_w = self.command_linvel_base_w + torch.cross(
            self.command_angvel_w,
            self.command_pos_ee_w - self.command_pos_base_w, 
            dim=-1,
        )
        self.command_pos_ee_b[:] = yaw_rotate(
            -self.command_yaw_w,
            self.command_pos_ee_w - self.command_pos_base_w,
        )
        self.command_linvel_ee_b[:] = yaw_rotate(
            -self.command_yaw_w,
            self.command_linvel_ee_w - des_coriolis_vel_ee_w,
        )
        self.command_pos_ee_diff_b[:] = self.command_pos_ee_b - self.pos_ee_b

        self.command_yaw_diff[:] = wrap_to_pi(
            self.command_yaw_w - self.asset.data.heading_w[:, None]
        )

        # populate command tensor
        self.command[:, 0:2] = self.command_setpoint_pos_base_diff_b[:, :2]
        self.command[:, 2:3] = self.command_setpoint_yaw_diff
        self.command[:, 3:6] = self.command_setpoint_pos_ee_diff_b

        self.command[:, 6:9] = self.kp_base
        self.command[:, 9:12] = self.kd_base
        self.command[:, 12:15] = self.kp_ee
        self.command[:, 15:18] = self.kd_ee
        self.command[:, 18:19] = self.kp_yaw
        self.command[:, 19:20] = self.kd_yaw
        if self.command_acc:
            self.command[:, 6:9] *= self.command_setpoint_pos_base_diff_b
            self.command[:, 12:15] *= self.command_setpoint_pos_ee_diff_b
            self.command[:, 18:19] *= self.command_setpoint_yaw_diff
        self.command[:, 20:21] = self.virtual_mass_base
        self.command[:, 21:22] = self.virtual_mass_ee
        self.command[:, 22:23] = self.virtual_inertia_z

        self.command_hidden[:, 0:2] = self.command_pos_base_diff_b[:, :2]
        self.command_hidden[:, 2:3] = self.command_yaw_diff
        self.command_hidden[:, 3:6] = self.command_pos_ee_diff_b
        self.command_hidden[:, 6:8] = self.command_linvel_base_b[:, :2]
        self.command_hidden[:, 8:9] = self.command_yaw_vel
        self.command_hidden[:, 9:12] = self.command_linvel_ee_b

    def update(self):
        self._update_buffers()
        self._compute_error()

        # if env is reset last step, reset cum error and the desired pos and linvel buffers
        env_ids = self.need_reset_mask.squeeze(-1)
        self._cum_error[env_ids] = 0.0
        self._cum_count[env_ids] = 0.0
        # print((self._cum_error / self._cum_count / self.env.step_dt).mean(0))
        self.desired_linacc_base_w[env_ids] = 0.0
        if not torch.all(self.desired_linvel_base_w[:, :, 2] == 0.0):
            breakpoint()
        self.desired_linvel_base_w[env_ids] = (
            self.asset.data.root_lin_vel_w[env_ids, None] * self.xy
        )
        if not torch.all(self.desired_linvel_base_w[:, :, 2] == 0.0):
            breakpoint()

        self.desired_pos_base_w[env_ids] = self.asset.data.root_pos_w[env_ids, None]

        self.desired_lin_acc_ee_w[env_ids] = 0.0
        self.desired_linvel_ee_w[env_ids] = self.asset.data.body_lin_vel_w[
            env_ids, None, self.ee_body_id
        ]
        self.desired_pos_ee_w[env_ids] = self.asset.data.body_pos_w[
            env_ids, None, self.ee_body_id
        ]
        if torch.isnan(self.desired_pos_ee_w).any():
            breakpoint()

        self.desired_yawacc_w[env_ids] = 0.0
        self.desired_yawvel_w[env_ids] = self.asset.data.root_ang_vel_w[
            env_ids, None, 2:3
        ]
        self.desired_yaw_w[env_ids] = self.asset.data.heading_w[env_ids, None, None]
        self.need_reset_mask[:] = False

        # resample command and force
        sample_command = (
            torch.rand(self.num_envs, device=self.device) < self.resample_prob
        )
        sample_command = sample_command.nonzero().squeeze(-1)
        if len(sample_command):
            self._sample_command(sample_command)

        sample_force_base = torch.rand(self.num_envs, device=self.device) < self.resample_prob
        force_type_base = torch.multinomial(self.force_type_probs, self.num_envs, replacement=True)
        wp.launch(
            maybe_sample_force,
            dim=self.num_envs,
            inputs=[
                self.cnt,
                self.constant_force_scale,
                self.constant_force_duration_range,
                self.impulse_force_velocity_scale * self.default_mass_base,
                self.impulse_force_duration_range,
                wp.from_torch(sample_force_base, dtype=wp.bool),
                wp.from_torch(force_type_base.int(), dtype=wp.int32),
                wp.from_torch(self.constant_force_base_struct, dtype=vec5f),
                wp.from_torch(self.impulse_force_base_struct, dtype=vec5f),
            ],
            device=str(self.device)
        )
        self.cnt += 1
        force_offset_b = torch.zeros(len(sample_force_base.nonzero()), 3, device=self.device)
        force_offset_b[:, 0].uniform_(-0.3, 0.3)
        force_offset_b[:, 1].uniform_(-0.2, 0.2)
        self.force_base_offset_b[sample_force_base] = force_offset_b

        sample_force_ee = torch.rand(self.num_envs, device=self.device) < self.resample_prob
        force_type_ee = torch.multinomial(self.force_type_probs, self.num_envs, replacement=True)
        wp.launch(
            maybe_sample_force,
            dim=self.num_envs,
            inputs=[
                self.cnt,
                self.constant_force_scale,
                self.constant_force_duration_range,
                self.impulse_force_velocity_scale * self.default_mass_ee,
                self.impulse_force_duration_range,
                wp.from_torch(sample_force_ee, dtype=wp.bool),
                wp.from_torch(force_type_ee.int(), dtype=wp.int32),
                wp.from_torch(self.constant_force_ee_struct, dtype=vec5f),
                wp.from_torch(self.impulse_force_ee_struct, dtype=vec5f),
            ],
            device=str(self.device)
        )
        self.cnt += 1

        # close-loop adjustments of desired quantities
        self.desired_linvel_base_w[:] = self.desired_linvel_base_w.roll(1, dims=1)
        self.desired_pos_base_w[:] = self.desired_pos_base_w.roll(1, dims=1)
        self.desired_linvel_ee_w[:] = self.desired_linvel_ee_w.roll(1, dims=1)
        self.desired_pos_ee_w[:] = self.desired_pos_ee_w.roll(1, dims=1)
        self.desired_yawvel_w[:] = self.desired_yawvel_w.roll(1, dims=1)
        self.desired_yaw_w[:] = self.desired_yaw_w.roll(1, dims=1)

        self.desired_linvel_base_w[:, 0] = self.asset.data.root_lin_vel_w * self.xy
        self.desired_pos_base_w[:, 0] = self.asset.data.root_pos_w
        self.desired_linvel_ee_w[:, 0] = self.asset.data.body_lin_vel_w[
            :, self.ee_body_id
        ]
        self.desired_pos_ee_w[:, 0] = self.asset.data.body_pos_w[:, self.ee_body_id]
        self.desired_yawvel_w[:, 0] = self.asset.data.root_ang_vel_w[:, 2:3]
        self.desired_yaw_w[:, 0] = self.asset.data.heading_w.unsqueeze(1)

        if torch.isnan(self.desired_pos_ee_w).any():
            breakpoint()

        # update forces and times
        constant_force_base = self.constant_force_base * (self.constant_force_base_time < self.constant_force_duration)
        impulse_force_base_t = (self.impulse_force_base_time / self.impulse_force_duration_base).clamp(0., 1.)
        impulse_force = torch.minimum(impulse_force_base_t, 1 - impulse_force_base_t)* self.impulse_force_base * 2
        self.force_ext_base_w[:] = constant_force_base + impulse_force
        self.constant_force_base_time.add_(self.env.step_dt)
        self.impulse_force_base_time.add_(self.env.step_dt)
        
        constant_force_ee = self.constant_force_ee * (self.constant_force_ee_time < self.constant_force_ee_duration)
        impulse_force_ee_t = (self.impulse_force_ee_time / self.impulse_force_ee_duration).clamp(0., 1.)
        impulse_force_ee = torch.minimum(impulse_force_ee_t, 1 - impulse_force_ee_t)* self.impulse_force_ee * 2
        self.force_ext_ee_w[:] = constant_force_ee + impulse_force_ee
        self.constant_force_ee_time.add_(self.env.step_dt)
        self.impulse_force_ee_time.add_(self.env.step_dt)
        self.force_ext_ee_w *= self.is_arm_activated

        for _ in range(int(self.env.step_dt / self.env.physics_dt)):
            self._integrate()

        yaw_diff = self.desired_yaw_w - self.desired_yaw_w[:, 0:1]
        self.desired_yaw_w[:] = self.desired_yaw_w[:, 0:1] + wrap_to_pi(yaw_diff)
        
        self._update_command()

    """
    --------------------------------------------------------------------------------
    |   setpoint_pos_ee_b --> desired_ee_pos_w --|--> desired_ee_pos_b --> reward  |
    | + desired_base_.*                          |  + real_base_pos                |
    ------------- physics integration ------------------ reward computation --------
    
    we visualize the first part to verify the correctness of physic model
    visualize the second part to verify the correctness of reward computation, to check whether the policy is learning the correct behavior
    
    """
    def _debug_draw_base(self):
        # setpoint pos for base (red/blue for stiff/compliant)
        self.env.debug_draw.point(
            self.command_setpoint_pos_base_w[~self.compliant_base.squeeze(-1)], color=(1.0, 0.0, 0.0, 1.0), size=40.0
        )
        self.env.debug_draw.point(
            self.command_setpoint_pos_base_w[self.compliant_base.squeeze(-1)], color=(0.0, 0.0, 1.0, 1.0), size=40.0
        )
        # desired pos and linvel for base (green)
        self.env.debug_draw.point(
            self.command_pos_base_w, color=(0.0, 1.0, 0.0, 1.0), size=40.0
        )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            self.command_linvel_base_w,
            color=(0.0, 1.0, 0.0, 1.0),
            size=2.0,
        )
        # real pos for base (yellow)
        self.env.debug_draw.point(
            self.asset.data.root_pos_w, color=(1.0, 1.0, 0.0, 1.0), size=40.0
        )
        # draw vector from desired pos to real pos (red/blue for stiff/compliant)
        self.env.debug_draw.vector(
            self.command_pos_base_w[~self.compliant_base.squeeze(-1)],
            self.command_setpoint_pos_base_w[~self.compliant_base.squeeze(-1)] - self.command_pos_base_w[~self.compliant_base.squeeze(-1)],
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.command_pos_base_w[self.compliant_base.squeeze(-1)],
            self.command_setpoint_pos_base_w[self.compliant_base.squeeze(-1)] - self.command_pos_base_w[self.compliant_base.squeeze(-1)],
            color=(0.0, 0.0, 1.0, 1.0),
        )
            
    
    def _debug_draw_yaw(self):
        # setpoint yaw (red/blue for stiff/compliant)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w[~self.compliant_yaw.squeeze(-1)],
            yaw_rotate(
                self.command_setpoint_yaw_w[~self.compliant_yaw.squeeze(-1)],
                torch.tensor([1.0, 0.0, 0.0], device=self.device),
            ),
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w[self.compliant_yaw.squeeze(-1)],
            yaw_rotate(
                self.command_setpoint_yaw_w[self.compliant_yaw.squeeze(-1)],
                torch.tensor([1.0, 0.0, 0.0], device=self.device),
            ),
            color=(0.0, 0.0, 1.0, 1.0),
        )
        # desired yaw (green)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            yaw_rotate(
                self.command_yaw_w,
                torch.tensor([1.0, 0.0, 0.0], device=self.device),
            ),
            color=(0.0, 1.0, 0.0, 1.0),
        )
        # real yaw (yellow)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            yaw_rotate(
                self.asset.data.heading_w.unsqueeze(1),
                torch.tensor([1.0, 0.0, 0.0], device=self.device),
            ),
            color=(1.0, 1.0, 0.0, 1.0),
        )
        
    def _debug_draw_ee(self):
        # setpoint pos for ee (red/blue for stiff/compliant)
        setpoint_ee_w = self.asset.data.root_pos_w + yaw_rotate(
            self.asset.data.heading_w.unsqueeze(1), self.command_setpoint_pos_ee_b
        )
        self.env.debug_draw.point(
            setpoint_ee_w[~self.compliant_ee.squeeze(-1)], 
            color=(1.0, 0.0, 0.0, 1.0), size=20.0
        )
        self.env.debug_draw.point(
            setpoint_ee_w[self.compliant_ee.squeeze(-1)], 
            color=(0.0, 0.0, 1.0, 1.0), size=20.0
        )
        # imaginary setpoint pos for ee (red/blue for stiff/compliant)
        command_setpoint_ee_w = self.command_pos_base_w + yaw_rotate(
            self.command_yaw_w, self.command_setpoint_pos_ee_b
        )
        self.env.debug_draw.point(
            command_setpoint_ee_w[~self.compliant_ee.squeeze(-1)], 
            color=(1.0, 0.0, 0.0, 0.5), size=30.0
        )
        self.env.debug_draw.point(
            command_setpoint_ee_w[self.compliant_ee.squeeze(-1)], 
            color=(0.0, 0.0, 1.0, 0.5), size=30.0
        )
        # desired pos and linvel for ee (green)
        command_pos_ee_w_rew = self.asset.data.root_pos_w + yaw_rotate(
            self.asset.data.heading_w[:, None], self.command_pos_ee_b
        )
        self.env.debug_draw.point(
            command_pos_ee_w_rew, color=(0.0, 1.0, 0.0, 1.0), size=20.0
        )
        self.env.debug_draw.vector(
            command_pos_ee_w_rew,
            self.command_linvel_ee_w,
            color=(0.0, 1.0, 0.0, 1.0),
        )
        # real pos for ee (yellow)
        self.env.debug_draw.point(
            self.asset.data.body_pos_w[:, self.ee_body_id], color=(1.0, 1.0, 0.0, 1.0), size=20.0
        )
        
    def _debug_draw_desired_ee(self):
        # desired pos and linvel for ee (green)
        self.env.debug_draw.point(
            self.desired_pos_ee_w.reshape(-1, 3), color=(0.0, 1.0, 0.0, 1.0), size=20.0
        )
        

    def _debug_draw_forces(self):
        # force on base (orange)
        force_acc_base = self.force_ext_base_w / self.virtual_mass_base
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w
            + yaw_rotate(self.asset.data.heading_w[:, None], self.force_base_offset_b),
            force_acc_base,
            color=(1.0, 0.8, 0.0, 1.0),
            size=4.0,
        )
        # force on ee (orange)
        force_acc_ee = self.force_ext_ee_w / self.virtual_mass_ee
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.ee_body_id],
            force_acc_ee,
            color=(1.0, 0.8, 0.0, 1.0),
            size=4.0,
        )
    
    def debug_draw(self):
        self._debug_draw_ee()
        self._debug_draw_base()
        self._debug_draw_yaw()
        self._debug_draw_forces()


def sample_disk(size, radius_range=(0., 1.0), device=None):
    r = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    theta = torch.rand(size, device=device) * 2 * torch.pi
    return torch.stack([
        r * torch.cos(theta), 
        r * torch.sin(theta), 
        torch.zeros_like(theta)
    ], dim=-1)


@wp.func
def sample_uniform_wp(rng: wp.uint32, range: wp.vec2) -> float:
    return wp.randf(rng) * (range[1] - range[0]) + range[0]

vec5f = wp.vec(length=5, dtype=wp.float32)

@wp.kernel
def maybe_sample_force(
    kernel_seed: int,
    const_force_scale: wp.vec3,
    const_force_duration_range: wp.vec2,
    impulse_force_scale: wp.vec3,
    impulse_force_duration_range: wp.vec2,
    force_offset_scale: wp.vec3,
    sample_force: wp.array(dtype=wp.bool),
    force_type: wp.array(dtype=wp.int32),
    const_force: wp.array(dtype=vec5f),
    impulse_force: wp.array(dtype=vec5f),
    force_offset: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if sample_force[tid]:
        if force_type[tid] == 0:
            pass
        elif (force_type[tid] == 1) and (const_force[tid][4] > const_force[tid][3]):
            rng = wp.rand_init(kernel_seed, tid)
            xy = wp.cw_mul(wp.sample_unit_cube(rng), const_force_scale)
            duration = sample_uniform_wp(rng, const_force_duration_range)
            const_force[tid] = vec5f(xy[0], xy[1], 0., duration, 0.)
        elif (force_type[tid] == 2) and (impulse_force[tid][4] > impulse_force[tid][3]):
            rng = wp.rand_init(kernel_seed, tid)
            xy = wp.cw_mul(wp.sample_unit_cube(rng), impulse_force_scale)
            duration = sample_uniform_wp(rng, impulse_force_duration_range)
            xy = xy / duration
            impulse_force[tid] = vec5f(xy[0], xy[1], 0., duration, 0.)
        
        rng = wp.rand_init(kernel_seed, tid + 1)
        force_offset[tid] = wp.cw_mul(wp.sample_unit_cube(rng), force_offset_scale)
