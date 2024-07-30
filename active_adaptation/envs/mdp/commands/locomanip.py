import torch
import einops

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat, wrap_to_pi, quat_from_euler_xyz, quat_mul, quat_inv, euler_xyz_from_quat, normalize
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify
from .locomotion import Command, sample_quat_yaw, sample_uniform, clamp_norm
from tensordict import TensorDict
from .generate_command_traj import generate_random_trajectories

from typing import Dict, Optional
import wandb

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

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
        # print(ee_pos_error)
        
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
        force_b = self.asset._external_force_b.clone()
        torque_b = self.asset._external_torque_b.clone()
        
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

        force_b[:, self.ee_id] = self.ee_force_b
        
        self.asset.set_external_force_and_torque(force_b, torque_b)

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
        
        