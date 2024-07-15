import torch
import einops

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat, wrap_to_pi, quat_from_euler_xyz, quat_mul, quat_inv, euler_xyz_from_quat, normalize
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify
from .locomotion import Command, sample_quat_yaw, sample_uniform, clamp_norm
from tensordict import TensorDict

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

def pitch_yaw_to_vec(pitch, yaw):
    """Convert pitch and yaw to a 3D unit vector.
    [...] [...] -> [..., 3]"""
    return torch.stack([
        torch.cos(yaw) * torch.cos(pitch),
        torch.sin(yaw) * torch.cos(pitch),
        -torch.sin(pitch)
    ], dim=-1)

def vec_to_pitch_yaw(vec):
    """Convert a 3D unit vector to pitch and yaw.
    [..., 3] -> [...], [...]"""
    pitch = -torch.asin(vec[..., 2])
    yaw = torch.atan2(vec[..., 1], vec[..., 0])
    return pitch, yaw

def slerp_vec(v1, v2, t):
    """Spherical linear interpolation for vectors.
    [..., 3] [..., 3] [...] -> [..., 3]
    """
    dot = torch.sum(v1 * v2, dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Handle cases where vectors are very close
    mask = sin_theta > 1e-6
    t1 = torch.where(mask, torch.sin((1 - t) * theta) / sin_theta, 1 - t)
    t2 = torch.where(mask, torch.sin(t * theta) / sin_theta, t)

    return normalize(v1 * t1.unsqueeze(-1) + v2 * t2.unsqueeze(-1))


def interpolate_position(last_target_pos_radius_pitch_yaw, next_target_pos_radius_pitch_yaw, command_alpha, yaw_range):
    last_r, last_pitch, last_yaw = last_target_pos_radius_pitch_yaw.unbind(dim=-1)
    next_r, next_pitch, next_yaw = next_target_pos_radius_pitch_yaw.unbind(dim=-1)

    current_r = last_r + command_alpha * (next_r - last_r)
    current_pitch = slerp_angle(last_pitch, next_pitch, command_alpha)
    current_yaw = slerp_angle_with_limit(last_yaw, next_yaw, command_alpha, yaw_range)
    
    return current_r, current_pitch, current_yaw

def interpolate_orientation(last_target_ori_roll_pitch_yaw, next_target_ori_roll_pitch_yaw, command_alpha):
    last_roll, last_ori_pitch, last_ori_yaw = last_target_ori_roll_pitch_yaw.unbind(dim=-1)
    next_roll, next_ori_pitch, next_ori_yaw = next_target_ori_roll_pitch_yaw.unbind(dim=-1)

    current_roll = last_roll + command_alpha * wrap_to_pi(next_roll - last_roll)
    last_ori_dir = pitch_yaw_to_vec(last_ori_pitch, last_ori_yaw)
    next_ori_dir = pitch_yaw_to_vec(next_ori_pitch, next_ori_yaw)
    current_ori_dir = slerp_vec(last_ori_dir, next_ori_dir, command_alpha)
    current_ori_pitch, current_ori_yaw = vec_to_pitch_yaw(current_ori_dir)

    return current_roll, current_ori_pitch, current_ori_yaw

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
