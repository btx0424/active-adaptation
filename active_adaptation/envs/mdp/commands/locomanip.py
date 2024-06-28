import torch
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat, wrap_to_pi, quat_from_euler_xyz, quat_mul
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from .locomotion import Command, sample_quat_yaw, sample_uniform, clamp_norm

class CommandEEPose(Command):
    def __init__(
        self, 
        env,
        ee_name: str,
        pitch_range: tuple = (-torch.pi / 3, 0.),
        angvel_range=(-1., 1.),
    ) -> None:
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
        assert len(self.ee_id) == 1
        self.ee_id = self.ee_id[0]

        self.pitch_range = pitch_range
        self.angvel_range = angvel_range
        self.resample_interval = 300
        self.resample_prob = 0.5

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 2 + 1 + 3 + 3)
            self.target_yaw = torch.zeros(self.num_envs)
            self._command_linvel = torch.zeros(self.num_envs, 3)
            self._command_speed = torch.zeros(self.num_envs, 1)
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
        yaw = torch.zeros(env_ids.shape, device=self.device).uniform_(-torch.pi, torch.pi)
        pitch = torch.zeros(env_ids.shape, device=self.device).uniform_(-torch.pi / 3, 0.)
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

