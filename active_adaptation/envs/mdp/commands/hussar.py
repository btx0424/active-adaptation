import torch


from .base import Command
from active_adaptation.envs.mdp import reward, termination, observation
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat,
    quat_from_yaw,
    wrap_to_pi
)


class LocoNavigation(Command):
    def __init__(self, env):
        super().__init__(env)

        self.resample_interval = 50
        self.resample_distance_thres = 0.2
        self.resample_yaw_thres = 0.2

        with torch.device(self.device):
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.target_rpy_w = torch.zeros(self.num_envs, 3)
            self.time_alloted = torch.zeros(self.num_envs, 1)
            self.time_elapsed = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)

        if self.env.sim.has_gui() and self.env.backend == "isaac":
            from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
            self.frame_marker = VisualizationMarkers(
                FRAME_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/target_pose",
                )
            )
            self.frame_marker.set_visibility(True)
        
        self.update()

    @property
    def command(self):
        target_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.target_pos_w - self.asset.data.root_pos_w
        )
        yaw_diff = wrap_to_pi(self.target_rpy_w[:, 2] - self.asset.data.heading_w)
        return torch.cat([
            target_pos_b[:, :2], # 2
            yaw_diff[:, None], # 1
            self.time_elapsed, # 1
            self.time_alloted - self.time_elapsed # 1
        ], dim=-1) # [num_envs, 5]
    
    def symmetry_transforms(self):
        return SymmetryTransform(
            perm=torch.arange(5),
            signs=torch.tensor([1, -1, -1, 1, 1])
        )
    
    @reward
    def velocity_direction(self) -> torch.Tensor:
        speed_xy = self.asset.data.root_lin_vel_w[:, :2].norm(dim=-1, keepdim=True)
        rew = torch.sum(self.pos_diff[:, :2] * self.asset.data.root_lin_vel_w[:, :2], dim=-1, keepdim=True)\
            .div(self.pos_diff_norm * speed_xy)
        return rew.reshape(self.num_envs, 1)

    @reward
    def reach_target_pos(self) -> torch.Tensor:
        Tr = 1
        diff = self.target_pos_w - self.asset.data.root_pos_w
        rew = 1 / (Tr * (1 + diff.square().sum(dim=-1, keepdim=True)))
        rew = torch.where(
            self.time_elapsed > self.time_alloted - Tr,
            rew,
            torch.zeros_like(self.time_alloted)
        )
        return rew.reshape(self.num_envs, 1)
    
    @reward
    def reach_target_yaw(self) -> torch.Tensor:
        Tr = 1
        diff = wrap_to_pi(self.target_rpy_w[:, 2] - self.asset.data.heading_w)
        rew = 1 / (Tr * (1 + diff.square()[:, None]))
        rew = torch.where(
            self.time_elapsed > self.time_alloted - Tr,
            rew,
            torch.zeros_like(self.time_alloted)
        )
        return rew.reshape(self.num_envs, 1)

    @reward
    def reaching_target(self) -> torch.Tensor:
        return self.target_reached.float().reshape(self.num_envs, 1)
    
    def reset(self, env_ids: torch.Tensor):
        self.time_alloted[env_ids] = 5.
        self.time_elapsed[env_ids] = 0.
        self.sample_target(env_ids)

    def update(self):
        interval_reached = (self.env.episode_length_buf+1) % self.resample_interval == 0
        
        self.pos_diff = self.target_pos_w - self.asset.data.root_pos_w
        self.pos_diff_norm = self.pos_diff[:, :2].norm(dim=-1, keepdim=True)
        self.yaw_diff = wrap_to_pi(self.target_rpy_w[:, 2] - self.asset.data.heading_w)

        self.target_reached = (self.pos_diff_norm.squeeze(1) < self.resample_distance_thres) \
            & (self.yaw_diff.abs() < self.resample_yaw_thres)
        resample = self.mask2id(interval_reached & self.target_reached)
        if len(resample) > 0:
            self.sample_target(resample)
        self.time_elapsed += self.env.step_dt

    def mask2id(self, mask: torch.Tensor):
        return mask.nonzero().squeeze(-1)
    
    def sample_target(self, env_ids: torch.Tensor):
        offset = torch.zeros(len(env_ids), 3, device=self.device)
        offset[:, :2].uniform_(1., 3.).mul_(torch.randn(len(env_ids), 2, device=self.device).sign())
        yaw = torch.rand(len(env_ids), device=self.device) * torch.pi * 2.
        self.target_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids] + offset
        self.target_rpy_w[env_ids, 2] = yaw
    
    def debug_draw(self):
        # self.env.sim.set_camera_view(
        #     self.asset.data.root_pos_w[0].cpu() + torch.tensor([2., 2., 1.]),
        #     self.asset.data.root_pos_w[0].cpu()
        # )
        if self.env.backend == "isaac":
            quat = quat_from_yaw(self.target_rpy_w[:, 2])
            scales = torch.tensor([0.2, 0.2, 0.2]).expand(len(self.target_pos_w), 3)
            self.frame_marker.visualize(self.target_pos_w, quat, scales=scales)
            # line from current position to target position
            self.env.debug_draw.vector(
                self.asset.data.root_pos_w,
                self.target_pos_w - self.asset.data.root_pos_w,
                color=(1, 0, 0, 1)
            )
    
    