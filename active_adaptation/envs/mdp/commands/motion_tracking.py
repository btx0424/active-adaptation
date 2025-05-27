import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor

from active_adaptation.envs.mdp import reward, observation
from active_adaptation.utils.motion import MotionDataset
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat
)
from .base import Command


class MotionTrackingCommand(Command):
    def __init__(self, env, data_path: str):
        super().__init__(env)
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]

        self.dataset = MotionDataset.create_from_path(
            data_path,
            target_fps=int(1/self.env.step_dt)
        )

        tracking_keypoint_names = [
            # "torso_link",
            ".*shoulder_pitch_link",
            ".*hip_pitch_link",
            ".*elbow.*",
            ".*knee.*",
            # ".*ankle.*"
        ]
        print(self.dataset.body_names)
        tracking_keypoint_names = self.asset.find_bodies(tracking_keypoint_names)[1]
        self.keypoint_idx_motion = []
        self.keypoint_idx_asset = []
        for body_name in tracking_keypoint_names:
            self.keypoint_idx_motion.append(self.dataset.body_names.index(body_name))
            self.keypoint_idx_asset.append(self.asset.body_names.index(body_name))
        print(self.keypoint_idx_motion)
        print(self.keypoint_idx_asset)

        tracking_joint_names = [
            "torso_joint",
            ".*shoulder.*",
            ".*elbow.*",
            ".*hip.*",
            ".*knee.*",
        ]
        tracking_joint_names = self.asset.find_joints(tracking_joint_names)[1]
        self.joint_idx_motion = []
        self.joint_idx_asset = []
        for joint_name in tracking_joint_names:
            self.joint_idx_motion.append(self.dataset.joint_names.index(joint_name))
            self.joint_idx_asset.append(self.asset.joint_names.index(joint_name))
        print(tracking_joint_names)
        print(self.joint_idx_motion)
        print(self.joint_idx_asset)

        feet_names = ".*ankle_roll_link"
        self.feet_ids_motion = self.dataset.find_bodies(feet_names)[0]
        self.feet_ids_asset = self.asset.find_bodies(feet_names)[0]
        self.feet_ids_sensor = self.contact_forces.find_bodies(feet_names)[0]
        print(feet_names)
        print(self.feet_ids_motion)
        print(self.feet_ids_asset)
        print(self.feet_ids_sensor)

        self._cum_error = torch.zeros(self.num_envs, 1, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)

        self.motion_ids = torch.randint(0, self.dataset.num_motions, size=(self.num_envs,))
        self.t = torch.zeros(self.num_envs, dtype=int)
        self.future_steps = torch.tensor([0, 12, 24, 36])
        self.update()
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        init_root_state = self.init_root_state[env_ids]
        origins = self.env.scene.env_origins[env_ids]
        motion = self.dataset.get_slice(self.motion_ids[env_ids.cpu()], 0, 1)
        init_root_state[:, :3] = origins + motion.root_pos_w[:, 0].to(self.device)
        init_root_state[:, 3:7] = motion.root_quat_w[:, 0].to(self.device)
        return init_root_state
    
    def reset(self, env_ids):
        self.t[env_ids] = 0

    @property
    def command(self):
        return torch.concat([
            self.target_pos_b.reshape(self.num_envs, -1),
            self.target_keypoints_b.reshape(self.num_envs, -1),
            self.relative_quat.reshape(self.num_envs, -1),
        ], dim=-1)
    
    @reward
    def root_pos_tracking(self):
        diff = self.target_pos_w[:, 0] - self.asset.data.root_pos_w
        error = diff.square().sum(-1, keepdim=True)
        return torch.exp(- error / 0.25)

    @reward
    def keypoint_tracking(self):
        diff = self.target_keypoints_w[:, 0] - self.asset.data.body_pos_w[:, self.keypoint_idx_asset]
        error = diff.square().sum(-1, keepdim=True)
        return torch.exp(- error / 0.1).mean(1)

    @reward
    def orientation_tracking(self):
        error = torch.norm(axis_angle_from_quat(self.relative_quat[:, 0]), dim=-1, keepdim=True)
        return torch.exp(- error)
    
    @reward
    def joint_pos_tracking(self):
        error = (self.target_joint_pos - self.asset.data.joint_pos[:, self.joint_idx_asset]).square()
        return torch.exp(- error / 0.5).mean(1, True)

    @reward
    def feet_tracking(self):
        # in_contact = self.contact_forces.data.current_contact_time[:, self.feet_ids_sensor] > 0.01
        first_contact = self.contact_forces.compute_first_contact(0.02)[:, self.feet_ids_sensor]
        diff = self.target_feet_pos_w - self.asset.data.body_pos_w[:, self.feet_ids_asset]
        error = diff.square().sum(-1)
        return - (error * first_contact).sum(1, True)

    def update(self):
        self._motion = self.dataset.get_slice(self.motion_ids, self.t, steps=self.future_steps)
        self.target_pos_w = self._motion.root_pos_w.to(self.device) \
            + self.env.scene.env_origins.reshape(self.num_envs, 1, 3)
        self.target_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            (self.target_pos_w - self.asset.data.root_pos_w.unsqueeze(1))
        )
        self.target_quat_w = self._motion.root_quat_w.to(self.device)
        self.relative_quat = quat_mul(
            self.asset.data.root_quat_w.unsqueeze(1).expand_as(self.target_quat_w),
            quat_conjugate(self.target_quat_w)
        )
        self.target_keypoints_b = self._motion.body_pos_b[:, :, self.keypoint_idx_motion].to(self.device)
        self.target_keypoints_w = self._motion.body_pos_w[:, :, self.keypoint_idx_motion].to(self.device)
        self.target_keypoints_w = self.target_keypoints_w \
            + self.env.scene.env_origins.reshape(self.num_envs, 1, 1, 3)
        self.target_feet_pos_w = self._motion.body_pos_w[:, 0, self.feet_ids_motion].to(self.device) \
            + self.env.scene.env_origins.reshape(self.num_envs, 1, 3)
        self.target_joint_pos = self._motion.joint_pos[:, 0, self.joint_idx_motion].to(self.device)
        
        self.t = torch.clamp_max(self.t + 1, self.dataset.lengths[self.motion_ids]-self.future_steps[-1])

    def debug_draw(self):
        target_keypoints_w = self._motion.body_pos_w[:, 0] + self.env.scene.env_origins.cpu().unsqueeze(1)
        self.env.debug_draw.point(target_keypoints_w.reshape(-1, 3), color=(1, 0, 0, 1))

        robot_keypoints_w = self.asset.data.body_pos_w[:, self.keypoint_idx_asset].cpu()
        self.env.debug_draw.point(robot_keypoints_w.reshape(-1, 3), color=(0, 1, 0, 1))

        self.env.debug_draw.vector(
            robot_keypoints_w.reshape(-1, 3),
            target_keypoints_w[:, self.keypoint_idx_motion].reshape(-1, 3) - robot_keypoints_w.reshape(-1, 3),
            color=(0, 0, 1, 1)
        )

        in_contact = self.contact_forces.data.current_contact_time[:, self.feet_ids_sensor] > 0.01
        diff = self.target_feet_pos_w - self.asset.data.body_pos_w[:, self.feet_ids_asset]
        
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.feet_ids_asset].reshape(-1, 3),
            (diff * in_contact.unsqueeze(-1)).reshape(-1, 3),
            color=(0, 1, 0, 1),
            size=5.
        )
