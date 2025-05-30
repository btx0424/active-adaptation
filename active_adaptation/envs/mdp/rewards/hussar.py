import torch

from typing import TYPE_CHECKING
from active_adaptation.envs.mdp.rewards import Reward
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation


class torques_scaled(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self) -> torch.Tensor:
        torques = self.asset.data.applied_torque / self.asset.data.joint_stiffness
        rew = torques.square().sum(1)
        return -rew.reshape(self.num_envs, 1)


class feet_stumble(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

    def compute(self) -> torch.Tensor:
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.body_ids, :]
        force_xy = torch.norm(contact_forces[:, :, :2], dim=-1)
        force_z = torch.abs(contact_forces[:, :, 2])
        rew = torch.any(force_xy > 3 * force_z, dim=1).float()
        return -rew.reshape(self.num_envs, 1).float()


class feet_contact_forces(Reward):
    def __init__(self, env, body_names, max_contact_force: float,weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.max_contact_force = max_contact_force
    
    def compute(self):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.body_ids, :].norm(dim=-1)
        return -torch.sum(contact_forces - self.max_contact_force, dim=1, keepdim=True).clamp_min(0.0)


class contact_momentum(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.sensor_ids, self.sensor_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = self.asset.find_bodies(body_names)[0]

    def compute(self):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.sensor_ids, 2]
        feet_vel = self.asset.data.body_vel_w[:, self.body_ids, 2]
        contact_momentum = torch.clip(feet_vel, max=0.0) * torch.clip(contact_forces-50.0, min=0.0)
        return contact_momentum.sum(1, keepdim=True)


class action_vanish(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.action_min = (self.asset.data.joint_pos_limits[:, :, 0] - self.asset.data.default_joint_pos) / self.env.action_manager.action_scaling
        self.action_max = (self.asset.data.joint_pos_limits[:, :, 1] - self.asset.data.default_joint_pos) / self.env.action_manager.action_scaling

    def compute(self) -> torch.Tensor:
        action = self.asset.data.applied_action[:, :, 0]
        upper_error = torch.clip(action - self.action_max, min=0.0)
        lower_error = torch.clip(-action + self.action_min, min=0.0)
        return -torch.sum(upper_error + lower_error, dim=1, keepdim=True)


class dof_vel(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        dof_vel = self.asset.data.joint_vel
        return -dof_vel.square().sum(1, True)


class dof_vel_limits(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.dof_vel_limits = self.asset.data.joint_velocity_limits[:, :] * 0.80
        
    def compute(self) -> torch.Tensor:
        dof_vel = self.asset.data.joint_vel
        return -((dof_vel - self.dof_vel_limits).clamp_min(0.0)).sum(1, True)


class torque_limits(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.torque_limits = self.asset.data.joint_effort_limits[:, :] * 0.95

    def compute(self) -> torch.Tensor:
        torque = self.asset.data.applied_torque
        return -((torque - self.torque_limits).clamp_min(0.0)).sum(1, True)


class dof_pos_limits(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.soft_joint_pos_limits = self.asset.data.soft_joint_pos_limits
    
    def compute(self) -> torch.Tensor:
        joint_pos = self.asset.data.joint_pos
        violation_min = (joint_pos - self.soft_joint_pos_limits[:, :, 0]).clamp_max(0.0)
        violation_max = (self.soft_joint_pos_limits[:, :, 1] - joint_pos).clamp_max(0.0)
        return (violation_min + violation_max).sum(1, keepdim=True)


class feet_clearance_height(Reward):
    def __init__(self, env, body_names, weight: float, height_target: float, relative_height: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.height_target = height_target
        self.relative_height = relative_height
        self.foot_ids = self.asset.find_bodies(body_names)[0]

    def compute(self) -> torch.Tensor:
        cur_feetvel_translated = self.asset.data.body_link_vel_w[:, self.foot_ids, :3] - self.asset.data.root_lin_vel_w.unsqueeze(1)
        feetvel_in_body_frame = torch.zeros(self.env.num_envs, 2, 3, device=self.env.device)
        for i in range(2):
            feetvel_in_body_frame[:, i, :] = quat_rotate_inverse(yaw_quat(self.asset.data.root_link_quat_w), cur_feetvel_translated[:, i, :])
        feet_pos = self.asset.data.body_pos_w[:, self.foot_ids, :]
        cur_feetpos_translated = feet_pos - self.asset.data.root_pos_w.unsqueeze(1)
        feetpos_in_body_frame = torch.zeros(self.env.num_envs, 2, 3, device=self.env.device)
        for i in range(2):
            feetpos_in_body_frame[:, i, :] = quat_rotate_inverse(yaw_quat(self.asset.data.root_link_quat_w), cur_feetpos_translated[:, i, :])
        feet_height_w = feet_pos[:, :, 2]
        feet_height = feet_height_w - self.env.get_ground_height_at(feet_pos)
        height_error = torch.square((feet_height - self.height_target).clamp(max=0.)).view(self.env.num_envs, -1)
        feet_relative_height = self.asset.data.root_pos_w[:, 2].unsqueeze(1) - feet_height_w
        height_error += torch.square((feet_relative_height - self.relative_height).clamp(min=0.)).view(self.env.num_envs, -1)
        feet_lateral_vel = torch.sqrt(torch.sum(torch.square(feetvel_in_body_frame[:, :, :2]), dim=2)).view(self.env.num_envs, -1)
        return -torch.sum(height_error * feet_lateral_vel, dim=1, keepdim=True)


class feet_distance_lateral(Reward):
    def __init__(self, env, body_names, least_distance: float, most_distance: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.least_distance = least_distance    
        self.most_distance = most_distance
    
    def compute(self):
        cur_feetpos_translated = self.asset.data.body_link_pos_w[:, self.body_ids, :] - self.asset.data.root_pos_w.unsqueeze(1)
        feetpos_in_body_frame = torch.zeros(self.env.num_envs, 2, 3, device=self.env.device)
        for i in range(2):
            feetpos_in_body_frame[:, i, :] = quat_rotate_inverse(yaw_quat(self.asset.data.root_link_quat_w), cur_feetpos_translated[:, i, :])
        foot_lateral_dis = torch.abs(feetpos_in_body_frame[:, 0, 1] - feetpos_in_body_frame[:, 1, 1])
        return -(torch.clamp(foot_lateral_dis - self.least_distance, max=0) + torch.clamp(-foot_lateral_dis + self.most_distance, max=0)).reshape(self.env.num_envs, -1)


class knee_distance_lateral(Reward):
    def __init__(self, env, body_names, least_distance: float, most_distance: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.least_distance = least_distance    
        self.most_distance = most_distance
    
    def compute(self):
        cur_feetpos_translated = self.asset.data.body_link_pos_w[:, self.body_ids, :] - self.asset.data.root_pos_w.unsqueeze(1)
        kneepos_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            kneepos_in_body_frame[:, i, :] = quat_rotate_inverse(yaw_quat(self.asset.data.root_link_quat_w), cur_feetpos_translated[:, i, :])
        foot_lateral_dis = torch.abs(kneepos_in_body_frame[:, 0, 1] - kneepos_in_body_frame[:, 1, 1]) + torch.abs(kneepos_in_body_frame[:, 2, 1] - kneepos_in_body_frame[:, 3, 1])
        return -(torch.clamp(foot_lateral_dis - 2 * self.least_distance, max=0) + torch.clamp(-foot_lateral_dis + 2 * self.most_distance, max=0)).reshape(self.env.num_envs, -1)


class stand_still(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
    
    def compute(self) -> torch.Tensor:
        is_standing = self.env.command_manager.is_standing_env.reshape(self.num_envs, 1)
        forces_z = self.contact_sensor.data.net_forces_w[:, self.body_ids, 2]
        contacts = torch.sum(forces_z < 0.1, dim=-1).reshape(self.num_envs, 1)
        return -(contacts) * is_standing


class feet_ground_parallel(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids_a, body_names_a = self.asset.find_bodies(body_names)
        self.body_ids_c, body_names_c = self.contact_forces.find_bodies(body_names)
        for name_a, name_c in zip(body_names_a, body_names_c):
            assert name_a == name_c

        self.thres = self.env.step_dt * 3

    def compute(self):
        feet_fwd_vec = quat_rotate(
            self.asset.data.body_quat_w[:, self.body_ids_a],
            torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, 2, 3)
        )
        toe_pos_w = self.asset.data.body_pos_w[:, self.body_ids_a] + feet_fwd_vec * 0.1
        heel_pos_w = self.asset.data.body_pos_w[:, self.body_ids_a] - feet_fwd_vec * 0.02
        first_contact = self.contact_forces.compute_first_air(self.thres)[:, self.body_ids_c]
        toe_height = toe_pos_w[:, :, 2] - self.env.get_ground_height_at(toe_pos_w)
        heel_height = heel_pos_w[:, :, 2] - self.env.get_ground_height_at(heel_pos_w)
        rew = torch.sum((toe_height - heel_height).square() * first_contact, dim=1)
        return - rew.reshape(self.num_envs, 1)


class feet_parallel(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        assert len(self.body_ids) == 2

    def compute(self):
        self.feet_fwd_vec = quat_rotate(
            yaw_quat(self.asset.data.body_quat_w[:, self.body_ids]),
            torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, 2, 3)
        )
        dot = torch.sum(self.feet_fwd_vec[:, 0] * self.feet_fwd_vec[:, 1], dim=1, keepdim=True)
        return dot - 1.
    
    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.body_ids].reshape(-1, 3),
            self.feet_fwd_vec.reshape(-1, 3),
            color=(0, 0, 1, 1),
        )


class feet_clearance_simple(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.target_height = 0.08
    
    def compute(self):
        feet_pos_w = self.asset.data.body_pos_w[:, self.body_ids]
        feet_vel_w = self.asset.data.body_vel_w[:, self.body_ids]
        feet_speed = torch.norm(feet_vel_w[:, :, :2], dim=2).square()
        feet_height = feet_pos_w[:, :, 2] - self.env.get_ground_height_at(feet_pos_w)
        error = (feet_height - self.target_height).clamp_max(0.0)
        return (feet_speed * error).sum(dim=1).reshape(self.num_envs, 1)


class waist_deviation_l2(Reward):
    def __init__(self, env, joint_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
    
    def compute(self):
        dev = self.asset.data.joint_pos[:, self.joint_ids] - self.default_joint_pos
        return -dev.square().sum(1, True)


class orientation_isaaclab(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        return -torch.sum(self.asset.data.projected_gravity_b[:, :2].square(), dim=1, keepdim=True)


class linvel_x_exp(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager = self.env.command_manager
    
    def compute(self):
        linvel_x = self.asset.data.root_lin_vel_b[:, 0]
        error = torch.square(self.command_manager.command_linvel[:, 0] - linvel_x)
        return torch.exp(-error / 0.25).reshape(self.num_envs, 1)


class linvel_y_exp(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager = self.env.command_manager
    
    def compute(self):
        linvel_y = self.asset.data.root_lin_vel_b[:, 1]
        error = torch.square(self.command_manager.command_linvel[:, 1] - linvel_y)
        return torch.exp(-error / 0.25).reshape(self.num_envs, 1)

