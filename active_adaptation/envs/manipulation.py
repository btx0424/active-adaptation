import torch
import logging

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat, quat_from_euler_xyz, wrap_to_pi, quat_inv, quat_mul
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

from collections import OrderedDict
from .locomotion import LocomotionEnv
from .mdp import Reward, Observation


class QuadrupedManip(LocomotionEnv):

    feet_name_expr = ".*_foot"
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.base_legs_indices = self.robot.actuators["base_legs"].joint_indices
        self.arm_indices = self.robot.actuators["arm"].joint_indices

    class ee_pos(Observation):
        def __init__(self, env, ee_name: str):
            super().__init__(env)
            self.asset: Articulation = self.env.scene["robot"]
        
        def compute(self) -> torch.Tensor:
            return self.asset.data.ee_pos_b

    class ee_ori(Observation):
        def __init__(self, env, ee_name: str):
            super().__init__(env)
            self.asset: Articulation = self.env.scene["robot"]
            self.body_id, self.body_names = self.asset.find_bodies(ee_name)
            self.body_id = self.body_id[0]
            self.fwd_vec = torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, -1)
            self.up_vec = torch.tensor([0., 0., 1.], device=self.device).expand(self.num_envs, -1)

        def compute(self) -> torch.Tensor:
            root_quat_yaw = yaw_quat(self.asset.data.root_quat_w)
            ee_quat = self.asset.data.body_quat_w[:, self.body_id]
            ee_fwd_w = quat_rotate(ee_quat, self.fwd_vec)
            ee_up_w = quat_rotate(ee_quat, self.up_vec)
            ee_fwd_b = quat_rotate_inverse(root_quat_yaw, ee_fwd_w)
            ee_up_b = quat_rotate_inverse(root_quat_yaw, ee_up_w)
            return torch.cat([ee_fwd_b, ee_up_b], dim=-1)

    class ee_vel(Observation):
        def __init__(self, env, ee_name: str):
            super().__init__(env)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_names = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]
        
        def compute(self) -> torch.Tensor:
            quat_yaw = yaw_quat(self.asset.data.root_quat_w)
            ee_linvel = quat_rotate_inverse(
                quat_yaw, 
                self.asset.data.body_lin_vel_w[:, self.ee_id]
            )
            return torch.cat([ee_linvel, self.asset.data.body_ang_vel_w[:, self.ee_id]], dim=-1)
    
    class ee_vel_hist(Observation):
        def __init__(self, env, ee_name: str, hist_len: int):
            super().__init__(env)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_names = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]
            self.hist_len = hist_len
            self.hist = torch.zeros(self.num_envs, hist_len, 6, device=self.device)
        
        def reset(self, env_ids: torch.Tensor):
            self.hist[env_ids] = 0.
        
        def compute(self) -> torch.Tensor:
            quat_yaw = yaw_quat(self.asset.data.root_quat_w)
            ee_linvel = quat_rotate_inverse(
                quat_yaw, 
                self.asset.data.body_lin_vel_w[:, self.ee_id]
            )
            ee_vel = torch.cat([ee_linvel, self.asset.data.body_ang_vel_w[:, self.ee_id]], dim=-1)
            self.hist[:, 1:] = self.hist[:, :-1]
            self.hist[:, 0] = ee_vel
            # return self.hist
            # flatten out last two dims
            return self.hist.reshape(self.num_envs, -1)
            
    
    class ee_pos_tracking_w(Reward):
    
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True, l: float = 0.25):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]
            self.l = l

        def compute(self) -> torch.Tensor:
            ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id]
            command_ee_pos_w = self.env.command_manager.command_ee_pos_w
            pos_error = (ee_pos_w - command_ee_pos_w).norm(dim=-1, keepdim=True)
            r = 1 - pos_error + torch.exp(- pos_error / self.l)
            return 0.5 * r

    class ee_pos_tracking_b(Reward):
    
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True, l: float = 0.25):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]
            self.l = l

        def compute(self) -> torch.Tensor:
            ee_pos_b = self.asset.data.ee_pos_b
            command_ee_pos_b = self.env.command_manager.command_ee_pos_b
            pos_error = ((ee_pos_b - command_ee_pos_b) / self.l).square().sum(1, True)
            r = torch.exp(- pos_error)
            return r


    class ee_failed_track_counts(Reward):
        def __init__(self, env, weight: float, enabled: bool = False):
            super().__init__(env, weight, enabled)
        
        def compute(self) -> torch.Tensor:
            return self.env.command_manager.failed_track_counts.unsqueeze(1).float()


    class ee_pos_error_w_l1(Reward):
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]

        def compute(self) -> torch.Tensor:
            ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id]
            command_ee_pos_w = self.env.command_manager.command_ee_pos_w
            pos_error = ((ee_pos_w - command_ee_pos_w)).abs().sum(1, True)
            return pos_error
    
    class ee_pos_error_b_l1(Reward):
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]

        def compute(self) -> torch.Tensor:
            ee_pos_b = self.asset.data.ee_pos_b
            command_ee_pos_b = self.env.command_manager.command_ee_pos_b
            pos_error = ((ee_pos_b - command_ee_pos_b)).abs().sum(1, True)
            return pos_error
    

    class ee_ori_tracking(Reward):

        l = 0.25

        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]

            with torch.device(self.device):
                self.fwd_vec = torch.tensor([1., 0., 0.]).expand(self.num_envs, -1)
                self.up_vec = torch.tensor([0., 0., 1.]).expand(self.num_envs, -1)
                self.ee_forward_w = torch.zeros(self.num_envs, 3)
                self.ee_up_w = torch.zeros(self.num_envs, 3)

        def compute(self) -> torch.Tensor:
            ee_quat_w = self.asset.data.body_quat_w[:, self.ee_id]
            self.ee_forward_w[:] = quat_rotate(ee_quat_w, self.fwd_vec)
            self.ee_up_w[:] = quat_rotate(ee_quat_w, self.up_vec)

            r1 = (self.ee_forward_w * self.env.command_manager.command_ee_forward_w).sum(-1, True)
            r2 = (self.ee_up_w * self.env.command_manager.command_ee_upward_w).sum(-1, True)
            r = 0.5 * (r1.sign() * r1.square() + r2.sign() * r2.square())
            return r
        
        def debug_draw(self):
            # draw real ee forward and up vector
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.ee_id],
                self.ee_forward_w * 0.2,
                color=(1., 0.1, 0.1, 1.)
            )
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.ee_id],
                self.ee_up_w * 0.2,
                color=(1, 0.1, 0.1, 1.)
            )
            # draw commanded ee forward and up vector
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.ee_id],
                self.env.command_manager.command_ee_forward_w * 0.2,
                color=(0.1, 0.1, 1., 1.)
            )
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.ee_id],
                self.env.command_manager.command_ee_upward_w * 0.2,
                color=(0.1, 0.1, 1., 1.)
            )

    
    class ee_ori_forward_tracking(Reward):

        l = 0.25

        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]

            with torch.device(self.device):
                self.fwd_vec = torch.tensor([1., 0., 0.]).expand(self.num_envs, -1)
                self.ee_forward_w = torch.zeros(self.num_envs, 3)

        def compute(self) -> torch.Tensor:
            ee_quat_w = self.asset.data.body_quat_w[:, self.ee_id]
            self.ee_forward_w[:] = quat_rotate(ee_quat_w, self.fwd_vec)

            r1 = (self.ee_forward_w * self.env.command_manager.command_ee_forward_w).sum(-1, True)
            r = r1.sign() * r1.square()
            return r
        
        def debug_draw(self):
            # draw real ee forward vector
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.ee_id],
                self.ee_forward_w * 0.2,
                color=(1., 0.1, 0.1, 1.)
            )
            # draw commanded ee forward vector
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.ee_id],
                self.env.command_manager.command_ee_forward_w * 0.2,
                color=(0.1, 0.1, 1., 1.)
            )

    
    class ee_ori_upward_tracking(Reward):

        l = 0.25

        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]

            with torch.device(self.device):
                self.up_vec = torch.tensor([0., 0., 1.]).expand(self.num_envs, -1)
                self.ee_upward_w = torch.zeros(self.num_envs, 3)

        def compute(self) -> torch.Tensor:
            ee_quat_w = self.asset.data.body_quat_w[:, self.ee_id]
            self.ee_upward_w[:] = quat_rotate(ee_quat_w, self.up_vec)

            r2 = (self.ee_upward_w * self.env.command_manager.command_ee_upward_w).sum(-1, True)
            r = r2.sign() * r2.square()
            return r
        
        def debug_draw(self):
            # draw real ee up vector
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.ee_id],
                self.ee_upward_w * 0.2,
                color=(1, 0.1, 0.1, 1.)
            )
            # draw commanded ee up vector
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.ee_id],
                self.env.command_manager.command_ee_upward_w * 0.2,
                color=(0.1, 0.1, 1., 1.)
            )

    class ee_ori_forward_error_l2(Reward):
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = False):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]

            with torch.device(self.device):
                self.fwd_vec = torch.tensor([1., 0., 0.]).expand(self.num_envs, -1)
                self.ee_forward_w = torch.zeros(self.num_envs, 3)

        def compute(self) -> torch.Tensor:
            self.ee_forward_w[:] = quat_rotate(
                self.asset.data.body_quat_w[:, self.ee_id],
                self.fwd_vec,
            )
            return (self.ee_forward_w - self.env.command_manager.command_ee_forward_w).square().sum(-1, True)
        
    class ee_ori_upward_error_l2(Reward):
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = False):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]

            with torch.device(self.device):
                self.up_vec = torch.tensor([0., 0., 1.]).expand(self.num_envs, -1)
                self.ee_up_w = torch.zeros(self.num_envs, 3)

        def compute(self) -> torch.Tensor:
            self.ee_up_w[:] = quat_rotate(
                self.asset.data.body_quat_w[:, self.ee_id],
                self.up_vec,
            )
            return (self.ee_up_w - self.env.command_manager.command_ee_upward_w).square().sum(-1, True)

    class base_joint_heading(Reward):
        """Encourage the rotation of the base joint of manipulator to align with the target ee_pos_b, projected onto the xy-plane."""
        def __init__(self, env, base_joint_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            base_joint_ids, base_joint_names = self.asset.find_joints(base_joint_name)
            self.base_joint_id = base_joint_ids[0]
        
        def compute(self) -> torch.Tensor:
            command_ee_pos_b_yaw = self.env.command_manager.command_ee_pos_b_yaw
            # get the yaw of the base joint
            base_joint_yaw = self.asset.data.joint_pos[:, self.base_joint_id]
            base_joint_yaw = wrap_to_pi(base_joint_yaw)
            # compute cosine dot product, give negative reward when cosine < 0, 0 reward when cosine > 0
            cosine = torch.cos(command_ee_pos_b_yaw - base_joint_yaw)
            # return cosine.clamp_max_(0.5).unsqueeze_(1) - 0.5
            return cosine.clamp_max_(0.0).unsqueeze_(1)
    
        def debug_draw(self):
            # draw the vector of the forward direction of the base joint, which is forward vector [1, 0, 0], applied with the yaw rotation of the base joint
            base_joint_yaw = self.asset.data.joint_pos[:, self.base_joint_id]
            fwd_vec_b = torch.stack([torch.cos(base_joint_yaw), torch.sin(base_joint_yaw), torch.zeros(self.num_envs, device=self.device)], dim=-1)
            fwd_vec_w = quat_rotate(self.asset.data.root_quat_w, fwd_vec_b)
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.base_joint_id],
                fwd_vec_w * 0.5,
                color=(1., 0.1, 0.1, 1.)
            )
        
    class base_joint_ang_vel_pd_exp(Reward):
        # target angvel = kp * (target_yaw - yaw) + kd * (0 - angvel)
        # use pd to compute the target angvel of the base joint
        def __init__(self, env, base_joint_name: str, weight: float, enabled: bool = True, kp: float = 2., kd: float = 0.5, l: float = 1.):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            base_joint_ids, base_joint_names = self.asset.find_joints(base_joint_name)
            self.base_joint_id = base_joint_ids[0]
            self.kp = kp
            self.kd = kd
            self.l = l
        
        def compute(self) -> torch.Tensor:
            command_ee_pos_b_yaw = self.env.command_manager.command_ee_pos_b_yaw
            base_joint_yaw = self.asset.data.joint_pos[:, self.base_joint_id]

            target_angvel = self.kp * (command_ee_pos_b_yaw - base_joint_yaw) - self.kd * self.asset.data.joint_vel[:, self.base_joint_id]
            base_joint_angvel = self.asset.data.joint_vel[:, self.base_joint_id]
            ang_vel_error = (target_angvel - base_joint_angvel).square().unsqueeze_(1)
            return torch.exp(- ang_vel_error / self.l)
        
    class ee_tracking_hybrid(Reward):
        def __init__(self, env, ee_name: str, base_joint_name: str, weight: float, enabled: bool = True, l: float = 0.25, pos_exp_weight: float = 1, ori_add_linear: bool = False):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            ee_ids, ee_names = self.asset.find_bodies(ee_name)
            self.ee_id = ee_ids[0]
            base_joint_ids, base_joint_names = self.asset.find_joints(base_joint_name)
            self.base_joint_id = base_joint_ids[0]
            self.l = l
            self.pos_exp_weight = pos_exp_weight
            self.ori_add_linear = ori_add_linear
            
            with torch.device(self.device):
                self.fwd_vec = torch.tensor([1., 0., 0.]).expand(self.num_envs, -1)
                self.up_vec = torch.tensor([0., 0., 1.]).expand(self.num_envs, -1)
                self.ee_forward_w = torch.zeros(self.num_envs, 3)
                self.ee_up_w = torch.zeros(self.num_envs, 3)

        def compute(self) -> torch.Tensor:
            command_ee_pos_b_yaw = self.env.command_manager.command_ee_pos_b_yaw
            base_joint_yaw = self.asset.data.joint_pos[:, self.base_joint_id]
            arm_base_ori_cosine = torch.cos(command_ee_pos_b_yaw - base_joint_yaw).unsqueeze_(1)
            
            ee_pos_w = self.asset.data.body_pos_w[:, self.ee_id]
            command_ee_pos_w = self.env.command_manager.command_ee_pos_w
            ee_pos_error = (ee_pos_w - command_ee_pos_w).norm(dim=-1, keepdim=True)
            ee_pos_rew = 1 - ee_pos_error + self.pos_exp_weight * torch.exp(- ee_pos_error / self.l)
            
            ee_quat_w = self.asset.data.body_quat_w[:, self.ee_id]
            self.ee_forward_w[:] = quat_rotate(ee_quat_w, self.fwd_vec)
            self.ee_up_w[:] = quat_rotate(ee_quat_w, self.up_vec)
            
            r1 = (self.ee_forward_w * self.env.command_manager.command_ee_forward_w).sum(-1, True)
            r2 = (self.ee_up_w * self.env.command_manager.command_ee_upward_w).sum(-1, True)
            if self.ori_add_linear:
                ee_ori_rew = 0.5 * (r1.sign() * r1.square() + r2.sign() * r2.square() + r1 + r2)
            else:
                ee_ori_rew = 0.5 * (r1.sign() * r1.square() + r2.sign() * r2.square())
            
            return arm_base_ori_cosine * (ee_pos_rew + ee_ori_rew)
            
    class ee_acc_penalty(Reward):
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            ee_ids, ee_names = self.asset.find_bodies(ee_name)
            self.ee_id = ee_ids[0]
        
        def compute(self) -> torch.Tensor:
            ee_acc = self.asset.data.body_lin_acc_w[:, self.ee_id]
            return - ee_acc.square().sum(1, True)

            # TODO, maybe try: when ee_pos_error is large, reduce this penalty? only apply this penalty when the ee_pos_error is small enough?
    
    class ee_acc_l2(Reward):
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = False):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            ee_ids, ee_names = self.asset.find_bodies(ee_name)
            self.ee_id = ee_ids[0]
        
        def compute(self) -> torch.Tensor:
            ee_acc = self.asset.data.body_lin_acc_w[:, self.ee_id]
            return ee_acc.square().sum(1, True)

    class joint_acc_penalty(Reward):
        def __init__(self, env, joint_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            joint_ids, joint_names = self.asset.find_joints(joint_name)
            self.joint_id = joint_ids[0]
        
        def compute(self) -> torch.Tensor:
            joint_acc = self.asset.data.joint_acc[:, self.joint_id]
            return - joint_acc.square().unsqueeze_(1)
    
    class joint_vel_penalty(Reward):
        def __init__(self, env, joint_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            joint_ids, joint_names = self.asset.find_joints(joint_name)
            self.joint_id = joint_ids[0]
        
        def compute(self) -> torch.Tensor:
            joint_vel = self.asset.data.joint_vel[:, self.joint_id]
            return - joint_vel.square().unsqueeze_(1)

    class joint_acc_l2(Reward):
        def __init__(self, env, joint_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            joint_ids, joint_names = self.asset.find_joints(joint_name)
            self.joint_id = joint_ids[0]
        
        def compute(self) -> torch.Tensor:
            joint_acc = self.asset.data.joint_acc[:, self.joint_id]
            return joint_acc.square().unsqueeze_(1)
    
    class arm_joint_acc_penalty(Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.joint_ids = self.asset.actuators["arm"].joint_indices
        
        def compute(self) -> torch.Tensor:
            joint_acc = self.asset.data.joint_acc[:, self.joint_ids]
            return - joint_acc.abs().sum(1, True)

    class arm_joint_acc_l1(Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.joint_ids = self.asset.actuators["arm"].joint_indices
        
        def compute(self) -> torch.Tensor:
            joint_acc = self.asset.data.joint_acc[:, self.joint_ids]
            return joint_acc.abs().sum(1, True)

    class base_lin_vel_exp(Reward):

        def __init__(self, env, weight: float, enabled: bool = True, mult_proj_gravity: bool = False, l: float = 0.25):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.mult_proj_gravity = mult_proj_gravity
            self.l = l
        
        def compute(self) -> torch.Tensor:
            base_lin_vel_xy = self.asset.data.root_lin_vel_b[:, :2]
            target_base_lin_vel_xy = self.env.command_manager.command_lin_vel[:, :2]
            lin_vel_error = (base_lin_vel_xy - target_base_lin_vel_xy).square().sum(-1, True)
            r = torch.exp(- lin_vel_error / self.l)
            self.asset.data.linvel_exp = r
            if self.mult_proj_gravity:
                r = r * -self.asset.data.projected_gravity_b[:, 2:3]
            return r

    class base_lin_vel_projection(Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
        
        def compute(self) -> torch.Tensor:
            base_lin_vel_xy = self.asset.data.root_lin_vel_b[:, :2]
            target_base_lin_vel_xy = self.env.command_manager.command_lin_vel[:, :2]
            projection = (base_lin_vel_xy * target_base_lin_vel_xy).sum(-1, True)
            return projection.clamp_max_(self.env.command_manager._command_speed)

    class base_ang_vel_exp(Reward):
            
        def __init__(self, env, weight: float, enabled: bool = True, l: float = 0.25):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.l = l
        
        def compute(self) -> torch.Tensor:
            base_ang_vel_z = self.asset.data.root_ang_vel_b[:, 2]
            target_base_ang_vel = self.env.command_manager.command[:, 2]
            ang_vel_error = (base_ang_vel_z - target_base_ang_vel).square().unsqueeze_(1)
            return torch.exp(- ang_vel_error / self.l)

    class joint_pos_l2_penalty(Reward):
        def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf)):
            super().__init__(env, weight, enabled, clip_range)
            self.asset: Articulation = self.env.scene["robot"]
            self.joint_ids = self.asset.actuators["base_legs"].joint_indices

        def compute(self):
            jpos_error = (
                self.asset.data.joint_pos[:, self.joint_ids] - 
                self.asset.data.default_joint_pos[:, self.joint_ids]
            ).square().sum(dim=1, keepdim=True)

            return - jpos_error
            # front_symmetry = self.asset.data.feet_pos_b[:, [0, 1], 1].sum(dim=1, keepdim=True).abs()
            # back_symmetry = self.asset.data.feet_pos_b[:, [2, 3], 1].sum(dim=1, keepdim=True).abs()
            # cost = - (jpos_error + front_symmetry + back_symmetry)

            # return cost * self.env.command_manager.is_standing_env.reshape(self.num_envs, 1)


    class ee_angvel_penalty(Reward):
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.body_id = self.asset.find_bodies(ee_name)[0]
            self.body_id = self.body_id[0]

        def compute(self) -> torch.Tensor:
            ee_angvel_w = self.asset.data.body_ang_vel_w[:, self.body_id]
            return - ee_angvel_w.square().sum(1, True)



def random_scale(x: torch.Tensor, low: float, high: float):
    return x * (torch.rand_like(x) * (high - low) + low)

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def square_norm(x: torch.Tensor):
    return x.square().sum(dim=-1, keepdim=True)

def noarmalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(dim=-1, keepdim=True)

def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def flip_lr(joints: torch.Tensor):
    return joints.reshape(-1, 3, 2, 2).flip(-1).reshape(-1, 12)

def flip_fb(joints: torch.Tensor):
    return joints.reshape(-1, 3, 2, 2).flip(-2).reshape(-1, 12)

