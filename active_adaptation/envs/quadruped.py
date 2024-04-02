import torch

from omni.isaac.orbit.sensors import ContactSensor, RayCaster
from omni.isaac.orbit.actuators import DCMotor
from omni.isaac.orbit.assets import Articulation
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

from collections import OrderedDict
from .locomotion import LocomotionEnv
from . import mdp

class Quadruped(LocomotionEnv):

    feet_name_expr = ".*_foot"
    
    @property
    def action_dim(self):
        return 12
    
    @mdp.observation_func
    def linvel_error(self):
        if not hasattr(self, "robot"):
            return torch.zeros(self.num_envs, 2, device=self.device)
        linvel_diff = self.command_manager._command_linvel[:, :2] - self.robot.data.root_lin_vel_b[:, :2]
        return linvel_diff

    # @observation_func
    # def feet_vel_b(self):
    #     feet_vel = quat_rotate_inverse(
    #         self.robot.data.root_quat_w.unsqueeze(1),
    #         self.robot.data.body_lin_vel_w[:, self.foot_indices]
    #     )
    #     return feet_vel.reshape(self.num_envs, -1)

    # @observation_func
    # def base_mass(self):
    #     rand = self.randomizations["body_masses"]
    #     return rand.randomized_masses.reshape(self.num_envs, -1)[:, [0]]

    # @observation_func
    # def base_inertia(self):
    #     rand = self.randomizations["body_inertias"]
    #     inertia = rand.randomized_inertias.reshape(self.num_envs, -1)[:, [0, 4, 8]]
    #     return inertia
    
    # @observation_func
    # def base_com(self):
    #     rand: BodyComs = self.randomizations["body_coms"]
    #     return rand.randomized_coms.reshape(self.num_envs, -1)
    
    # @observation_func
    # def body_materials(self):
    #     rand = self.randomizations["body_material"]
    #     return rand.material_properties.reshape(self.num_envs, -1) * 2.0 - 1.0

    # @observation_func
    # def motor_failure(self):
    #     rand: MotorFailure = self.randomizations["motor_failure"]
    #     return rand.motor_failure.reshape(self.num_envs, -1)
    
    # @observation_func
    # def body_inertias(self):
    #     rand: BodyInertias = self.randomizations["body_inertias"]
    #     return rand.randomized_inertias.reshape(self.num_envs, -1)
    
    # @reward_func
    # def base_height(self):
    #     height = self.robot.data.root_pos_w[:, [2]]
    #     height = height - self.robot.data.body_pos_w[:, self.foot_indices, 2].mean(1, keepdim=True)
    #     return (height / self.target_base_height).square().clamp_max(0.8)
    
    @mdp.reward_func
    def stand(self):
        if not hasattr(self, "robot"):
            return torch.zeros(self.num_envs, 1)
        jpos_error = (self.robot.data.joint_pos - self.robot.data.default_joint_pos).abs().sum(dim=1, keepdim=True)
        front_symmetry = self.robot.data.feet_pos_b[:, [0, 1], 1].sum(dim=1, keepdim=True).abs()
        back_symmetry = self.robot.data.feet_pos_b[:, [2, 3], 1].sum(dim=1, keepdim=True).abs()
        cost = - (jpos_error + front_symmetry + back_symmetry) 
        return cost * self.command_manager.is_standing_env.reshape(self.num_envs, 1)


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

