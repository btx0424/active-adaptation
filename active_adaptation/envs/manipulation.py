import torch
import logging

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat
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
            self.ee_pos = self.asset.data.ee_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
            self.ee_id, self.ee_names = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]
        
        def update(self):
            quat_yaw = yaw_quat(self.asset.data.root_quat_w)
            self.ee_pos[:] = quat_rotate_inverse(
                quat_yaw, 
                self.asset.data.body_pos_w[:, self.ee_id] - self.asset.data.root_pos_w
            )
            
        def __call__(self) -> torch.Tensor:
            return self.ee_pos


    class ee_vel(Observation):
        def __init__(self, env, ee_name: str):
            super().__init__(env)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_names = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]
        
        def __call__(self) -> torch.Tensor:
            quat_yaw = yaw_quat(self.asset.data.root_quat_w)
            ee_linvel = quat_rotate_inverse(
                quat_yaw, 
                self.asset.data.body_lin_vel_w[:, self.ee_id]
            )
            return torch.cat([ee_linvel, self.asset.data.body_ang_vel_w[:, self.ee_id]], dim=-1)
    

    class ee_pos_tracking(Reward):
    
        l: float = 0.25

        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]

        def compute(self) -> torch.Tensor:
            ee_pos = self.asset.data.ee_pos_b
            ee_pos_target = self.env.command_manager.command_ee_pos_b
            pos_error = ((ee_pos - ee_pos_target) / self.l).square().sum(1, True)
            r = torch.exp(- pos_error)
            return r


    class ee_pos_error_l1(Reward):
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]

        def compute(self) -> torch.Tensor:
            ee_pos = self.asset.data.ee_pos_b
            ee_pos_target = self.env.command_manager.command_ee_pos_b
            pos_error = ((ee_pos - ee_pos_target)).abs().sum(1, True)
            return pos_error
    

    class ee_ori_tracking(Reward):
        def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.ee_id, self.ee_name = self.asset.find_bodies(ee_name)
            self.ee_id = self.ee_id[0]

            with torch.device(self.device):
                self.fwd_vec = torch.tensor([1., 0., 0.]).expand(self.num_envs, -1)
                self.up_vec = torch.tensor([0., 0., 1.]).expand(self.num_envs, -1)
                self.ee_forward_w = torch.zeros(self.num_envs, 3)

        def compute(self) -> torch.Tensor:
            self.ee_forward_w[:] = quat_rotate(
                self.asset.data.body_quat_w[:, self.ee_id],
                self.fwd_vec,
            )
            r = (self.ee_forward_w * self.env.command_manager.command_ee_forward_w).sum(-1, True)
            r = r.sign() * r.square()
            return r
        
        def debug_draw(self):
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.ee_id],
                self.ee_forward_w * 0.2,
                color=(1., 0.1, 0.1, 1.)
            )


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

