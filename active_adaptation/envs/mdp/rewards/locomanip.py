import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import wrap_to_pi, quat_rotate
from active_adaptation.utils.helpers import batchify
from ..commands import BaseEEImpedance, Impedance
from .locomotion import Reward

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

class impedance_base_pos(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, l: float = 0.25):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: BaseEEImpedance | Impedance = self.env.command_manager
        self.l = l
    
    def compute(self) -> torch.Tensor:
        diff = (self.command_manager.command_pos_base_w[:, :2] - self.asset.data.root_pos_w[:, :2])
        r = torch.exp(- diff.norm(dim=-1, keepdim=True) / self.l)
        return r
    
class impedance_base_vel(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, l: float = 0.25):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: BaseEEImpedance | Impedance = self.env.command_manager
        self.l = l
    
    def compute(self) -> torch.Tensor:
        diff = (self.command_manager.command_linvel_base_w[:, :2] - self.asset.data.root_lin_vel_w[:, :2])
        command_speed = self.command_manager.command_speed
        diff = torch.where(command_speed > 1.0, diff / command_speed, diff)
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = torch.exp(- error_l2 / self.l) - error_l2
        return r
    
class impedance_ee_pos_b(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, l: float = 0.05):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: BaseEEImpedance = self.env.command_manager
        self.pos_ee_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.l = l

    def update(self):
        pos_ee_w = self.asset.data.body_pos_w[:, self.command_manager.ee_body_id] - self.asset.data.root_pos_w
        self.pos_ee_b[:] = yaw_rotate(
            -self.asset.data.heading_w[:, None],
            pos_ee_w
        )

    def compute(self) -> torch.Tensor:
        diff = (self.command_manager.command_pos_ee_b - self.pos_ee_b)
        r = torch.exp(- diff.norm(dim=-1, keepdim=True) / self.l)
        return r * self.command_manager.is_arm_activated

class impedance_ee_vel_b(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, l: float = 0.05):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: BaseEEImpedance = self.env.command_manager
        self.linvel_ee_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.l = l
    
    def update(self):
        pos_ee_w = self.asset.data.body_pos_w[:, self.command_manager.ee_body_id] - self.asset.data.root_pos_w
        root_ang_vel_w_only_yaw = self.asset.data.root_ang_vel_w.clone()
        root_ang_vel_w_only_yaw[:, :2] = 0.0
        coriolis_vel_ee_w = self.asset.data.root_lin_vel_w + torch.cross(
            root_ang_vel_w_only_yaw, 
            pos_ee_w,
            dim=-1,
        )

        self.linvel_ee_b[:] = yaw_rotate(
            -self.asset.data.heading_w[:, None], self.asset.data.body_lin_vel_w[:, self.command_manager.ee_body_id] - coriolis_vel_ee_w
        )

    def compute(self) -> torch.Tensor:
        diff = (self.command_manager.command_linvel_ee_b - self.linvel_ee_b)
        r = torch.exp(- diff.square().sum(dim=-1, keepdim=True) / self.l)
        return r * self.command_manager.is_arm_activated

# class impedance_yaw_pos(Reward):
#     def __init__(self, env, weight: float, enabled: bool = True, l: float = 0.25):
#         super().__init__(env, weight, enabled)
#         self.asset: Articulation = self.env.scene["robot"]
#         self.command_manager: BaseEEImpedance | Impedance = self.env.command_manager
#         self.l = l
    
#     def compute(self) -> torch.Tensor:
#         diff = wrap_to_pi(self.command_manager.command_yaw_w - self.asset.data.heading_w[:, None])
#         r = torch.exp(- diff.abs() / self.l)
#         return r

# class impedance_yaw_vel(Reward):
#     def __init__(self, env, weight: float, enabled: bool = True, l: float = 0.25):
#         super().__init__(env, weight, enabled)
#         self.asset: Articulation = self.env.scene["robot"]
#         self.command_manager: BaseEEImpedance | Impedance = self.env.command_manager
#         self.l = l
    
#     def compute(self) -> torch.Tensor:
#         diff = (self.command_manager.command_yaw_vel - self.asset.data.root_ang_vel_w[:, 2:3])
#         r = torch.exp(- diff.abs() / self.l)
#         return r

class ee_forward(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: BaseEEImpedance = self.env.command_manager

        self.ee_fwd_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.base_fwd_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.ee_fwd_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_fwd_w = torch.zeros(self.num_envs, 3, device=self.device)
    
    def update(self):
        self.ee_fwd_w[:] = quat_rotate(self.asset.data.body_quat_w[:, self.command_manager.ee_body_id], self.ee_fwd_vec)
        self.base_fwd_w[:] = quat_rotate(self.asset.data.root_quat_w, self.base_fwd_vec)
    
    def compute(self) -> torch.Tensor:
        diff = self.base_fwd_w - self.ee_fwd_w
        r = - diff.norm(dim=-1, keepdim=True)
        return r
    
    def debug_draw(self):
        # draw ee forward vector (yellow)
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.command_manager.ee_body_id],
            self.ee_fwd_w,
            color=(1.0, 1.0, 0.0, 1.0),
        )
        # draw body forward vector
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.command_manager.ee_body_id],
            self.base_fwd_w,
            color=(1.0, 0.0, 0.0, 1.0),
        )