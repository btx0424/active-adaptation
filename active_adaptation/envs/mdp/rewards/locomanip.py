import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import wrap_to_pi
from ..commands import *
from .locomotion import Reward

class impedance_base_pos(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: BaseEEImpedance = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        diff = (self.command_manager.command_pos_base_w - self.asset.data.root_pos_w)
        r = torch.exp(- diff.norm(dim=-1, keepdim=True) / 0.25)
        return r
    
    
class impedance_ee_pos(Reward):
    def __init__(self, env, weight: float, ee_name: str = "arm_link06", enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id = self.asset.find_bodies(ee_name)[0][0]
        self.command_manager: BaseEEImpedance = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        diff = (self.command_manager.command_pos_ee_w - self.asset.data.body_pos_w[:, self.body_id])
        r = torch.exp(- diff.norm(dim=-1, keepdim=True) / 0.05)
        return r

class impedance_yaw_pos(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: BaseEEImpedance = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        diff = wrap_to_pi(self.command_manager.command_yaw_w - self.asset.data.heading_w[:, None])
        r = torch.exp(- diff.abs() / 0.25)
        return r

class impedance_base_vel(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: BaseEEImpedance = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        diff = (self.command_manager.command_linvel_base_w - self.asset.data.root_lin_vel_w)
        r = torch.exp(- diff.square().sum(dim=-1, keepdim=True) / 0.25)
        return r
    
class impedance_ee_vel(Reward):
    def __init__(self, env, weight: float, ee_name: str = "arm_link06", enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id = self.asset.find_bodies(ee_name)[0][0]
        self.command_manager: BaseEEImpedance = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        diff = (self.command_manager.command_linvel_ee_w - self.asset.data.body_lin_vel_w[:, self.body_id])
        r = torch.exp(- diff.square().sum(dim=-1, keepdim=True) / 0.05)
        return r

class impedance_yaw_vel(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: BaseEEImpedance = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        diff = wrap_to_pi(self.command_manager.command_yawvel - self.asset.data.root_ang_vel_w[:, 2:3])
        r = torch.exp(- diff.abs() / 0.25)
        return r
