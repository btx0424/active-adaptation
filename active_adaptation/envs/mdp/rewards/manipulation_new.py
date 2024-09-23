import torch
from .locomotion import Reward
from omni.isaac.lab.assets import Articulation
from ..commands import EEImpedance, EEPosition


class impedance_ee_pos(Reward):
    def __init__(
        self, env, weight: float, ee_name: str = "arm_link06", enabled: bool = True, l: float = 0.05
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id = self.asset.find_bodies(ee_name)[0][0]
        self.command_manager: EEImpedance | EEPosition = self.env.command_manager
        self.l = l

    def compute(self) -> torch.Tensor:
        diff = (
            self.command_manager.command_pos_ee_w
            - self.asset.data.body_pos_w[:, self.body_id]
        )
        r = torch.exp(-diff.norm(dim=-1, keepdim=True) / self.l)
        return r


class impedance_ee_vel(Reward):
    def __init__(
        self, env, weight: float, ee_name: str = "arm_link06", enabled: bool = True, l: float = 0.05
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id = self.asset.find_bodies(ee_name)[0][0]
        self.command_manager: EEImpedance = self.env.command_manager
        self.l = l

    def compute(self) -> torch.Tensor:
        diff = (
            self.command_manager.command_linvel_ee_w
            - self.asset.ee_lin_vel_w
        )
        r = torch.exp(-diff.norm(dim=-1, keepdim=True) / self.l)
        return r

class impedance_ee_pos_err(Reward):
    def __init__(
        self, env, weight: float, ee_name: str = "arm_link06", enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id = self.asset.find_bodies(ee_name)[0][0]
        self.command_manager: EEImpedance | EEPosition = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = (
            self.command_manager.command_pos_ee_w
            - self.asset.data.body_pos_w[:, self.body_id]
        )
        r = diff.norm(dim=-1, keepdim=True)
        return r


class impedance_ee_vel_err(Reward):
    def __init__(
        self, env, weight: float, ee_name: str = "arm_link06", enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id = self.asset.find_bodies(ee_name)[0][0]
        self.command_manager: EEImpedance = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = (
            self.command_manager.command_linvel_ee_w
            - self.asset.ee_lin_vel_w
        )
        r = diff.norm(dim=-1, keepdim=True)
        return r


class ee_angvel_penalty(Reward):
    def __init__(self, env, ee_name: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id = self.asset.find_bodies(ee_name)[0]
        self.body_id = self.body_id[0]

    def compute(self) -> torch.Tensor:
        ee_angvel_w = self.asset.data.body_ang_vel_w[:, self.body_id]
        return - ee_angvel_w.square().sum(1, True)