import torch
from .locomotion import Reward
from omni.isaac.lab.assets import Articulation
from ..commands import EEImpedance


class impedance_ee_pos(Reward):
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
            - self.asset.data.body_lin_vel_w[:, self.body_id]
        )
        r = torch.exp(-diff.norm(dim=-1, keepdim=True) / self.l)
        return r
