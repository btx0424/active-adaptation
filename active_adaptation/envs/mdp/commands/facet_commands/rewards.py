import torch
import einops

from active_adaptation.envs.mdp import Reward
from active_adaptation.utils.math import wrap_to_pi
from .impedance import Impedance
from .impedance_manip import ImpedanceCommandManager


class impedance_pos(Reward[Impedance]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.impedance: Impedance = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = self.impedance.surrogate_pos_target - self.impedance.get_pos_w().unsqueeze(1)
        error_l2 = diff[:, :, :2].square().sum(dim=-1, keepdim=True)
        return error_l2.mean(1)


class impedance_vel(Reward[Impedance]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.impedance: Impedance = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = einops.rearrange(self.impedance.surrogate_lin_vel_target, "n t1 d -> n t1 1 d") \
            - einops.rearrange(self.impedance.lin_vel_ema.ema, "n t2 d-> n 1 t2 d")
        error_l2 = (diff * self.impedance.dim_weights).square().sum(dim=-1, keepdim=True)
        r = ((- error_l2 / 0.25).exp() - 0.25 * error_l2).mean(1)
        return r.max(dim=1).values

class impedance_acc(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.command_manager: Impedance = self.env.command_manager
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        lin_acc_w = self.asset.data.body_acc_w[:, 0, :2]
        error_l2 = (self.command_manager.ref_lin_acc_w[:, 0, :2] - lin_acc_w).square().sum(1, True)
        return torch.exp(- error_l2 / 2.0)


class impedance_yaw_pos(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.command_manager: Impedance = self.env.command_manager
        self.asset = self.command_manager.asset
        
    def compute(self) -> torch.Tensor:
        target_yaw = self.command_manager.surrogate_yaw_target 
        diff = target_yaw - self.asset.data.heading_w.reshape(-1, 1, 1)
        diff = wrap_to_pi(diff)
        error_l2 = diff.square()
        r = torch.exp(-error_l2 / 0.25).mean(1)
        return r


class impedance_yaw_vel(Reward[Impedance]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.command_manager: Impedance = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = einops.rearrange(self.command_manager.surrogate_yaw_vel_target, "n t1 d -> n t1 1 d") \
            - einops.rearrange(self.command_manager.ang_vel_ema.ema[:, :, 2:3], "n t2 d-> n 1 t2 d")
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = ((- error_l2 / 0.25).exp() - 0.25 * error_l2).mean(1)
        return r.max(dim=1).values


class impedance_pos_error(Reward[Impedance]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.impedance: Impedance = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = self.impedance.ref_pos_w[:, [*self.impedance.surr_steps, -1]] - self.impedance.get_pos_w().unsqueeze(1)
        error_l2 = diff[:, :, :2].square().sum(dim=-1, keepdim=True)
        return error_l2.mean(1)


class impedance_vel_error(Reward[Impedance]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.impedance: Impedance = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = self.impedance.ref_lin_vel_w[:, [*self.impedance.surr_steps, -1]] - self.impedance.get_lin_vel_w().unsqueeze(1)
        error_l2 = diff[:, :, :2].square().sum(dim=-1, keepdim=True)
        return error_l2.mean(1)


class impedance_acc_error(Reward[Impedance]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.impedance: Impedance = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = self.impedance.ref_lin_acc_w[:, 0] - self.impedance.asset.data.body_acc_w[:, 0, :3]
        error_l2 = diff[:, :2].square().sum(dim=-1, keepdim=True)
        return error_l2

