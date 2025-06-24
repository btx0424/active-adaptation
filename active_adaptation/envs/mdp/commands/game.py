import torch

from active_adaptation.envs.mdp.base import Reward
from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_rotate,
    normalize,
)


class Game(Command):
    def __init__(self, env) -> None:
        super().__init__(env)
        with torch.device(self.device):
            self.role = torch.arange(self.num_envs, device=self.device) % 2
    
    @property
    def command(self):
        arange = torch.arange(self.num_envs, device=self.device)
        return torch.cat([
            quat_rotate_inverse(self.asset.data.root_quat_w, self.target_diff),
            quat_rotate_inverse(self.asset.data.root_quat_w, self.target_lin_vel_w),
            (arange % 2 == 0).reshape(self.num_envs, 1),
            (arange % 2 == 1).reshape(self.num_envs, 1),
        ], dim=-1)
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        return super().sample_init(env_ids)

    def reset(self, env_ids: torch.Tensor):
        return super().reset(env_ids)
    
    def update(self):
        self.target_pos_w = torch.cat([
            self.asset.data.root_pos_w[1::2],
            self.asset.data.root_pos_w[::2],
        ], dim=-1)
        self.target_lin_vel_w = torch.cat([
            self.asset.data.root_lin_vel_w[1::2],
            self.asset.data.root_lin_vel_w[::2],
        ], dim=-1)
        self.target_diff = self.target_pos_w - self.asset.data.root_pos_w
        self.distance = self.target_diff[:, :2].norm(dim=-1)
    
    def debug_draw(self):
        return super().debug_draw()
    

class chase(Reward[Game]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.last_distance = torch.zeros(self.num_envs, device=self.device)
        self.distance_change = torch.zeros(self.num_envs, device=self.device)
    
    def update(self):
        self.distance_change = self.command_manager.distance - self.last_distance
        self.last_distance = self.command_manager.distance

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_active = torch.arange(self.num_envs, device=self.device) % 2 == 0
        rew = self.distance_change.reshape(self.num_envs, 1)
        return rew, is_active


class evade(Reward[Game]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
    
    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_active = torch.arange(self.num_envs, device=self.device) % 2 == 1
        rew = 1 - torch.exp(-self.command_manager.distance).reshape(self.num_envs, 1)
        return rew, is_active


class target_in_sight(Reward[Game]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
    
    def compute(self) -> torch.Tensor:
        forward_vec = quat_rotate(
            self.asset.data.root_quat_w,
            torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, 3)
        )
        diff = normalize(self.command_manager.target_diff)
        rew = torch.sum(forward_vec[:, :2] * diff[:, :2], dim=1, keepdim=True)
        rew = torch.where(self.command_manager.role == 0, rew, -rew)
        return rew

