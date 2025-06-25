import torch

from active_adaptation.envs.mdp.base import Reward
from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_rotate,
    normalize,
)
from active_adaptation.utils.symmetry import SymmetryTransform


class Game(Command):
    def __init__(self, env) -> None:
        super().__init__(env)
        with torch.device(self.device):
            self.role = torch.arange(self.num_envs, device=self.device) % 2
        
        self.update()
    
    @property
    def command(self):
        arange = torch.arange(self.num_envs, device=self.device)
        return torch.cat([
            quat_rotate_inverse(self.asset.data.root_quat_w, self.target_diff),
            quat_rotate_inverse(self.asset.data.root_quat_w, self.target_lin_vel_w),
            (arange % 2 == 0).reshape(self.num_envs, 1),
            (arange % 2 == 1).reshape(self.num_envs, 1),
        ], dim=-1)
    
    def symmetry_transforms(self):
        return SymmetryTransform(
            perm=torch.arange(8),
            signs=torch.tensor([1, -1, 1, 1, -1, 1, 1, 1])
        )
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        return super().sample_init(env_ids)

    def reset(self, env_ids: torch.Tensor):
        return super().reset(env_ids)
    
    def update(self):
        self.target_pos_w = torch.cat([
            self.asset.data.root_pos_w[1::2],
            self.asset.data.root_pos_w[::2],
        ])
        self.target_lin_vel_w = torch.cat([
            self.asset.data.root_lin_vel_w[1::2],
            self.asset.data.root_lin_vel_w[::2],
        ])
        self.target_diff = self.target_pos_w - self.asset.data.root_pos_w
        self.distance = self.target_diff[:, :2].norm(dim=-1, keepdim=True)
    
    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w[::2],
            self.asset.data.root_pos_w[1::2] - self.asset.data.root_pos_w[::2],
            color=(1, 0, 0, 1),
        )
    

class chase_distance(Reward[Game]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.last_distance = torch.zeros(self.num_envs, 1, device=self.device)
        self.distance_change = torch.zeros(self.num_envs, 1, device=self.device)
    
    def update(self):
        self.distance_change = self.command_manager.distance - self.last_distance
        self.last_distance = self.command_manager.distance

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        rew = torch.where(
            self.command_manager.role[:, None] == 0,
            -self.distance_change, # closer is better
            self.distance_change, # further is better
        )
        return rew.reshape(self.num_envs, 1)


class chase_velocity(Reward[Game]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_active = torch.arange(self.num_envs, device=self.device) % 2 == 0
        direction = normalize(self.command_manager.target_diff[:, :2])
        velocity = self.asset.data.root_lin_vel_w[:, :2]
        rew = torch.sum(direction * velocity, dim=1, keepdim=True)
        rew = torch.where(rew > 0, rew.log1p(), rew)
        return rew.reshape(self.num_envs, 1), is_active.reshape(self.num_envs, 1)


class evade(Reward[Game]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
    
    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_active = torch.arange(self.num_envs, device=self.device) % 2 == 1
        rew = 1 - torch.exp(-self.command_manager.distance * 0.5).reshape(self.num_envs, 1)
        return rew.reshape(self.num_envs, 1), is_active.reshape(self.num_envs, 1)


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
        rew = torch.where(self.command_manager.role[:, None] == 0, rew, -rew)
        return rew.reshape(self.num_envs, 1)

