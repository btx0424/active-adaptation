import torch
import abc

from omni.isaac.orbit.assets import Articulation

class Reward:
    def __init__(self, env, weight: float, enabled: bool=True):
        self.env = env
        self.weight = weight
        self.enabled = enabled
    
    def update(self):
        pass

    def reset(self, env_ids):
        pass
    
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return self.weight * self.compute()
    
    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError


def reward_func(func):
    class RewFunc(Reward):
        def compute(self):
            return func(self.env)
    return RewFunc


@reward_func
def energy_l1(self):
    asset: Articulation = self.scene["robot"]
    energy = (
        (asset.data.joint_vel * asset.data.applied_torque)
        .abs()
        .sum(dim=-1, keepdim=True)
    )
    return - energy


@reward_func
def energy_l2(self):
    asset: Articulation = self.scene["robot"]
    energy = (
        (asset.data.joint_vel * asset.data.applied_torque)
        .square()
        .sum(dim=-1, keepdim=True)
    )
    return - energy


@reward_func
def joint_acc_l2(self):
    asset: Articulation = self.scene["robot"]
    return - asset.data.joint_acc.square().sum(dim=-1, keepdim=True)


@reward_func
def joint_torques_l2(self):
    asset: Articulation = self.scene["robot"]
    return - asset.data.applied_torque.square().sum(dim=-1, keepdim=True)


@reward_func
def survival(self):
    return torch.ones(self.num_envs, 1, device=self.device)

@reward_func
def linvel_z_l2(self):
    return - self.scene["robot"].data.root_lin_vel_b[:, [2]].square()

@reward_func
def angvel_xy_l2(self):
    return - self.scene["robot"].data.root_ang_vel_b[:, :2].square().sum(-1, True)