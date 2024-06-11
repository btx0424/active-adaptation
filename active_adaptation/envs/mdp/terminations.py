import torch
import abc

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sensors import ContactSensor

class Termination:
    def __init__(self, env):
        self.env = env
    
    def update(self):
        pass

    def reset(self, env_ids):
        pass
    
    @abc.abstractmethod
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError


def termination_func(func):
    class TermFunc(Termination):
        def __call__(self):
            return func(self.env)
    return TermFunc


class crash(Termination):
    def __init__(self, env, body_names_expr, t_thres: float = 0., z_thres: float=-0.3):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = self.contact_sensor.find_bodies(body_names_expr)
        self.z_thres = z_thres
        self.t_thres = t_thres
        print(f"Terminate upon contact on {self.body_names}")
    
    def __call__(self):
        fall_over = self.asset.data.projected_gravity_b[:, 2] >= self.z_thres
        contact_force = self.contact_sensor.data.net_forces_w[:, self.body_indices]
        contact_times = self.contact_sensor.data.current_contact_time[:, self.body_indices]
        undesired_contact = (
            (contact_force.norm(dim=-1) > 1.)
            | (contact_times > self.t_thres)
        ).any(dim=1)
        terminated = (fall_over | undesired_contact).unsqueeze(1)
        return terminated


class tracking_error(Termination):
    def __init__(self, env, tracking_error_threshold):
        super().__init__(env)
        self.tracking_error_threshold = tracking_error_threshold
        self.asset: Articulation = self.env.scene["robot"]
    
    def __call__(self) -> torch.Tensor:
        return self.asset.data._tracking_error > self.tracking_error_threshold


class cum_error(Termination):
    def __init__(self, env, thres: float = 0.85):
        super().__init__(env)
        from .commands import Command2
        self.thres = thres
        self.command_manager: Command2 = self.env.command_manager
    
    def __call__(self) -> torch.Tensor:
        return self.command_manager._cum_error > self.thres


class joint_acc_exceeds(Termination):
    def __init__(self, env, thres: float):
        super().__init__(env)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
    
    def __call__(self) -> torch.Tensor:
        valid = (self.env.episode_length_buf > 2).unsqueeze(-1)
        return (
            valid & 
            (self.asset.data.joint_acc.abs() > self.thres).any(1, True)
        )

