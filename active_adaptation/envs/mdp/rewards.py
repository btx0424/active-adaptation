import torch
import abc

from omni.isaac.orbit.sensors import ContactSensor
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

    def debug_draw(self):
        pass


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


class undesired_contact(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        
        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        print(f"Penalizing contacts on {self.body_names}.")
    
    def compute(self) -> torch.Tensor:
        contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.
        return - contact.float().sum(1, keepdim=True)

    def debug_draw(self):
        self.env.debug_draw.point(
            # self.contact_sensor.data.pos_w[:, self.body_ids],
            self.asset.data.body_pos_w[:, self.articulation_body_ids],
            color=(1., .6, .4, 1.),
            size=20,
        )

class impact_force(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        print(f"Penalizing impact forces on {self.body_names}.")
    
    def compute(self) -> torch.Tensor:
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.body_ids]
        force = self.contact_sensor.data.net_forces_w.norm(dim=-1)[:, self.body_ids] / self.default_mass_total
        return - (force * first_contact).sum(1, True)


class linvel_rational(Reward):
    def __init__(
        self, 
        env, 
        weight: float, 
        enabled: bool = True, 
        frame:str = "body", 
        sigma: float=0.25, 
        dim: int=3
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        assert frame.startswith("w") or frame.startswith("b")
        self.frame = frame
        self.sigma = sigma
        self.dim = dim
    
    def compute(self) -> torch.Tensor:
        if self.frame.startswith("w"):
            linvel = self.asset.data.root_lin_vel_w[:, :self.dim]
        else:
            linvel = self.asset.data.root_lin_vel_b[:, :self.dim]
        linvel_error = (
            (linvel - self.env.command_manager._command_linvel[:, :self.dim])
            .square()
            .sum(-1, True)
        )
        return 1 / (1. + linvel_error / self.sigma)


class heading_projection(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self) -> torch.Tensor:
        target_heading_b = normalize(self.env.command_manager._command_heading)
        return target_heading_b[:, [0]]


def normalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)