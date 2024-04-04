import torch
import abc

from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.assets import Articulation

class Reward:
    def __init__(self, env, weight: float, enabled: bool=True):
        self.env = env
        self.weight = weight
        self.enabled = enabled
    
    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device
    
    def step(self, substep: int):
        pass

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
    r = - asset.data.joint_acc.square().sum(dim=-1, keepdim=True)
    return 0.5 * r + 0.5 * self.scene["robot"].data.linvel_exp


@reward_func
def joint_torques_l2(self):
    asset: Articulation = self.scene["robot"]
    return - asset.data.applied_torque.square().sum(dim=-1, keepdim=True)


@reward_func
def survival(self):
    return torch.ones(self.num_envs, 1, device=self.device)

@reward_func
def linvel_z_l2(self):
    asset: Articulation = self.scene["robot"]
    return - asset.data.root_lin_vel_b[:, [2]].square()

@reward_func
def angvel_xy_l2(self):
    return - self.scene["robot"].data.root_ang_vel_b[:, :2].square().sum(-1, True)

@reward_func
def heading_yaw(self):
    asset: Articulation = self.scene["robot"]
    yaw_diff = self.command_manager.command[:, 2].abs()
    return  - yaw_diff.unsqueeze(1)


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

    # def debug_draw(self):
    #     self.env.debug_draw.point(
    #         # self.contact_sensor.data.pos_w[:, self.body_ids],
    #         self.asset.data.body_pos_w[:, self.articulation_body_ids],
    #         color=(1., .6, .4, 1.),
    #         size=20,
    #     )

class impact_force_l2(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        print(f"Penalizing impact forces on {self.body_names}.")
    
    def compute(self) -> torch.Tensor:
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.body_ids]
        contact_forces = self.contact_sensor.data.net_forces_w_history.norm(dim=-1).mean(1)
        force = contact_forces[:, self.body_ids] / self.default_mass_total
        return - (force.square() * first_contact).sum(1, True)


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


class linvel_exp(Reward):
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
        self.asset.data.linvel_exp = torch.exp(- linvel_error / self.sigma)
        return self.asset.data.linvel_exp * -self.asset.data.projected_gravity_b[:, 2].unsqueeze(1)


class angvel_z_exp(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        angvel_error = (
            self.env.command_manager.command[:, 2] 
            - self.asset.data.root_ang_vel_b[:, 2]
        ).square().unsqueeze(1)
        r = torch.exp(- angvel_error / 0.25)
        r = (0.5 + 0.5 * self.asset.data.linvel_exp) * r
        return r


class linvel_and_height(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        command = self.env.command_manager.command
        linvel = self.asset.data.root_lin_vel_b
        linvel_error = (
            (linvel - self.env.command_manager._command_linvel)
            .square()
            .sum(-1, True)
        )
        print(command)
        height_error = (
            (self.asset.data.root_pos_w[:, 2] - command[:, 3])
            .square()
            .unsqueeze(1)
        )
        return torch.exp(- linvel_error / 0.25) * torch.exp(- height_error / 0.25)


class heading_projection(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self) -> torch.Tensor:
        target_heading_b = normalize(self.env.command_manager._command_heading)
        return target_heading_b[:, [0]]


class linacc_z_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.prev_linvel_z = torch.zeros(self.env.num_envs, device=self.env.device)
        self.linacc_z = torch.zeros(self.env.num_envs, device=self.env.device)
    
    def reset(self, env_ids):
        self.prev_linvel_z[env_ids] = 0.
        self.linacc_z[env_ids] = 0.
    
    def update(self):
        self.linacc_z[:] = (self.asset.data.root_lin_vel_b[:, 2] - self.prev_linvel_z) / self.env.step_dt
        self.prev_linvel_z[:] = self.asset.data.root_lin_vel_b[:, 2]

    def compute(self) -> torch.Tensor:
        return - self.linacc_z.square().unsqueeze(1)


class test_joint_acc(Reward):
    def __init__(self, env, weight: float, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]

        with torch.device(self.device):
            self.joint_acc = torch.zeros(self.num_envs, self.asset.num_joints)

    def reset(self, env_ids):
        self.joint_acc[env_ids] = 0.
    
    def step(self, substep: int):
        self.joint_acc.lerp_(self.asset.data.joint_acc, 0.9)
    
    def compute(self) -> torch.Tensor:
        return (self.joint_acc - self.asset.data.joint_acc).square().sum(-1, True)


class tracking_error_exp(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        return torch.exp(- self.asset.data._tracking_error / 0.5)


class feet_slip(Reward):
    def __init__(self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool=True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
    
    def compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.02
        feet_vel = self.asset.data.body_lin_vel_w[:, self.articulation_body_ids, :2]
        slip = (in_contact * feet_vel.norm(dim=-1).square()).sum(dim=1, keepdim=True)
        return - slip * self.asset.data.linvel_exp


class feet_air_time(Reward):
    def __init__(
        self, 
        env: "LocomotionEnv", 
        body_names: str, 
        thres: float, 
        weight: float, 
        enabled: bool=True,
        condition_on_linvel: bool=True,
    ):
        super().__init__(env, weight, enabled)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.condition_on_linvel = condition_on_linvel

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.reward = torch.zeros(self.num_envs, 1, device=self.env.device)

    def compute(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.body_ids]
        last_air_time = self.contact_sensor.data.last_air_time[:, self.body_ids]
        self.reward = torch.sum((last_air_time - self.thres) * first_contact, dim=1, keepdim=True)
        self.reward *= (~self.env.command_manager.is_standing_env)
        if self.condition_on_linvel:
            self.reward *= self.asset.data.linvel_exp
        return self.reward


class feet_contact_count(Reward):
    def __init__(self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool=True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.first_contact = torch.zeros(self.num_envs, len(self.body_ids), device=self.env.device)

    def compute(self):
        self.first_contact[:] = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.body_ids]
        return self.first_contact.sum(1, keepdim=True)



def normalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)