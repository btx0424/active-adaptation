from math import inf
import torch
import abc

from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.assets import Articulation
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse


class Reward:
    def __init__(self, env, weight: float, enabled: bool=True, clip_range=(-torch.inf, +torch.inf)):
        self.env = env
        self.weight = weight
        self.enabled = enabled
        self.clip_range = clip_range
    
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
        return (self.weight * self.compute()).clip(*self.clip_range)
    
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
    return 0.5 * r + 0.5 * asset.data.linvel_exp


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
    asset: Articulation = self.scene["robot"]
    r = - asset.data.root_ang_vel_b[:, :2].square().sum(-1, True)
    return 0.5 * r + 0.5 * asset.data.linvel_exp

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

        self.undesired_contact_cum = \
            self.asset.data.undesired_contact_cum \
            = torch.zeros(self.num_envs, 1, device=self.device)

        print(f"Penalizing contacts on {self.body_names}.")
    
    def reset(self, env_ids):
        self.undesired_contact_cum[env_ids] = 0.
    
    def update(self):
        contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.
        self.undesired_contact = - contact.float().sum(1, keepdim=True)
        self.undesired_contact_cum.add_(self.undesired_contact)

    def compute(self) -> torch.Tensor:
        return self.undesired_contact

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
        body_names: str = None, 
        sigma: float=0.25, 
        dim: int=3
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.sigma = sigma
        self.dim = dim
        if body_names is not None:
            self.body_ids, self.body_names = self.asset.find_bodies(body_names)
            self.body_masses = self.asset.root_physx_view.get_masses()[0, self.body_ids]
            self.body_masses = (self.body_masses / self.body_masses.sum()).unsqueeze(-1).to(self.device)
            self.body_ids = torch.tensor(self.body_ids, device=self.device)
        else:
            self.body_ids = None
    
    def compute(self) -> torch.Tensor:
        if self.body_ids is None:
            linvel = self.asset.data.root_lin_vel_b[:, :self.dim]
        else:
            linvel = quat_rotate_inverse(
                self.asset.data.root_quat_w,
                (self.asset.data.body_lin_vel_w[:, self.body_ids] * self.body_masses).mean(1)
            )
        linvel_error = (
            (linvel[:, :self.dim] - self.env.command_manager._command_linvel[:, :self.dim])
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
        body_names: str = None,
        sigma: float=0.25, 
        dim: int=3
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.sigma = sigma
        self.dim = dim
        if body_names is not None:
            self.body_ids, self.body_names = self.asset.find_bodies(body_names)
            self.body_masses = self.asset.root_physx_view.get_masses()[0, self.body_ids]
            self.body_masses = (self.body_masses / self.body_masses.sum()).unsqueeze(-1).to(self.device)
            self.body_ids = torch.tensor(self.body_ids, device=self.device)
        else:
            self.body_ids = None
        self.linvel = torch.zeros(self.num_envs, 3, device=self.device)
    
    def update(self):
        if self.body_ids is None:
            linvel = self.asset.data.root_lin_vel_b
        else:
            linvel = quat_rotate_inverse(
                self.asset.data.root_quat_w,
                (self.asset.data.body_lin_vel_w[:, self.body_ids] * self.body_masses).sum(1)
            )
        self.linvel[:] = linvel
        
    def compute(self) -> torch.Tensor:
        linvel_error = (
            (self.linvel[:, :self.dim] - self.env.command_manager._command_linvel[:, :self.dim])
            .square()
            .sum(-1, True)
        )
        self.asset.data.linvel_exp = torch.exp(- linvel_error / self.sigma)
        return self.asset.data.linvel_exp * -self.asset.data.projected_gravity_b[:, 2].unsqueeze(1)

    def debug_draw(self):
        if self.body_ids is None:
            linvel = self.asset.data.root_lin_vel_w
        else:
            linvel = (self.asset.data.body_lin_vel_w[:, self.body_ids] * self.body_masses).sum(1)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            linvel,
            color=(0.8, 0.1, 0.8, 1.)
        )

class angvel_z_exp(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_angvel: torch.Tensor = self.env.command_manager.command_angvel
    
    def compute(self) -> torch.Tensor:
        angvel_error = (
            (self.target_angvel - self.asset.data.root_ang_vel_b[:, 2])
            .square()
            .unsqueeze(1)
        )
        r = torch.exp(- angvel_error / 0.25)
        return r


class angvel_z_exp_shaped(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_angvel: torch.Tensor = self.env.command_manager.command_angvel
    
    def compute(self) -> torch.Tensor:
        angvel_error = (self.target_angvel - self.asset.data.root_ang_vel_b[:, 2]).unsqueeze(1)
        angvel_error = shaped_error(angvel_error)
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


class step_up(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_height_map: torch.Tensor = self.asset.data.feet_height_map
    
    def compute(self) -> torch.Tensor:
        is_standing = self.env.command_manager.is_standing_env
        r = self.feet_height_map.clamp_max(0.).sum((1, 2)).unsqueeze(1)
        return r  * (~is_standing)


class base_height_l1(Reward):
    def __init__(self, env, target_height: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        if isinstance(target_height, str) and target_height == "command":
            self.target_height = self.env.command_manager._target_base_height
        else:
            self.target_height = float(target_height)
        self.scale = self.asset.cfg.spawn.scale
        if isinstance(self.scale, torch.Tensor):
            self.scale = self.scale.to(self.device)
        else:
            self.scale = torch.tensor(1., device=self.device)

    def compute(self) -> torch.Tensor:
        target_height = self.target_height * self.scale
        height = self.asset.data.feet_pos_b[:, :, 2].min(1, keepdim=True)[0].abs()
        height_errot = (height - target_height) / target_height
        return - height_errot.abs()

class quadruped_stand(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.actuators["base_legs"].joint_indices

    def compute(self):
        jpos_error = (
            self.asset.data.joint_pos[:, self.joint_ids] - 
            self.asset.data.default_joint_pos[:, self.joint_ids]
        ).abs().sum(dim=1, keepdim=True)

        front_symmetry = self.asset.data.feet_pos_b[:, [0, 1], 1].sum(dim=1, keepdim=True).abs()
        back_symmetry = self.asset.data.feet_pos_b[:, [2, 3], 1].sum(dim=1, keepdim=True).abs()
        cost = - (jpos_error + front_symmetry + back_symmetry)

        return cost * self.env.command_manager.is_standing_env.reshape(self.num_envs, 1)


class stand_up(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        return (
            self.asset.data.root_pos_w[:, 2].clamp(max=0.7).square()
            -self.asset.data.projected_gravity_b[:, 0]
        ).unsqueeze(-1)

def normalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def shaped_error(error: torch.Tensor):
    """
    Shaped error for reward shaping.
    """
    return torch.maximum(error.abs(), error.square())