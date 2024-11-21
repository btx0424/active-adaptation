from math import inf
import torch
import abc
from typing import TYPE_CHECKING

from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat, wrap_to_pi
import omni.isaac.lab.utils.string as string_utils
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify
from ..commands import *

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

class Reward:
    def __init__(self, env, weight: float, enabled: bool=True, clip_range=(-torch.inf, +torch.inf)):
        self.env: Env = env
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

    def post_step(self, substep: int):
        pass

    def update(self):
        pass

    def reset(self, env_ids):
        pass
    
    def __call__(self) -> torch.Tensor:
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
    if hasattr(asset.data, "linvel_exp"):
        return r * (0.5 + 0.5 * asset.data.linvel_exp)
    else:
        return r


@reward_func
def survival(self):
    return torch.ones(self.num_envs, 1, device=self.device)


class linvel_z_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager = self.env.command_manager
        if isinstance(self.command_manager, Command2):
            self.coeff = self.command_manager.command[:, 3].unsqueeze(1)
        else:
            self.coeff = 1.

    def compute(self) -> torch.Tensor:
        # command_speed = self.command_manager.command_linvel[:, 2].norm(dim=-1, keepdim=True)
        linvel_z = self.asset.data.root_lin_vel_b[:, 2].unsqueeze(1)
        return - linvel_z.square() * self.coeff


class angvel_xy_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, body_name: str=None):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        if body_name is not None:
            self.body_id = self.asset.find_bodies(body_name)[0][0]
        else:
            self.body_id = None
        self.world_frame = False
            
    def update(self):
        if self.body_id is not None:
            angvel = self.asset.data.body_ang_vel_w[:, self.body_id]
            if not self.world_frame:
                angvel = quat_rotate_inverse(self.asset.data.root_quat_w, angvel)
        else:
            if self.world_frame:
                angvel = self.asset.data.root_ang_vel_w
            else:
                angvel = self.asset.data.root_ang_vel_b
        self.angvel = angvel
    
    def compute(self) -> torch.Tensor:
        r = - self.angvel[:, :2].square().sum(-1, True)
        return r
        

@reward_func
def heading_yaw(self):
    asset: Articulation = self.scene["robot"]
    yaw_diff = self.command_manager.command[:, 2].abs()
    return  - yaw_diff.unsqueeze(1)


class energy_l1(Reward):
    
    decay: float = 0.99

    def __init__(self, env, weight: float, enabled: bool = True, a={".*": 1.0}):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.a = string_utils.resolve_matching_names_values(dict(a), self.asset.joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.a = torch.tensor(self.a, device=self.device)
        
        self.power = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        self.energy = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        self.count = torch.zeros(self.num_envs, 1, device=self.device)
        self.asset.data.energy_ema = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)

    def reset(self, env_ids):
        self.energy[env_ids] = 0.
        self.count[env_ids] = 0.

    def update(self):
        torques = self.asset.data.applied_torque[:, self.joint_ids]
        joint_vel = self.asset.data.joint_vel[:, self.joint_ids]
        self.power[:] = (torques * joint_vel).abs()
        
        self.energy.add_(self.power).mul_(self.decay)
        self.count.add_(1.).mul_(self.decay)
        self.asset.data.energy_ema[:] = self.energy / self.count

    def compute(self) -> torch.Tensor:
        return - (self.power * self.a).sum(1, keepdim=True)


class energy_dist_lr(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.energy_ema: Articulation = self.asset.data.energy_ema
        self.left_joint_ids = self.asset.find_joints("[F,R]L_.*_joint")[0]
        self.right_joint_ids = self.asset.find_joints("[F,R]R_.*_joint")[0]

    def compute(self) -> torch.Tensor:
        energy_left = self.energy_ema[:, self.left_joint_ids]
        energy_right = self.energy_ema[:, self.right_joint_ids]
        return - (energy_left - energy_right).square().sum(1, keepdim=True)


class energy_dist_fb(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.energy_ema: Articulation = self.asset.data.energy_ema
        self.front_joint_ids = self.asset.find_joints("F[L,R]_.*_joint")[0]
        self.rear_joint_ids = self.asset.find_joints("R[L,R]_.*_joint")[0]

    def compute(self) -> torch.Tensor:
        energy_front = self.energy_ema[:, self.front_joint_ids]
        energy_rear = self.energy_ema[:, self.rear_joint_ids]
        return - (energy_front - energy_rear).square().sum(1, keepdim=True)


class joint_torques_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids =  self.asset.find_joints(joint_names)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
    
    def compute(self) -> torch.Tensor:
        return - self.asset.data.applied_torque[:, self.joint_ids].square().sum(1, keepdim=True)


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
            (linvel[:, :self.dim] - self.env.command_manager.command_linvel[:, :self.dim])
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
        dim: int=3,
        yaw_only: bool=False,
        gamma: float=0.0,
        upright: bool=False
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.sigma = sigma
        self.dim = dim
        self.yaw_only = yaw_only
        self.gamma = gamma
        self.upright = upright
        if body_names is not None:
            self.body_ids, self.body_names = self.asset.find_bodies(body_names)
            self.body_masses = self.asset.root_physx_view.get_masses()[0, self.body_ids]
            self.body_masses = (self.body_masses / self.body_masses.sum()).unsqueeze(-1).to(self.device)
            self.body_ids = torch.tensor(self.body_ids, device=self.device)
        else:
            self.body_ids = None
        self.linvel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.linvel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.count = torch.zeros(self.num_envs, 1, device=self.device)
        self.command_manager: Command2 = self.env.command_manager

    def reset(self, env_ids):
        self.linvel_w[env_ids] = 0.
        self.count[env_ids] = 0.

    def update(self):
        if self.body_ids is None:
            linvel_w = self.asset.data.root_lin_vel_w
        else:
            linvel_w = (self.asset.data.body_lin_vel_w[:, self.body_ids] * self.body_masses).sum(1)
        if self.yaw_only:
            quat = yaw_quat(self.asset.data.root_quat_w)
        else:
            quat = self.asset.data.root_quat_w
        
        self.linvel_w.mul_(self.gamma).add_(linvel_w)
        self.count.mul_(self.gamma).add_(1.)
        self.linvel_b[:] = quat_rotate_inverse(quat, self.linvel_w / self.count)
        
    def compute(self) -> torch.Tensor:
        linvel_error = (
            (self.linvel_b[:, :self.dim] - self.command_manager.command_linvel[:, :self.dim])
            .square()
            .sum(-1, True)
        )
        self.asset.data.linvel_exp = torch.exp(- linvel_error / self.sigma)
        if self.upright:
            return self.asset.data.linvel_exp * -self.asset.data.projected_gravity_b[:, 2].unsqueeze(1)
        else:
            return self.asset.data.linvel_exp

    def debug_draw(self):
        # draw smoothed lin vel (purple)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            self.linvel_w / self.count,
            color=(0.8, 0.1, 0.8, 1.)
        )


class linvel_projection(Reward):
    def __init__(
        self, 
        env, 
        weight: float, 
        enabled: bool = True,
        dim: int=2,
        yaw_only: bool=False,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.dim = dim
        self.yaw_only = yaw_only
        self.linvel = torch.zeros(self.num_envs, 3, device=self.device)
    
    def update(self):
        linvel_w = self.asset.data.root_lin_vel_w
        if self.yaw_only:
            quat = yaw_quat(self.asset.data.root_quat_w)
        else:
            quat = self.asset.data.root_quat_w
        self.linvel[:] = quat_rotate_inverse(quat, linvel_w)
    
    def compute(self) -> torch.Tensor:
        command_linvel_b: torch.Tensor = self.env.command_manager.command_linvel[:, :self.dim]
        command_linvel_b = command_linvel_b / command_linvel_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # or
        # command_linvel_b.nan_to_num_(nan=0., posinf=0., neginf=0.)
        projection = (self.linvel[:, :self.dim] * command_linvel_b).sum(dim=-1, keepdim=True)
        reward = projection.clamp_max(self.env.command_manager.command_speed)
        return reward.reshape(self.num_envs, 1)


class linvel_yaw_exp(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        command_linvel_b = self.env.command_manager.command_linvel[:, :2]
        linvel_yaw_b = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_quat_w),
            self.asset.data.root_lin_vel_w
        )
        linvel_error = (command_linvel_b - linvel_yaw_b[:, :2]).square().sum(-1, True)
        self.asset.data.linvel_exp = torch.exp(-linvel_error / 0.25)
        return self.asset.data.linvel_exp


class angvel_z_exp(Reward):
    def __init__(
        self, 
        env, 
        weight: float, 
        enabled: bool = True, 
        world_frame: bool=False,
        body_name: str=None,
        gamma: float = 0.0
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.world_frame = world_frame
        self.gamma = gamma
        self.target_angvel: torch.Tensor = self.env.command_manager.command_angvel
        self.count = torch.zeros(self.num_envs, 1, device=self.device)
        self.angvel_sum = torch.zeros(self.num_envs, 3, device=self.device)
        if body_name is not None:
            self.body_id = self.asset.find_bodies(body_name)[0][0]
        else:
            self.body_id = None

    def update(self):
        if self.body_id is not None:
            angvel = self.asset.data.body_ang_vel_w[:, self.body_id]
            if not self.world_frame:
                angvel = quat_rotate_inverse(self.asset.data.root_quat_w, angvel)
        else:
            if self.world_frame:
                angvel = self.asset.data.root_ang_vel_w
            else:
                angvel = self.asset.data.root_ang_vel_b
        self.angvel_sum.mul_(self.gamma).add_(angvel)
        self.count.mul_(self.gamma).add_(1)
        self.angvel = self.angvel_sum / self.count

    def compute(self) -> torch.Tensor:
        angvel_error = (
            (self.target_angvel - self.angvel[:, 2])
            .square()
            .unsqueeze(1)
        )
        r = torch.exp(- angvel_error / 0.25)
        return r

    def debug_draw(self):
        if self.body_id is not None:
            fwd = torch.tensor([1., 0., 0.], device=self.device)
            body_quat_w = self.asset.data.body_quat_w[:, self.body_id]
            fwd = quat_rotate(body_quat_w, fwd.expand(self.num_envs, 3))
            
            self.env.debug_draw.vector(
                self.asset.data.root_pos_w,
                fwd,
                color=(1., 0., 0., 1.),
                size=2.0
            )

class angvel_z_exp_shaped(Reward):
    def __init__(
        self, 
        env, 
        weight: float, 
        enabled: bool = True,
        world_frame: bool=False
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_angvel: torch.Tensor = self.env.command_manager.command_angvel
        self.world_frame = world_frame
    
    def compute(self) -> torch.Tensor:
        if self.world_frame:
            angvel_z = self.asset.data.root_ang_vel_w[:, 2]
        else:
            angvel_z = self.asset.data.root_ang_vel_b[:, 2]
        angvel_error = (self.target_angvel - angvel_z).unsqueeze(1)
        angvel_error = shaped_error(angvel_error)
        r = torch.exp(- angvel_error / 0.25)
        return r


class linvel_and_height(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        command = self.env.command_manager.command
        linvel = self.asset.data.root_lin_vel_b
        linvel_error = (
            (linvel - self.env.command_manager.command_linvel)
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
        self.last_air_time = torch.zeros(self.num_envs, len(self.body_ids), device=self.env.device)

    def compute(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.body_ids]
        last_air_time = self.contact_sensor.data.last_air_time[:, self.body_ids]
        contact = self.last_air_time != last_air_time
        self.last_air_time = last_air_time
        self.reward = torch.sum((last_air_time - self.thres).clamp_max(0.) * contact, dim=1, keepdim=True)
        self.reward *= (~self.env.command_manager.is_standing_env)
        # if self.condition_on_linvel and hasattr(self.asset.data, "linvel_exp"):
        #     self.reward *= self.asset.data.linvel_exp
        return self.reward

class max_feet_height(Reward):
    def __init__(
        self,
        env,
        body_names: str,
        target_height: float,
        weight: float,
        enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.target_height = target_height

        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self.asset_body_ids, self.asset_body_names = self.asset.find_bodies(body_names)

        self.in_contact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.impact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.detach = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.has_impact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.max_height = torch.zeros(self.num_envs, len(self.body_ids), device=self.device)
        self.impact_point = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)
        self.detach_point = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)

    def reset(self, env_ids):
        self.has_impact[env_ids] = False
    
    def update(self):
        contact_force = self.contact_sensor.data.net_forces_w_history[:, :, self.body_ids]
        feet_pos_w = self.asset.data.body_pos_w[:, self.asset_body_ids]
        in_contact = (contact_force.norm(dim=-1) > 0.01).any(dim=1)
        self.impact = (~self.in_contact) & in_contact
        self.detach = self.in_contact & (~in_contact)
        self.in_contact = in_contact
        self.has_impact.logical_or_(self.impact)
        self.impact_point[self.impact] = feet_pos_w[self.impact]
        self.detach_point[self.detach] = feet_pos_w[self.detach]
        self.max_height = torch.where(
            self.detach,
            feet_pos_w[:, :, 2],
            torch.maximum(self.max_height, feet_pos_w[:, :, 2])
        )

    def compute(self) -> torch.Tensor:
        reference_height = torch.maximum(self.impact_point[:, :, 2], self.detach_point[:, :, 2])
        max_height = self.max_height - reference_height
        r = self.impact * (max_height / self.target_height).clamp_max(1.0) 
        return r.sum(dim=1, keepdim=True)

    def debug_draw(self):
        feet_pos_w = self.asset.data.body_pos_w[:, self.asset_body_ids]
        self.env.debug_draw.point(
            feet_pos_w[self.impact],
            color=(1.0, 0., 0., 1.),
            size=30,
        )


# class linvel_exp2(Reward):
#     def __init__(self, env, weight: float, enabled: bool = True):
#         super().__init__(env, weight, enabled)
#         self.feet_names = ".*_foot"
#         self.asset: Articulation = self.env.scene["robot"]
#         self.body_ids = self.asset.find_bodies(self.feet_names)[0]
#         self.center_pos_history = torch.zeros(self.num_envs, 3, 8, device=self.device)
#         self.command_manager: Command2 = self.env.command_manager

#     def update(self):
#         root_quat = self.asset.data.root_quat_w
#         feet_pos_w = self.asset.data.body_pos_w[:, self.body_ids]
#         feet_pos = feet_pos_w # quat_rotate_inverse(root_quat.unsqueeze(1), feet_pos_w)
#         feet_pos = quat_rotate_inverse(root_quat.unsqueeze(1), feet_pos - self.asset.data.root_pos_w.unsqueeze(1))
#         # find the bounding rectangle
#         min_x = feet_pos[:, :, 0].min(1)[0]
#         max_x = feet_pos[:, :, 0].max(1)[0]
#         min_y = feet_pos[:, :, 1].min(1)[0]
#         max_y = feet_pos[:, :, 1].max(1)[0]
#         y = 0.5 * torch.ones_like(min_y)
#         # rotate back to world frame
#         self.ul_corner = quat_rotate(root_quat, torch.stack([min_x, min_y, y], dim=-1)) + self.asset.data.root_pos_w
#         self.ur_corner = quat_rotate(root_quat, torch.stack([max_x, min_y, y], dim=-1)) + self.asset.data.root_pos_w
#         self.lr_corner = quat_rotate(root_quat, torch.stack([max_x, max_y, y], dim=-1)) + self.asset.data.root_pos_w
#         self.ll_corner = quat_rotate(root_quat, torch.stack([min_x, max_y, y], dim=-1)) + self.asset.data.root_pos_w
#         self.center = torch.stack([self.ul_corner, self.ur_corner, self.lr_corner, self.ll_corner], dim=1).mean(1)
#         self.center = 0.5 * (self.center + self.asset.data.root_pos_w)
#         self.center_pos_history[:, :, 1:] = self.center_pos_history[:, :, :-1]
#         self.center_pos_history[:, :, 0] = self.center
#         self.center_vel = torch.mean((self.center_pos_history[:, :, :-2] - self.center_pos_history[:, :, 2:]) / (self.env.step_dt * 2), dim=2)

#     def compute(self) -> torch.Tensor:
#         linvel_diff = self.center_vel - self.command_manager.command_linvel_w
#         linvel_error = linvel_diff[:, :2].square().sum(-1, keepdim=True)
#         return torch.exp(- linvel_error / 0.25)

#     def debug_draw(self):
#         # draw
#         self.env.debug_draw.vector(self.ul_corner, self.ur_corner - self.ul_corner, color=(1.0, 0., 0., 1.))
#         self.env.debug_draw.vector(self.ur_corner, self.lr_corner - self.ur_corner, color=(1.0, 0., 0., 1.))
#         self.env.debug_draw.vector(self.lr_corner, self.ll_corner - self.lr_corner, color=(1.0, 0., 0., 1.))
#         self.env.debug_draw.vector(self.ll_corner, self.ul_corner - self.ll_corner, color=(1.0, 0., 0., 1.))
#         self.env.debug_draw.vector(self.center, self.center_vel, color=(1.0, 0., 0., 1.))


class max_feet_height(Reward):
    def __init__(
        self,
        env,
        body_names: str,
        target_height: float,
        weight: float,
        enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.target_height = target_height

        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self.asset_body_ids, self.asset_body_names = self.asset.find_bodies(body_names)

        self.in_contact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.impact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.detach = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.has_impact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.max_height = torch.zeros(self.num_envs, len(self.body_ids), device=self.device)
        self.impact_point = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)
        self.detach_point = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)

    def reset(self, env_ids):
        self.has_impact[env_ids] = False
    
    def update(self):
        contact_force = self.contact_sensor.data.net_forces_w_history[:, :, self.body_ids]
        feet_pos_w = self.asset.data.body_pos_w[:, self.asset_body_ids]
        in_contact = (contact_force.norm(dim=-1) > 0.01).any(dim=1)
        self.impact[:] = (~self.in_contact) & in_contact
        self.detach[:] = self.in_contact & (~in_contact)
        self.in_contact[:] = in_contact
        self.has_impact.logical_or_(self.impact)
        self.impact_point[self.impact] = feet_pos_w[self.impact]
        self.detach_point[self.detach] = feet_pos_w[self.detach]
        self.max_height[:] = torch.where(
            self.detach,
            feet_pos_w[:, :, 2],
            torch.maximum(self.max_height, feet_pos_w[:, :, 2])
        )

    def compute(self) -> torch.Tensor:
        reference_height = torch.maximum(self.impact_point[:, :, 2], self.detach_point[:, :, 2])
        max_height = self.max_height - reference_height
        r = (self.impact * (max_height / self.target_height).clamp_max(1.0)).sum(dim=1, keepdim=True)
        is_standing = self.env.command_manager.is_standing_env.squeeze(1)
        r[~is_standing] -= r[~is_standing].mean()
        r[is_standing] = 0
        return r

    def debug_draw(self):
        feet_pos_w = self.asset.data.body_pos_w[:, self.asset_body_ids]
        self.env.debug_draw.point(
            feet_pos_w[self.impact],
            color=(1.0, 0., 0., 1.),
            size=30,
        )

    # def _reward_feet_max_height_for_this_air(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.

    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     from_air_to_contact = torch.logical_and(contact_filt, ~self.last_contacts_filt)

    #     self.last_contacts = contact
    #     self.last_contacts_filt = contact_filt

    #     self.feet_air_max_height = torch.max(self.feet_air_max_height, self._rigid_body_pos[:, self.feet_indices, 2])
        
    #     rew_feet_max_height = torch.sum((torch.clamp_min(self.cfg.rewards.desired_feet_max_height_for_this_air - self.feet_air_max_height, 0)) * from_air_to_contact, dim=1) # reward only on first contact with the ground
    #     self.feet_air_max_height *= ~contact_filt
    #     return rew_feet_max_height



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
        r = torch.where(
            self.feet_height_map > - 0.03,
            0,
            - self.feet_height_map.abs().sqrt()
        ).mean((1, 2)).unsqueeze(1)
        return r  * (~is_standing)


class step_up_needed(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_height_map: torch.Tensor = self.asset.data.feet_height_map
    
    def compute(self) -> torch.Tensor:
        is_standing = self.env.command_manager.is_standing_env
        cnt = (self.feet_height_map < - 0.03).float().mean((1, 2)).unsqueeze(1)
        return cnt  * (~is_standing)


from ..observations import _initialize_warp_meshes, raycast_mesh

class step_lift(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.mesh = _initialize_warp_meshes("/World/ground", "cuda")
        self.command_manager: Command2 = self.env.command_manager
        self.feet_ids = self.asset.find_bodies(".*foot")[0]

    def compute(self) -> torch.Tensor:
        ray_dir = self.command_manager.command_linvel_w
        ray_start_w = self.asset.data.body_pos_w[:, self.feet_ids]
        _, distance, normal, _ = raycast_mesh(
            ray_start_w,
            ray_dir,
            max_dist=0.2,
            mesh=self.mesh,
            return_distance=True,
            return_normal=True
        )
        distance = distance.nan_to_num(nan=1.0, posinf=1.0)
        hit_stair = (distance < 0.05) & (normal[:, :, 2].abs() < 0.05)
        feet_vel_z = self.asset.data.body_lin_vel_w[:, self.feet_ids, 2]
        r = hit_stair * feet_vel_z.clamp_min(0.0)
        return r.max(1, True).values


class com_linvel(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]

        with torch.device(self.device):
            self.com_pos_w = torch.zeros(self.num_envs, 3)
            self.com_pos_w_prev = torch.zeros(self.num_envs, 3)
            self.com_linvel_w = torch.zeros(self.num_envs, 3)
            self.coms = self.asset.root_physx_view.get_coms()[:, :, :3].to(self.device)
            self.masses = self.asset.root_physx_view.get_masses().to(self.device)
            self.masses = self.masses / self.masses.sum(1, True)

    def update(self):
        self.com_pos_w_prev[:] = self.com_pos_w
        com_pos_w = (self.asset.data.body_pos_w + quat_rotate(self.asset.data.body_quat_w, self.coms))
        self.com_pos_w[:] = (com_pos_w * self.masses.unsqueeze(-1)).sum(1)
        self.com_linvel_w[:] = (self.com_pos_w - self.com_pos_w_prev) / self.env.step_dt
    
    def compute(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, device=self.device)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.com_pos_w,
            self.com_linvel_w,
            color=(1.0, 0.1, 0.7, 1.)
        )


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
        height = (self.asset.data.feet_height - self.asset.data.feet_pos_b[:, :, 2]).max(1, keepdim=True)[0]
        return (height - target_height).clamp_max(0.)


class quadruped_stand_always(Reward):
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

        return cost


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

        is_standing = self.env.command_manager.is_standing_env.squeeze(1)
        cost[~is_standing] = 0
        cost[is_standing] -= cost[is_standing].mean()
        return cost


class quadruped_stand_feet_contact_force(Reward):
    # expecting the foot to contact the ground firmly but not with too much force

    def __init__(self, env, weight, body_names, enabled=True, force_range=(10., 80.)):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.force_range = force_range
    
        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]

        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
    
    def compute(self):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.body_ids]
        lower_bound, upper_bound = self.force_range
        force_penalty = (contact_forces < lower_bound).float() + (contact_forces > upper_bound).float()
        # force_penalty = (contact_forces - contact_forces.clamp(lower_bound, upper_bound)).abs()
        total_penalty = torch.sum(force_penalty, dim=(1, 2)).reshape(self.num_envs, 1)
        is_standing = self.env.command_manager.is_standing_env.squeeze(1)
        total_penalty[~is_standing] = 0
        total_penalty[is_standing] -= total_penalty[is_standing].mean(0)
        return - total_penalty
    
    def debug_draw(self):
        # draw contact forces on each of the body (orange)
        contact_forces = self.contact_sensor.data.net_forces_w_history.mean(1)[:, self.body_ids]
        body_pos_w = self.asset.data.body_pos_w[:, self.articulation_body_ids]
        is_standing = self.env.command_manager.is_standing_env.squeeze(1)
        self.env.debug_draw.vector(
            body_pos_w[is_standing].view(-1, 3),
            contact_forces[is_standing].view(-1, 3),
            # orange
            color=(1., 0.1, 0.1, 1.),
            size=5.0
        )
    
class is_standing_env(Reward):
    def __init__(self, env, weight: float, enabled: bool = False, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
    
    def compute(self) -> torch.Tensor:
        return self.env.command_manager.is_standing_env.reshape(self.num_envs, 1)

class stance_width(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf), target_width=0.15):
        """penalize stance width smaller than target_width"""
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_width = target_width
    
    def compute(self) -> torch.Tensor:
        front_width = self.asset.data.feet_pos_b[:, [0, 1], 0].diff(dim=1).norm(dim=1, keepdim=True)
        back_width = self.asset.data.feet_pos_b[:, [2, 3], 0].diff(dim=1).norm(dim=1, keepdim=True)
        width = torch.cat([front_width, back_width], dim=1)
        return -(self.target_width - width).clamp_min(0.).sum(1, keepdim=True)
    

class stand_up_height(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        return (
            self.asset.data.root_pos_w[:, 2].clamp(max=0.7).square()
        ).unsqueeze(-1)

class stand_up_orientation(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        return (
            -self.asset.data.projected_gravity_b[:, 0].clamp_min_(-0.8)
        ).unsqueeze(-1)


class joint_vel_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _ = self.asset.find_joints(joint_names)
        self.joint_vel = torch.zeros(self.num_envs, 2, len(self.joint_ids), device=self.device)
    
    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        joint_vel = self.joint_vel.mean(1)
        return - joint_vel.square().sum(1, True)


class impedance_pos(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: Impedance = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = (self.command_manager.command_pos_w - self.asset.data.root_pos_w)
        r = torch.exp(- diff.norm(dim=-1, keepdim=True) / 0.25)
        return r
    
from ..commands.impedance import rpy_from_quat, ImpedanceBase

class impedance_vel(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: Impedance = self.env.command_manager

        self.decay = torch.tensor([0.0, 0.5], device=self.device)
        self.cnt = torch.zeros(self.num_envs, len(self.decay), device=self.device)
        self.lin_vel_sum = torch.zeros(self.num_envs, 3, len(self.decay), device=self.device)

    def reset(self, env_ids):
        self.lin_vel_sum[env_ids] = 0.
        self.cnt[env_ids] = 0

    def update(self):
        self.cnt.mul_(self.decay).add_(1.)
        self.lin_vel_sum.mul_(self.decay).add_(self.asset.data.root_lin_vel_w.unsqueeze(-1))

    def compute(self) -> torch.Tensor:
        lin_vel_w = self.lin_vel_sum / self.cnt.unsqueeze(1)
        command_lin_vel_w = self.command_manager.command_linvel_w.unsqueeze(-1)
        command_speed = self.command_manager.command_speed.reshape(self.num_envs, 1)
        diff = (lin_vel_w - command_lin_vel_w)
        error_l2 = diff[:, :2].square().sum(dim=1, keepdim=True)
        error_l2 = error_l2.min(-1)[0]
        p = torch.exp(- error_l2 / 0.25)
        r = p - 0.5 * error_l2
        self.asset.data.linvel_exp = p
        return r

class impedance_acc(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: ImpedanceBase = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        lin_acc_w = self.asset.data.body_acc_w[:, 0, :3]
        command_lin_acc_w = self.command_manager.des_lin_acc_w
        error_l2 = (lin_acc_w - command_lin_acc_w)[:, :2].square().sum(dim=1, keepdim=True)
        r = torch.exp(- error_l2 / 2.)
        return r

class impedance_yaw_pos(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: Impedance = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        diff = wrap_to_pi(self.command_manager.command_yaw_w - self.asset.data.heading_w.unsqueeze(1))
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = torch.exp(- error_l2 / 0.25) - error_l2
        return r


class impedance_pitch_pos(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: ImpedanceBase = self.env.command_manager

    def compute(self):
        rpy = rpy_from_quat(self.asset.data.root_quat_w)
        diff = self.command_manager.setpoint_rpy_w[:, 1].unsqueeze(1) - rpy[:, 1].unsqueeze(1)
        error_l1 = diff.abs()
        r = torch.exp(- error_l1 / 0.4)
        return r


class impedance_yaw_vel(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: Impedance = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        command_angvel = self.command_manager.command_angvel.reshape(self.num_envs, 1)
        diff = (command_angvel - self.asset.data.root_ang_vel_b[:, 2:3])
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = torch.exp(- error_l2 / 0.25) - 0.5 * error_l2
        return r


class feet_swing_height(Reward):
    def __init__(self, env, target_height: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_height = target_height
        self.feet_ids = self.asset.find_bodies(".*foot.*")[0]

    def update(self):
        self.feet_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.asset.data.body_pos_w[:, self.feet_ids] - self.asset.data.root_pos_w.unsqueeze(1)
        )
        self.feet_vel_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.asset.data.body_lin_vel_w[:, self.feet_ids]
        )

    def compute(self) -> torch.Tensor:
        hight_error = (self.asset.data.feet_height - self.target_height).abs()
        lateral_speed = (
            self.feet_vel_b[:, :, :2].square().sum(-1)
            + self.asset.data.body_ang_vel_w[:, self.feet_ids, 2].square()
        )
        return - (hight_error * lateral_speed).sum(1, keepdim=True)


class head_clearance(Reward):
    def __init__(self, env, target_height: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.target_height = target_height
        self.asset: Articulation = self.env.scene["robot"]
        self.head_height: torch.Tensor = self.asset.data.head_height

    def compute(self) -> torch.Tensor:
        return (self.head_height - self.target_height).clamp_max(0.)


class com_support(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_ids = self.asset.find_bodies(".*foot")[0]

    def compute(self) -> torch.Tensor:
        feet_center = self.asset.data.body_pos_w[:, self.feet_ids].mean(1)
        error = (self.asset.data.root_pos_w[:, :2] - feet_center[:, :2]).square().sum(1, keepdim=True)
        return torch.exp(- error / 0.2)


class com_linvel_exp(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        with torch.device(self.device):
            self.com_pos_w = torch.zeros(self.num_envs, 3)
            self.com_pos_w_last = torch.zeros(self.num_envs, 3)
            self.com_vel_w = torch.zeros(self.num_envs, 3)
        self.feet_ids = self.asset.find_bodies(".*foot")[0]

    def update(self):
        self.com_pos_w_last[:] = self.com_pos_w
        self.com_pos_w[:] = self.asset.data.body_pos_w[:, self.feet_ids].mean(1)
        self.com_vel_w[:] = (self.com_pos_w - self.com_pos_w_last) / self.env.step_dt
        self.com_vel_w[:, 2] = 0.

    def compute(self) -> torch.Tensor:
        com_linvel_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.com_vel_w
        )
        error = (com_linvel_b[:, :2] - self.env.command_manager.command_linvel[:, :2]).square().sum(1, keepdim=True)
        return torch.exp(- error / 0.25)

class joint_limits(Reward):
    def __init__(self, env, joint_names: str, offset: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_limits = self.asset.data.joint_limits[:, self.joint_ids].clone()
        self.joint_limits_max = self.joint_limits[:, :, 1] - offset
        self.joint_limits_min = self.joint_limits[:, :, 0] + offset

    def compute(self) -> torch.Tensor:
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (joint_pos - self.joint_limits_min).clamp_max(0.)
        violation_max = (self.joint_limits_max - joint_pos).clamp_max(0.)
        return (violation_min + violation_max).sum(1, keepdim=True)


from active_adaptation.assets.quadruped import Quadruped
class step_vel(Reward):
    def __init__(self, env, weight, enabled = True, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Quadruped = self.env.scene["robot"]
        self.prev_root_pos_w = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.last_impact_time = torch.zeros(self.num_envs, 4, 1, device=self.device)
        self.command_manager: Impedance = self.env.command_manager

    def reset(self, env_ids):
        self.prev_root_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids].unsqueeze(1)
        self.last_impact_time[env_ids] = 0.

    def update(self):
        root_pos_w = self.asset.data.root_pos_w.unsqueeze(1)
        t = self.env.episode_length_buf.reshape(-1, 1, 1) * self.env.step_dt
        self.step_vel = (root_pos_w - self.prev_root_pos_w) / (t - self.last_impact_time)
        self.step_vel = torch.nan_to_num(self.step_vel, nan=0., posinf=0., neginf=0.)
        self.prev_root_pos_w = torch.where(self.asset.impact.unsqueeze(-1), root_pos_w, self.prev_root_pos_w)
        self.last_impact_time = torch.where(self.asset.impact.unsqueeze(-1), t, self.last_impact_time)

    def compute(self):
        error_l1 = (self.command_manager.command_linvel_w[:, :2].unsqueeze(1) - self.step_vel[:, :, :2]).norm(dim=-1)
        r = torch.exp(- error_l1) * self.asset.impact
        return r.sum(1, keepdim=True)


def normalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def shaped_error(error: torch.Tensor):
    """
    Shaped error for reward shaping.
    """
    return torch.maximum(error.abs(), error.square())