import torch
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.actuators import DCMotor, ImplicitActuator
from typing import Union
import logging
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from typing import Dict, Tuple

class Randomization:
    def __init__(self, env):
        self.env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device
    
    def startup(self):
        pass
    
    def reset(self, env_ids: torch.Tensor):
        pass
    
    def step(self, substep):
        pass

    def debug_draw(self):
        pass


class motor_params(Randomization):
    def __init__(
        self, 
        env,
        actuator_name,
        stiffness_range = (1.0, 1.0),
        damping_range = (1.0, 1.0),
        strength_range = (1.0, 1.0),
        homogeneous: bool = False,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.actuator_name = actuator_name
        self.stiffness_range = stiffness_range
        self.damping_range = damping_range
        self.strength_range = strength_range
        self.homogeneous = homogeneous
        self.motors: Union[DCMotor, ImplicitActuator] = self.asset.actuators[self.actuator_name]
        
        self.default_stiffness = self.motors.stiffness.clone()
        self.default_damping = self.motors.damping.clone()
        if isinstance(self.motors, DCMotor):
            self.default_strength = torch.full_like(self.default_stiffness, self.motors._saturation_effort)
            self.motors._saturation_effort = self.default_strength.clone()
        elif isinstance(self.motors, ImplicitActuator):
            self.default_strength = self.motors.effort_limit

        if self.homogeneous:
            self.randomized_stiffness = torch.ones_like(self.default_damping.mean(-1, True))
            self.randomized_damping = torch.ones_like(self.default_damping.mean(-1, True))
            self.randomized_strength = torch.ones_like(self.default_strength.mean(-1, True))
        else:
            self.randomized_stiffness = torch.ones_like(self.default_stiffness)
            self.randomized_damping = torch.ones_like(self.default_damping)
            self.randomized_strength = torch.ones_like(self.default_strength)
        
        setattr(self.motors, "_randomized_stiffness", self.randomized_stiffness)
        setattr(self.motors, "_randomized_damping", self.randomized_damping)
        setattr(self.motors, "_randomized_strength", self.randomized_strength)
        
    def reset(self, env_ids: torch.Tensor=slice(None)):
        stiffness, self.randomized_stiffness[env_ids] = random_scale(
            self.default_stiffness[env_ids], *self.stiffness_range, self.homogeneous
        )
        damping, self.randomized_damping[env_ids] = random_scale(
            self.default_damping[env_ids], *self.damping_range, self.homogeneous
        )
        strength, self.randomized_strength[env_ids] = random_scale(
            self.default_strength[env_ids], *self.strength_range, self.homogeneous
        )
        self.motors.stiffness[env_ids] = stiffness
        self.motors.damping[env_ids] = damping

        if isinstance(self.motors, DCMotor):
            self.motors._saturation_effort[env_ids] = strength
        elif isinstance(self.motors, ImplicitActuator):
            self.asset.write_joint_stiffness_to_sim(self.motors.stiffness, self.motors.joint_indices)
            self.asset.write_joint_damping_to_sim(self.motors.damping, self.motors.joint_indices)
            self.motors.effort_limit[env_ids] = strength


class MotorFailure(Randomization):
    def __init__(
        self, 
        env,
        joint_indices,
        failure_prob: float = 0.2,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_indices = torch.as_tensor(joint_indices, device=self.env.device)
        self.failure_prob = failure_prob
    
    def startup(self):
        self.motors: DCMotor = self.asset.actuators["base_legs"]
        self.motor_failure = torch.zeros_like(self.motors.stiffness)
        
    def reset(self, env_ids: torch.Tensor):
        self.motor_failure[env_ids] = 0.0
        with torch.device(self.env.device):
            env_ids = env_ids[torch.rand(len(env_ids)) < self.failure_prob]
            joint_id = self.joint_indices[torch.randint(0, len(self.joint_indices), env_ids.shape)]
        self.motors.stiffness[env_ids, joint_id] = 0.1
        self.motors.damping[env_ids, joint_id] = 0.1
        self.motor_failure[env_ids, joint_id] = 1.0


class perturb_body_materials(Randomization):
    def __init__(
        self,
        env,
        body_names,
        static_friction_range = (0.6, 1.0),
        dynamic_friction_range = (0.6, 1.0),
        restitution_range=(0.0, 0.0),
        homogeneous: bool=True
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.homogeneous = homogeneous

        self.default_materials = (
            self.asset.body_physx_view.get_material_properties()
            .reshape(self.num_envs, self.asset.num_bodies, -1, 3)
        )

    def startup(self):
        logging.info(f"Randomize body materials of {self.body_names} upon startup.")
        
        materials = self.default_materials.clone()
        if self.homogeneous:
            shape = (self.num_envs, 1, 1)
        else:
            shape = (self.num_envs, len(self.body_ids), 1)
        materials[:, self.body_ids, :, 0] = sample_uniform(shape, *self.static_friction_range)
        materials[:, self.body_ids, :, 1] = sample_uniform(shape, *self.dynamic_friction_range)
        materials[:, self.body_ids, :, 2] = sample_uniform(shape, *self.restitution_range)

        indices = torch.arange(self.asset.body_physx_view.count)
        self.asset.body_physx_view.set_material_properties(materials.flatten(), indices)
        self.asset.data.body_materials = materials.mean(2).to(self.device)


class perturb_body_mass(Randomization):
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_masses = (
            self.asset.body_physx_view
            .get_masses()
            .reshape(self.num_envs, self.asset.num_bodies)[0]
        )
        self.mass_ranges = torch.ones(self.asset.num_bodies, 2)

        for body_name_expr, (low, high) in perturb_ranges.items():
            body_ids, body_names = self.asset.find_bodies(body_name_expr)
            print(f"Default mass of {body_names}: \n"
                  f"{[round(i, 2) for i in self.default_masses[body_ids].tolist()]}")
            self.mass_ranges[body_ids, 0] = low
            self.mass_ranges[body_ids, 1] = high

    def startup(self):
        logging.info("Randomize body masses upon startup.")
        shape = (self.env.num_envs, self.asset.num_bodies)
        masses, _ = random_scale(
            self.default_masses.expand(self.env.num_envs, self.asset.num_bodies), 
            self.mass_ranges[:, 0], 
            self.mass_ranges[:, 1], 
        )
        indices = torch.arange(self.asset.body_physx_view.count)
        self.asset.body_physx_view.set_masses(masses, indices)


class BodyInertias(Randomization):
    def __init__(
        self,
        env,
        inertia_range=(0.7, 1.3),
        body_indices=None
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.inertia_range = inertia_range

        self.default_inertias: torch.Tensor = None
        self.body_inertias: torch.Tensor = None
        if body_indices is None:
            body_indices = slice(None)
        self.body_indices = body_indices

    def startup(self):
        logging.info("Randomize body inertias upon starup.")
        shape = (self.env.num_envs, self.asset.num_bodies)
        default_inertias_all = self.asset.body_physx_view.get_inertias().reshape(*shape, -1).clone()
        default_inertias = default_inertias_all[:, self.body_indices]
        randomized_inertias, _ = random_scale(default_inertias, *self.inertia_range)
        
        bodies_per_env = self.asset.body_physx_view.count // self.env.num_envs        
        indices = self.body_indices.repeat(self.env.num_envs, 1)
        indices += torch.arange(self.env.num_envs).unsqueeze(1) * bodies_per_env
        default_inertias_all[:, self.body_indices] = randomized_inertias
        self.asset.body_physx_view.set_inertias(default_inertias_all.flatten(), indices.flatten())
        
        self.default_inertias = default_inertias.to(self.env.device)
        self.randomized_inertias = randomized_inertias.to(self.env.device)

class JointFriction(Randomization):
    def __init__(
        self,
        env,
        friction_range=(0.01, 0.1),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.friction_range = friction_range

    def startup(self):
        logging.info("Randomize joint frictions upon starup.")
        frictions = torch.zeros(self.env.num_envs, 1)
        frictions.uniform_(*self.friction_range)
        self.asset.root_physx_view.set_dof_friction_coefficients(
            frictions.expand(-1, self.asset.num_joints), 
            indices=self.asset._ALL_INDICES.cpu()
        )


class BodyComs(Randomization):
    def __init__(self, env, com_range=(-0.1, 0.1), body_indices=None):
        super().__init__(env)
        self.com_range = com_range
        self.asset: Articulation = self.env.scene["robot"]
        if body_indices is None:
            body_indices = slice(None)
        self.body_indices = body_indices
    
    def startup(self):
        logging.info(f"Randomize body coms {self.body_indices} upon starup.")
        shape = (self.env.num_envs, self.asset.num_bodies)
        coms: torch.Tensor = self.asset.body_physx_view.get_coms().clone().reshape(*shape, -1)
        offset = torch.zeros_like(coms)
        offset[:, :, :3].uniform_(*self.com_range)
        coms = coms + offset

        bodies_per_env = self.asset.body_physx_view.count // self.env.num_envs        
        indices = self.body_indices.repeat(self.env.num_envs, 1)
        indices += torch.arange(self.env.num_envs).unsqueeze(1) * bodies_per_env
        self.asset.body_physx_view.set_coms(coms.flatten(), indices.flatten())

        self.randomized_coms = coms[:, self.body_indices, :3].to(self.env.device)


class reset_joint_states_uniform(Randomization):
    def __init__(self, env, pos_ranges: Dict[str, tuple]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.joint_ids = []
        self.pos_ranges = []
        for joint_name, (low, high) in pos_ranges.items():
            joint_ids, joint_names = self.asset.find_joints(joint_name)
            self.joint_ids.extend(joint_ids)
            self.pos_ranges.append(torch.tensor([low, high], device=self.env.device).expand(len(joint_ids), 2))
            print(f"Reset {joint_names} to U({low}, {high})")
        self.pos_ranges = torch.cat(self.pos_ranges, 0).unbind(1)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids]
    
    def reset(self, env_ids: torch.Tensor):
        shape = (len(env_ids), len(self.joint_ids))
        init_pos = sample_uniform(shape, *self.pos_ranges, self.env.device)
        init_vel = self.default_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(
            init_pos, init_vel, self.joint_ids, env_ids.unsqueeze(1)
        )


class reset_joint_states_scale(Randomization):
    def __init__(self, env, pos_scales: Dict[str, tuple]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.joint_ids = []
        self.pos_scales = []
        for joint_name, (low, high) in pos_scales.items():
            joint_ids, joint_names = self.asset.find_joints(joint_name)
            self.joint_ids.extend(joint_ids)
            self.pos_scales.append(torch.tensor([low, high], device=self.env.device).expand(len(joint_ids), 2))
            print(f"Reset {joint_names} to scales of U({low}, {high})")
        self.pos_scales = torch.cat(self.pos_scales, 0).unbind(1)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids]
    
    def reset(self, env_ids: torch.Tensor):
        init_pos = random_scale(
            self.default_joint_pos[env_ids], 
            *self.pos_scales, 
            self.env.device
        )[0]
        init_vel = self.default_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(
            init_pos, init_vel, self.joint_ids, env_ids.unsqueeze(1)
        )
        

import omni.isaac.orbit.utils.math as math_utils

class CommandManager:
    def __init__(self, env, speed_range=(0.5, 2.0)):
        self.env = env
        self.robot: Articulation = env.scene["robot"]
        self.device = env.device
        self.speed_range = speed_range

        with torch.device(env.device):
            # world frame
            self._target_yaw = torch.zeros(env.num_envs)
            self._command_stand = torch.zeros(env.num_envs, 1)
            self._command_linvel = torch.zeros(env.num_envs, 3)
            self._command_yaw = torch.zeros(env.num_envs)
            self._command_heading = torch.zeros(env.num_envs, 3)
            self._command_speed = torch.zeros(env.num_envs, 1)
        self.is_standing_env = self._command_stand

    def reset(self, env_ids: torch.Tensor):
        self.sample_commands(env_ids)

    def update(self, resample: torch.Tensor=None):
        if resample is not None:
            self.sample_commands(resample)
        heading_w = quat_rotate(
            self.robot.data.root_quat_w,
            torch.tensor([[1., 0., 0.]], device=self.device).expand(self.env.num_envs, 3)
        )
        yaw = torch.atan2(heading_w[:, 1], heading_w[:, 0])
        self._command_yaw[:] = self._target_yaw
        self._command_heading[:, 0] = self._command_yaw.cos()
        self._command_heading[:, 1] = self._command_yaw.sin()
        self._command_heading[:, 2] = 0.

    def sample_commands(self, env_ids: torch.Tensor):
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        stand = torch.rand(len(env_ids), device=self.device) < 0.2
        speed = torch.zeros(len(env_ids), device=self.device).uniform_(*self.speed_range)
        speed = speed * (~stand).float()
        
        self._command_stand[env_ids] = stand.float().unsqueeze(1)
        self._command_speed[env_ids] = speed.unsqueeze(1)
        self._command_linvel[env_ids, 0] = speed * a.cos()
        self._command_linvel[env_ids, 1] = speed * a.sin()
        
        yaw = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        self._target_yaw[env_ids] = yaw


class push(Randomization):
    def __init__(self, env, body_names, min_interval=100):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.min_interval = min_interval
        
        with torch.device(self.env.device):
            self.last_push = torch.zeros(self.env.num_envs, len(self.body_indices), 1)
            self.forces = torch.zeros(self.env.num_envs, len(self.body_indices), 3)
            self.torques = torch.zeros(self.env.num_envs, len(self.body_indices), 3)

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.
        self.last_push[env_ids] = 0.

    def step(self, substep):
        if substep == 0:
            t = self.env.episode_length_buf.view(self.env.num_envs, 1, 1)
            i = torch.rand(self.env.num_envs, len(self.body_indices), 1, device=self.env.device) < 0.02
            i = i & ((t - self.last_push) > self.min_interval)
            self.last_push = torch.where(i, t, self.last_push)

            push_forces = torch.rand_like(self.forces) * self.default_mass_total
            push_forces[:, :, 2] = 0.
            self.forces = torch.where(i, push_forces, self.forces * 0.8)
        self.asset.set_external_force_and_torque(self.forces, self.torques, body_ids=self.body_indices)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.body_indices],
            self.forces / self.default_mass_total,
            color=(1., 0.8, 1., 1.)
        )

class CommandManager1:
    def __init__(
        self, 
        env, 
        speed_range=(0.5, 2.0),
        angvel_range=(-1.0, 1.0),
        stand_prob=0.1
    ):
        self.env = env
        self.robot: Articulation = env.scene["robot"]
        self.device = env.device
        self.speed_range = speed_range
        self.angvel_range = angvel_range
        self.sand_prob = stand_prob

        with torch.device(env.device):
            self._target_yaw = torch.zeros(env.num_envs)
            self._command_stand = torch.zeros(env.num_envs, 1)
            self._command_linvel = torch.zeros(env.num_envs, 3)
            self._command_angvel_yaw = torch.zeros(env.num_envs)
            self._command_heading = torch.zeros(env.num_envs, 3)
            self._command_speed = torch.zeros(env.num_envs, 1)
            self.command = torch.zeros(env.num_envs, 3)
            self.command_prev = torch.zeros(env.num_envs, 3)
        self.is_standing_env = self._command_stand

    def reset(self, env_ids: torch.Tensor):
        self.sample_commands(env_ids)
        self.command_prev[env_ids] = self.command[env_ids]

    def update(self, resample: torch.Tensor=None):
        if resample is not None and len(resample) > 0:
            self.sample_commands(resample)
        
        yaw_diff = self._target_yaw - self.robot.data.heading_w
        self._command_angvel_yaw[:] = math_utils.wrap_to_pi(yaw_diff).clamp(*self.angvel_range)

        self.command_prev[:] = self.command
        self.command[:, :2] = self._command_linvel[:, :2]
        self.command[:, 2] = self._command_angvel_yaw

    def sample_commands(self, env_ids: torch.Tensor):
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        stand = torch.rand(len(env_ids), device=self.device) < self.sand_prob
        speed = torch.zeros(len(env_ids), device=self.device).uniform_(*self.speed_range)
        speed = speed * (~stand).float()
        
        self._command_stand[env_ids] = stand.float().unsqueeze(1)
        self._command_speed[env_ids] = speed.unsqueeze(1)
        self._command_linvel[env_ids, 0] = speed * a.cos()
        self._command_linvel[env_ids, 1] = speed * a.sin()
        
        yaw = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        self._target_yaw[env_ids] = yaw
        self._command_heading[env_ids, 0] = yaw.cos()
        self._command_heading[env_ids, 1] = yaw.sin()


def random_scale(x: torch.Tensor, low: float, high: float, homogeneous: bool=False):
    if homogeneous:
        u = torch.rand(*x.shape[:1], 1, device=x.device)
    else:
        u = torch.rand_like(x)
    return x * (u * (high - low) + low), u

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def angle_mix(a: torch.Tensor, b: torch.Tensor, weight: float=0.1):
    d = a - b
    d[d > torch.pi] -= 2 * torch.pi
    d[d < -torch.pi] += 2 * torch.pi
    return a - d * weight