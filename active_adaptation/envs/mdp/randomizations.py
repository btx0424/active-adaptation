import torch
import numpy as np
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.actuators import DCMotor, ImplicitActuator
from omni.isaac.orbit.sensors import RayCaster
import omni.isaac.orbit.utils.string as string_utils
from typing import Union
import logging
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify
from typing import Dict, Tuple

quat_rotate_inverse = batchify(quat_rotate_inverse)

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
        
        self.default_stiffness = self.motors.default_stiffness = self.motors.stiffness.clone()
        self.default_damping = self.motors.default_damping = self.motors.damping.clone()
        self.default_strength = torch.ones_like(self.default_stiffness)
        if isinstance(self.motors, DCMotor):
            if isinstance(self.motors._saturation_effort, float):
                self.default_strength.fill_(self.motors._saturation_effort)
                self.motors._saturation_effort = self.default_strength.clone()
        elif isinstance(self.motors, ImplicitActuator):
            self.default_strength[:] = self.motors.effort_limit
        
    def reset(self, env_ids: torch.Tensor=slice(None)):
        stiffness, _ = random_scale(
            self.default_stiffness[env_ids], *self.stiffness_range, self.homogeneous
        )
        damping, _ = random_scale(
            self.default_damping[env_ids], *self.damping_range, self.homogeneous
        )
        strength, _ = random_scale(
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


class random_motor_failure(Randomization):
    def __init__(
        self,
        env,
        actuator_name: str,
        joint_names: str,
        failure_prob: float = 0.2,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.motors: DCMotor = self.asset.actuators[actuator_name]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names, self.motors.joint_names)
        self.joint_ids = torch.as_tensor(self.joint_ids, device=self.device)
        self.failure_prob = failure_prob
        assert not hasattr(self.motors, "motor_failure")
        self.motor_failure = self.motors.motor_failure = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        logging.info(f"Randomly disable one joint from {self.joint_names} with prob. {self.failure_prob}.")
        self.failure_prob = failure_prob

        # hard-coded
        self._body_ids = self.asset.find_bodies(".*calf.*")[0]
        
    def reset(self, env_ids: torch.Tensor):
        self.motor_failure[env_ids] = -1.0
        with torch.device(self.device):
            env_ids = env_ids[torch.rand(len(env_ids)) < self.failure_prob]
            i = torch.randint(0, len(self.joint_ids), env_ids.shape)
            joint_id = self.joint_ids[i]
        self.motors.stiffness[env_ids, joint_id] = 0.02
        self.motors.damping[env_ids, joint_id] = 0.02
        self.motor_failure[env_ids, i] = 1.0

    def debug_draw(self):
        x = self.asset.data.body_pos_w[:, self._body_ids]
        x = x[self.motor_failure > 0.]
        self.env.debug_draw.point(x, color=(0.1, 1.0, 0.1, 0.8), size=20)


class perturb_body_materials(Randomization):
    def __init__(
        self,
        env,
        body_names,
        static_friction_range = (0.6, 1.0),
        dynamic_friction_range = (0.6, 1.0),
        restitution_range=(0.0, 0.2),
        homogeneous: bool=False
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.homogeneous = homogeneous
        
        self.default_materials = (
            self.asset.root_physx_view.get_material_properties()
        )
        
        num_shapes_per_body = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        cumsum = np.cumsum([0,] + num_shapes_per_body)
        self.shape_ids = torch.cat([
            torch.arange(cumsum[i], cumsum[i+1]) 
            for i in self.body_ids
        ])

    def startup(self):
        logging.info(f"Randomize body materials of {self.body_names} upon startup.")
        
        materials = self.default_materials.clone()
        if self.homogeneous:
            shape = (self.num_envs, 1)
        else:
            shape = (self.num_envs, len(self.shape_ids))
        materials[:, self.shape_ids, 0] = sample_uniform(shape, *self.static_friction_range)
        materials[:, self.shape_ids, 1] = sample_uniform(shape, *self.dynamic_friction_range)
        materials[:, self.shape_ids, 2] = sample_uniform(shape, *self.restitution_range)

        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_material_properties(materials.flatten(), indices)
        self.asset.data.body_materials = materials.to(self.device)


class perturb_body_mass(Randomization):
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        self.body_ids, self.body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        self.mass_ranges = torch.tensor(values)
        print(self.body_names)

    def startup(self):
        logging.info(f"Randomize body masses of {self.body_names} upon startup.")
        masses = self.asset.root_physx_view.get_masses().clone()
        inertias = self.asset.root_physx_view.get_inertias().clone()
        print(f"Default masses: {masses[0]}")
        scale = uniform(
            self.mass_ranges[:, 0].expand_as(masses[:, self.body_ids]),
            self.mass_ranges[:, 1].expand_as(masses[:, self.body_ids])
        )
        masses[:, self.body_ids] *= scale
        inertias[:, self.body_ids] *= scale.unsqueeze(-1)
        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_masses(masses, indices)
        self.asset.root_physx_view.set_inertias(inertias, indices)
        assert torch.allclose(self.asset.root_physx_view.get_masses(), masses)


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


class reset_joint_states_uniform(Randomization):
    def __init__(self, env, pos_ranges: Dict[str, tuple], rel: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.rel = rel
        
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
        if self.rel:
            init_pos += self.default_joint_pos[env_ids]
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


class push(Randomization):
    def __init__(self, env, body_names, force_range = (0.2, 0.9), min_interval=100):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.num_bodies = len(self.body_indices)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.force_range = force_range
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

            push_forces = torch.zeros_like(self.forces)
            push_forces[:, :, 0].uniform_(*self.force_range)
            push_forces[:, :, 1].uniform_(*self.force_range)
            self.forces = torch.where(i, push_forces * self.default_mass_total, self.forces * 0.8)
        self.asset.set_external_force_and_torque(self.forces, self.torques, body_ids=self.body_indices)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.body_indices],
            self.forces / self.default_mass_total,
            color=(1., 0.8, 1., 1.)
        )


class stumble(Randomization):
    def __init__(
        self, 
        env,
        body_names: str,
        stumble_height: float=0.05,
        friction_range=(0.0, 0.2),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.num_feet = len(self.body_ids)

        self.body_ids = torch.as_tensor(self.body_ids, device=self.device)
        self.stumble_height = stumble_height
        self.friction_range = friction_range
        self.friction_coef = torch.zeros(self.num_envs, 1, 1, device=self.device)
    
    def startup(self):
        self.feet_height: torch.Tensor = self.asset.data.feet_height

    def reset(self, env_ids: torch.Tensor):
        self.friction_coef[env_ids] = sample_uniform((len(env_ids), 1, 1), 0.0, 0.2, self.device)

    def step(self, substep):
        # feet_height = self.asset.data.feet_height_map.mean(-1).reshape(-1)
        feet_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.body_ids]
        feet_quat_w = self.asset.data.body_quat_w[:, self.body_ids]
        stumble_prob = ((self.stumble_height - self.feet_height) / self.stumble_height).clamp(0., 1.)
        self.forces_w = - self.friction_coef * feet_lin_vel_w / self.env.physics_dt
        self.forces_w[..., 2] = 0.
        forces = torch.where(
            (torch.rand_like(self.feet_height) < stumble_prob).unsqueeze(-1),
            quat_rotate_inverse(feet_quat_w, self.forces_w),
            torch.zeros(self.num_envs, self.num_feet, 3, device=self.env.device)
        )
        torques = torch.zeros_like(forces)
        self.asset.set_external_force_and_torque(forces, torques, body_ids=self.body_ids)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.body_ids],
            self.forces_w * self.env.physics_dt,
            color=(1., 0.6, 0., 1.)
        )

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

def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)

def log_uniform(low: torch.Tensor, high: torch.Tensor):
    return uniform(low.log(), high.log()).exp()

def angle_mix(a: torch.Tensor, b: torch.Tensor, weight: float=0.1):
    d = a - b
    d[d > torch.pi] -= 2 * torch.pi
    d[d < -torch.pi] += 2 * torch.pi
    return a - d * weight