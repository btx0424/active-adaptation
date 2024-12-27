import torch
import numpy as np
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.actuators import DCMotor, ImplicitActuator
from omni.isaac.lab.sensors import RayCaster
from active_adaptation.envs.actuator import HybridActuator
import omni.isaac.lab.utils.string as string_utils
from typing import Union
import logging
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify
from typing import Dict, Tuple, Union

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

    def update(self):
        pass

    def debug_draw(self):
        pass


RangeType = Tuple[float, float]
NestedRangeType = Union[RangeType, Dict[str, RangeType]]


class motor_params(Randomization):
    """
    2024.10.28
    - refactor to grouped randomization.


    Example usage in the config file:

      actuator_name: base_legs
        stiffness_range:  
          F[L,R]_hip:   [0.8, 1.2]
          F[L,R]_thigh: [0.8, 1.2]
          ...
        damping_range:
          F[L,R]_hip:   [0.8, 1.2]
          F[L,R]_thigh: [0.8, 1.2]
          ...
        scale_factor_range:
          .*_hip:   [0.8, 1.2]
          .*_thigh: [0.8, 1.2]
          .*_calf:  [0.8, 1.2]
    
    """
    def __init__(
        self, 
        env,
        actuator_name,
        stiffness_range: NestedRangeType = (1.0, 1.0),
        damping_range: NestedRangeType = (1.0, 1.0),
        scale_factor_range: NestedRangeType = (1.0, 1.0),
        armature_range: NestedRangeType = (0.0, 0.0),
        strength_range: NestedRangeType = (1.0, 1.0),
        **kwargs
    ):
        super().__init__(env)
        if len(kwargs) > 0:
            import warnings
            warnings.warn(f"Got unexpected keyword arguments: {kwargs}")
        
        self.asset: Articulation = self.env.scene["robot"]
        self.actuator_name = actuator_name
        self.stiffness_range = stiffness_range
        self.damping_range = damping_range
        self.strength_range = strength_range

        self.actuator: Union[DCMotor, ImplicitActuator] = self.asset.actuators[self.actuator_name]
        self.num_joints = len(self.actuator.joint_names)
        self.default_stiffness  = self.actuator.stiffness[0].clone()
        self.default_damping    = self.actuator.damping[0].clone()
        
        from omegaconf import ListConfig
        def parse(range: NestedRangeType, default: torch.Tensor):
            if isinstance(range, (tuple, list, ListConfig)):
                range = {".*": range}
            result = {}
            for key, value in range.items():
                ids, names = string_utils.resolve_matching_names(key, self.actuator.joint_names)
                result[key] = (ids, names, value, default[ids])
            return result

        self.stiffness_range    = parse(stiffness_range, self.default_stiffness)
        self.damping_range      = parse(damping_range, self.default_damping)
        self.scale_factor_range = parse(scale_factor_range, torch.ones(self.num_joints, device=self.device))
        self.armature_range     = parse(armature_range, torch.zeros(self.num_joints, device=self.device))
        
    def reset(self, env_ids: torch.Tensor=slice(None)):

        scale_factor = torch.ones(len(env_ids), self.num_joints, device=self.device)
        for key, (ids, names, value, default) in self.scale_factor_range.items():
            r = (value[1] - value[0]) * torch.rand(len(env_ids), 1, device=self.device) + value[0]
            scale_factor[:, ids] = default * r

        stiffness = self.default_stiffness.expand(len(env_ids), -1).clone()
        for key, (ids, names, value, default) in self.stiffness_range.items():
            r = (value[1] - value[0]) * torch.rand(len(env_ids), 1, device=self.device) + value[0]
            stiffness[:, ids] = default * r
        self.actuator.stiffness[env_ids] = stiffness * scale_factor
        
        damping = self.default_damping.expand(len(env_ids), -1).clone()
        for key, (ids, names, value, default) in self.damping_range.items():
            r = (value[1] - value[0]) * torch.rand(len(env_ids), 1, device=self.device) + value[0]
            damping[:, ids] = default * r
        self.actuator.damping[env_ids] = damping * scale_factor

        armature = torch.zeros(len(env_ids), self.num_joints, device=self.device)
        for key, (ids, names, value, default) in self.armature_range.items():
            r = (value[1] - value[0]) * torch.rand(len(env_ids), 1, device=self.device) + value[0]
            armature[:, ids] = r
        self.asset.write_joint_armature_to_sim(armature, env_ids=env_ids)

        # apply randomization
        if isinstance(self.actuator, DCMotor):
            pass
            # self.actuator._saturation_effort[env_ids] = strength
        elif isinstance(self.actuator, HybridActuator):
            implicit = self.actuator.implicit[env_ids]
            self.asset.write_joint_stiffness_to_sim(stiffness * implicit, self.actuator.joint_indices, env_ids)
            self.asset.write_joint_damping_to_sim(damping * implicit, self.actuator.joint_indices, env_ids)
        elif isinstance(self.actuator, ImplicitActuator):
            self.asset.write_joint_stiffness_to_sim(stiffness, self.actuator.joint_indices, env_ids)
            self.asset.write_joint_damping_to_sim(damping, self.actuator.joint_indices, env_ids)


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

        self.joint_ids, self.joint_names, self.pos_ranges = string_utils.resolve_matching_names_values(
            dict(pos_ranges), self.asset.joint_names
        )
        self.pos_ranges = torch.as_tensor(self.pos_ranges, device=self.device).unbind(-1)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids]
        self.joint_limits = self.asset.data.joint_limits[0, self.joint_ids].unbind(-1)

    def reset(self, env_ids: torch.Tensor):
        shape = (len(env_ids), len(self.joint_ids))
        init_pos = sample_uniform(shape, *self.pos_ranges, self.device)
        if self.rel:
            init_pos += self.default_joint_pos[env_ids]
        init_vel = self.default_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(
            init_pos.clamp(*self.joint_limits), 
            init_vel, self.joint_ids, env_ids #.unsqueeze(1)
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
            init_pos, init_vel, self.joint_ids, env_ids #.unsqueeze(1)
        )


class push(Randomization):
    def __init__(self, env, body_names, force_range = (0.2, 0.9), min_interval=100, decay: float=0.9):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.num_bodies = len(self.body_indices)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.force_range = force_range
        self.min_interval = min_interval
        self.decay = decay
        
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
            self.forces = torch.where(i, push_forces * self.default_mass_total, self.forces * self.decay)
        self.asset.set_external_force_and_torque(self.forces, self.torques, body_ids=self.body_indices)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.body_indices],
            self.forces / self.default_mass_total,
            color=(1., 0.8, .4, 1.)
        )

class drag(Randomization):
    def __init__(self, env, body_names, drag_range=(0.0, 0.1)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.num_bodies = len(self.body_indices)
        self.drag_coeffs = sample_uniform((self.num_envs, self.num_bodies, 1), *drag_range, self.device).expand(self.num_envs, self.num_bodies, 3)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81

        with torch.device(self.env.device):
            self.forces = torch.zeros(self.env.num_envs, len(self.body_indices), 3)
            self.torques = torch.zeros(self.env.num_envs, len(self.body_indices), 3)

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.

    def step(self, substep):
        lin_vel = self.asset.data.body_lin_vel_w[:, self.body_indices]
        drag_forces = - lin_vel * self.drag_coeffs
        self.forces = drag_forces * self.default_mass_total
        self.asset.set_external_force_and_torque(self.forces, self.torques, body_ids=self.body_indices)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.body_indices],
            self.forces / self.default_mass_total * 100,
            color=(0.6, 0.8, 0.6, 1.)
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
        friction = torch.empty(len(env_ids), 1, 1, device=self.device)
        friction.uniform_(*self.friction_range)
        self.friction_coef[env_ids] = friction

    def step(self, substep):
        # feet_height = self.asset.data.feet_height_map.mean(-1).reshape(-1)
        feet_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.body_ids]
        feet_quat_w = self.asset.data.body_quat_w[:, self.body_ids]
        stumble_prob = ((self.stumble_height - self.feet_height) / self.stumble_height).clamp(0., 1.)
        self.forces_w = - self.friction_coef * feet_lin_vel_w / self.env.physics_dt
        self.forces_w[..., 2] = 0.
        friction_forces = torch.where(
            (torch.rand_like(self.feet_height) < stumble_prob).unsqueeze(-1),
            quat_rotate_inverse(feet_quat_w, self.forces_w),
            torch.zeros(self.num_envs, self.num_feet, 3, device=self.env.device)
        )
        forces_b = self.asset._external_force_b.clone()
        torques_b = self.asset._external_torque_b.clone()
        forces_b[:, self.body_ids] += friction_forces
        self.asset.set_external_force_and_torque(forces_b, torques_b)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.body_ids],
            self.forces_w * self.env.physics_dt,
            color=(1., 0.6, 0., 1.)
        )


class random_joint_friction(Randomization):
    def __init__(self, env, actuator_name: str, friction_range):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joints = self.asset.actuators[actuator_name]
        self.friction_range = friction_range
    
    def startup(self):
        print(f"Randomize joint friction of joints {self.joints.joint_names}, {self.joints.joint_indices}")
        friction = torch.zeros_like(self.joints.friction)
        friction.uniform_(*self.friction_range)
        self.asset.write_joint_friction_to_sim(friction, joint_ids=self.joints.joint_indices)


class pull(Randomization):
    def __init__(
        self, 
        env,
        drag_prob: float = 0.2,
        drag_range=(0.0, 0.2)
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.drag_prob = drag_prob
        self.drag_range = drag_range
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum().to(self.device) * 9.81
        
        with torch.device(self.device):
            self.forces = torch.zeros(self.num_envs, 3)
            self.axis = torch.zeros(self.num_envs, 3)
            self.apply_drag = torch.zeros(self.num_envs, 1, dtype=bool)
            self.drag_magnitude = torch.zeros(self.num_envs, 1)

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.
        
        # pull direction
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        axis = torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], -1)
        self.axis[env_ids] = axis

        drag_magnitude = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.drag_range)
        self.drag_magnitude[env_ids] = drag_magnitude * self.default_mass_total
        self.apply_drag[env_ids] = (torch.rand(len(env_ids), 1, device=self.device) < self.drag_prob)
    
    def update(self):
        pass

    def step(self, substep):
        force =  self.axis * self.drag_magnitude
        self.forces[:] = torch.where(self.apply_drag, force, torch.zeros_like(self.forces))
        self.asset.set_external_force_and_torque(
            quat_rotate_inverse(self.asset.data.root_quat_w, self.forces).unsqueeze(1), 
            torch.zeros_like(force).unsqueeze(1), [0])

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w, 
            self.forces / self.default_mass_total, 
            color=(0.6, 0.8, 0.6, 1.)
        )


class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.offset_range = string_utils.resolve_matching_names_values(dict(offset_range), self.asset.joint_names)
        
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.offset_range = torch.tensor(self.offset_range, device=self.device)

        self.action_manager = self.env.action_manager

    def reset(self, env_ids: torch.Tensor):
        offset = uniform(self.offset_range[:, 0], self.offset_range[:, 1])
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset


class random_pull(Randomization):
    def __init__(self, env, force_xy_range, force_z_range, prob=0.01, duration = (0.5, 0.5)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.force_xy_range = force_xy_range
        self.force_z_range = force_z_range
        self.prob = prob
        self.duration = (duration, duration) if isinstance(duration, (int, float)) else duration
        self.mass_total = self.asset.root_physx_view.get_masses()[0].sum().to(self.device)

        self.force_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.offset_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.time_remaining = torch.zeros(self.num_envs, 1, device=self.device)

    def step(self, substep):
        force_w = self.force_w * (self.time_remaining > 0)
        self.asset._external_force_b[:, 0] += quat_rotate_inverse(self.asset.data.root_quat_w, force_w)
        self.asset._external_torque_b[:, 0] += self.offset_b.cross(force_w, dim=-1)
        self.asset.has_external_wrench = True
    
    def update(self):
        sample_force = (torch.rand(self.num_envs, device=self.device) < self.prob).nonzero().squeeze(-1)
        if len(sample_force) > 0:
            force_w = torch.zeros(len(sample_force), 3, device=self.device)
            force_w[:, 0].uniform_(*self.force_xy_range)
            force_w[:, 1].uniform_(*self.force_xy_range)
            force_w[:, 2].uniform_(*self.force_z_range) 
            offset_b = torch.zeros(len(sample_force), 3, device=self.device)
            offset_b[:, 0].uniform_(-0.25, 0.25)
            offset_b[:, 1].uniform_(-0.15, 0.15)
            offset_b[:, 2].uniform_(-0.15, 0.15)
            self.offset_b[sample_force] = offset_b
            self.force_w[sample_force] = force_w
            duration = torch.zeros(len(sample_force), 1, device=self.device)
            duration.uniform_(*self.duration)
            self.time_remaining[sample_force] = duration / self.env.step_dt
        self.time_remaining -= 1.

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w, 
            (self.force_w / self.mass_total) * (self.time_remaining > 0), 
            color=(0.6, 0.8, 0.6, 1.)
        )


class spring_grf(Randomization):
    def __init__(self, env, feet_names: str = ".*_foot", thres_range = (0.1, 0.2), kp_range = (200, 300)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.thres_range = thres_range
        self.kp_range = kp_range

        self.feet_ids = self.asset.find_bodies(feet_names)[0]
        self.kp = torch.zeros(self.num_envs, 4, device=self.device)
        self.thres = torch.zeros(self.num_envs, 4, device=self.device)
        self.forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.flag = torch.zeros(self.num_envs, 4, dtype=bool, device=self.device)
        self.axis = torch.zeros(self.num_envs, 4, 3, device=self.device)

    def update(self):
        resample = (self.env.episode_length_buf % 100 == 0).unsqueeze(1) # [num_envs, 1]
        self.flag = torch.where(resample, torch.rand(self.flag.shape, device=self.device) < 0.2, self.flag)
        self.kp = torch.where(resample, uniform_like(self.kp, *self.kp_range), self.kp)
        self.thres = torch.where(resample, uniform_like(self.thres, *self.thres_range), self.thres)
        axis = torch.zeros(self.num_envs, 4, 3, device=self.device)
        axis[:, :, 1].uniform_(-0.3, 0.3)
        axis[:, :, 0].uniform_(-0.3, 0.3)
        axis[:, :, 2] = 1.
        axis = axis / axis.norm(dim=-1, keepdim=True)
        self.axis = torch.where(resample.unsqueeze(-1), axis, self.axis)

    def step(self, substep):
        feet_height = self.asset.data.feet_height
        feet_quat = self.asset.data.body_quat_w[:, self.feet_ids]
        feet_lin_vel = self.asset.data.body_lin_vel_w[:, self.feet_ids]
        forces = (
            self.kp * (self.thres - feet_height) + 
            5. * (0. - feet_lin_vel[:, :, 2])
        ) * self.flag
        self.forces = forces.unsqueeze(-1) * self.axis 
        self.asset._external_force_b[:, self.feet_ids] += quat_rotate_inverse(feet_quat, self.forces)
        self.asset.has_external_wrench = True

    def debug_draw(self):
        feet_pos = self.asset.data.body_pos_w[:, self.feet_ids]
        self.env.debug_draw.vector(feet_pos, self.forces / 9.81, color=(0.8, 0.6, 0.6, 1.))


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

def uniform_like(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(x)
    return low + r * (high - low)

def log_uniform(low: torch.Tensor, high: torch.Tensor):
    return uniform(low.log(), high.log()).exp()

def angle_mix(a: torch.Tensor, b: torch.Tensor, weight: float=0.1):
    d = a - b
    d[d > torch.pi] -= 2 * torch.pi
    d[d < -torch.pi] += 2 * torch.pi
    return a - d * weight