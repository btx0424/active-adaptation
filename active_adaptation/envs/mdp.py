import torch
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.actuators import DCMotor

import logging

class Randomization:
    def __init__(self, env):
        self.env = env

    def startup(self):
        pass
    
    def reset(self, env_ids: torch.Tensor):
        pass


class MotorParams(Randomization):
    def __init__(
        self, 
        env, 
        stiffness_range = (0.7, 1.3),
        damping_range = (0.7, 1.3)
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.stiffness_range = stiffness_range
        self.damping_range = damping_range
    
    def startup(self):
        self.motors: DCMotor = self.asset.actuators["base_legs"]
        self.default_stiffness = self.motors.stiffness.clone()
        self.default_damping = self.motors.damping.clone()
        
    def reset(self, env_ids: torch.Tensor):
        self.motors.stiffness[env_ids] = random_scale(
            self.default_stiffness[env_ids], *self.stiffness_range
        )
        self.motors.damping[env_ids] = random_scale(
            self.default_damping[env_ids], *self.damping_range
        )


class BodyMaterial(Randomization):
    def __init__(
        self,
        env,
        body_indices,
        static_friction_range = (0.6, 1.0),
        dynamic_friction_range = (0.6, 1.0),
        restitution_range=(0.0, 0.0)
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices = body_indices
        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range

        self.default_material_properties: torch.Tensor = None
        self.material_properties: torch.Tensor = None

    def startup(self):
        logging.info("Randomize body materials upon starup.")
        default_material_properties = self.asset.body_physx_view.get_material_properties()
        shape = (self.asset.body_physx_view.count, self.asset.body_physx_view.max_shapes, 3)
        materials = torch.zeros(shape)
        materials[:, :, 0].uniform_(*self.static_friction_range)
        materials[:, :, 1].uniform_(*self.dynamic_friction_range)
        materials[:, :, 2].uniform_(*self.restitution_range)

        bodies_per_env = self.asset.body_physx_view.count // self.env.num_envs  # - number of bodies per spawned asset
        indices = torch.tensor(self.body_indices, dtype=torch.int).repeat(self.env.num_envs, 1)
        indices += torch.arange(self.env.num_envs).unsqueeze(1) * bodies_per_env
        # indices = torch.arange(self.asset.body_physx_view.count)
        self.asset.body_physx_view.set_material_properties(materials, indices)

        self.default_material_properties = default_material_properties[self.body_indices].to(self.env.device)
        self.material_properties = materials[self.body_indices].to(self.env.device)


class BodyMasses(Randomization):
    def __init__(
        self,
        env,
        mass_range=(0.7, 1.3),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.mass_range = mass_range

        self.default_masses: torch.Tensor = None
        self.body_masses: torch.Tensor = None

    def startup(self):
        logging.info("Randomize body masses upon starup.")
        default_masses = self.asset.root_physx_view.get_masses()
        body_masses = random_scale(default_masses, *self.mass_range)
        indices = self.asset._ALL_INDICES.cpu()
        self.asset.root_physx_view.set_masses(body_masses, indices=indices)
        
        self.default_masses = default_masses.to(self.env.device)
        self.body_masses = body_masses.to(self.env.device)

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
        

def random_scale(x: torch.Tensor, low: float, high: float):
    return x * (torch.rand_like(x) * (high - low) + low)

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)