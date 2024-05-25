import torch
import numpy as np
import abc
import einops

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.sensors import ContactSensor, RayCaster, patterns, RayCasterData
from omni.isaac.orbit.sensors import Camera
import omni.isaac.orbit.sim as sim_utils
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from omni.isaac.orbit.terrains.trimesh.utils import make_plane
from omni.isaac.orbit.utils.math import convert_quat, quat_apply, quat_apply_yaw, yaw_quat
from omni.isaac.orbit.utils.warp import convert_to_warp_mesh, raycast_mesh
from pxr import UsdGeom, UsdPhysics

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

class Buffer:
    def __init__(self, shape, size, device):
        self.data = torch.zeros(*shape, size, device=device)
        self.time_stamp = 0

    def reset(self, env_ids: torch.Tensor, value=0.):
        self.data[env_ids] = value
    
    def update(self, value: torch.Tensor, time_stamp: int):
        if time_stamp > self.time_stamp:
            self.data[..., :-1] = self.data[..., 1:]
            self.data[..., -1] = value
            self.time_stamp = time_stamp


class Observation:
    def __init__(self, env):
        self.env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def __call__(self) ->  torch.Tensor:
        raise NotImplementedError

    def startup(self):
        pass
    
    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

class BufferedObs(Observation):
    def __init__(self, env, shape, size):
        super().__init__(env)
        self.buffer = Buffer(shape, size, self.env.device)
        setattr(self.env, f"_{self.__class__.__name__}")
    
    def __call__(self):
        return self.buffer.data.reshape(self.env.num_envs, -1)

    def reset(self, env_ids: torch.Tensor):
        self.buffer.reset(env_ids)


class linvel_b_buffer(BufferedObs):
    def __init__(self, env, size: int=4):
        self.asset = env.scene["robot"]
        super().__init__(env, self.asset.data.root_lin_vel_b.shape, size)

    def update(self):
        self.buffer.update(self.asset.data.root_linvel_b, self.env.time_stamp)


class joint_pos_buffer(BufferedObs):
    def __init__(self, env, size: int=4):
        self.asset = env.scene["robot"]
        super().__init__(env, self.asset.data.joint_pos.shape, size)

    def update(self):
        self.buffer.update(self.asset.data.joint_pos, self.env.time_stamp)


class joint_vel_buffer(BufferedObs):
    def __init__(self, env, size: int=4):
        self.asset = env.scene["robot"]
        super().__init__(env, self.asset.data.joint_pos.shape, size)

    def update(self):
        self.buffer.update(self.asset.data.joint_vel, self.env.time_stamp)


class body_pos(Observation):
    def __init__(self, env, body_names, yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        print(f"Track body pos for {self.body_names}")
        self.body_pos_b = torch.zeros(self.env.num_envs, len(self.body_indices), 3, device=self.env.device)

    def update(self):
        if self.yaw_only:
            quat = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            quat = self.asset.data.root_quat_w.unsqueeze(1)
        body_pos = self.asset.data.body_pos_w[:, self.body_indices]
        body_pos = body_pos - self.asset.data.root_pos_w.unsqueeze(1)
        self.body_pos_b[:] = quat_rotate_inverse(quat, body_pos)
        
    def __call__(self):
        return self.body_pos_b.reshape(self.env.num_envs, -1)


class body_vel(Observation):
    def __init__(self, env, body_names, yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        print(f"Track body vel for {self.body_names}")
        self.body_vel_b = torch.zeros(self.env.num_envs, len(self.body_indices), 3, device=self.env.device)

    def update(self):
        if self.yaw_only:
            quat = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            quat = self.asset.data.root_quat_w.unsqueeze(1)
        body_vel_w = self.asset.data.body_lin_vel_w[:, self.body_indices]
        self.body_vel_b[:] = quat_rotate_inverse(quat, body_vel_w)
        
    def __call__(self):
        return self.body_vel_b.reshape(self.env.num_envs, -1)


def observation_func(func):

    class ObsFunc(Observation):
        def __init__(self, env, **params):
            super().__init__(env)
            self.params = params

        def __call__(self):
            return func(self.env, **self.params)
    
    return ObsFunc


@observation_func
def root_quat_w(self):
    return self.scene["robot"].data.root_quat_w


class root_angvel_b(Observation):
    def __init__(self, env, noise_std: float=0., yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.yaw_only = yaw_only
    
    def __call__(self) -> torch.Tensor:
        ang_vel_b = random_noise(self.asset.data.root_ang_vel_b, self.noise_std) 
        if self.yaw_only:
            return ang_vel_b[:, 2].reshape(self.num_envs, 1)
        else:
            return ang_vel_b


class projected_gravity_b(Observation):
    def __init__(self, env, noise_std: float=0.):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
    
    def __call__(self):
        projected_gravity_b = self.asset.data.projected_gravity_b
        noise = torch.randn_like(projected_gravity_b).clip(-3., 3.) * self.noise_std
        projected_gravity_b += noise
        return projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)


class root_linvel_b(Observation):
    def __init__(self, env, body_names: str=None):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
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
    
    def __call__(self) -> torch.Tensor:
        return self.linvel

    def debug_draw(self):
        if self.body_ids is None:
            linvel = self.asset.data.root_lin_vel_w
        else:
            linvel = (self.asset.data.body_lin_vel_w[:, self.body_ids] * self.body_masses).mean(1)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            linvel,
            color=(0.8, 0.1, 0.1, 1.)
        )
    

class joint_pos(Observation):
    def __init__(self, env, noise_std: float=0.0):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std

    def __call__(self) -> torch.Tensor:
        return random_noise(self.asset.data.joint_pos, self.noise_std)


class joint_vel(Observation):
    def __init__(self, env, noise_std: float=0.0):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
    
    def __call__(self) -> torch.Tensor:
        return random_noise(self.asset.data.joint_vel, self.noise_std)

class joint_acc(Observation):
    
    smoothing_length = 5
    
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_acc_buf = (
            torch.zeros_like(self.asset.data.joint_acc)
            .unsqueeze(-1)
            .expand(-1, -1, self.smoothing_length)
            .clone()
        )
        self.smoothing_weights = torch.arange(self.smoothing_length, device=self.device).flipud()
        self.smoothing_weights = self.smoothing_weights / self.smoothing_weights.sum()

    def reset(self, env_ids: torch.Tensor):
        self.joint_acc_buf[env_ids] = 0.

    def update(self):
        self.joint_acc_buf[..., 1:] = self.joint_acc_buf[..., :-1]
        self.joint_acc_buf[..., 0] = self.asset.data.joint_acc

    def __call__(self) -> torch.Tensor:
        joint_acc = (self.joint_acc_buf * self.smoothing_length).mean(-1)
        joint_acc *= self.env.step_dt
        return joint_acc


class applied_torques(Observation):
    def __init__(self, env, actuator_name: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.actuator = self.asset.actuators[actuator_name]
        
        self.joint_indices = self.actuator.joint_indices
        self.effort_limit = self.actuator.effort_limit
    
    def __call__(self) -> torch.Tensor:
        applied_efforts = self.asset.data.applied_torque
        return applied_efforts[:, self.joint_indices] / self.effort_limit


class contact_indicator(Observation):
    def __init__(self, env, body_names: str, timing: bool=True):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        
        self.artc_ids, names = self.asset.find_bodies(body_names, preserve_order=True)
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names, preserve_order=True)
        self.timing = timing
        
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum().to(self.env.device) * 9.81
        self.forces = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)

    def update(self):
        self.forces[:] = self.contact_sensor.data.net_forces_w_history[:, :, self.body_ids].mean(1)
        # self.forces[:] = self.contact_sensor.data.net_forces_w[:, self.body_ids]

    def __call__(self):
        forces = quat_rotate_inverse(self.asset.data.root_quat_w.unsqueeze(1), self.forces) / self.default_mass_total
        if self.timing:
            current_air_time = self.contact_sensor.data.current_air_time[:, self.body_ids].clamp_max(1.)
            current_contact_time = self.contact_sensor.data.current_contact_time[:, self.body_ids].clamp_max(1.)
            return torch.cat([
                current_air_time,
                current_contact_time,
                forces.reshape(self.num_envs, -1).clip(-5., 5.)
            ], dim=-1)
        else:
            return forces.reshape(self.num_envs, -1).clip(-5., 5.)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.artc_ids],
            self.forces / self.default_mass_total,
            color=(1., 1., 1., 1.)
        )


class motor_params(Observation):
    def __init__(self, env, actuator_name: str, homogeneous: bool=False):
        super().__init__(env)
        self.homogeneous = homogeneous
        self.asset: Articulation = self.env.scene["robot"]
        self.motors = self.asset.actuators[actuator_name]
        self.defalut_stiffness = self.motors.stiffness.clone()
        self.default_damping = self.motors.damping.clone()
        self.stiffness = self.motors.stiffness
        self.damping = self.motors.damping

        if self.homogeneous:
            self.defalut_stiffness = self.defalut_stiffness[..., 0].unsqueeze(-1)
            self.default_damping = self.default_damping[..., 0].unsqueeze(-1)
            self.stiffness = self.stiffness[..., 0].unsqueeze(-1)
            self.damping = self.damping[..., 0].unsqueeze(-1)
    
    def __call__(self) -> torch.Tensor:
        stiffness = (self.stiffness / self.defalut_stiffness) - 1.
        damping  = (self.damping / self.default_damping) - 1.
        return torch.cat([stiffness, damping], dim=-1)


class motor_failure(Observation):
    def __init__(self, env, actuator_name: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.motors = self.asset.actuators[actuator_name]
        self.motor_failure = self.motors.motor_failure
    
    def __call__(self) -> torch.Tensor:
        return self.motor_failure


class com(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        default_masses = self.asset.root_physx_view.get_masses()[0].to(self.env.device)
        self.default_mass_distribution = (default_masses / default_masses.sum()).unsqueeze(-1)
        self.default_coms = self.asset.root_physx_view.get_coms()[0, :, :3].to(self.env.device)

        self.com_b = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        self.com_w_buffer = Buffer((self.env.num_envs, 3), 4, self.env.device)

    def update(self):
        body_com_pos_w = self.asset.data.body_pos_w + quat_rotate(self.asset.data.body_quat_w, self.default_coms.unsqueeze(0))
        self.com_w = (body_com_pos_w * self.default_mass_distribution).sum(1)
        self.com_w_buffer.update(self.com_w, self.env.time_stamp)
        com_diff = self.com_w_buffer.data[:, :, 1:] - self.com_w_buffer.data[:, :, :-1]
        self.com_vel_w = (com_diff / self.env.step_dt).mean(dim=-1)

    def __call__(self) -> torch.Tensor:
        return self.com_b

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.com_w,
            self.com_vel_w,
            color=(0., 1., 1., 1.)
        )


class external_forces(Observation):
    def __init__(self, env, body_names):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81

    def __call__(self) -> torch.Tensor:
        forces_b = self.asset._external_force_b[:, self.body_indices]
        return (forces_b / self.default_mass_total).reshape(self.env.num_envs, -1)


class body_materials(Observation):
    def __init__(self, env, body_names, homogeneous: bool=False):
        super().__init__(env)
        self.homogeneous = homogeneous
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        num_shapes_per_body = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        cumsum = np.cumsum([0,] + num_shapes_per_body)
        self.shape_ids = torch.cat([
            torch.arange(cumsum[i], cumsum[i+1]) 
            for i in self.body_ids
        ]).to(self.device)

        if self.homogeneous:
            self.shape_ids = self.shape_ids[0]
        
    def __call__(self):
        return self.asset.data.body_materials[:, self.shape_ids].reshape(self.num_envs, -1)


class body_mass(Observation):
    def __init__(self, env, body_names, homogeneous: bool=False):
        super().__init__(env)
        self.homogeneous = homogeneous
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        
        masses = self.asset.root_physx_view.get_masses()[0]
        self.default_mass_total = masses.sum()
        self.masses = torch.zeros_like(masses[self.body_ids], device=self.device)
    
    def startup(self):
        self.masses = (
            self.asset.root_physx_view.get_masses()[:, self.body_ids]
            / self.default_mass_total.sum()
        ).to(self.device)
    
    def __call__(self) -> torch.Tensor:
        return self.masses.reshape(self.num_envs, -1)
    

class body_momentum(Observation):
    def __init__(self, env, body_names, homogeneous: bool=False):
        super().__init__(env)
        self.homogeneous = homogeneous
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        masses = self.asset.root_physx_view.get_masses()[0]
        self.default_mass_total = masses.sum()
        self.masses = torch.zeros_like(masses[self.body_ids], device=self.device)
    
    def startup(self):
        self.masses = (
            self.asset.root_physx_view.get_masses()[:, self.body_ids].unsqueeze(-1)
            / self.default_mass_total.sum()
        ).to(self.device)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
    
    def __call__(self) -> torch.Tensor:
        velocity = self.asset.data.body_lin_vel_w[:, self.body_ids]
        momentum = self.masses * quat_rotate_inverse(self.asset.data.root_quat_w.unsqueeze(1), velocity)
        return momentum.reshape(self.num_envs, -1)


class feet_height(Observation):
    def __init__(self, env, feet_names=".*_foot", nomial_height=0.3):
        super().__init__(env)
        self.nominal_height = nomial_height
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(feet_names)
        self.num_feet = len(self.body_ids)
    
    def __call__(self) -> torch.Tensor:
        return self.asset.data.body_pos_w[:, self.body_ids, 2].reshape(self.num_envs, -1) / self.nominal_height


class feet_height_map(Observation):
    def __init__(
        self, 
        env, 
        feet_names=".*_foot", 
        nomial_height=0.3,
        resolution: float=0.1,
        size=[0.15, 0.15],
    ):
        super().__init__(env)
        self.nominal_height = nomial_height
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(feet_names)
        self.num_feet = len(self.body_ids)
        
        self._init_raycaster(resolution, size)
    
    def _init_raycaster(self, resolution, size):
        _initialize_warp_meshes("/World/ground", "cuda")

        # pattern_cfg = patterns.GridPatternCfg(resolution=resolution, size=size)
        # self.ray_starts, self.ray_directions = pattern_cfg.func(pattern_cfg, self.device)
        # self.ray_starts[:, 2] += 10.
        self.ray_starts = torch.tensor(
            [
                [0., 0., 10.], 
                # [0., 0.1, 10.],
                # [0., -0.1, 10.],
                # [0.1, 0., 10.],
                # [-0.1, 0., 10.],
                [0.1, 0.1, 10.],
                [0.1, -.1, 10.],
                [-.1, -.1, 10.],
                [-.1, 0.1, 10.],
            ],
            device=self.device
        )
        self.ray_directions = torch.tensor([0., 0., -1.], device=self.device)
        self.num_rays = len(self.ray_starts)

        shape = (self.num_envs, self.num_feet, self.num_rays)

        # fill the data buffer
        # self._data.pos_w = torch.zeros(*shape, 3, device=self.device)
        # self._data.quat_w = torch.zeros(*shape, 4, device=self.device)
        self.ray_hits_w = torch.zeros(*shape, 3, device=self.device)
        self.feet_height_map = torch.zeros(shape, device=self.device)
        self.asset.data.feet_height = self.feet_height_map[:, :, 0]
        self.asset.data.feet_height_map = self.feet_height_map
    
    def update(self):
        self.feet_pos_w = self.asset.data.body_pos_w[:, self.body_ids]
        self.feet_quat_w = self.asset.data.body_quat_w[:, self.body_ids]
        shape = (self.num_envs, self.num_feet, self.num_rays, -1)
        ray_starts_w = quat_apply_yaw(
            self.feet_quat_w.unsqueeze(-2).expand(shape),
            self.ray_starts.reshape(1, 1, -1, 3).expand(shape),
        )
        ray_starts_w += self.feet_pos_w.unsqueeze(-2)
        self.ray_hits_w[:] = raycast_mesh(
            ray_starts_w,
            self.ray_directions.expand_as(ray_starts_w).clone(),
            max_dist=100.,
            mesh=RayCaster.meshes["/World/ground"],
        )[0]

        self.feet_height_map[:] = self.feet_pos_w.unsqueeze(-2)[..., 2] - self.ray_hits_w[..., 2]

    def __call__(self):
        return self.feet_height_map.reshape(self.num_envs, -1) / self.nominal_height
    
    def debug_draw(self):
        x = self.ray_hits_w.clone()
        x[..., 2] = self.feet_pos_w.unsqueeze(-2)[..., 2]
        d = self.ray_hits_w - x
        self.env.debug_draw.vector(x, d)


class height_scan(Observation):
    def __init__(self, env, prim_path, flatten: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.height_scanner: RayCaster = self.env.scene["height_scanner"]
        self.height_scan = torch.zeros(self.num_envs, 3, 11, 17, device=self.device)
        self.update()
        self.flatten = flatten

    def reset(self, env_ids):
        self.height_scan[env_ids] = 0.

    def update(self):
        self.height_scan[:, :-1] = self.height_scan[:, 1:]
        self.height_scan[:, -1] = (
            self.asset.data.root_pos_w[:, 2].unsqueeze(1)
            - self.height_scanner.data.ray_hits_w[:, :, 2]
        ).reshape(self.num_envs, 11, 17).clamp(-1., 1.)

    def __call__(self):
        if self.flatten:
            return self.height_scan.reshape(self.num_envs, -1)
        else:
            return self.height_scan

    # def debug_draw(self):
    #     to = self.height_scan.data.ray_hits_w.reshape(-1, 11, 17, 3)[:, :, 0].reshape(-1, 11, 3)
    #     start = self.root_pos_w.unsqueeze(-2).expand_as(to)
    #     self.env.debug_draw.vector(
    #         start,
    #         to - start,
    #     )

class prev_actions(Observation):
    def __init__(self, env, steps: int=1):
        super().__init__(env)
        self.steps = steps

        with torch.device(self.device):
            self.prev_action = torch.zeros(self.num_envs, self.env.action_spec.shape[-1], self.steps)
    
    def update(self):
        self.prev_action[:] = self.env.action_buf[:, :, :self.steps]
    
    def __call__(self):
        return self.prev_action.reshape(self.num_envs, -1)


class last_contact(Observation):
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]

        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        with torch.device(self.device):
            self.body_ids = torch.as_tensor(self.body_ids)
            self.has_contact = torch.zeros(self.num_envs, len(self.body_ids), 1, dtype=bool)
            self.last_contact_pos_w = torch.zeros(self.num_envs, len(self.body_ids), 3)
        self.body_pos_w = self.asset.data.body_pos_w[:, self.articulation_body_ids]
        
    def reset(self, env_ids: torch.Tensor):
        self.has_contact[env_ids] = False
    
    def update(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.body_ids].unsqueeze(-1)
        self.has_contact.logical_or_(first_contact)
        self.body_pos_w = self.asset.data.body_pos_w[:, self.articulation_body_ids]
        self.last_contact_pos_w = torch.where(
            first_contact,
            self.body_pos_w,
            self.last_contact_pos_w
        )
    
    def __call__(self):
        distance_xy = (self.body_pos_w[:, :, :2] - self.last_contact_pos_w[:, :, :2]).norm(dim=-1)
        distance_z = self.body_pos_w[:, :, 2] - self.last_contact_pos_w[:, :, 2]
        distance = torch.stack([distance_xy, distance_z], dim=-1)
        return (distance * self.has_contact).reshape(self.num_envs, -1)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.body_pos_w,
            torch.where(self.has_contact, self.last_contact_pos_w, self.body_pos_w) - self.body_pos_w
        )


class body_scale(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.scales = getattr(self.asset.cfg, "scale", torch.ones(self.num_envs, 1)).to(self.device)

    def __call__(self) -> torch.Tensor:
        return self.scales


class action_delay(Observation):
    def __call__(self) -> torch.Tensor:
        if hasattr(self.env, "delay"):
            return self.env.delay.float()
        else:
            return torch.zeros(self.num_envs, 1, device=self.device)


class rewards(Observation):
    def __call__(self) -> torch.Tensor:
        return self.env._reward_buf


# class incoming_wrench(Observation):
#     def __init__(self, env):
#         super().__init__(env)
#         self.asset: Articulation = self.env.scene["robot"]
#         self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum().to(self.env.device) * 9.81

#     def __call__(self) -> torch.Tensor:
#         self.forces = self.asset.root_physx_view.get_link_incoming_joint_force()[:, :, :3]
#         return self.forces / self.default_mass_total

#     def debug_draw(self):
#         self.env.debug_draw.vector(
#             self.asset.data.body_pos_w[:, [1, 2]],
#             self.forces[:, [1, 2]],
#             color=(1., 1., 0., 1.)
#         )

class cum_error(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager
    
    def __call__(self) -> torch.Tensor:
        return self.command_manager._cum_error

import imageio

class camera(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.camera: Camera = self.env.scene["camera"]
        self.frame_count = 0
    
    def __call__(self) -> torch.Tensor:
        image = self.camera.data.output["rgb"]
        imageio.imwrite(f"frame-{self.frame_count}.png", image[0, :, :, :3].cpu())
        self.frame_count += 1
        return image / 255.0


class clock(Observation):
    def __init__(self, env, frequencies: list[int]=[1, 2, 4]):
        super().__init__(env)
        self.frequencies = torch.as_tensor(frequencies, device=self.device).unsqueeze(0)
    
    def __call__(self) -> torch.Tensor:
        t = (self.env.episode_length_buf * self.env.step_dt)
        t = t.reshape(self.num_envs, 1) * self.frequencies
        return torch.cat([t.sin(), t.cos()], dim=1)

class phase(Observation):
    def __init__(self, env, cycle: float=1.2):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.freq = torch.pi * 2 / cycle
        self.asset.data.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase: torch.Tensor = self.asset.data.phase
        self.offset: torch.Tensor = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        offset = torch.full(env_ids.shape, torch.pi/3, device=self.device)
        offset[torch.rand_like(offset) > 0.5] += torch.pi
        self.offset[env_ids] = offset

    def update(self):
        self.phase[:] = self.offset + self.env.episode_length_buf * self.freq * self.env.step_dt

    def __call__(self) -> torch.Tensor:
        return torch.stack([self.phase.sin(), self.phase.cos()], 1)
        
    
def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std


def _initialize_warp_meshes(mesh_prim_path, device):
    if mesh_prim_path in RayCaster.meshes:
        return

    # check if the prim is a plane - handle PhysX plane as a special case
    # if a plane exists then we need to create an infinite mesh that is a plane
    mesh_prim = sim_utils.get_first_matching_child_prim(
        mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
    )
    # if we did not find a plane then we need to read the mesh
    if mesh_prim is None:
        # obtain the mesh prim
        mesh_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        # check if valid
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # read the vertices and faces
        points = np.asarray(mesh_prim.GetPointsAttr().Get())
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
        wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    else:
        mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
        wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
    # add the warp mesh to the list
    RayCaster.meshes[mesh_prim_path] = wp_mesh