import torch
import abc

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.sensors import ContactSensor

from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

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
    def __init__(self, env, body_names):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        print(f"Track body pos for {self.body_names}")
        self.body_pos_b = torch.zeros(self.env.num_envs, len(self.body_indices), 3, device=self.env.device)

    def update(self):
        body_pos_w = self.asset.data.body_pos_w[:, self.body_indices]
        self.body_pos_b[:] = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            body_pos_w - self.asset.data.root_pos_w.unsqueeze(1)
        )
        
    def __call__(self):
        return self.body_pos_b.reshape(self.env.num_envs, -1)


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


@observation_func
def root_angvel_b(self):
    return self.scene["robot"].data.root_ang_vel_b


@observation_func
def projected_gravity_b(self):
    return self.scene["robot"].data.projected_gravity_b


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
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        if hasattr(self.env, "_joint_vel_buffer"):
            self.buffer = self.env._joint_vel_buffer
        else:
            self.buffer = Buffer(self.asset.data.joint_pos.shape, 4, self.env.device)

    def reset(self, env_ids: torch.Tensor):
        self.buffer.reset(env_ids)

    def update(self):
        self.buffer.update(self.asset.data.joint_vel, self.env.time_stamp)

    def __call__(self) -> torch.Tensor:
        vel_diff = self.buffer.data[:, :, 1:] - self.buffer.data[:, :, :-1]
        acc = (vel_diff / self.env.step_dt).mean(dim=-1)
        acc = acc * self.env.step_dt
        return symlog(acc)


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
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.timing = timing
        
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum().to(self.env.device) * 9.81
        self.forces = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)

    def update(self):
        self.forces[:] = self.contact_sensor.data.net_forces_w_history[:, :, self.body_ids].mean(1)

    def __call__(self):
        if self.timing:
            current_air_time = self.contact_sensor.data.current_air_time[:, self.body_ids].clamp_max(1.)
            current_contact_time = self.contact_sensor.data.current_contact_time[:, self.body_ids].clamp_max(1.)
            return torch.cat([
                current_air_time,
                current_contact_time,
                (self.forces / self.default_mass_total).reshape(self.num_envs, -1)
            ], dim=-1)
        else:
            return (self.forces / self.default_mass_total).reshape(self.num_envs, -1)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.body_ids],
            self.forces / self.default_mass_total,
            color=(1., 1., 1., 1.)
        )

class motor_params(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.randomized_stiffness = self.env._randomized_stiffness
        self.randomized_damping = self.env._randomized_damping
        self.randomized_strength = self.env._randomized_strength
    
    def __call__(self) -> torch.Tensor:
        stiffness = self.randomized_stiffness
        damping = self.randomized_damping
        return torch.cat([stiffness, damping], dim=-1)


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
    def __init__(self, env, body_names):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

    def __call__(self):
        return self.asset.data.body_materials[:, self.body_ids, :2].reshape(self.num_envs, -1)


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


def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std