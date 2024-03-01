import torch
import abc

from omni.isaac.orbit.assets import Articulation

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

    @abc.abstractmethod
    def __call__(self) ->  torch.Tensor:
        raise NotImplementedError

    def startup(self):
        pass
    
    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor):
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

    def __call__(self):
        body_pos_w = self.asset.data.body_pos_w[:, self.body_indices]
        body_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            body_pos_w - self.asset.data.root_pos_w.unsqueeze(1)
        )
        return body_pos_b.reshape(self.env.num_envs, -1)


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
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum().to(self.env.device) * 9.81
    
    def __call__(self):
        force_history = self.contact_sensor.data.net_forces_w_history[:, :, self.body_indices]
        force_norm = force_history.norm(dim=-1).mean(dim=1)
        return (force_norm / self.default_mass_total).clamp_max(1.)


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


def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std