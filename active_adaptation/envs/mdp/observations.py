import torch
import numpy as np
import abc
import einops
from typing import Tuple, TYPE_CHECKING
from torchvision.utils import make_grid
from torchvision.io import write_jpeg

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sensors import ContactSensor, RayCaster, patterns, RayCasterData, Imu
from omni.isaac.lab.sensors import Camera, TiledCamera
import omni.isaac.lab.sim as sim_utils
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.assets import Quadruped
from omni.isaac.lab.terrains.trimesh.utils import make_plane
from omni.isaac.lab.utils.math import convert_quat, quat_apply, quat_apply_yaw, yaw_quat
from omni.isaac.lab.utils.warp import convert_to_warp_mesh, raycast_mesh
from omni.isaac.lab.utils.string import resolve_matching_names
from pxr import UsdGeom, UsdPhysics

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env

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
    def __init__(self, env, mask_ratio: float=0.):
        """
        For each episode, with probability mask_ratio, the observation will be masked.
        Note that `True` means the observation is masked.

        """
        self.env: Env = env
        self.mask_ratio = mask_ratio
        self.mask = torch.zeros(self.num_envs, device=self.device, dtype=bool)

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError
    
    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def lerp(self, obs_tm1: torch.Tensor, obs_t: torch.Tensor, t) -> torch.Tensor:
        return torch.lerp(obs_tm1, obs_t, t)
    
    def __call__(self) ->  Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.compute()
        if self.mask_ratio > 0.:
            tensor[self.mask] = 0.
        return tensor, self.mask
    
    def startup(self):
        pass
    
    def post_step(self, substep: int):
        pass

    def update(self):
        """Called at each step **after** simulation"""
        pass

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination"""
        if self.mask_ratio > 0.:
            self.mask[env_ids] = torch.rand(env_ids.shape[0], device=self.device) < self.mask_ratio

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled"""
        pass


class BufferedObs(Observation):
    def __init__(self, env, shape, size):
        super().__init__(env)
        self.buffer = Buffer(shape, size, self.env.device)
        setattr(self.env, f"_{self.__class__.__name__}")
    
    def compute(self):
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


class CartesianObs(Observation):

    def __init__(
        self,
        env,
        body_names: str,
        left_bodies: str=None,
        right_bodies: str=None,
        mask_ratio: float=0.
    ):
        super().__init__(env, mask_ratio)
        self.asset: Articulation = self.env.scene["robot"]

        self.body_indices, self.body_names = self.asset.find_bodies(body_names)

        if left_bodies is not None and left_bodies is not False:
            self.left_ids, self.left_names = resolve_matching_names(left_bodies, self.body_names)
            self.right_ids, self.right_names = resolve_matching_names(right_bodies, self.body_names)
        
    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        obs_flipped = obs.reshape(self.num_envs, -1, 3).clone()
        left = obs_flipped[:, self.left_ids]
        right = obs_flipped[:, self.right_ids]
        fliplr = torch.tensor([1., -1., 1.], device=self.device)
        obs_flipped[:, self.left_ids] = right * fliplr
        obs_flipped[:, self.right_ids] = left * fliplr
        return obs_flipped.reshape(self.num_envs, -1)


class root_linacc_b(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.lin_vel_w = self.asset.data.root_lin_vel_w.clone()
        self.lin_acc_w = torch.zeros(self.num_envs, 3, device=self.env.device)

    def update(self):
        lin_vel_w = self.asset.data.root_lin_vel_w
        self.lin_acc_w = (lin_vel_w - self.lin_vel_w) / self.env.step_dt
        self.lin_vel_w = lin_vel_w

    def compute(self):
        lin_acc_b = quat_rotate_inverse(self.asset.data.root_quat_w, self.lin_acc_w)
        return lin_acc_b.reshape(self.num_envs, -1)


class root_linacc_debug(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.lin_vel_w = self.asset.data.root_lin_vel_w.clone()
        self.lin_acc_w = torch.zeros(self.num_envs, 3, device=self.env.device)
        self.root_vel_w_substep = torch.zeros(self.num_envs, self.env.cfg.decimation, 3, device=self.env.device)
        self.body_acc_w_substep = torch.zeros(self.num_envs, self.env.cfg.decimation, 3, device=self.env.device)

    def post_step(self, substep):
        self.root_vel_w_substep[:, substep] = self.asset.data.root_lin_vel_w
        self.body_acc_w_substep[:, substep] = self.asset.data.body_acc_w[:, 0, :3]

    def update(self):
        lin_vel_w = self.asset.data.root_lin_vel_w
        self.lin_acc_w = (lin_vel_w - self.lin_vel_w) / self.env.step_dt
        self.lin_vel_w = lin_vel_w

    def compute(self):
        print(self.asset.data.root_lin_vel_w[0], self.asset.data.body_lin_vel_w[0, 0])

        lin_acc_b0 = quat_rotate_inverse(self.asset.data.root_quat_w, self.lin_acc_w)
        lin_acc_b1 = quat_rotate_inverse(self.asset.data.root_quat_w, self.root_vel_w_substep.diff(dim=1).mean(1) / self.env.physics_dt)
        lin_acc_b2 = quat_rotate_inverse(self.asset.data.root_quat_w, self.asset.data.body_acc_w[:, 0, :3])
        lin_acc_b3 = quat_rotate_inverse(self.asset.data.root_quat_w, self.body_acc_w_substep.mean(1))
        return torch.cat([lin_acc_b0, lin_acc_b1, lin_acc_b2, lin_acc_b3], dim=-1)


# class root_angvel_debug(Observation):
#     def __init__(self, env):
#         super().__init__(env)
#         self.asset: Articulation = self.env.scene["robot"]
#         self.rpy_w = torch.zeros(self.num_envs, 3, device=self.env.device)
#         self.angvel_w = torch.zeros(self.num_envs, 3, device=self.env.device)

#     def update(self):
#         rpy_w = rpy_from_quat(self.asset.data.root_quat_w)
#         self.angvel_w = (rpy_w - self.rpy_w) / self.env.step_dt
#         self.rpy_w = rpy_w


class body_pos(CartesianObs):
    def __init__(
        self,
        env,
        body_names: str,
        left_bodies: str=None,
        right_bodies: str=None,
        yaw_only: bool=False
    ):
        super().__init__(env, body_names, left_bodies, right_bodies)
        self.yaw_only = yaw_only
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
        
    def compute(self):
        return self.body_pos_b.reshape(self.num_envs, -1)


class body_vel(CartesianObs):
    def __init__(
        self,
        env,
        body_names: str,
        left_bodies: str=None,
        right_bodies: str=None,
        yaw_only: bool=False
    ):
        super().__init__(env, body_names, left_bodies, right_bodies)
        self.yaw_only = yaw_only
        print(f"Track body vel for {self.body_names}")
        self.body_vel_b = torch.zeros(self.num_envs, len(self.body_indices), 3, device=self.env.device)

    def update(self):
        if self.yaw_only:
            quat = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            quat = self.asset.data.root_quat_w.unsqueeze(1)
        body_vel_w = self.asset.data.body_lin_vel_w[:, self.body_indices]
        self.body_vel_b[:] = quat_rotate_inverse(quat, body_vel_w)
        
    def compute(self):
        return self.body_vel_b.reshape(self.num_envs, -1)


class body_acc(Observation):
    
    def __init__(self, env, body_names, yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        print(f"Track body acc for {self.body_names}")
        self.body_acc_b = torch.zeros(self.env.num_envs, len(self.body_indices), 3, device=self.env.device)

    def update(self):
        if self.yaw_only:
            quat = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            quat = self.asset.data.root_quat_w.unsqueeze(1)
        body_acc_w = self.asset.data.body_lin_acc_w[:, self.body_indices]
        self.body_acc_b[:] = quat_rotate_inverse(quat, body_acc_w)
        
    def compute(self):
        return self.body_acc_b.reshape(self.env.num_envs, -1)


class imu_acc(Observation):
    def __init__(self, env, smoothing_window: int=3):
        super().__init__(env)
        self.imu: Imu = self.env.scene["imu"]
        self.smoothing_window = smoothing_window
        self.acc_buf = torch.zeros(self.env.num_envs, 3, smoothing_window, device=self.env.device)

    def reset(self, env_ids):
        self.acc_buf[env_ids] = 0.0

    def update(self):
        self.acc_buf[:, :, 1:] = self.acc_buf[:, :, :-1]
        self.acc_buf[:, :, 0] = self.imu.data.lin_acc_b

    def compute(self):
        return self.acc_buf.mean(dim=2).view(self.env.num_envs, -1)
    

class imu_angvel(Observation):
    def __init__(self, env, smoothing_window: int=3):
        super().__init__(env)
        self.imu: Imu = self.env.scene["imu"]
        self.smoothing_window = smoothing_window
        self.angvel_buf = torch.zeros(self.env.num_envs, 3, smoothing_window, device=self.env.device)
    
    def reset(self, env_ids):
        self.angvel_buf[env_ids] = 0.0

    def update(self):
        self.angvel_buf[:, :, 1:] = self.angvel_buf[:, :, :-1]
        self.angvel_buf[:, :, 0] = self.imu.data.ang_vel_b

    def compute(self):
        return self.angvel_buf.mean(dim=2).view(self.env.num_envs, -1)


def observation_func(func):

    class ObsFunc(Observation):
        def __init__(self, env, **params):
            super().__init__(env)
            self.params = params

        def compute(self):
            return func(self.env, **self.params)
    
    return ObsFunc


@observation_func
def root_quat_w(self):
    return self.scene["robot"].data.root_quat_w


class command(Observation):
    def __init__(self, env, mask_ratio: float = 0):
        super().__init__(env, mask_ratio)
        self.command_manager = self.env.command_manager

    def compute(self):
        return self.command_manager.command
    
    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        return self.command_manager.fliplr(obs)


class command_hidden(Observation):
    def __init__(self, env, mask_ratio: float = 0):
        super().__init__(env, mask_ratio)
        self.command_manager = self.env.command_manager
    
    def compute(self):
        return self.command_manager.command_hidden


class joint_pos_target(Observation):
    def __init__(self, env, mask_ratio = 0, subtract_offset: bool=False):
        super().__init__(env, mask_ratio)
        self.subtract_offset = subtract_offset
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self):
        joint_pos_target = self.asset.data.joint_pos_target
        if self.subtract_offset:
            joint_pos_target = joint_pos_target - self.asset.data.default_joint_pos
        return joint_pos_target.reshape(self.num_envs, -1)


class root_angvel_b(Observation):
    def __init__(self, env, noise_std: float=0., yaw_only: bool=False, mask_ratio: float=0.):
        super().__init__(env, mask_ratio=mask_ratio)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.yaw_only = yaw_only
    
    def compute(self) -> torch.Tensor:
        ang_vel_b = random_noise(self.asset.data.root_ang_vel_b, self.noise_std) 
        if self.yaw_only:
            return ang_vel_b[:, 2].reshape(self.num_envs, 1)
        else:
            return ang_vel_b

    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        # assume the robot is symmetric left-right
        return obs * torch.tensor([-1., 1., -1.], device=self.device)


class root_gyro_substep(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        shape = (self.num_envs, self.env.cfg.decimation, 3)
        self.gyro = torch.zeros(shape, device=self.device)

    def post_step(self, substep):
        self.gyro[:, substep] = self.asset.data.root_ang_vel_b
    
    def compute(self):
        return self.gyro

class root_gyro_multistep(Observation):
    def __init__(self, env, steps: int=4, noise_std: float=0.):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.gyro_multistep = torch.zeros((self.num_envs, steps, 3), device=self.device)
    
    def update(self):
        self.gyro_multistep = self.gyro_multistep.roll(1, dims=1)
        self.gyro_multistep[:, 0] = random_noise(self.asset.data.root_ang_vel_b, self.noise_std)
    
    def compute(self):
        return self.gyro_multistep.reshape(self.num_envs, -1)


class projected_gravity_b(Observation):
    def __init__(self, env, noise_std: float=0.):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.init_quat = self.asset.data.root_quat_w.clone()
        self.noise_std = noise_std
    
    def compute(self):
        # projected_gravity_b = quat_rotate_inverse(self.init_quat, self.asset.data.projected_gravity_b)
        projected_gravity_b = self.asset.data.projected_gravity_b
        noise = torch.randn_like(projected_gravity_b).clip(-3., 3.) * self.noise_std
        projected_gravity_b += noise
        return projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)

    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        return obs * torch.tensor([1., -1., 1.], device=self.device)
    
    def lerp(self, obs_tm1, obs_t, t):
        gravity = torch.lerp(obs_tm1, obs_t, t)
        gravity = gravity / gravity.norm(dim=-1, keepdim=True)
        return gravity


class root_linvel_b(Observation):
    def __init__(self, env, body_names: str=None, yaw_only: bool=False, mask_ratio: float=0):
        super().__init__(env, mask_ratio=mask_ratio)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
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
            if self.yaw_only:
                root_quat = yaw_quat(self.asset.data.root_quat_w)
                linvel = quat_rotate_inverse(
                    root_quat,
                    self.asset.data.root_lin_vel_w
                )
            else:
                linvel = self.asset.data.root_lin_vel_b
        else:
            if self.yaw_only:
                root_quat = yaw_quat(self.asset.data.root_quat_w)
            else:
                root_quat = self.asset.data.root_quat_w
            linvel = quat_rotate_inverse(
                root_quat,
                (self.asset.data.body_lin_vel_w[:, self.body_ids] * self.body_masses).sum(1)
            )
        self.linvel[:] = linvel
    
    def compute(self) -> torch.Tensor:
        return self.linvel
    
    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        return obs * torch.tensor([1., -1., 1.], device=self.device)

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
    
class JointObs(Observation):
    def __init__(
        self, 
        env,
        joint_names: str=".*", 
        left_joints = None,
        right_joints = None,
        asym_joints = None,
        mask_ratio: float = 0
    ):
        super().__init__(env, mask_ratio)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        if left_joints is not None:
            self.left_joint_ids, self.left_joint_names = resolve_matching_names(left_joints, self.joint_names)
            self.right_joint_ids, self.right_joint_names = resolve_matching_names(right_joints, self.joint_names)
        else:
            self.left_joint_ids = None
            self.right_joint_ids = None
        self.signs = torch.ones(len(self.joint_ids), device=self.device)
        
        if asym_joints is not None:
            self.asym_joint_ids = resolve_matching_names(asym_joints, self.joint_names)[0]
            self.signs[self.asym_joint_ids] = -1.

    def fliplr(self, obs: torch.Tensor):
        if self.left_joint_ids is None and self.middle_joint_ids is None:
            raise ValueError(f"Flipping is not supported for this {self.__class__.__name__}.")
        obs_flipped = obs.clone()
        if self.left_joint_ids is not None:
            obs_flipped[:, self.left_joint_ids] = obs[:, self.right_joint_ids]
            obs_flipped[:, self.right_joint_ids] = obs[:, self.left_joint_ids]
        return obs_flipped * self.signs


class joint_pos(JointObs):
    def __init__(
        self, 
        env, 
        joint_names: str=".*",
        left_joints = None,
        right_joints = None,
        asym_joints = None,
        subtract_offset: bool=False,
        noise_std: float=0.0,
    ):
        super().__init__(env, joint_names, left_joints, right_joints, asym_joints)
        self.noise_std = noise_std
        self.subtract_offset = subtract_offset
        self.offset = self.asset.data.default_joint_pos[:, self.joint_ids]
        shape = (self.num_envs, 2, self.asset.num_joints)
        self.joint_pos = torch.zeros(shape, device=self.device)
    
    def post_step(self, substep):
        self.joint_pos[:, substep % 2] = self.asset.data.joint_pos[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        joint_pos = self.joint_pos.mean(1)
        if self.subtract_offset:
            joint_pos = joint_pos - self.offset
        return random_noise(joint_pos, self.noise_std)


class joint_pos_substep(Observation):
    def __init__(self, env, mask_ratio = 0):
        super().__init__(env, mask_ratio)
        self.asset: Articulation = self.env.scene["robot"]
        shape = (self.num_envs, self.env.cfg.decimation, self.asset.num_joints)
        self.joint_pos = torch.zeros(shape, device=self.device)
    
    def post_step(self, substep):
        self.joint_pos[:, substep] = self.asset.data.joint_pos
    
    def compute(self):
        return self.joint_pos

class joint_pos_multistep(Observation):
    def __init__(self, env, steps: int=4, noise_std: float=0., diff: bool=False, mask_ratio = 0):
        super().__init__(env, mask_ratio)
        self.steps = steps
        self.noise_std = noise_std
        self.diff = diff
        self.asset: Articulation = self.env.scene["robot"]
        shape = (self.num_envs, steps, self.asset.num_joints)
        self.joint_pos_multistep = torch.zeros(shape, device=self.device)
        self.joint_pos = torch.zeros(self.num_envs, 2, self.asset.num_joints, device=self.device)
    
    def post_step(self, substep):
        self.joint_pos[:, substep % 2] = self.asset.data.joint_pos
    
    def update(self):
        self.joint_pos_multistep = self.joint_pos_multistep.roll(1, 1)
        joint_pos = self.joint_pos.mean(1)
        if self.noise_std > 0:
            joint_pos = random_noise(joint_pos, self.noise_std)
        self.joint_pos_multistep[:, 0] = joint_pos
    
    def compute(self):
        joint_pos = self.joint_pos_multistep.clone()
        if self.diff:
            joint_pos[:, 1:] = joint_pos[:, 1:] - joint_pos[:, :-1]
        return joint_pos.reshape(self.num_envs, -1)


class joint_vel_multistep(Observation):
    def __init__(self, env, steps: int=4, noise_std: float=0., diff: bool=False, mask_ratio = 0):
        super().__init__(env, mask_ratio)
        self.steps = steps
        self.noise_std = noise_std
        self.diff = diff
        self.from_pos = True
        self.asset: Articulation = self.env.scene["robot"]
        shape = (self.num_envs, steps, self.asset.num_joints)
        
        self.joint_vel_multistep = torch.zeros(shape, device=self.device)
        
        if self.from_pos:
            shape = (self.num_envs, self.env.cfg.decimation, self.asset.num_joints)
            self.joint_pos_substep = torch.zeros(shape, device=self.device)
        else:
            shape = (self.num_envs, 2, self.asset.num_joints)
            self.joint_vel_substep = torch.zeros(shape, device=self.device)
    
    def post_step(self, substep):
        if self.from_pos:
            self.joint_pos_substep[:, substep] = self.asset.data.joint_pos
        else:
            self.joint_vel_substep[:, substep % 2] = self.asset.data.joint_vel
    
    def update(self):
        self.joint_vel_multistep = self.joint_vel_multistep.roll(1, 1)
        if self.from_pos:
            joint_vel = self.joint_pos_substep.diff(dim=1).mean(dim=1) / self.env.physics_dt
        else:
            joint_vel = self.joint_vel_substep.mean(dim=1)
        if self.noise_std > 0:
            joint_vel = random_noise(joint_vel, self.noise_std)
        self.joint_vel_multistep[:, 0] = joint_vel
    
    def compute(self):
        joint_vel = self.joint_vel_multistep.clone()
        if self.diff:
            joint_vel[:, 1:] = joint_vel[:, 1:] - joint_vel[:, :-1]
        return joint_vel.reshape(self.num_envs, -1)


class joint_vel_substep(Observation):
    def __init__(self, env, mask_ratio = 0):
        super().__init__(env, mask_ratio)
        self.asset: Articulation = self.env.scene["robot"]
        shape = (self.num_envs, self.env.cfg.decimation, self.asset.num_joints)
        self.joint_vel = torch.zeros(shape, device=self.device)

    def post_step(self, substep):
        self.joint_vel[:, substep] = self.asset.data.joint_vel
    
    def compute(self):
        return self.joint_vel


class joint_pos_des_substep(Observation):
    def __init__(self, env, mask_ratio = 0):
        super().__init__(env, mask_ratio)
        self.asset: Articulation = self.env.scene["robot"]
        shape = (self.num_envs, self.env.cfg.decimation, self.asset.num_joints)
        self.joint_pos_des = torch.zeros(shape, device=self.device)
    
    def post_step(self, substep):
        self.joint_pos_des[:, substep] = self.asset.data.joint_pos_target
    
    def compute(self):
        return self.joint_pos_des


class joint_vel(JointObs):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        left_joints = None,
        right_joints = None,
        asym_joints = None,
        noise_std: float=0.0,
        from_pos: bool=False
    ):
        super().__init__(env, joint_names, left_joints, right_joints, asym_joints)
        self.noise_std = noise_std
        self.from_pos = from_pos

        if self.from_pos:
            shape = (self.num_envs, self.env.cfg.decimation, self.asset.num_joints)
            self.joint_pos_substep = torch.zeros(shape, device=self.device)
        else:
            shape = (self.num_envs, 2, self.asset.num_joints)
            self.joint_vel = torch.zeros(shape, device=self.device)

    def post_step(self, substep):
        if self.from_pos:
            self.joint_pos_substep[:, substep] = self.asset.data.joint_pos
        else:
            self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        if self.from_pos:
            joint_vel = self.joint_pos_substep.diff(dim=1).mean(dim=1) / self.env.physics_dt
        else:
            joint_vel = self.joint_vel.mean(dim=1)
        return random_noise(joint_vel, self.noise_std)


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

    def compute(self) -> torch.Tensor:
        joint_acc = (self.joint_acc_buf * self.smoothing_length).mean(-1)
        joint_acc *= self.env.step_dt
        return joint_acc


class applied_torques(JointObs):
    def __init__(
        self, 
        env,
        actuator_name: str,
        left_joints: str = None,
        right_joints: str = None,
        asym_joints: str = None
    ):
        self.asset: Articulation = env.scene["robot"]
        self.actuator = self.asset.actuators[actuator_name]
        super().__init__(
            env, 
            joint_names=self.actuator.joint_names,
            left_joints=left_joints,
            right_joints=right_joints,
            asym_joints=asym_joints
        )
        
        self.joint_indices = self.actuator.joint_indices
        self.effort_limit = self.actuator.effort_limit.clamp_min(1e-6)
    
    def compute(self) -> torch.Tensor:
        applied_efforts = self.asset.data.applied_torque
        return applied_efforts[:, self.joint_indices]


class contact_indicator(Observation):
    def __init__(self, env, body_names: str, timing: bool=True):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        
        self.artc_ids, names = self.asset.find_bodies(body_names)
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names, preserve_order=True)
        self.timing = timing
        
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum()
        self.giravity = self.default_mass_total.to(self.env.device) * 9.81
        self.force_substep = torch.zeros(self.num_envs, self.env.cfg.decimation, len(self.body_ids), 3, device=self.device)

    def post_step(self, substep):
        force = self.contact_sensor.data.net_forces_w[:, self.body_ids]
        self.force_substep[:, substep] = force

    def compute(self):
        forces = self.force_substep.mean(1).norm(dim=-1)
        forces = torch.where(forces < 2., 0., forces)
        forces = torch.where(forces > 2., 1., forces)
        return forces.reshape(self.num_envs, -1)

    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        if self.timing:
            obs = obs.reshape(self.num_envs, len(self.body_ids), 5)[:, [1, 0, 3, 2]] * torch.tensor([1., 1., 1., -1., 1.], device=obs.device)
        else:
            obs = obs.reshape(self.num_envs, len(self.body_ids), 2)[:, [1, 0, 3, 2]] * torch.tensor([1., 1.], device=obs.device)
        return obs.reshape(self.num_envs, -1)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.artc_ids],
            self.force_substep.mean(1) / self.giravity,
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
    
    def compute(self) -> torch.Tensor:
        stiffness = (self.stiffness / self.defalut_stiffness) - 1.
        damping  = (self.damping / self.default_damping) - 1.
        return torch.cat([stiffness, damping], dim=-1)


class motor_failure(Observation):
    def __init__(self, env, actuator_name: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.motors = self.asset.actuators[actuator_name]
        self.motor_failure = self.motors.motor_failure
    
    def compute(self) -> torch.Tensor:
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

    def compute(self) -> torch.Tensor:
        return self.com_b

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.com_w,
            self.com_vel_w,
            color=(0., 1., 1., 1.)
        )


class external_forces(Observation):
    def __init__(self, env, body_names, divide_by_mass: bool=True, scale: float = 1.0):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.forces_w = torch.zeros(self.env.num_envs, len(self.body_indices) * 3, device=self.device)
        self.forces_b = torch.zeros(self.env.num_envs, len(self.body_indices) * 3, device=self.device)
    
        default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.denom = default_mass_total if divide_by_mass else torch.tensor(scale, device=self.device)

    def update(self):
        forces_b = self.asset._external_force_b[:, self.body_indices]
        forces_w = quat_rotate(self.asset.data.body_quat_w[:, self.body_indices], forces_b)
        forces_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            forces_w
        )
        forces_b /= self.denom
        self.forces_w[:] = forces_w.reshape(self.env.num_envs, -1)
        self.forces_b[:] = forces_b.reshape(self.env.num_envs, -1)

        # print("forces:", self.forces_b.abs().mean(0))

    def compute(self) -> torch.Tensor:
        return self.forces_b

    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        return obs * torch.tensor([1., -1., 1.], device=self.device)


class external_torques(Observation):
    def __init__(self, env, body_names, divide_by_mass: bool=True, scale: float = 0.2):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.torques_b = torch.zeros(self.env.num_envs, len(self.body_indices) * 3, device=self.device)
        default_inertia = self.asset.root_physx_view.get_inertias()[0, 0, [0, 4, 8]].to(self.device)
        self.denom = default_inertia if divide_by_mass else torch.tensor(scale, device=self.device)
    
    def update(self):
        torques_b = self.asset._external_torque_b[:, self.body_indices]
        torques_b = torques_b / self.denom
        self.torques_b[:] = torques_b.reshape(self.env.num_envs, -1)
        # print("torque:", self.torques_b.abs().mean(0))
    
    def compute(self) -> torch.Tensor:
        return self.torques_b

    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        return obs * torch.tensor([1., 1., -1.], device=self.device)

class contact_forces(Observation):
    def __init__(self, env, body_names, divide_by_mass: bool=True):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.denom = self.default_mass_total if divide_by_mass else 1.
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

    def compute(self) -> torch.Tensor:
        contact_forces = self.contact_sensor.data.net_forces_w_history.mean(1)
        force = contact_forces[:, self.body_ids] / self.denom
        return force.view(self.num_envs, -1)


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
        
    def compute(self):
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
    
    def compute(self) -> torch.Tensor:
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
    
    def compute(self) -> torch.Tensor:
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
    
    def compute(self) -> torch.Tensor:
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
        self.mesh = _initialize_warp_meshes("/World/ground", "cuda")

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
            mesh=self.mesh,
        )[0]

        self.feet_height_map[:] = (self.feet_pos_w.unsqueeze(-2)[..., 2] - self.ray_hits_w[..., 2]).nan_to_num(nan=0., posinf=0., neginf=0.)

    def compute(self):
        return self.feet_height_map.reshape(self.num_envs, -1) / self.nominal_height
    
    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.reshape(self.num_envs, self.num_feet, 5)[:, [1, 0, 3, 2]]
        obs = obs[:, :, [0, 2, 1, 4, 3]]
        return obs.reshape(self.num_envs, -1)
    
    def debug_draw(self):
        x = self.ray_hits_w.clone()
        x[..., 2] = self.feet_pos_w.unsqueeze(-2)[..., 2]
        d = self.ray_hits_w - x
        self.env.debug_draw.vector(x, d)


class head_height(Observation):
    def __init__(self, env, mask_ratio: float = 0):
        super().__init__(env, mask_ratio)
        self.asset: Articulation = self.env.scene["robot"]
        self.head_id = self.asset.find_bodies("Head_lower")[0][0]
        
        self.head_height = torch.zeros(self.num_envs, 1, device=self.device)
        self.asset.data.head_height = self.head_height
        self.mesh = _initialize_warp_meshes("/World/ground", "cuda")
        self.ray_direction = torch.tensor([0., 0., -1.], device=self.device).expand(self.num_envs, 3)

    def update(self):
        self.ray_start_w = self.asset.data.body_pos_w[:, self.head_id]
        self.ray_hit_w = raycast_mesh(
            self.ray_start_w,
            self.ray_direction,
            max_dist=100.,
            mesh=self.mesh,
        )[0]
        self.head_height[:] = (self.ray_start_w[:, 2] - self.ray_hit_w[:, 2]).nan_to_num(nan=0., posinf=0., neginf=0.).unsqueeze(1)

    def compute(self):
        return self.head_height.reshape(self.num_envs, -1)

    def debug_draw(self):
        self.env.debug_draw.vector(self.ray_start_w, self.ray_hit_w - self.ray_start_w, color=(1., 0., 1., 1.))


class path_integrator(Observation):
    
    decimation: int = 3

    def __init__(self, env, mask_ratio: float = 0):
        super().__init__(env, mask_ratio)
        self.asset: Articulation = self.env.scene["robot"]
        _initialize_warp_meshes("/World/ground", "cuda")
        with torch.device(self.device):
            self.ray_starts = torch.tensor(
                [
                    [0., 0., 10.], 
                    [0.1, 0.1, 10.],
                    [0.1, -.1, 10.],
                    [-.1, -.1, 10.],
                    [-.1, 0.1, 10.],
                ],
                device=self.device
            )
            self.ray_directions = torch.tensor([0., 0., -1.], device=self.device)
            
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.target_pos_w_hist = torch.zeros(self.num_envs, 3, 40)
            self.pos_w_hist = torch.zeros(self.num_envs, 3, 40)

            self.ray_hits_height = torch.zeros(self.num_envs)
        
        self.num_rays = len(self.ray_starts)
        self.command_manager = self.env.command_manager
        self.step_cnt = 0

    def reset(self, env_ids: torch.Tensor):
        root_pos_w = self.asset.data.root_pos_w[env_ids]
        self.target_pos_w[env_ids] = root_pos_w
        self.target_pos_w_hist[env_ids, :, :] = root_pos_w.unsqueeze(2)
        self.pos_w_hist[env_ids, :, :] = root_pos_w.unsqueeze(2)

    def update(self):
        root_quat = yaw_quat(self.asset.data.root_quat_w)
        command_linvel_b = self.command_manager.command_linvel
        command_linvel_w = quat_rotate(root_quat, command_linvel_b)
        
        ray_starts_w = self.target_pos_w.unsqueeze(1) + self.ray_starts
        ray_hits_w = raycast_mesh(
            ray_starts_w,
            self.ray_directions.expand_as(ray_starts_w).clone(),
            max_dist=100.,
            mesh=RayCaster.meshes["/World/ground"],
        )[0]
        self.ray_hits_height[:] = ray_hits_w[:, :, 2].mean(1)

        self.target_pos_w.add_(command_linvel_w * self.env.step_dt)
        self.target_pos_w[:, 2] = self.ray_hits_height + 0.35
        self.target_pos_w[:, :2].lerp_(self.asset.data.root_pos_w[:, :2], 0.01)
        if self.step_cnt % self.decimation == 0:
            self.target_pos_w_hist[:, :, 1:] = self.target_pos_w_hist[:, :, :-1]
            self.target_pos_w_hist[:, :, 0] = self.target_pos_w
            self.pos_w_hist[:, :, 1:] = self.pos_w_hist[:, :, :-1]
            self.pos_w_hist[:, :, 0] = self.asset.data.root_pos_w
        self.step_cnt += 1
    
    def compute(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, device=self.device)
    
    def debug_draw(self):
        mix = self.pos_w_hist.lerp(self.target_pos_w_hist, 0.5)
        for x in self.target_pos_w_hist.unbind(0):
            self.env.debug_draw.plot(
                x.T,
                color=(0., 1., 1., 1.)
            )
        for x in mix.unbind(0):
            # use purple
            self.env.debug_draw.plot(
                x.T,
                color=(1., 0., 1., 1.)
            )


class height_scan(Observation):
    def __init__(self, env, prim_path, flatten: bool=False, noise_scale = 0.005):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.height_scanner: RayCaster = self.env.scene["height_scanner"]
        self.height_scan = torch.zeros(self.num_envs, 11, 17, device=self.device)
        self.flatten = flatten
        self.noise_scale = noise_scale
        self.asset.data.height_scan = self.height_scan

    def update(self):
        height_scan = (
            self.asset.data.root_pos_w[:, 2].unsqueeze(1)
            - self.height_scanner.data.ray_hits_w[..., 2]
        )
        self.height_scan[:] = height_scan.reshape(self.num_envs, 11, 17)

    def compute(self):
        height_scan = self.height_scan.clamp(-1., 1.)
        if self.noise_scale > 0:
            noise = torch.randn_like(height_scan) * self.noise_scale
            height_scan = height_scan + noise
        if self.flatten:
            return height_scan.reshape(self.num_envs, -1)
        else:
            return height_scan.reshape(self.num_envs, 1, 11, 17)

    # def debug_draw(self):
    #     to = self.height_scan.data.ray_hits_w.reshape(-1, 11, 17, 3)[:, :, 0].reshape(-1, 11, 3)
    #     start = self.root_pos_w.unsqueeze(-2).expand_as(to)
    #     self.env.debug_draw.vector(
    #         start,
    #         to - start,
    #     )

class prev_actions(Observation):
    def __init__(self, env, steps: int=1, flatten: bool=True):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.action_manager = self.env.action_manager
    
    def compute(self):
        action_buf = self.action_manager.action_buf[:, :, :self.steps]
        if self.flatten:
            return action_buf.reshape(self.num_envs, -1)
        else:
            return action_buf

    def fliplr(self, obs: torch.Tensor):
        return self.action_manager.fliplr(obs)


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
    
    def compute(self):
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

    def compute(self) -> torch.Tensor:
        return self.scales


class action_delay(Observation):
    def compute(self) -> torch.Tensor:
        if hasattr(self.env, "delay"):
            return self.env.delay.float()
        else:
            return torch.zeros(self.num_envs, 1, device=self.device)


class rewards(Observation):
    def compute(self) -> torch.Tensor:
        _ = torch.cat([group.rew_buf for group in self.env.reward_groups.values()], dim=-1)
        return _


class incoming_wrench(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = (
            self.asset.root_physx_view.get_masses()[0]
            .sum().to(self.env.device) * 9.81
        )
        self.child_ids = self.asset.find_bodies(".*_hip")[0]

    def update(self):
        self.forces = self.asset.root_physx_view.get_link_incoming_joint_force()
        self.child_forces = self.forces[:, self.child_ids, :3]
        self.child_forces = quat_rotate(self.asset.data.body_quat_w[:, self.child_ids], self.child_forces)
    
    def compute(self) -> torch.Tensor:
        # measured_forces = self.asset.root_physx_view.get_dof_projected_joint_forces()
        self.forces = self.asset.root_physx_view.get_link_incoming_joint_force()
        return (self.forces / self.default_mass_total).reshape(self.num_envs, -1)

    def debug_draw(self):
        self.env.debug_draw.vector(
            # self.asset.data.body_pos_w[:, self.child_ids],
            self.asset.data.root_pos_w,
            self.child_forces.sum(1),
            color=(0., 0., 1., 1.),
            size=10.
        )

class applied_action(JointObs):

    def compute(self) -> torch.Tensor:
        return self.env.action_manager.applied_action

    def fliplr(self, obs: torch.Tensor):
        return self.env.action_manager.fliplr(obs)

class joint_forces(JointObs):

    def compute(self) -> torch.Tensor:
        measured_forces = self.asset.root_physx_view.get_dof_projected_joint_forces()
        return measured_forces[:, self.joint_ids]


class jacobians(Observation):
    def __init__(self, env, body_names: str=".*"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        if self.env.fix_root_link:
            self.body_ids = self.body_ids - 1
    
    def compute(self) -> torch.Tensor:
        jacobian = self.asset.root_physx_view.get_jacobians()[:, self.body_ids]
        return jacobian.reshape(self.num_envs, -1)


class jacobians_b(Observation):
    """The jacobians relative to the root link in body frame. The shape of returned jacobian is (num_envs, num_bodies * 6 * num_joints)"""
    def __init__(self, env, body_names: str, joint_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        if self.env.fix_root_link:
            self.body_ids = self.body_ids - 1
        else:
            self.joint_ids = self.joint_ids + 6
    
    def compute(self) -> torch.Tensor:
        jacobian_all = self.asset.root_physx_view.get_jacobians() # [N, B, 6, J]
        jacobian = jacobian_all[:, self.body_ids.unsqueeze(1), :, self.joint_ids.unsqueeze(0)].permute(2, 0, 3, 1) # [N, b, j, 6]
        root_quat_w = self.asset.data.root_quat_w # [N, 4]
        # [N, b, 6, j] -> [N, b, j, 6] -> [N, b * j * 2, 3] then rotate
        jacobian_b = jacobian.permute(0, 1, 3, 2).reshape(self.num_envs, -1, 3)
        jacobian_b = quat_rotate_inverse(root_quat_w.unsqueeze(1), jacobian_b)

        # # [N, b * j * 2, 3] -> [N, b * j, 6] -> [N, b, j, 6] -> [N, b, 6, j]
        # jacobian_b = jacobian_b.reshape(self.num_envs, len(self.body_ids), -1, 6).permute(0, 1, 3, 2)
        # arm_joint_ids, _ = self.asset.find_joints("arm_joint[1-6]")
        # breakpoint()

        return jacobian_b.reshape(self.num_envs, -1)

class cum_error(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        return self.command_manager._cum_error

    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class clock(Observation):
    def __init__(self, env, frequencies: list[int]=[1, 2, 4]):
        super().__init__(env)
        self.frequencies = torch.as_tensor(frequencies, device=self.device).unsqueeze(0)
    
    def compute(self) -> torch.Tensor:
        t = (self.env.episode_length_buf * self.env.step_dt)
        t = t.reshape(self.num_envs, 1) * self.frequencies
        return torch.cat([t.sin(), t.cos()], dim=1)

class phase(Observation):
    def __init__(self, env, cycle_range = (1.0, 1.2), deriv: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.cycle_range = cycle_range
        self.deriv = deriv
        self.offset_range= [torch.pi/3, 2 * torch.pi/3]
        self.asset.data.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase: torch.Tensor = self.asset.data.phase
        self.omega = torch.zeros(self.num_envs, device=self.device)
        self.offset= torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        offset = torch.zeros(env_ids.shape, device=self.device)
        offset.uniform_(*self.offset_range)
        offset[torch.rand_like(offset) > 0.5] += torch.pi
        cycle = torch.zeros(env_ids.shape, device=self.device)
        cycle.uniform_(*self.cycle_range)

        self.offset[env_ids] = offset
        self.omega[env_ids] = torch.pi * 2 / cycle

    def update(self):
        self.phase[:] = self.offset + self.env.episode_length_buf * self.omega * self.env.step_dt

    def compute(self) -> torch.Tensor:
        phase_sin = self.phase.sin()
        phase_cos = self.phase.cos()
        if self.deriv:
            return torch.stack([
                phase_sin, self.omega * phase_cos,
                phase_cos, -self.omega * phase_sin
            ], 1)
        else:
            return torch.stack([phase_sin, phase_cos], 1)
    
    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        phase_sin = (self.phase + torch.pi).sin()
        phase_cos = (self.phase + torch.pi).cos()
        if self.deriv:
            return torch.stack([
                phase_sin, self.omega * phase_cos,
                phase_cos, -self.omega * phase_sin
            ], 1)
        else:
            return torch.stack([phase_sin, phase_cos], 1)
        
    
class dummy(Observation):
    def __init__(self, env, load_path: str):
        super().__init__(env)
        self.obs: torch.Tensor = torch.load(load_path).to(self.device)
    
    def compute(self) -> torch.Tensor:
        return self.obs.expand(self.num_envs, -1)


class camera(Observation):
    def __init__(self, env, name: str, key: str="depth"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.camera: TiledCamera = self.env.scene[name]
        self.key = key
        self.offset = torch.tensor([1.25, 0.0, 0.75], device=self.device)
        self.frame_count = 0
    
    def update(self):
        self.camera.set_world_poses_from_view(
            eyes=self.asset.data.root_pos_w + self.offset,
            targets=self.asset.data.root_pos_w,
        )

    def compute(self):
        img = self.camera.data.output[self.key]
        img = einops.rearrange(img, "n h w c -> n c h w")
        return img


def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std

meshes = {}

def _initialize_warp_meshes(mesh_prim_path, device):
    if mesh_prim_path in meshes:
        return meshes[mesh_prim_path]

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
    meshes[mesh_prim_path] = wp_mesh
    return wp_mesh


class root_pos_w(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Quadruped = self.env.scene["robot"]

    def compute(self):
        return self.asset.data.root_pos_w

class root_quat_w(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Quadruped = self.env.scene["robot"]

    def compute(self):
        return self.asset.data.root_quat_w

class impact_point_w(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Quadruped = self.env.scene["robot"]

    def compute(self):
        impact_point = self.asset.impact_point_w.reshape(self.num_envs, -1)
        return torch.cat([impact_point, self.asset.impact], dim=1)

class feet_orientation(Observation):
    def __init__(self, env, feet_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id = self.asset.find_bodies(feet_names)[0]
        self.heading_feet = torch.tensor([[[1., 0., 0.]]], device=self.device)
    
    def compute(self):
        self.quat_feet = yaw_quat(self.asset.data.body_quat_w[:, self.feet_id])
        feet_fwd = quat_rotate(self.quat_feet, self.heading_feet)
        return feet_fwd.reshape(self.num_envs, -1)


class symmetry_quad(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.flip_y = torch.tensor([1., -1., 1.], device=self.device)
        self.flip_ry = torch.tensor([-1., 1., -1.], device=self.device)
    
    def compute(self):
        jpos = self.asset.data.joint_pos
        jvel = self.asset.data.joint_vel
        
        gravity = self.asset.data.projected_gravity_b
        linvel = self.asset.data.root_lin_vel_b
        angvel = self.asset.data.root_ang_vel_b

        left = torch.cat([jpos, linvel , gravity], dim=1)
        right = torch.cat([self.mirror(jpos), linvel * self.flip_y, gravity * self.flip_y], dim=1)
        return torch.stack([left, right], dim=1)
    
    def mirror(self, jnt: torch.Tensor):
        return jnt.reshape(self.num_envs, 4, 3)[:, [1, 0, 3, 2]].reshape(self.num_envs, -1)


class oscillator(Observation):
    
    def __init__(self, env, history: bool=False,mask_ratio = 0):
        super().__init__(env, mask_ratio)
        self.history = history
        self.asset: Quadruped = self.env.scene["robot"]        
        self.phi_history = torch.zeros(self.num_envs, 4, 4, device=self.device)

    def update(self):
        if self.history:
            self.phi_history = self.phi_history.roll(1, dims=1)
            self.phi_history[:, 0] = self.asset.phi

    def compute(self):
        if self.history:
            phi_sin = self.phi_history.sin().reshape(self.num_envs, -1)
            phi_cos = self.phi_history.cos().reshape(self.num_envs, -1)
        else:
            phi_sin = self.asset.phi.sin()
            phi_cos = self.asset.phi.cos()
        obs = torch.concat([phi_sin, phi_cos, self.asset.phi_dot], dim=-1)
        return obs.reshape(self.num_envs, -1)


class feet_contact_multistep(Observation):
    def __init__(self, env, steps: int=4, thres: float=1.):
        super().__init__(env)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_sensor"]
        self.feet_id = self.asset.find_bodies(".*_foot")[0]
        self.contact = torch.zeros(self.num_envs, steps, device=self.device, dtype=bool)
        self.grf_substep = torch.zeros(self.num_envs, self.env.cfg.decimation, device=self.device)
    
    def post_step(self, substep):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.feet_id]
        self.grf_substep[:, substep] = contact_forces.norm(dim=-1)
    
    def update(self):
        self.contact = self.contact.roll(1, dims=1) 
        self.contact[:, 0] = self.grf_substep.mean(dim=1) > self.thres
    
    def compute(self):
        return self.contact.reshape(self.num_envs, -1)


class actuator_type(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.actuator = self.asset.actuators["base_legs"]

    def compute(self):
        return self.actuator.implicit.reshape(self.num_envs, -1)