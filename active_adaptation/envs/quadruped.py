import torch

from omni.isaac.orbit.sensors import ContactSensor, RayCaster
from omni.isaac.orbit.actuators import DCMotor
from active_adaptation.envs.base import Env, observation_func, reward_func, termination_func
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

from tensordict.tensordict import TensorDictBase
from torchrl.data import (
    CompositeSpec, 
    UnboundedContinuousTensorSpec
)

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

from .mdp import BodyMasses, BodyMaterial, MotorParams, BodyInertias, MotorFailure, CommandManager1, BodyComs
from collections import OrderedDict
from .locomotion import LocomotionEnv

class Quadruped(LocomotionEnv):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.action_scaling = 0.5
        self.target_base_height = self.cfg.target_base_height
        
        self.foot_indices, _ = self.robot.find_bodies(".*_foot")
        self.calf_indices, _ = self.robot.find_bodies(".*_calf")
        self.thigh_indices, _ = self.robot.find_bodies(".*_thigh")
        self.main_body_indices = list(
            set(range(self.robot.num_bodies)) 
            - set(self.calf_indices)
            # - set(self.thigh_indices)
            - set(self.foot_indices)
        )

        self.init_root_state = self.robot.data.default_root_state.clone()
        self.init_joint_pos = self.robot.data.default_joint_pos.clone()
        self.init_joint_vel = self.robot.data.default_joint_vel.clone()
        
        self.motor_joint_indices = self.robot.actuators["base_legs"].joint_indices
        self.default_joint_pos = self.robot.data.default_joint_pos[:, self.motor_joint_indices]

        self._feet_pos_b = self.robot.data.body_pos_w[:, self.foot_indices].clone()

        self.command_manager = CommandManager1(self, speed_range=(0.5, 2.0))

        self.randomizations = OrderedDict({
            "body_masses": BodyMasses(self, (0.7, 1.3), body_indices=torch.arange(19)),
            "body_coms": BodyComs(self, (-0.1, 0.1), body_indices=torch.tensor([0])),
            "body_inertias": BodyInertias(self, (0.7, 1.3), body_indices=torch.tensor([0])),
            # "payload_mass": BodyMasses(self, (0.01, 4.), body_indices=torch.tensor([19])),
            # "payload_inertia": BodyInertias(self, (0.01, 4.0), body_indices=torch.tensor([19])),
            "body_material": BodyMaterial(self, self.foot_indices, (0.6, 1.0), (0.6, 1.0)),
            "motor_params": MotorParams(self, "base_legs", (0.7, 1.3), (0.6, 1.4)),
            "motor_failure": MotorFailure(self, [8, 9, 10, 11], failure_prob=0.0),
        })
        # self.randomizations = OrderedDict({
        #     "body_masses": BodyMasses(self, (0.7, 1.3), body_indices=torch.arange(19)),
        #     "payload_mass": BodyMasses(self, (0.01, 4.), body_indices=torch.tensor([19])),
        #     "payload_inertia": BodyInertias(self, (0.01, 4.0), body_indices=torch.tensor([19])),
        #     "body_material": BodyMaterial(self, self.foot_indices, (0.6, 2.0), (0.6, 2.0)),
        #     "motor_params": MotorParams(self, "base_legs", (0.7, 1.3), (0.6, 1.4)),
        #     "motor_failure": MotorFailure(self, [8, 9, 10, 11], failure_prob=1.0),
        # })
        for _, randomization in self.randomizations.items():
            randomization.startup()
        self.sim.physics_sim_view.flush()

        obs = super()._compute_observation()
        reward = self._compute_reward()

        self.action_spec = CompositeSpec(
            {
                "action": UnboundedContinuousTensorSpec((self.num_envs, 12))
            }, 
            shape=[self.num_envs]
        ).to(self.device)

        for key, value in obs.items(True, True):
            if key not in self.observation_spec.keys(True, True):
                self.observation_spec[key] = UnboundedContinuousTensorSpec(value.shape, device=self.device)
        
        self.reward_spec = CompositeSpec(
            {
                key: UnboundedContinuousTensorSpec(value.shape)
                for key, value in reward.items()
            },
            shape=[self.num_envs]
        ).to(self.device)

        self.packet_loss = 0.0

        self.resample_interval = 300
        self.resample_prob = 0.6
    
    @observation_func
    def joint_pos(self):
        joint_pos = self.robot.data.joint_pos[:, self.motor_joint_indices]
        return random_noise(joint_pos - self.default_joint_pos, 0.05)
    
    @observation_func
    def joint_vel(self):
        all_joint_vel = self.robot.data.joint_vel
        return all_joint_vel[:, self.motor_joint_indices]
    
    @observation_func
    def applied_torques(self):
        return self.robot.data.applied_torque / 30.
    
    @observation_func
    def motor_params(self):
        rand: MotorParams = self.randomizations["motor_params"]
        stiffness = rand.randomized_stiffness - 1.
        damping = rand.randomized_damping - 1.
        return torch.cat([damping, stiffness], dim=-1).reshape(self.num_envs, -1)

    @observation_func
    def body_masses(self):
        rand = self.randomizations["body_masses"]
        return rand.randomized_masses.reshape(self.num_envs, -1)[:, [0]]

    @observation_func
    def payload_mass(self):
        rand = self.randomizations["payload_mass"]
        return rand.randomized_masses.reshape(self.num_envs, -1)
    
    @observation_func
    def payload_inertia(self):
        rand = self.randomizations["payload_inertia"]
        inertia = rand.randomized_inertias.reshape(self.num_envs, -1)[:, [0, 4, 8]]
        return inertia
    
    @observation_func
    def body_materials(self):
        rand = self.randomizations["body_material"]
        return rand.material_properties.reshape(self.num_envs, -1)

    @observation_func
    def motor_failure(self):
        rand: MotorFailure = self.randomizations["motor_failure"]
        return rand.motor_failure.reshape(self.num_envs, -1)
    
    @observation_func
    def incoming_wrench(self):
        link_incoming_forces = self.robot.root_physx_view.get_link_incoming_joint_force()
        link_incoming_forces[:, :, :3] *= 0.01
        return link_incoming_forces.reshape(self.num_envs, -1)
    
    @observation_func
    def body_inertias(self):
        rand: BodyInertias = self.randomizations["body_inertias"]
        return rand.randomized_inertias.reshape(self.num_envs, -1)
    
    @observation_func
    def body_coms(self):
        rand: BodyComs = self.randomizations["body_coms"]
        return rand.randomized_coms.reshape(self.num_envs, -1)
    
    
    @reward_func
    def base_height(self):
        height = self.robot.data.root_pos_w[:, [2]]
        height = height - self.robot.data.body_pos_w[:, self.foot_indices, 2].mean(1, keepdim=True)
        return (height / self.target_base_height).square().clamp_max(0.8)
    
    @reward_func
    def stand(self):
        jpos_error = square_norm(self.robot.data.joint_pos - self.robot.data.default_joint_pos)
        front_symmetry = self._feet_pos_b[:, [0, 1], 1].sum(dim=1, keepdim=True).abs()
        back_symmetry = self._feet_pos_b[:, [2, 3], 1].sum(dim=1, keepdim=True).abs()
        cost = - (jpos_error + front_symmetry + back_symmetry) 
        return cost * self.command_manager.is_standing_env.reshape(self.num_envs, 1)

    @reward_func
    def feet_air_time(self):
        last_air_time = self.contact_sensor.data.last_air_time[:, self.foot_indices]
        first_contact = last_air_time > 0.0
        reward = torch.sum((last_air_time - 0.5) * first_contact, dim=1)
        reward *= (self.command_manager.command[:, :2].norm(dim=-1)>0.1)
        return reward.reshape(self.num_envs, -1)

    @reward_func
    def undesired_contact(self):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.calf_indices]
        return - (contact_forces.norm(dim=-1) > 1.).sum(dim=1, keepdim=True).float()
    
    @termination_func
    def crash(self):
        fall_over = (self.robot.data.projected_gravity_b[:, 2] >= -0.1)
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.main_body_indices]
        undesired_contact = (contact_forces.norm(dim=-1) > 1.).any(dim=1)
        terminated = (fall_over | undesired_contact).unsqueeze(1)
        return terminated


def random_scale(x: torch.Tensor, low: float, high: float):
    return x * (torch.rand_like(x) * (high - low) + low)

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def square_norm(x: torch.Tensor):
    return x.square().sum(dim=-1, keepdim=True)

def noarmalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(dim=-1, keepdim=True)

def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def flip_lr(joints: torch.Tensor):
    return joints.reshape(-1, 3, 2, 2).flip(-1).reshape(-1, 12)

def flip_fb(joints: torch.Tensor):
    return joints.reshape(-1, 3, 2, 2).flip(-2).reshape(-1, 12)

