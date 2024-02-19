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

from .mdp import BodyMasses, BodyMaterial, MotorParams, BodyInertias, MotorFailure, CommandManager1
from .locomotion import LocomotionEnv

class Biped(LocomotionEnv):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.action_scaling = 0.5
        self.target_base_height = self.cfg.target_base_height
        
        self.foot_indices, _ = self.robot.find_bodies(".*_toe")
        self.main_body_indices = list(set(range(self.robot.num_bodies)) - set(self.foot_indices))

        self.init_root_state = self.robot.data.default_root_state.clone()
        self.init_joint_pos = self.robot.data.default_joint_pos.clone()
        self.init_joint_vel = self.robot.data.default_joint_vel.clone()
        
        self.leg_joint_indices = self.robot.actuators["legs"].joint_indices
        self.toe_joint_indices = self.robot.actuators["toes"].joint_indices
        self.motor_joint_indices = self.leg_joint_indices + self.toe_joint_indices
        self.default_joint_pos = self.robot.data.default_joint_pos[:, self.motor_joint_indices]

        self.command_manager = CommandManager1(self)

        with torch.device(self.device):
            # self.action_scale = torch.ones(self.num_envs, 1)
            self.action_alpha = torch.ones(self.num_envs, 1)
            self.action_buf = torch.zeros(self.num_envs, 12, 4)
            self.last_action = torch.zeros(self.num_envs, 12)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

        self.randomizations = {
            "body_masses": BodyMasses(self, (0.7, 1.3), body_indices=torch.arange(13)),
            "body_material": BodyMaterial(self, self.foot_indices, (0.6, 2.0), (0.6, 2.0)),
            "motor_params": MotorParams(self, "legs", (0.6, 1.4), (0.6, 1.4), homogeneous=True),
        }
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
        all_joint_pos = self.robot.data.joint_pos
        return random_noise(all_joint_pos[:, self.motor_joint_indices], 0.05)
    
    @observation_func
    def joint_vel(self):
        all_joint_vel = self.robot.data.joint_vel
        return random_noise(all_joint_vel[:, self.motor_joint_indices], 0.2)
    
    @observation_func
    def prev_actions(self):
        return self.action_buf.reshape(self.num_envs, -1)
    
    @observation_func
    def applied_torques(self):
        return self.robot.data.applied_torque / 30.
    
    @observation_func
    def contact_forces(self):
        forces = self.contact_sensor.data.net_forces_w_history[:, :, self.foot_indices].mean(dim=1)
        return (forces * self.step_dt).reshape(self.num_envs, -1)
        forces_norm = forces.norm(dim=-1, keepdim=True)
        return (forces / forces_norm.clamp_min(1e-6) * symlog(forces_norm)).reshape(self.num_envs, -1)
    
    @observation_func
    def contact_indicator(self):
        force_history = self.contact_sensor.data.net_forces_w_history[:, :, self.foot_indices]
        force_norm = force_history.norm(dim=-1)
        return (force_norm.mean(dim=1) > 1.).float()

    @observation_func
    def feet_pos_b(self):
        feet_pos_w = self.robot.data.body_pos_w[:, self.foot_indices]
        self._feet_pos_b = quat_rotate_inverse(
            self.robot.data.root_quat_w.unsqueeze(1),
            feet_pos_w - self.robot.data.root_pos_w.unsqueeze(1)
        )
        return self._feet_pos_b.reshape(self.num_envs, -1)
    
    @observation_func
    def motor_params(self):
        rand: MotorParams = self.randomizations["motor_params"]
        stiffness = rand.randomized_stiffness - 1.
        damping = rand.randomized_damping - 1.
        return torch.cat([damping, stiffness], dim=-1).reshape(self.num_envs, -1)

    @observation_func
    def body_masses(self):
        rand = self.randomizations["body_masses"]
        return rand.randomized_masses.reshape(self.num_envs, -1)
    
    @observation_func
    def body_materials(self):
        rand = self.randomizations["body_material"]
        return rand.material_properties.reshape(self.num_envs, -1)

    @observation_func
    def motor_failure(self):
        rand: MotorFailure = self.randomizations["motor_failure"]
        return rand.motor_failure.reshape(self.num_envs, -1)
    
    @reward_func
    def orientation(self):
        return self.robot.data.projected_gravity_b[:, [2]].square()
    
    @reward_func
    def feet_slip(self):
        i = self.contact_indicator()
        feet_vel = self.robot.data.body_lin_vel_w[:, self.foot_indices]
        return - (i * feet_vel.norm(dim=-1)).sum(dim=1, keepdim=True)
    
    @reward_func
    def stand(self):
        jpos_error = square_norm(self.robot.data.joint_pos - self.robot.data.default_joint_pos)
        cost = - (jpos_error) * self.command_manager._command_stand
        return cost

    @termination_func
    def crash(self):
        fall_over = (self.robot.data.projected_gravity_b[:, 2] >= -0.3)
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

