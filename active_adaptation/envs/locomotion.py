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
from collections import OrderedDict

class LocomotionEnv(Env):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.action_scaling = 0.5

        self.robot = self.scene.articulations["robot"]
        body_masses = self.robot.body_physx_view.get_masses().reshape(self.num_envs, -1)[0]
        for name, mass in zip(self.robot.body_names, body_masses):
            print(name, mass)

        self.contact_sensor: ContactSensor = self.scene.sensors.get("contact_forces", None)
        self.height_scanner: RayCaster = self.scene.sensors.get("height_scanner", None)

        self.init_root_state = self.robot.data.default_root_state.clone()
        self.init_joint_pos = self.robot.data.default_joint_pos.clone()
        self.init_joint_vel = self.robot.data.default_joint_vel.clone()
        
        self.default_joint_pos = self.init_joint_pos.clone()
        # assume all joints are actuated for now
        self.action_dim = self.default_joint_pos.shape[-1]
        
        try:
            from omni_drones.envs.isaac_env import DebugDraw
            self.debug_draw = DebugDraw()
            print("[INFO] Debug Draw API enabled.")
        except ModuleNotFoundError:
            pass
        
        self.lookat_env_i = (
            self.scene._default_env_origins.cpu() 
            - torch.tensor(self.cfg.viewer.lookat)
        ).norm(dim=-1).argmin()

        self.target_base_height = self.cfg.target_base_height
        self.command_manager = CommandManager1(self)

        with torch.device(self.device):
            # self.action_scale = torch.ones(self.num_envs, 1)
            self.action_alpha = torch.ones(self.num_envs, 1)
            self.action_buf = torch.zeros(self.num_envs, self.action_dim, 4)
            self.last_action = torch.zeros(self.num_envs, self.action_dim)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

        self.randomizations = OrderedDict()

        # set by subclass
        self.resample_interval = 300
        self.resample_prob = 0.6
        self.foot_indices: torch.Tensor
        self._feet_pos_b: torch.Tensor
        self.main_body_indices: torch.Tensor
        self.motor_joint_indices: torch.Tensor

    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.init_root_state[env_ids]
        init_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        self.robot.write_root_state_to_sim(
            init_root_state, 
            env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            random_scale(self.init_joint_pos[env_ids], 0.8, 1.2),
            self.init_joint_vel[env_ids],
            env_ids=env_ids
        )
        self.stats[env_ids] = 0.
        self.action_buf[env_ids] = 0.
        self.delay[env_ids] = torch.randint(0, 4, (len(env_ids), 1), device=self.device)

        self.scene.reset(env_ids)
        self.scene.update(dt=self.physics_dt)
        for _, randomization in self.randomizations.items():
            randomization.reset(env_ids)
        
        self.command_manager.reset(env_ids=env_ids)
    
    def _update(self):
        super()._update()
        should_resample = (
            (self.episode_length_buf % self.resample_interval == 0)
            & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        )
        self.command_manager.update(resample=should_resample.nonzero().squeeze(-1))

        feet_pos_w = self.robot.data.body_pos_w[:, self.foot_indices]
        self._feet_pos_b[:] = quat_rotate_inverse(
            self.robot.data.root_quat_w.unsqueeze(1),
            feet_pos_w - self.robot.data.root_pos_w.unsqueeze(1)
        )

        if self.sim.has_gui() and hasattr(self, "debug_draw"):
            self.debug_draw.clear()
            robot_pos = (
                self.robot.data.root_pos_w.cpu()
                + torch.tensor([0., 0., 0.2])
            )
            self.debug_draw.clear()
            command_linvel_w = quat_rotate(
                self.robot.data.root_quat_w,
                self.command_manager._command_linvel
            )
            self.debug_draw.vector(
                robot_pos, 
                command_linvel_w,
                color=(1., 1., 1., 1.)
            )
            self.debug_draw.vector(
                robot_pos,
                self.command_manager._command_heading,
                color=(.2, .2, 1., 1.)
            )
            self.debug_draw.vector(
                robot_pos, 
                self.robot.data.root_lin_vel_w,
                color=(1., .5, .5, 1.)
            )

    def render(self, mode: str = "human"):
        robot_pos = self.robot.data.root_pos_w[self.lookat_env_i].cpu()
        if mode == "rgb_array":
            eye = torch.tensor(self.cfg.viewer.eye) + robot_pos
            lookat = torch.tensor(self.cfg.viewer.lookat) + robot_pos
            self.sim.set_camera_view(eye, lookat)
        return super().render(mode)

    def apply_action(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            # random packet loss: repeat previous actions
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = tensordict["action"]
            action = self.action_buf.take_along_dim(self.delay.unsqueeze(1), dim=-1)
            self.last_action[:] = action.squeeze(-1)

            pos_target = self.last_action * self.action_scaling + self.default_joint_pos
            self.robot.set_joint_position_target(pos_target, self.motor_joint_indices)
        self.robot.write_data_to_sim()

    @observation_func
    def height_scan(self):
        root_pos_w = self.robot.data.root_pos_w
        ray_hits_w = self.height_scanner.data.ray_hits_w
        height_scan = root_pos_w[:, [2]].unsqueeze(1) - ray_hits_w[:, :, [2]]
        return height_scan.reshape(self.num_envs, 1, 11, 17).clamp(-2., 2.)
    
    @observation_func
    def command(self):
        return self.command_manager.command
    
    @observation_func
    def prev_command(self):
        return self.command_manager.command_prev
    
    @observation_func
    def root_quat_w(self):
        return self.robot.data.root_quat_w
    
    @observation_func
    def root_angvel_b(self):
        return self.robot.data.root_ang_vel_b
    
    @observation_func
    def joint_pos(self):
        all_joint_pos = self.robot.data.joint_pos
        return random_noise(all_joint_pos[:, self.motor_joint_indices], 0.05)
    
    @observation_func
    def joint_vel(self):
        all_joint_vel = self.robot.data.joint_vel
        return random_noise(all_joint_vel[:, self.motor_joint_indices], 0.2)
    
    @observation_func
    def joint_acc(self):
        return self.robot.data.joint_acc * self.step_dt

    @observation_func
    def projected_gravity_b(self):
        return self.robot.data.projected_gravity_b
    
    @observation_func
    def root_linvel_b(self):
        return self.robot.data.root_lin_vel_b
    
    @observation_func
    def prev_actions(self):
        return self.last_action.reshape(self.num_envs, -1)
    
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
    def linvel_projection(self):
        linvel_b = self.robot.data.root_lin_vel_b[:, :2]
        command_linvel_b = self.command_manager._command_linvel[:, :2]
        projection = (linvel_b * command_linvel_b).sum(dim=-1, keepdim=True) 
        return projection.clamp_max(self.command_manager._command_speed)
    
    @reward_func
    def linvel_exp(self):
        linvel_b = self.robot.data.root_lin_vel_b[:, :2]
        linvel_error = square_norm(linvel_b - self.command_manager._command_linvel[:, :2])
        return torch.exp( - linvel_error / 0.25)
    
    @reward_func
    def angvel_z_exp(self):
        angvel_error = (self.command_manager.command[:, [2]] - self.robot.data.root_ang_vel_b[:, [2]]).square()
        return torch.exp( - angvel_error / 0.25)

    @reward_func
    def linvel_z_l2(self):
        return -self.robot.data.root_lin_vel_b[:, [2]].square()
    
    @reward_func
    def angvel_xy_l2(self):
        return - self.robot.data.root_ang_vel_b[:, :2].square().sum(-1, True)
    
    @reward_func
    def heading(self):
        root_quat = self.robot.data.root_quat_w
        heading_b_x = quat_rotate_inverse(root_quat, self.command_manager._command_heading)[:, [0]]
        return 0.5 * (heading_b_x + heading_b_x.sign() * heading_b_x.square())
    
    @reward_func
    def base_height(self):
        height = self.robot.data.root_pos_w[:, [2]]
        height = height - self.robot.data.body_pos_w[:, self.foot_indices, 2].mean(1, keepdim=True)
        return (height / self.target_base_height).square().clamp_max(1.)

    @reward_func
    def energy_l2(self):
        energy = (
            (self.robot.data.joint_vel * self.robot.data.applied_torque)
            .square()
            .sum(dim=-1, keepdim=True)
        )
        return - energy
    
    @reward_func
    def energy_l1(self):
        energy = (
            (self.robot.data.joint_vel * self.robot.data.applied_torque)
            .abs()
            .sum(dim=-1, keepdim=True)
        )
        return - energy
    
    @reward_func
    def joint_acc_l2(self):
        return - self.robot.data.joint_acc.square().sum(dim=-1, keepdim=True)
    
    @reward_func
    def joint_torques_l2(self):
        return - self.robot.data.applied_torque.square().sum(dim=-1, keepdim=True)

    @reward_func
    def action_rate_l2(self):
        action_diff = self.action_buf[:, :, 0] - self.action_buf[:, :, 1]
        return - action_diff.square().sum(dim=-1, keepdim=True)

    @reward_func
    def survival(self):
        return torch.ones(self.num_envs, 1, device=self.device)
    
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

