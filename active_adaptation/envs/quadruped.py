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

from .mdp import BodyMasses, BodyMaterial, MotorParams, BodyInertias, MotorFailure, CommandManager
from collections import OrderedDict

class Quadruped(Env):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.permute_actions = False
        self.action_scaling = 0.5

        self.robot = self.scene.articulations["robot"]
        body_masses = self.robot.body_physx_view.get_masses().reshape(self.num_envs, -1)[0]
        for name, mass in zip(self.robot.body_names, body_masses):
            print(name, mass)
        self.foot_indices, _ = self.robot.find_bodies(".*_foot")
        self.calf_indices, _ = self.robot.find_bodies(".*_calf")
        self.thigh_indices, _ = self.robot.find_bodies(".*_thigh")
        self.main_body_indices = list(set(range(self.robot.num_bodies)) - set(self.calf_indices) - set(self.foot_indices))

        self.contact_sensor: ContactSensor = self.scene.sensors.get("contact_forces", None)
        self.height_scanner: RayCaster = self.scene.sensors.get("height_scanner", None)

        self.init_root_state = self.robot.data.default_root_state.clone()
        self.init_joint_pos = self.robot.data.default_joint_pos.clone()
        self.init_joint_vel = self.robot.data.default_joint_vel.clone()
        
        self.motor_joint_indices = self.robot.actuators["base_legs"].joint_indices
        self.default_joint_pos = self.robot.data.default_joint_pos[:, self.motor_joint_indices]
        
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
        self.command_manager = CommandManager(self)

        with torch.device(self.device):
            # self.action_scale = torch.ones(self.num_envs, 1)
            self.action_alpha = torch.ones(self.num_envs, 1)
            self._actions_t = torch.zeros(self.num_envs, 12)
            self._actions_tm1 = torch.zeros_like(self._actions_t)
            self._actions_tm2 = torch.zeros_like(self._actions_t)
            if self.permute_actions:
                self._flip_lr = torch.randn(self.num_envs, 1) > 0.
                self._flip_fb = torch.randn(self.num_envs, 1) > 0.

        self.randomizations = OrderedDict({
            "body_masses": BodyMasses(self, (0.7, 1.3), body_indices=torch.arange(19)),
            "payload_mass": BodyMasses(self, (0.01, 4.), body_indices=torch.tensor([19])),
            "payload_inertia": BodyInertias(self, (0.01, 4.0), body_indices=torch.tensor([19])),
            "body_material": BodyMaterial(self, self.foot_indices, (0.6, 2.0), (0.6, 2.0)),
            "motor_params": MotorParams(self, "base_legs", (0.7, 1.3), (0.6, 1.4)),
            "motor_failure": MotorFailure(self, [8, 9, 10, 11], failure_prob=0.2),
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

    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.init_root_state[env_ids]
        init_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        self.robot.write_root_state_to_sim(
            init_root_state, 
            env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            # random_scale(self.init_joint_pos[env_ids], 0.8, 1.2),
            self.init_joint_pos[env_ids],
            self.init_joint_vel[env_ids],
            env_ids=env_ids
        )
        self.stats[env_ids] = 0.
        self._actions_t[env_ids] = 0.
        self._actions_tm1[env_ids] = 0.
        self._actions_tm2[env_ids] = 0.

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
        
        if self.sim.has_gui() and hasattr(self, "debug_draw"):
            self.debug_draw.clear()
            robot_pos = (
                self.robot.data.root_pos_w.cpu()
                + torch.tensor([0., 0., 0.2])
            )
            self.debug_draw.clear()
            self.debug_draw.vector(
                robot_pos, 
                self.command_manager._command_linvel,                
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


    def apply_action(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            # random packet loss: repeat previous actions
            actions = tensordict["action"]

            self._actions_tm2[:] = self._actions_tm1
            self._actions_tm1[:] = self._actions_t
            # self._actions_t.lerp_(actions, self.action_alpha)

            if self.permute_actions:
                actions = torch.where(self._flip_lr, flip_lr(actions), actions)
                actions = torch.where(self._flip_fb, flip_fb(actions), actions)

            pos_target = actions * self.action_scaling + self.default_joint_pos
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
        quat_w = self.robot.data.root_quat_w
        command_linvel = quat_rotate_inverse(quat_w, self.command_manager._command_linvel)
        command_heading = quat_rotate_inverse(quat_w, self.command_manager._command_heading)
        return torch.cat([command_linvel, command_heading], dim=-1)
    
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
        return all_joint_vel[:, self.motor_joint_indices]
    
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
        return self._actions_t
    
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
    def action_flip(self):
        return self._flip_lr.long() * 2 + self._flip_fb.long()
    
    @observation_func
    def flip_lr(self):
        return self._flip_lr.float()
    
    @observation_func
    def flip_fb(self):
        return self._flip_fb.float()

    @observation_func
    def motor_failure(self):
        rand: MotorFailure = self.randomizations["motor_failure"]
        return rand.motor_failure.reshape(self.num_envs, -1)
    
    @reward_func
    def linvel_projection(self):
        linvel_w = self.robot.data.root_lin_vel_w
        return (
            (linvel_w * self.command_manager._command_linvel)
            .sum(dim=1, keepdim=True)
            .clamp_max(self.command_manager._command_speed)
        )
    
    @reward_func
    def linvel_exp(self):
        linvel_w = self.robot.data.root_lin_vel_w
        linvel_error = square_norm(linvel_w - self.command_manager._command_linvel)
        return 1. / (1. + linvel_error / 0.25)
    
    @reward_func
    def linvel_z_l2(self):
        return -self.robot.data.root_lin_vel_b[:, [2]].square()
    
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
        return - (self._actions_t - self._actions_tm1).square().sum(dim=-1, keepdim=True)

    @reward_func
    def action_rate2_l2(self):
        return - (
            (self._actions_t - self._actions_tm1 - self._actions_tm1 + self._actions_tm2)
            .square()
            .sum(dim=-1, keepdim=True)
        )

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
        front_symmetry = self._feet_pos_b[:, [0, 1], 1].sum(dim=1, keepdim=True).abs()
        back_symmetry = self._feet_pos_b[:, [2, 3], 1].sum(dim=1, keepdim=True).abs()
        cost = - (jpos_error + front_symmetry + back_symmetry) * self.command_manager._command_stand
        return cost

    @reward_func
    def feet_air_time(self):
        return (self.contact_sensor.data.current_air_time[:, self.foot_indices] > 0.1).sum(-1, True)

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

