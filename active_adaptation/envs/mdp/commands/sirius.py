import torch
import math

from tensordict import TensorClass
from typing import TYPE_CHECKING
from active_adaptation.utils.math import (
    euler_from_quat,
    quat_from_euler_xyz,
    wrap_to_pi,
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat
)
from active_adaptation.envs.mdp import reward, termination
from active_adaptation.envs.mdp.base import Observation, Reward
from active_adaptation.utils.symmetry import SymmetryTransform
if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor

from .base import Command


JUMP_PREPARE_TIME = 0.3
JUMP_LANDING_TIME = 0.3

class SiriusCommand(TensorClass):
    cmd_lin_vel: torch.Tensor # body frame
    cmd_ang_vel: torch.Tensor
    cmd_rpy: torch.Tensor
    yaw_stiffness: torch.Tensor

    des_height: torch.Tensor # [N, 2], fore and hind hip height
    des_contact: torch.Tensor # [N, 4]
    des_rpy: torch.Tensor
    des_stand_vel: torch.Tensor
    
    time: torch.Tensor # [N, 1]
    duration: torch.Tensor # [N, 1]
    mode: torch.Tensor # [N]

    @property
    def phase(self):
        return self.time / self.duration
    
    @property
    def is_jumping(self):
        return (
            (self.mode[:, None] == 2) 
            & (self.time > JUMP_PREPARE_TIME) 
            & (self.time < self.duration - JUMP_LANDING_TIME)
        )
    
    @classmethod
    def zero(cls, size: int, device: str):
        return cls(
            cmd_lin_vel=torch.zeros(size, 3, device=device),
            cmd_ang_vel=torch.zeros(size, 3, device=device),
            cmd_rpy=torch.zeros(size, 3, device=device),
            des_rpy=torch.zeros(size, 3, device=device),
            des_height=torch.zeros(size, 2, device=device),
            des_contact=torch.zeros(size, 4, device=device),
            des_stand_vel=torch.zeros(size, 1, device=device),
            yaw_stiffness=torch.zeros(size, 1, device=device),
            time=torch.zeros(size, 1, device=device),
            duration=torch.full((size, 1), torch.inf, device=device),
            mode=torch.zeros(size, dtype=int, device=device),
        )
        


class SiriusCommandManager(Command):
        
    CMD_WALK = 0
    CMD_STAND = 1
    CMD_JUMP = 2
    CMD_FLIP = 3
    
    WHEEL_RADIUS = 0.0825

    def __init__(
        self,
        env,
        lin_vel_x_range,
        lin_vel_y_range,
        ang_vel_z_range = (-2.0, 2.0),
        transitions = None,
        teleop = False,
    ):
        super().__init__(env, teleop)
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_z_range = ang_vel_z_range

        # virtual spring model for standing
        self.stand_kp = 12.
        self.stand_kd = 2.0 * math.sqrt(self.stand_kp) * 0.8

        self.wheel_joint_ids = self.asset.find_joints(".*_WHEEL")[0]
        self.leg_joint_ids = self.asset.find_joints(".*(HAA|HFE)")[0]
        self.thigh_joint_ids = self.asset.find_joints(".*HFE")[0]
        self.fore_hip_ids = self.asset.find_bodies("[L,R]F_hip")[0]
        self.hind_hip_ids = self.asset.find_bodies("[L,R]H_hip")[0]
        
        self.pitch_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.roll_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.lin_vel_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.ang_vel_z_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.ang_vel_x_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        
        self.front_height_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.back_height_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.stand_height_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        
        self.target_jump_lin_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self.rew_jump_inertia = torch.zeros(self.num_envs, 1, device=self.device)
        self.rew_stand_lin_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self.rew_stand_height = torch.zeros(self.num_envs, 1, device=self.device)

        self.target_base_height = torch.zeros(self.num_envs, 1, device=self.device)
        self.default_mass = self.asset.data.default_mass.to(self.device)

        self._stand_height = self.asset.data.root_pos_w[:, 2:3]
        self._cum_error = torch.zeros(self.num_envs, 1, device=self.device)
        
        self._command = SiriusCommand.zero(self.num_envs, self.device)
        with torch.device(self.device):
            if transitions is None:
                self.transition = torch.eye(4) 
                self.transition[self.CMD_WALK]  = torch.tensor([.2, .8, 0., .0]) # normal to others
                self.transition[self.CMD_STAND] = torch.tensor([1., 0., 0., 0.]) # stand to others
                self.transition[self.CMD_JUMP]  = torch.tensor([1., 0., 0., 0.]) # jump to others
                self.transition[self.CMD_FLIP]  = torch.tensor([1., 0., 0., 0.]) # flip to others
            else:
                self.transition = torch.as_tensor(transitions)
            self.transition /= self.transition.sum(1, keepdim=True)
            
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            from isaaclab.markers import RED_ARROW_X_MARKER_CFG, VisualizationMarkers
            self.frame_marker = VisualizationMarkers(
                RED_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/frame",
                )
            )
            self.frame_marker.set_visibility(True)
        elif self.env.sim.has_gui() and self.env.backend == "mujoco":
            self.arrow_marker_0 = self.env.scene.create_arrow_marker(radius=0.02, rgba=[1, 0, 0, 0.8])
            self.arrow_marker_1 = self.env.scene.create_arrow_marker(radius=0.02, rgba=[0, 0, 1, 0.8])
    
    # @reward
    # def jump_lin_vel(self):
    #     is_active = (self._command.mode==self.CMD_JUMP).unsqueeze(1)
    #     front_lin_vel = self.asset.data.body_lin_vel_w[:, [self.front_body_id, self.back_body_id], 2:3]
    #     rew = front_lin_vel.clamp_max(self.target_jump_lin_vel.unsqueeze(1))
    #     rew = (rew + rew.square()).mean(1) * (self._command.phase < 0.5)
    #     return rew, is_active
    
    # @reward
    # def jump_height(self):
    #     is_active = (self._command.mode==self.CMD_JUMP).unsqueeze(1)
    #     rew = torch.exp(-self.front_height_error_l2 / 0.1) + torch.exp(-self.back_height_error_l2 / 0.1)
    #     return rew, is_active

    # @reward
    # def flip_ang_vel(self):
    #     is_active = (self._command.mode==self.CMD_FLIP).unsqueeze(1)
    #     cmd_ang_vel_roll = self._command.cmd_ang_vel[:, 0:1]
    #     ang_vel_roll = self.asset.data.root_ang_vel_b[:, 0:1]
    #     rew_jump_ang_vel = torch.clamp(
    #         ang_vel_roll * cmd_ang_vel_roll.sign(),
    #         torch.zeros_like(ang_vel_roll),
    #         cmd_ang_vel_roll.abs()
    #     )
    #     return rew_jump_ang_vel, is_active
    
    # @reward
    # def flip_roll(self):
    #     is_active = (self._command.mode==self.CMD_FLIP).unsqueeze(1)
    #     return torch.exp( -self.roll_error_l2 / 0.25), is_active
    
    # @reward
    # def walk_angvel_xy_l2(self):
    #     rew = -self.asset.data.root_ang_vel_b[:, :2].square().sum(1, True)
    #     is_active = (self._command.mode==self.CMD_WALK).unsqueeze(1)
    #     return rew, is_active

    # @reward
    # def walk_linvel_z_l2(self):
    #     rew = -self.asset.data.root_lin_vel_b[:, 2:3].square()
    #     is_active = (self._command.mode==self.CMD_WALK).unsqueeze(1)
    #     return rew, is_active
    
    # @reward
    # def walk_joint_devi_l2(self):
    #     diff = self.asset.data.joint_pos - self.asset.data.default_joint_pos
    #     rew = - (diff[:, self.hip_joint_ids]).square().sum(1, True)
    #     is_active = (self._command.mode==self.CMD_WALK).unsqueeze(1)
    #     return rew, is_active

    @termination
    def stand_error_exceeds(self):
        return (self._command.mode == self.CMD_STAND).unsqueeze(1) & (self.stand_height_error_l2 > 0.2)
    
    @termination
    def flip_error_exceeds(self):
        flip_mode = (self._command.mode == self.CMD_FLIP).unsqueeze(1)
        return  flip_mode & ((self.roll_error_l2 > 0.5) | (self.asset.data.root_pos_w[:, 2:3] < 0.3))

    @termination
    def jump_error_exceeds(self):
        error = (self.front_height_error_l2 > 0.12) | (self.back_height_error_l2 > 0.12)
        return (self._command.mode == self.CMD_JUMP).unsqueeze(1) & error
    
    @property
    def command(self):
        # only episodic commands (jump and flip) have phase
        jump_mode = (self._command.mode == self.CMD_JUMP).reshape(-1, 1)
        result = torch.cat([
            self._command.cmd_lin_vel[:, :2],
            self._command.cmd_ang_vel,
            self._command.cmd_rpy[:, :2],
            torch.where(jump_mode, self._command.time, torch.zeros_like(self._command.time)),
            torch.where(jump_mode, self._command.duration - self._command.time, torch.zeros_like(self._command.time)),
            torch.nn.functional.one_hot(self._command.mode, num_classes=4),
            self._command.des_contact,
        ], dim=-1)
        return result
    
    @property
    def des_height(self):
        return self._command.des_height + self.env.get_ground_height_at(self.asset.data.root_pos_w).unsqueeze(1)

    def symmetry_transforms(self):
        return SymmetryTransform.cat([
            SymmetryTransform(perm=torch.arange(2), signs=torch.tensor([1, -1])), # flip y
            SymmetryTransform(perm=torch.arange(3), signs=torch.tensor([-1, 1, -1])), # flip roll and yaw
            SymmetryTransform(perm=torch.arange(2), signs=torch.tensor([-1, 1])), # flip roll
            SymmetryTransform(perm=torch.arange(6), signs=torch.ones(6)), # do nothing
            SymmetryTransform(perm=torch.tensor([2, 3, 0, 1]), signs=torch.ones(4))
        ])
    
    def reset(self, env_ids):
        command = self.sample_command_normal(len(env_ids))
        self._command[env_ids] = command
        self._cum_error[env_ids] = 0.
        self.target_base_height[env_ids] = 0.4

    def update(self):
        if self.teleop:
            # print(self.key_pressed)
            self._command.mode[:] = self.CMD_WALK
            self._command.des_contact[:] = torch.zeros(4, device=self.device)
            self._command.cmd_lin_vel[:, 0] = self.key_pressed["W"] - self.key_pressed["S"]
            self._command.cmd_lin_vel[:, 1] = self.key_pressed["A"] - self.key_pressed["D"]
            self._command.cmd_ang_vel[:, 2] = self.key_pressed["LEFT"] - self.key_pressed["RIGHT"]
            return
        r, p, y = euler_from_quat(self.asset.data.root_quat_w).unbind(-1)
        self.pitch_error_l2 = torch.square( wrap_to_pi(self._command.cmd_rpy[:, 1:2] - p.unsqueeze(1)) )

        quat_yaw = yaw_quat(self.asset.data.root_quat_w)
        
        self._cmd_lin_vel_w = quat_rotate(quat_yaw, self._command.cmd_lin_vel)
        self.lin_vel_error_l2 = torch.square(
            self._cmd_lin_vel_w[:, :2] - self.asset.data.root_lin_vel_w[:, :2] ).sum(1, keepdim=True)
        
        self.ang_vel_z_error_l2 = torch.square(
            self._command.cmd_ang_vel[:, 2:3] - self.asset.data.root_ang_vel_w[:, 2:3])

        self.roll_error_l2 = (
            self.asset.data.projected_gravity_b[:, 0:1].square()
            + (self._command.cmd_rpy[:, 0:1].sin() + self.asset.data.projected_gravity_b[:, 1:2]).square()
            + (self._command.cmd_rpy[:, 0:1].cos() + self.asset.data.projected_gravity_b[:, 2:3]).square()
        )        
        
        cmd_yaw_diff = (self._command.mode == self.CMD_WALK) \
            .mul(wrap_to_pi(self._command.des_rpy[:, 2] - self.asset.data.heading_w)) \
            .unsqueeze(1)

        self._command.time.add_(self.env.step_dt)

        # update jump command
        is_jumping = self._command.is_jumping
        self._command.des_contact[:] = torch.where(
            self._command.mode[:, None] == self.CMD_JUMP,
            torch.where(
                is_jumping,
                torch.tensor([-1., -1., -1., -1.], device=self.device),
                torch.zeros(4, device=self.device),
            ),
            self._command.des_contact,
        )
        self._command.des_height[:] = torch.where(
            self._command.mode[:, None] == self.CMD_JUMP,
            torch.where(is_jumping, 0.65, 0.45),
            self._command.des_height
        )

        # self._command.cmd_rpy[:, 0:1] = torch.where(
        #     (self._command.mode == self.CMD_FLIP).reshape(-1, 1),
        #     self._command.cmd_ang_vel[:, 0:1] * (self._command.time-self.jump_prepare_time).clamp_min(0.),
        #     self._command.cmd_rpy[:, 0:1]
        # )
        self._command.cmd_rpy[:, 1:2].lerp_(self._command.des_rpy[:, 1:2], 0.4)
        self._command.cmd_ang_vel[:, 2:3] = (self._command.yaw_stiffness * cmd_yaw_diff).clamp(*self.ang_vel_z_range)

        sample = (self.env.episode_length_buf-20) % 175 == 0
        sample |= self._command.phase.squeeze(1) > 1.0
        
        next_mode_prob = self.transition[self._command.mode]
        next_mode = next_mode_prob.multinomial(1, replacement=True).squeeze(-1)

        self._command: SiriusCommand = self.sample_command_normal(self.num_envs) \
            .where(sample & (next_mode==self.CMD_WALK), self._command)
        self._command: SiriusCommand = self.sample_command_flip(self.num_envs) \
            .where(sample & (next_mode==self.CMD_FLIP), self._command)
        self._command: SiriusCommand = self.sample_command_stand(self.num_envs) \
            .where(sample & (next_mode==self.CMD_STAND), self._command)
        self._command: SiriusCommand = self.sample_command_jump(self.num_envs) \
            .where(sample & (next_mode==self.CMD_JUMP), self._command)

    # def step(self, substep):
    #     self.asset._external_torque_b[:, 0]
    #     self.asset.has_external_wrench = True

    def sample_command_normal(self, size):
        command = SiriusCommand.zero(size, self.device)
        command.cmd_lin_vel[:, 0].uniform_(*self.lin_vel_x_range)
        command.cmd_lin_vel[:, 1].uniform_(*self.lin_vel_y_range)
        direction = torch.randn(size, 3, device=self.device).sign()
        command.cmd_lin_vel = command.cmd_lin_vel * command.cmd_lin_vel.norm(dim=-1, keepdim=True) > 0.15
        command.cmd_lin_vel = command.cmd_lin_vel * direction
        command.yaw_stiffness.uniform_(0.8, 1.2)
        command.des_rpy[:, 2].uniform_(-torch.pi, torch.pi)
        command.des_rpy[:, 1].uniform_(-0.2 * torch.pi, 0.2 * torch.pi) # pitch
        command.des_height[:, 0] = 0.45 - command.des_rpy[:, 1].sin() * 0.314
        command.des_height[:, 1] = 0.45 + command.des_rpy[:, 1].sin() * 0.314
        command.mode[:] = self.CMD_WALK
        return command

    def sample_command_stand(self, size):
        command = SiriusCommand.zero(size, self.device)
        # command.cmd_lin_vel[:, 0].uniform_(-0.4, 0.4)
        command.cmd_lin_vel[:, 0] = torch.randint(-2, 4, (size,), device=self.device) * 0.5
        command.des_rpy[:, 1].uniform_(0.42 * torch.pi, 0.48 * torch.pi) # pitch
        command.des_rpy[:, 1].mul_(torch.randn(size, device=self.device).sign())
        command.des_rpy[:, 2] = self.asset.data.heading_w
        command.des_height[:, 0] = 0.71 - command.des_rpy[:, 1].sin() * 0.314
        command.des_height[:, 1] = 0.71 + command.des_rpy[:, 1].sin() * 0.314
        command.des_contact = torch.where(
            command.des_rpy[:, 1, None] > 0.0,
            torch.tensor([0., -1., 0., -1.], device=self.device), # stand on fore legs
            torch.tensor([-1., 0., -1., 0.], device=self.device), # stand on hind legs
        )
        command.mode[:] = self.CMD_STAND
        return command
    
    def sample_command_flip(self, size):
        command = SiriusCommand.zero(size, self.device)
        command.des_rpy[:, 2] = self.asset.data.heading_w
        command.cmd_ang_vel[:, 0].uniform_(1.8 * torch.pi, 2.4 * torch.pi)
        command.duration[:] = JUMP_PREPARE_TIME + 2 * torch.pi / command.cmd_ang_vel[:, 0:1]
        command.cmd_ang_vel[:, 0].mul_(-1.)
        # command.cmd_ang_vel[:, 0].mul_(torch.randn(size, device=self.device).sign())
        command.mode[:] = self.CMD_FLIP
        return command
    
    def sample_command_jump(self, size):
        command = SiriusCommand.zero(size, self.device)
        command.cmd_lin_vel[:, 0] = self._command.cmd_lin_vel[:, 0] * 0.8
        command.des_rpy[:, 2] = self.asset.data.heading_w
        command.duration.uniform_(0.9, 1.1)
        command.des_height[:] = 0.45
        command.mode[:] = self.CMD_JUMP
        return command

    def debug_draw(self):
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            # r, p, y = euler_xyz_from_quat(self.asset.data.root_quat_w)
            quat = quat_from_euler_xyz(
                self._command.cmd_rpy[:, 0:1].squeeze(1),
                self._command.cmd_rpy[:, 1:2].squeeze(1),
                self._command.des_rpy[:, 2],
            )
            self.frame_marker.visualize(
                translations=self.asset.data.root_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                orientations=quat,
                scales=torch.tensor([[4., 1., 0.1]]).expand(self.num_envs, 3),
            )

            self.env.debug_draw.vector(
                self.asset.data.root_pos_w,
                quat_rotate(quat, torch.tensor([0., 0., 1.], device=self.device).expand(self.num_envs, 3))
            )

            des_height = self.des_height
            point = self.asset.data.body_pos_w[:, self.fore_hip_ids].mean(dim=1)
            point[:, 2] = des_height[:, 0]
            self.env.debug_draw.point(point, color=(0, 0, 1, 1))

            point = self.asset.data.body_pos_w[:, self.hind_hip_ids].mean(dim=1)
            point[:, 2] = des_height[:, 1]
            self.env.debug_draw.point(point, color=(0, 1, 0, 1))

            # start = self.asset.data.body_pos_w[self.stand_fore_legs.squeeze(1)][:, self.back_body_id]
            # vector = torch.zeros(self.stand_fore_legs.sum().item(), 3, device=self.device)
            # vector[:, 2:3] = - self._stand_height[self.stand_fore_legs.squeeze(1)]
            # self.env.debug_draw.vector(start, vector, color=(0.1, 1.0, 0.1, 1.0))

            # start = self.asset.data.body_pos_w[self.stand_hind_legs.squeeze(1)][:, self.front_body_id]
            # vector = torch.zeros(self.stand_hind_legs.sum().item(), 3, device=self.device)
            # vector[:, 2:3] = - self._stand_height[self.stand_hind_legs.squeeze(1)]
            # self.env.debug_draw.vector(start, vector, color=(0.1, 1.0, 0.1, 1.0))

        elif self.env.sim.has_gui() and self.env.backend == "mujoco":
            from_ = self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2])
            self.arrow_marker_0.from_to(from_, from_ + self._cmd_lin_vel_w)
            self.arrow_marker_1.from_to(from_, from_ + self.asset.data.root_lin_vel_w)


class command_mode(Observation[SiriusCommandManager]):

    def compute(self) -> torch.Tensor:
        return self.command_manager._command.mode.reshape(self.num_envs, 1)


class no_drift(Reward[SiriusCommandManager]):
    """Penalize undesired drifting when the command velocity is zero"""
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.wheel_joint_ids = self.command_manager.wheel_joint_ids

    def compute(self) -> torch.Tensor:
        wheel_joint_vel = self.asset.data.joint_vel[:, self.wheel_joint_ids]
        cmd_speed = self.command_manager._command.cmd_lin_vel[:, :2].norm(dim=-1)
        rew = - wheel_joint_vel.abs().sum(dim=1) * (cmd_speed < 0.05)
        return rew.reshape(self.num_envs, 1)


class sirius_base_height(Reward[SiriusCommandManager]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.fore_hip_ids = self.asset.find_bodies("[L,R]F_hip")[0]
        self.hind_hip_ids = self.asset.find_bodies("[L,R]H_hip")[0]

    def compute(self) -> torch.Tensor:
        fore_height = self.asset.data.body_pos_w[:, self.fore_hip_ids, 2].mean(dim=1)
        hind_height = self.asset.data.body_pos_w[:, self.hind_hip_ids, 2].mean(dim=1)
        des_height = self.command_manager._command.des_height
        rew = 0.5 * (
            torch.exp( - (fore_height - des_height[:, 0]).square() / 0.1)
            + torch.exp( - (hind_height - des_height[:, 1]).square() / 0.1)
        )
        return rew.reshape(self.num_envs, 1)


class wheel_contact_direction(Reward[SiriusCommandManager]):
    """Penalize contacts where the wheels are not upright"""
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.wheel_ids = self.asset.find_bodies(".*_FOOT")[0]
        self.wheel_ids_contact = self.contact_forces.find_bodies(".*_FOOT")[0]
        self.gravity = self.asset.data.default_mass[0].sum(-1).to(self.device) * 9.81

    def compute(self) -> torch.Tensor:
        wheel_contact_forces = self.contact_forces.data.net_forces_w[:, self.wheel_ids_contact] / self.gravity
        wheel_normal = quat_rotate(
            self.asset.data.body_quat_w[:, self.wheel_ids],
            torch.tensor([0., 0., 1.], device=self.device).expand(self.num_envs, 4, 3)
        )
        rew = - (wheel_contact_forces * wheel_normal).sum(dim=-1).abs()
        return rew.sum(1, True)


class contact_pattern(Reward[SiriusCommandManager]):
    """Reward adherence to the commanded contact pattern"""
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.wheel_ids_contact, self.wheel_names_contact = self.contact_forces.find_bodies(".*_FOOT")
        self.des_air_time = torch.zeros(self.num_envs, 4, device=self.device)

    def update(self):
        self.des_air_time = torch.where(
            self.command_manager._command.des_contact != 0,
            self.des_air_time - self.command_manager._command.des_contact,
            0.0,
        )
    
    def reset(self, env_ids):
        self.des_air_time[env_ids] = 0.0
    
    def compute(self) -> torch.Tensor:
        des_contact = self.command_manager._command.des_contact
        contact_forces = self.contact_forces.data.net_forces_w[:, self.wheel_ids_contact].norm(dim=-1)
        in_contact = contact_forces > 1.
        rew = (des_contact * in_contact).sum(dim=-1, keepdim=True)
        # current_air_time = self.contact_forces.data.current_air_time[:, self.wheel_ids_contact] / 0.02
        # rew = (current_air_time - self.des_air_time).clamp(-4, 0) * (des_contact != 0)
        # rew = (current_air_time / 0.02 < self.des_air_time) * -(des_contact != 0)
        self.env.discount.mul_(torch.exp(0.25 * rew))
        return rew.reshape(self.num_envs, 1)


class sirius_joint_deviation(Reward[SiriusCommandManager]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.joint_ids = self.asset.find_joints(".*(HAA|HFE)")[0]

    def compute(self) -> torch.Tensor:
        is_active = self.command_manager._command.mode == self.command_manager.CMD_JUMP
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        joint_dev = (joint_pos - self.asset.data.default_joint_pos[:, self.joint_ids]).square().sum(1, True)
        return -joint_dev, is_active.reshape(self.num_envs, 1)

