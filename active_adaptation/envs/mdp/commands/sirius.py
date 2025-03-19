import torch

from tensordict import TensorClass
from active_adaptation.utils.math import (
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    wrap_to_pi,
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat
)
from .base import Command


def reward(func):
    func.is_reward = True
    return func

def observation(func):
    func.is_observation = True
    return func

def termination(func):
    func.is_termination = True
    return func


class SiriusCommand(TensorClass):
    cmd_lin_vel_xy: torch.Tensor
    cmd_ang_vel: torch.Tensor
    cmd_roll: torch.Tensor
    cmd_pitch: torch.Tensor
    yaw_stiffness: torch.Tensor

    des_rpy: torch.Tensor
    des_height: torch.Tensor
    
    time: torch.Tensor
    duration: torch.Tensor
    mode: torch.Tensor

    @property
    def phase(self):
        return self.time / self.duration
    
    @classmethod
    def zero(cls, size: int, device: str):
        return cls(
            cmd_lin_vel_xy=torch.zeros(size, 2, device=device),
            cmd_ang_vel=torch.zeros(size, 3, device=device),
            cmd_roll=torch.zeros(size, 1, device=device),
            des_rpy=torch.zeros(size, 3, device=device),
            des_height=torch.zeros(size, 1, device=device),
            yaw_stiffness=torch.zeros(size, 1, device=device),
            cmd_pitch=torch.zeros(size, 1, device=device),
            time=torch.zeros(size, 1, device=device),
            duration=torch.full((size, 1), torch.inf, device=device),
            mode=torch.zeros(size, dtype=int, device=device),
        )
    
    def get_command(self):
        return torch.cat([
            self.cmd_lin_vel_xy,
            self.cmd_ang_vel,
            self.cmd_roll,
            self.cmd_pitch,
            self.phase
        ], dim=-1)


class SiriusCommandManager(Command):
    
    jump_prep: float = 0.1
    
    CMD_NORMAL = 0
    CMD_FLIP = 1
    CMD_STAND = 2

    def __init__(self, env):
        super().__init__(env)
        self.front_body_id = self.asset.find_bodies("front")[0][0]
        self.back_body_id = self.asset.find_bodies("back")[0][0]
        
        self.pitch_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.roll_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.lin_vel_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.ang_vel_z_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.ang_vel_x_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.base_height_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        self.stand_height_error_l2 = torch.zeros(self.num_envs, 1, device=self.device)
        
        self.rew_jump_lin_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self.rew_jump_inertia = torch.zeros(self.num_envs, 1, device=self.device)
        self.rew_stand_lin_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self.rew_stand_height = torch.zeros(self.num_envs, 1, device=self.device)

        self.target_base_height = torch.zeros(self.num_envs, 1, device=self.device)
        self.default_mass = self.asset.data.default_mass.to(self.device)

        self._cum_error = torch.zeros(self.num_envs, 1, device=self.device)
        
        self._command = SiriusCommand.zero(self.num_envs, self.device)
        with torch.device(self.device):
            self.transition = torch.eye(3) # normal, flip, stand
            self.transition[0] = torch.tensor([.2, .8, 0.]) # normal to others
            self.transition[1] = torch.tensor([1., 0., 0.]) # flip to others
            self.transition[2] = torch.tensor([1., 0., 0.]) # stand to others

        if self.env.sim.has_gui():
            from omni.isaac.lab.markers import RED_ARROW_X_MARKER_CFG, VisualizationMarkers
            self.frame_marker = VisualizationMarkers(
                RED_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/frame",
                )
            )
            self.frame_marker.set_visibility(True)
    
    # @reward
    # def jump_lin_vel(self):
    #     is_active = (self._command.mode==self.CMD_FLIP).unsqueeze(1)
    #     return self.rew_jump_lin_vel, is_active
    
    @reward
    def jump_height(self):
        is_active = (self._command.mode==self.CMD_FLIP).unsqueeze(1)
        return -self.base_height_error_l2, is_active

    @reward
    def jump_ang_vel(self):
        is_active = (self._command.mode==self.CMD_FLIP).unsqueeze(1)
        cmd_ang_vel_roll = self._command.cmd_ang_vel[:, 0:1]
        ang_vel_roll = self.asset.data.root_ang_vel_b[:, 0:1]
        rew_jump_ang_vel = torch.clamp(
            ang_vel_roll * cmd_ang_vel_roll.sign(),
            torch.zeros_like(ang_vel_roll),
            cmd_ang_vel_roll.abs()
        )
        return rew_jump_ang_vel, is_active
    
    @reward
    def jump_roll(self):
        is_active = (self._command.mode==self.CMD_FLIP).unsqueeze(1)
        return torch.exp( -self.roll_error_l2 / 0.25), is_active
    
    @reward
    def walk_angvel_xy_l2(self):
        rew = -self.asset.data.root_ang_vel_b[:, :2].square().sum(1, True)
        is_active = (self._command.mode==self.CMD_NORMAL).unsqueeze(1)
        return rew, is_active

    @reward
    def walk_linvel_z_l2(self):
        rew = -self.asset.data.root_lin_vel_b[:, 2:3].square()
        is_active = (self._command.mode==self.CMD_NORMAL).unsqueeze(1)
        return rew, is_active

    @reward
    def jump_inertia(self):
        return self.rew_jump_inertia
    
    @reward
    def stand_lin_vel(self):
        is_active = (self._command.mode==self.CMD_STAND).unsqueeze(1)
        return self.rew_stand_lin_vel, is_active
    
    @reward
    def stand_height(self):
        is_active = (self._command.mode==self.CMD_STAND).unsqueeze(1)
        return self.rew_stand_height, is_active

    @observation
    def command_mode(self):
        return self._command.mode

    @observation
    def command_end(self):
        return self._command.time == 0.
    
    @termination
    def stand_error_exceeds(self):
        return self.stand_height_error_l2 > 0.2
    
    @termination
    def jump_error_exceeds(self):
        return self.roll_error_l2 > 0.5

    @property
    def command(self):
        return self._command.get_command()
    
    def reset(self, env_ids):
        command = self.sample_command_normal(len(env_ids))
        self._command[env_ids] = command
        self._cum_error[env_ids] = 0.
        self.target_base_height[env_ids] = 0.4

    def update(self):
        r, p, y = euler_xyz_from_quat(self.asset.data.root_quat_w)
        self.pitch_error_l2 = torch.square( wrap_to_pi(self._command.cmd_pitch - p.unsqueeze(1)) )

        quat_yaw = yaw_quat(self.asset.data.root_quat_w)
        root_lin_vel_b = quat_rotate_inverse(quat_yaw, self.asset.data.root_lin_vel_w)
        
        self.lin_vel_error_l2 = torch.square(
            self._command.cmd_lin_vel_xy - root_lin_vel_b[:, :2] ).sum(1, keepdim=True)
        self.lin_vel_error_l2 *= (self._command.mode == self.CMD_NORMAL).unsqueeze(1)
        
        self.ang_vel_z_error_l2 = torch.square(self._command.cmd_ang_vel[:, 2:3] - self.asset.data.root_ang_vel_w[:, 2:3])
        self.ang_vel_z_error_l2 *= (self._command.mode == self.CMD_NORMAL).unsqueeze(1) # only apply to normal mode

        self.roll_error_l2 = (
            self.asset.data.projected_gravity_b[:, 0:1].square()
            + (self._command.cmd_roll.sin() + self.asset.data.projected_gravity_b[:, 1:2]).square()
            + (self._command.cmd_roll.cos() + self.asset.data.projected_gravity_b[:, 2:3]).square()
        )
        # print(self.roll_error_l2.squeeze(1))

        jump_height = torch.clamp_min(0.5 - .5*9.81*(self._command.time - self._command.duration/2)**2, 0.)
        target_base_height = 0.4 + jump_height

        self.jump_vel_z = (target_base_height - self.target_base_height) / self.env.step_dt
        
        body_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.asset.data.body_pos_w - self.asset.data.root_pos_w.unsqueeze(1)
        )
        
        inertia_x = (body_pos_b[:, :, 2].square() * self.default_mass).sum(1, True)
        in_air = (self._command.phase > 0.25) & (self._command.phase < 0.75)
        self.rew_jump_inertia = (1. - inertia_x.square()) * in_air

        self.target_base_height = target_base_height
        self.base_height_error_l2 = (self.target_base_height - self.asset.data.root_pos_w[:, 2:3]).square()

        self.front_height = self.asset.data.body_pos_w[:, self.front_body_id, 2:3]
        self.back_height = self.asset.data.body_pos_w[:, self.back_body_id, 2:3]
        
        self._stand_height = self.asset.data.root_pos_w[:, 2:3] # base height
        self.stand_fore_legs = self._command.cmd_pitch > +0.3*torch.pi
        self.stand_hind_legs = self._command.cmd_pitch < -0.3*torch.pi
        self._stand_height = torch.where(self.stand_fore_legs, self.back_height, self._stand_height)
        self._stand_height = torch.where(self.stand_hind_legs, self.front_height, self._stand_height)
        
        stand_lin_vel = torch.zeros(self.num_envs, 1, device=self.device)
        stand_lin_vel_front = self.asset.data.body_lin_vel_w[:, self.front_body_id, 2:3]
        stand_lin_vel_back = self.asset.data.body_lin_vel_w[:, self.back_body_id, 2:3]
        stand_lin_vel = torch.where(self.stand_fore_legs, stand_lin_vel_back, stand_lin_vel)
        stand_lin_vel = torch.where(self.stand_hind_legs, stand_lin_vel_front, stand_lin_vel)

        target_stand_lin_vel = 4. * (1.3 - self._stand_height)
        
        self.stand_height_error_l2 = (self._command.mode == self.CMD_STAND).unsqueeze(1) \
            * (self._command.des_height - self._stand_height).square()
        stand_lin_vel_error_l2 = (target_stand_lin_vel - stand_lin_vel).square()

        self.rew_stand_lin_vel = torch.exp(- stand_lin_vel_error_l2 )
        self.rew_stand_height = torch.exp(- self.stand_height_error_l2 / 0.25)
        
        cmd_yaw_diff = (self._command.mode == self.CMD_NORMAL) \
            .mul(wrap_to_pi(self._command.des_rpy[:, 2] - self.asset.data.heading_w)) \
            .unsqueeze(1)

        self._command.time.add_(self.env.step_dt)
        self._command.cmd_roll = self._command.cmd_ang_vel[:, 0:1] * (self._command.time-self.jump_prep).clamp_min(0.)
        self._command.cmd_ang_vel[:, 2:3] = (self._command.yaw_stiffness * cmd_yaw_diff).clamp(-2., 2.)
        self._command.cmd_pitch.lerp_(self._command.des_rpy[:, 1:2], 0.4)
        self._command.des_height.add_(self.env.step_dt * 4. * (1.3 - self._command.des_height))

        sample = (self.env.episode_length_buf-20) % 175 == 0
        sample |= self._command.phase.squeeze(1) > 1.0
        
        next_mode_prob = self.transition[self._command.mode]
        next_mode = next_mode_prob.multinomial(1, replacement=True).squeeze(-1)

        self._command: SiriusCommand = self.sample_command_normal(self.num_envs) \
            .where(sample & (next_mode==self.CMD_NORMAL), self._command)
        self._command: SiriusCommand = self.sample_command_flip(self.num_envs) \
            .where(sample & (next_mode==self.CMD_FLIP), self._command)
        self._command: SiriusCommand = self.sample_command_stand(self.num_envs) \
            .where(sample & (next_mode==self.CMD_STAND), self._command)

    def step(self, substep):
        self.asset._external_torque_b[:, 0]
        self.asset.has_external_wrench = True

    def sample_command_normal(self, size):
        command = SiriusCommand.zero(size, self.device)
        command.cmd_lin_vel_xy[:, 0].uniform_(-1.0, 1.2)
        command.cmd_lin_vel_xy[:, 1].uniform_(-0.8, 0.8)
        command.yaw_stiffness.uniform_(0.8, 1.2)
        command.des_rpy[:, 2].uniform_(-torch.pi, torch.pi)
        # command.des_rpy[:, 1].uniform_(-0.2 * torch.pi, 0.2 * torch.pi)
        command.mode[:] = self.CMD_NORMAL
        return command

    def sample_command_stand(self, size):
        command = SiriusCommand.zero(size, self.device)
        command.cmd_lin_vel_xy[:, 0].uniform_(-0.4, 0.4)
        command.des_rpy[:, 1].uniform_(0.4 * torch.pi, 0.45 * torch.pi)
        command.des_rpy[:, 1].mul_(torch.randn(size, device=self.device).sign())
        command.des_height[:] = 0.4
        command.mode[:] = self.CMD_STAND
        return command
    
    def sample_command_flip(self, size):
        command = SiriusCommand.zero(size, self.device)
        command.des_rpy[:, 2] = self.asset.data.heading_w
        command.cmd_ang_vel[:, 0].uniform_(1.8 * torch.pi, 2.4 * torch.pi)
        command.duration[:] = self.jump_prep + 2 * torch.pi / command.cmd_ang_vel[:, 0:1]
        command.cmd_ang_vel[:, 0].mul_(-1.)
        # command.cmd_ang_vel[:, 0].mul_(torch.randn(size, device=self.device).sign())
        command.mode[:] = self.CMD_FLIP
        return command

    def debug_draw(self):
        if self.env.sim.has_gui():
            # r, p, y = euler_xyz_from_quat(self.asset.data.root_quat_w)
            quat = quat_from_euler_xyz(
                self._command.cmd_roll.squeeze(1),
                self._command.cmd_pitch.squeeze(1),
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
            point = self.asset.data.root_pos_w.clone()
            point[:, 2] = self.target_base_height.squeeze(1)
            self.env.debug_draw.point(point)

            point = self.asset.data.body_pos_w[self.stand_fore_legs.squeeze(1), self.back_body_id]
            point[:, 2] = self._command.des_height[self.stand_fore_legs.squeeze(1)].squeeze(1)
            self.env.debug_draw.point(point, color=(0, 0, 1, 1))

            point = self.asset.data.body_pos_w[self.stand_hind_legs.squeeze(1), self.front_body_id]
            point[:, 2] = self._command.des_height[self.stand_hind_legs.squeeze(1)].squeeze(1)
            self.env.debug_draw.point(point, color=(0, 1, 0, 1))

            # start = self.asset.data.body_pos_w[self.stand_fore_legs.squeeze(1)][:, self.back_body_id]
            # vector = torch.zeros(self.stand_fore_legs.sum().item(), 3, device=self.device)
            # vector[:, 2:3] = - self._stand_height[self.stand_fore_legs.squeeze(1)]
            # self.env.debug_draw.vector(start, vector, color=(0.1, 1.0, 0.1, 1.0))

            # start = self.asset.data.body_pos_w[self.stand_hind_legs.squeeze(1)][:, self.front_body_id]
            # vector = torch.zeros(self.stand_hind_legs.sum().item(), 3, device=self.device)
            # vector[:, 2:3] = - self._stand_height[self.stand_hind_legs.squeeze(1)]
            # self.env.debug_draw.vector(start, vector, color=(0.1, 1.0, 0.1, 1.0))

