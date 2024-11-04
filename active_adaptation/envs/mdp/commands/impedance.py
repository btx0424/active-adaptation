import torch

from active_adaptation.envs.mdp.commands.base import Command
from omni.isaac.lab.utils.math import quat_rotate_inverse, quat_rotate, yaw_quat, wrap_to_pi, euler_xyz_from_quat


def rpy_from_quat(quat: torch.Tensor):
    q_w, q_x, q_y, q_z = quat.unbind(-1)
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(
        torch.abs(sin_pitch) >= 1,
        (torch.pi / 2.0) * torch.sign(sin_pitch),
        torch.asin(sin_pitch)
    )

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return torch.stack([roll, pitch, yaw], -1)


class ImpedanceBase(Command):
    def __init__(
        self, 
        env, 
        temporal_smoothing: int = 16,
        teleop: bool = False
    ):
        super().__init__(env, teleop=teleop)
        self.temporal_smoothing = temporal_smoothing
        
        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 10)
            # linvel_xy, angvel_z
            self.command_hidden = torch.zeros(self.num_envs, 8)

            self.des_pos_w = torch.zeros(self.num_envs, 3)
            self.des_lin_vel_w = torch.zeros(self.num_envs, 3)
            self.des_yaw_w = torch.zeros(self.num_envs, 1)
            self.des_yaw_vel_w = torch.zeros(self.num_envs, 1)

            self._des_lin_acc_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self._des_lin_vel_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self._des_pos_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)

            self._des_ang_acc_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self._des_ang_vel_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self._des_rpy_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)

            self.setpoint_pos_w = torch.zeros(self.num_envs, 3)
            self.setpoint_pos_w_next = torch.zeros(self.num_envs, 3)
            self.setpoint_rpy_w = torch.zeros(self.num_envs, 3)

            self.virtual_mass = torch.zeros(self.num_envs, 1)
            self.lin_kp = torch.zeros(self.num_envs, 1)
            self.lin_kd = torch.zeros(self.num_envs, 1)
            self.ang_kp = torch.zeros(self.num_envs, 1)
            self.ang_kd = torch.zeros(self.num_envs, 1)

            self.force_ext_w = torch.zeros(self.num_envs, 3)
            self.torque_ext_w = torch.zeros(self.num_envs, 3)

            self._cum_error = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.target_lin_vel = torch.zeros(self.num_envs, 3)

        # for backward compatibility
        self.command_pos_w = self.des_pos_w
        self.command_linvel_w = self.des_lin_vel_w
        self.command_yaw_w = self.des_yaw_w
        self.command_angvel = self.des_yaw_vel_w
        self.command_speed = self.des_lin_vel_w.norm(dim=-1, keepdim=True)

        self.substeps = self.env.cfg.decimation
        self.lin_vel_error_sum = torch.tensor(0., device=self.device)
        self.ang_vel_error_sum = torch.tensor(0., device=self.device)
        self.cnt = 0.
    
    def reset(self, env_ids):
        self._des_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids].reshape(len(env_ids), 1, 3)
        self._des_lin_vel_w[env_ids] = 0.
        rpy = rpy_from_quat(self.asset.data.root_quat_w[env_ids])
        self._des_rpy_w[env_ids] = rpy.reshape(len(env_ids), 1, 3)
        self._des_ang_vel_w[env_ids] = 0.
        # reset to zero command
        self.target_lin_vel[env_ids] = 0.
        root_pos_w = self.asset.data.root_pos_w[env_ids]
        self.setpoint_pos_w[env_ids] = root_pos_w
        self.setpoint_pos_w_next[env_ids] = root_pos_w
        self.setpoint_rpy_w[env_ids] = 0.
        self.force_ext_w[env_ids] = 0.

        self.lin_kp[env_ids] = 1.0
        self.lin_kd[env_ids] = 2.0
        self.ang_kp[env_ids] = 1.0
        self.ang_kd[env_ids] = 2.0
        self.virtual_mass[env_ids] =  2.0
    
    def step(self, substep):
        root_quat = self.asset.data.root_quat_w
        self.asset._external_force_b[:, 0] = quat_rotate_inverse(root_quat, self.force_ext_w)
        self.asset.has_external_wrench = True
    
    def update(self):
        t = self.env.episode_length_buf - 20
        sample_command = ((t % 300 == 0)).nonzero().squeeze(-1)
        if len(sample_command) > 0:
            self.sample_command(sample_command)

        self.setpoint_pos_w_next = torch.where(
            self.lin_kp > 0.1,
            self.target_lin_vel * self.lin_kd / self.lin_kp + self.asset.data.root_pos_w,
            self.asset.data.root_pos_w
        )
        self.setpoint_pos_w = self.setpoint_pos_w + 0.2 * (self.setpoint_pos_w_next - self.setpoint_pos_w)
        
        # closed-loop adjustment
        # linear velocity and position
        self._des_lin_vel_w = self._des_lin_vel_w.roll(1, 1)
        self._des_lin_vel_w[:, 0] = self.asset.data.root_lin_vel_w
        self._des_pos_w = self._des_pos_w.roll(1, 1)
        self._des_pos_w[:, 0] = self.asset.data.root_pos_w
        # angular velocity and heading
        self._des_ang_vel_w = self._des_ang_vel_w.roll(1, 1)
        self._des_ang_vel_w[:, 0] = self.asset.data.root_ang_vel_w
        self._des_rpy_w = self._des_rpy_w.roll(1, 1)
        self._des_rpy_w[:, 0] = rpy_from_quat(self.asset.data.root_quat_w)

        lin_kp = self.lin_kp.unsqueeze(1)
        lin_kd = self.lin_kd.unsqueeze(1)
        ang_kp = self.ang_kp.unsqueeze(1)
        ang_kd = self.ang_kd.unsqueeze(1)
        for _ in range(self.substeps):
            self._des_lin_acc_w = (
                lin_kp * (self.setpoint_pos_w.unsqueeze(1) - self._des_pos_w)
                + lin_kd * (0. - self._des_lin_vel_w)
                + (self.force_ext_w / self.virtual_mass).unsqueeze(1)
            )
            self._des_lin_vel_w.add_(self._des_lin_acc_w * self.env.physics_dt)
            self._des_lin_vel_w[:, :, 2] = 0. # only xy
            self._des_pos_w.add_(self._des_lin_vel_w * self.env.physics_dt)

            ang_error = wrap_to_pi((self.setpoint_rpy_w.unsqueeze(1) - self._des_rpy_w))
            self._des_ang_acc_w = (
                ang_kp * ang_error
                + ang_kd * (0. - self._des_ang_vel_w)
            )
            self._des_ang_vel_w.add_(self._des_ang_acc_w * self.env.physics_dt)
            self._des_ang_vel_w[:, :, :2] = 0. # only yaw
            self._des_rpy_w.add_(self._des_ang_vel_w * self.env.physics_dt)
        
        self.des_lin_vel_w  = self._des_lin_vel_w[:, 4:].mean(1)
        self.des_pos_w      = self._des_pos_w[:, 4:].mean(1)
        self.des_ang_vel_w  = self._des_ang_vel_w[:, 4:].mean(1)
        # special handling for yaw since cannot directly average
        heading = self.asset.data.heading_w.unsqueeze(1)
        des_yaw = (heading + wrap_to_pi(self._des_rpy_w[:, :, 2] - heading))
        self.des_yaw_w      = des_yaw[:, :4].mean(1).reshape(-1, 1)
        
        # for backward compatibility
        self.command_pos_w = self.des_pos_w
        self.command_linvel_w = self.des_lin_vel_w
        self.command_yaw_w = self.des_yaw_w
        self.command_angvel = self.des_ang_vel_w[:, 2:3]
        self.command_speed = self.des_lin_vel_w.norm(dim=-1, keepdim=True)

        root_quat = self.asset.data.root_quat_w
        yaw_diff = wrap_to_pi(self.setpoint_rpy_w[:, 2:3] - self.asset.data.heading_w.unsqueeze(1))
        pos_diff = self.setpoint_pos_w - self.asset.data.root_pos_w
        pos_diff = quat_rotate_inverse(root_quat, pos_diff)
        
        self.command[:, 0:2] = pos_diff[:, 0:2]
        self.command[:, 2:3] = yaw_diff
        self.command[:, 3:5] = self.lin_kp * pos_diff[:, :2] # 2
        self.command[:, 5:6] = self.lin_kd # 1
        self.command[:, 6:7] = self.ang_kp # 1
        self.command[:, 7:8] = self.ang_kd
        self.command[:, 8:9] = self.virtual_mass

        self.command_hidden[:, 0:3] = quat_rotate_inverse(root_quat, self.des_pos_w - self.asset.data.root_pos_w)
        self.command_hidden[:, 3:6] = quat_rotate_inverse(root_quat, self.des_lin_vel_w)
        self.command_hidden[:, 6:7] = self.des_ang_vel_w[:, 2:3]

        # compute errors
        lin_vel_error = (self.des_lin_vel_w - self.asset.data.root_lin_vel_w).square().sum()
        self.lin_vel_error_sum.mul_(0.995).add_(lin_vel_error)
        self.cnt = self.cnt * 0.995 + self.num_envs

        sample_force = (
            (torch.rand(self.num_envs, device=self.device) < 0.02)
            & (self.env.episode_length_buf > self.temporal_smoothing)
        )
        self.sample_force(sample_force.nonzero().squeeze(-1))
    
    def sample_command(self, env_ids: torch.Tensor):
        empty = torch.empty(len(env_ids), 1, device=self.device)
        
        # critical damping
        lin_kd = empty.uniform_(2., 10.).clone()
        lin_kp = torch.square(0.5 * lin_kd)
        
        ang_kd = empty.uniform_(2., 10.).clone()
        ang_kp = torch.square(0.5 * ang_kd)

        self.lin_kp[env_ids] = lin_kp
        self.lin_kd[env_ids] = lin_kd
        self.ang_kp[env_ids] = ang_kp
        self.ang_kd[env_ids] = ang_kd

        target_lin_vel = torch.zeros(len(env_ids), 3, device=self.device)
        target_lin_vel[:, 0].uniform_(0.6, 1.6).mul_(torch.rand(len(env_ids), device=self.device) < 0.1)
        target_lin_vel[:, 1].uniform_(-1.0, 1.0)
        yaw = torch.atan2(target_lin_vel[:, 1], target_lin_vel[:, 0])
        
        self.target_lin_vel[env_ids] = target_lin_vel
        self.setpoint_rpy_w[env_ids, 2] = yaw * (torch.rand(len(env_ids), device=self.device) < 0.4)
        self.virtual_mass[env_ids] = empty.uniform_(2.0, 4.0)
    
    def sample_force(self, env_ids: torch.Tensor):
        force_ext_w = torch.zeros(len(env_ids), 3, device=self.device)
        force_ext_w[:, 0].uniform_(-30., 30.)
        force_ext_w[:, 1].uniform_(-30., 30.)
        has_force = torch.rand(len(env_ids), 1, device=self.device) < 0.5
        self.force_ext_w[env_ids] = force_ext_w * has_force

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.des_lin_vel_w,
            color=(0., 1., 0., 1.)
        )
        # draw line to setpoint pos (red)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            self.setpoint_pos_w - self.asset.data.root_pos_w,
            color=(1., 0., 0., 1.)
        )
        self.env.debug_draw.point(self.setpoint_pos_w, color=(1., 0., 0., 1.), size=40.)

        # draw setpoint rpy (blue)
        heading = torch.zeros(self.num_envs, 3, device=self.device)
        heading[:, 0] = self.setpoint_rpy_w[:, 2].cos()
        heading[:, 1] = self.setpoint_rpy_w[:, 2].sin()
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            heading,
            color=(0., 0., 1., 1.)
        )

        # draw external forces (orange)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.force_ext_w / (self.virtual_mass * 9.81),
            color=(1., 0.5, 0., 1.),
            size=4.0
        )

