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

@torch.compile
def clamp_norm_(x: torch.Tensor, max_norm: float):
    norm = x.norm(dim=-1, keepdim=True)
    return x.div_(norm.clamp(min=1e-6)).mul_(norm.clamp_max(max_norm))


@torch.compile
def down_scale(x: torch.Tensor, a: float):
    norm = x.norm(dim=-1, keepdim=True)
    return x / norm.clamp(1e-6) * torch.log1p(norm / a) * a


@torch.compile
def minnorm(a: torch.Tensor, b: torch.Tensor):
    a_norm = a.norm(dim=-1, keepdim=True)
    b_norm = b.norm(dim=-1, keepdim=True)
    return torch.where(a_norm < b_norm, a, b)


class ImpedanceBase(Command):
    def __init__(
        self, 
        env,
        compliant_ratio: float = 0.1,
        temporal_smoothing: int = 32,
        teleop: bool = False,
    ):
        super().__init__(env, teleop=teleop)
        self.compliant_ratio = compliant_ratio
        self.temporal_smoothing = temporal_smoothing
        
        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 11)
            # linvel_xy, angvel_z
            self.command_hidden = torch.zeros(self.num_envs, 8)

            self.des_pos_w = torch.zeros(self.num_envs, 3)
            self.des_lin_vel_w = torch.zeros(self.num_envs, 3)
            self.des_lin_acc_w = torch.zeros(self.num_envs, 3)
            self.des_yaw_w = torch.zeros(self.num_envs, 1)
            self.des_yaw_vel_w = torch.zeros(self.num_envs, 1)

            self._des_lin_acc_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self._des_lin_vel_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self._des_pos_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)

            self._des_ang_acc_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self._des_ang_vel_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)
            self._des_rpy_w = torch.zeros(self.num_envs, self.temporal_smoothing, 3)

            self.setpoint_pos_b = torch.zeros(self.num_envs, 3)
            self.setpoint_pos_w = torch.zeros(self.num_envs, 3)
            self.setpoint_pos_w_next = torch.zeros(self.num_envs, 3)
            self.setpoint_rpy_w = torch.zeros(self.num_envs, 3)
            self.setpoint_rpy_w_next = torch.zeros(self.num_envs, 3)

            self.virtual_mass = torch.zeros(self.num_envs, 1)
            self.lin_kp = torch.zeros(self.num_envs, 1)
            self.lin_kd = torch.zeros(self.num_envs, 1)
            self.ang_kp = torch.zeros(self.num_envs, 1)
            self.ang_kd = torch.zeros(self.num_envs, 1)
            self.compliant = torch.zeros(self.num_envs, 1, dtype=bool)

            self.force_ext_w = torch.zeros(self.num_envs, 3)
            self.force_impulse_struct = torch.zeros(self.num_envs, 4)
            self.force_impulse_w = self.force_impulse_struct[:, :3]
            self.force_constant_w = torch.zeros(self.num_envs, 3)

            self.constant_force_offset_b = torch.zeros(self.num_envs, 3)
            self.impulse_force_offset_b = torch.zeros(self.num_envs, 3)
            self.torque_ext_w = torch.zeros(self.num_envs, 3)

            self._cum_error = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.target_lin_vel = torch.zeros(self.num_envs, 3)
            self.target_yaw_vel = torch.zeros(self.num_envs, 1)

        # for backward compatibility
        self.command_pos_w = self.des_pos_w
        self.command_linvel_w = self.des_lin_vel_w
        self.command_linvel = self.des_lin_vel_w
        self.command_yaw_w = self.des_yaw_w
        self.command_angvel = self.des_yaw_vel_w
        self.command_speed = self.des_lin_vel_w.norm(dim=-1, keepdim=True)

        self.substeps = self.env.cfg.decimation
        self.lin_vel_error_sum = torch.tensor(0., device=self.device)
        self.ang_vel_error_sum = torch.tensor(0., device=self.device)
        self.cnt = 0.
        self.step_cnt = 0

        if self.teleop:
            self.key_mappings_pos = {
                "W": torch.tensor([1., 0., 0.], device=self.device),
                "S": torch.tensor([-1., 0., 0.], device=self.device),
                "A": torch.tensor([0., 1., 0.], device=self.device),
                "D": torch.tensor([0., -1., 0.], device=self.device),
            }
            self.key_mappings_rpy = {
                "Q": torch.tensor([0., 0., +torch.pi], device=self.device),
                "E": torch.tensor([0., 0., -torch.pi], device=self.device),
            }
        
        if self.env.scene.terrain.cfg.terrain_type == "generator":
            self.terrain_generator = self.env.scene.terrain.terrain_generator
            self.env_stairs = torch.as_tensor(
                (
                    (self.terrain_generator.terrain_types == "MeshInvertedPyramidStairsTerrainCfg") 
                    | (self.terrain_generator.terrain_types == "MeshPyramidStairsTerrainCfg")
                ), 
                device=self.device
            )
            self.terrain_centers = torch.tensor(self.terrain_generator.terrain_origins, device=self.device)
            self.terrain_origin = torch.tensor(self.terrain_centers[0, 0, :2], device=self.device)
            self.terrain_origin -= 0.5 * torch.tensor(self.terrain_generator.cfg.size, device=self.device)
            self.terrain_size = self.terrain_centers.shape[:2]
        else:
            self.terrain_generator = None

    def reset(self, env_ids):
        self._des_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids].reshape(len(env_ids), 1, 3)
        self._des_lin_vel_w[env_ids] = 0.
        rpy = rpy_from_quat(self.asset.data.root_quat_w[env_ids])
        self._des_rpy_w[env_ids] = rpy.reshape(len(env_ids), 1, 3)
        self._des_ang_vel_w[env_ids] = 0.
        
        # reset to zero command
        root_pos_w = self.asset.data.root_pos_w[env_ids]
        self.target_lin_vel[env_ids] = 0.
        self.target_yaw_vel[env_ids] = 0.
        self.target_pos_w[env_ids] = root_pos_w
        
        self.setpoint_pos_w[env_ids] = root_pos_w
        self.setpoint_pos_w_next[env_ids] = root_pos_w
        self.setpoint_rpy_w[env_ids, 1] = 0.
        self.setpoint_rpy_w[env_ids, 2] = self.asset.data.heading_w[env_ids]
        self.setpoint_rpy_w_next[env_ids] = 0.
        self.force_ext_w[env_ids] = 0.
        self.is_standing_env[env_ids] = True

        self.lin_kp[env_ids] = 1.0
        self.lin_kd[env_ids] = 2.0
        self.ang_kp[env_ids] = 1.0
        self.ang_kd[env_ids] = 2.0
        self.virtual_mass[env_ids] =  2.0

        self.lin_vel_error_avg = self.lin_vel_error_sum / self.cnt
        self.env.extra["stats/lin_vel_error_avg"] = self.lin_vel_error_avg.item()
    
    def step(self, substep):
        # apply force
        root_quat = self.asset.data.root_quat_w
        constant_force_b = quat_rotate_inverse(root_quat, self.force_constant_w)
        impulse_force_b = quat_rotate_inverse(root_quat, self.force_impulse_w)
        self.asset._external_force_b[:, 0] += quat_rotate_inverse(root_quat, self.force_ext_w)
        self.asset._external_torque_b[:, 0] += (
            self.constant_force_offset_b.cross(constant_force_b, dim=-1) 
            + self.impulse_force_offset_b.cross(impulse_force_b, dim=-1)
        )
        self.asset.has_external_wrench = True

        self.force_impulse_w[:, :3].mul_(self.force_impulse_struct[:, 3].unsqueeze(1))
        self.step_cnt += 1
    
    def update_setpoint(self):
        if self.teleop:
            for key, vec in self.key_mappings_pos.items():
                if self.key_pressed[key]:
                    self.setpoint_pos_b.add_(vec * self.env.step_dt)
        else:
            pass
        
        self.setpoint_pos_w = quat_rotate(self.root_quat_yaw, self.setpoint_pos_b) + self.asset.data.root_pos_w
    
    def update(self):
        self.root_pos_w = self.asset.data.root_pos_w
        self.root_quat = self.asset.data.root_quat_w
        self.root_quat_yaw = yaw_quat(self.root_quat)

        if self.terrain_generator is not None:
            terrain_idx = ((self.asset.data.root_pos_w[:, :2] - self.terrain_origin) / 8).int() # [num_envs, 2]
            x, y = terrain_idx.unbind(-1)
            x = x.clamp(min=0, max=self.terrain_size[0] -1)
            y = y.clamp(min=0, max=self.terrain_size[1] -1)
            self.terrain_idx = (x, y)
            self.among_stairs = self.env_stairs[x, y]

        # integrate
        lin_kp = self.lin_kp.unsqueeze(1)
        lin_kd = self.lin_kd.unsqueeze(1)
        ang_kp = self.ang_kp.unsqueeze(1)
        ang_kd = self.ang_kd.unsqueeze(1)
        
        setpoint_pos_w_that_moves = quat_rotate(self.asset.data.root_quat_w, self.setpoint_pos_b).unsqueeze(1) + self._des_pos_w
        setpoint_pos_w = setpoint_pos_w_that_moves
        
        self._des_lin_acc_w = (
            lin_kp * (setpoint_pos_w - self._des_pos_w)
            + lin_kd * (0. - self._des_lin_vel_w)
            + down_scale(self.force_ext_w, 40.).unsqueeze(1)
        ) / self.virtual_mass.unsqueeze(1)

        self._des_lin_vel_w.add_(self._des_lin_acc_w * self.env.step_dt)
        self._des_lin_vel_w[:, :, 2] = 0. # only xy
        clamp_norm_(self._des_lin_vel_w, 1.6)
        self._des_pos_w.add_(self._des_lin_vel_w * self.env.step_dt)

        ang_error = wrap_to_pi((self.setpoint_rpy_w.unsqueeze(1) - self._des_rpy_w))
        self._des_ang_acc_w = (
            ang_kp * ang_error
            + ang_kd * (0. - self._des_ang_vel_w)
        )
        self._des_ang_vel_w.add_(self._des_ang_acc_w * self.env.physics_dt)
        self._des_ang_vel_w[:, :, 0] = 0. # only pitch and yaw
        self._des_ang_vel_w.clamp_(-torch.pi / 2, torch.pi / 2)
        self._des_rpy_w.add_(self._des_ang_vel_w * self.env.step_dt)

        # closed-loop adjustment
        # linear velocity and position
        self._des_lin_vel_w = self._des_lin_vel_w.roll(1, 1)
        self._des_lin_vel_w[:, 0] = self.asset.data.root_lin_vel_w
        self._des_pos_w= self._des_pos_w.roll(1, 1)
        self._des_pos_w[:, 0] = self.asset.data.root_pos_w
        # angular velocity and heading
        self._des_ang_vel_w = self._des_ang_vel_w.roll(1, 1)
        self._des_ang_vel_w[:, 0] = self.asset.data.root_ang_vel_w
        self._des_rpy_w = self._des_rpy_w.roll(1, 1)
        self._des_rpy_w[:, 0] = rpy_from_quat(self.asset.data.root_quat_w)
        
        self.des_lin_acc_w = self._des_lin_acc_w[:, 0]
        self.des_lin_vel_w  = self._des_lin_vel_w[:, -1]
        self.des_pos_w      = self._des_pos_w[:, -1]
        self.des_ang_vel_w  = self._des_ang_vel_w[:, -1]
        # special handling for yaw since cannot directly average
        heading = self.asset.data.heading_w.unsqueeze(1)
        des_yaw = (heading + wrap_to_pi(self._des_rpy_w[:, :, 2] - heading))
        self.des_yaw_w      = des_yaw[:, 8:].mean(1).reshape(-1, 1)
        
        # for backward compatibility
        self.command_pos_w = self.des_pos_w
        self.command_linvel_w = self.des_lin_vel_w
        self.command_linvel = quat_rotate_inverse(self.root_quat, self.command_linvel_w)
        self.command_yaw_w = self.des_yaw_w
        self.command_angvel = self.des_ang_vel_w[:, 2:3]
        self.command_speed = self.des_lin_vel_w.norm(dim=-1, keepdim=True)

        rpy = rpy_from_quat(self.root_quat) 
        pitch_yaw_diff = wrap_to_pi(self.setpoint_rpy_w[:, 1:3] - rpy[:, 1:3])
        pos_diff = self.setpoint_pos_w - self.asset.data.root_pos_w
        pos_diff = quat_rotate_inverse(self.root_quat, pos_diff)
        
        self.command[:, 0:2] = pos_diff[:, 0:2] # 2
        self.command[:, 2:4] = pitch_yaw_diff   # 2
        self.command[:, 4:6] = self.lin_kp * pos_diff[:, :2] # 2
        self.command[:, 6:7] = self.lin_kd # 1
        self.command[:, 7:9] = self.ang_kp * pitch_yaw_diff # 2
        self.command[:, 9:10] = self.ang_kd # 1
        self.command[:, 10:11] = self.virtual_mass

        self.command_hidden[:, 0:3] = quat_rotate_inverse(self.root_quat, self.des_pos_w - self.asset.data.root_pos_w)
        self.command_hidden[:, 3:6] = quat_rotate_inverse(self.root_quat, self.des_lin_vel_w)
        self.command_hidden[:, 6:7] = self.des_ang_vel_w[:, 2:3]

        # compute errors
        lin_vel_error = (self.des_lin_vel_w - self.asset.data.root_lin_vel_w).square().sum()
        self.lin_vel_error_sum.mul_(0.995).add_(lin_vel_error)
        self.cnt = self.cnt * 0.995 + self.num_envs

        # sample forces
        valid = (self.env.episode_length_buf > self.temporal_smoothing)
        sample_force = (torch.rand(self.num_envs, device=self.device) < 0.02) & valid
        self.sample_constant_force(sample_force.nonzero().squeeze(-1))
        sample_force = (torch.rand(self.num_envs, device=self.device) < 0.01) & valid
        self.sample_impulse_force(sample_force.nonzero().squeeze(-1))
        self.force_ext_w = self.force_constant_w + self.force_impulse_w
        
        eps = 0.05
        has_force = self.force_ext_w.any(dim=1, keepdim=True)
        has_cmd = (pos_diff.abs() > eps).any(1, True) | (pitch_yaw_diff.abs() > eps).any(1, True)
        
        des_lin_speed = self.des_lin_vel_w.norm(dim=-1, keepdim=True)
        des_yaw_speed = self.des_ang_vel_w[:, 2:3]
        has_des_speed = (des_lin_speed > eps) | (des_yaw_speed > eps)
        self.is_standing_env[:] = torch.where(
            self.is_standing_env,
            ~((has_force | has_cmd) & has_des_speed),
            ~has_des_speed
        )

        self.update_setpoint()
        
        t = self.env.episode_length_buf - 20
        sample_command = ((t % 300 == 0)).nonzero().squeeze(-1)
        if len(sample_command) > 0:
            self.sample_command(sample_command)
    
    def sample_command(self, env_ids: torch.Tensor):
        empty = torch.empty(len(env_ids), 1, device=self.device)
        
        virtual_mass = empty.uniform_(1.0, 4.0).clone()
        compliant = (
            (torch.rand(len(env_ids), 1, device=self.device) < self.compliant_ratio)
            & (self.env.episode_length_buf[env_ids].unsqueeze(1) > 300)
        )
        lin_kp = empty.uniform_(2., 20.).clone()
        lin_kd = 2 * lin_kp.sqrt()

        ang_kp = empty.uniform_(2., 20.).clone()
        ang_kd = 2 * ang_kp.sqrt()

        self.lin_kp[env_ids] = lin_kp
        self.lin_kd[env_ids] = lin_kd
        self.ang_kp[env_ids] = ang_kp
        self.ang_kd[env_ids] = ang_kd

        setpoint_pos_b = torch.zeros(len(env_ids), 3, device=self.device)
        setpoint_pos_b[:, 0].uniform_(-1.0, 1.0)
        setpoint_pos_b[:, 1].uniform_(-0.7, 0.7)
        self.setpoint_pos_b[env_ids] = setpoint_pos_b

        if self.terrain_generator is not None:
            among_stairs = self.among_stairs[env_ids]
            # target_lin_vel[:, 1] *= (~among_stairs.reshape(-1))

        self.compliant[env_ids] = compliant
        # self.target_lin_vel[env_ids] = target_lin_vel
        self.target_yaw_vel[env_ids] = 0. #(torch.rand(len(env_ids), 1, device=self.device) * 2 - 1.0) * (~among_stairs.reshape(-1, 1))
        # self.target_pos_w[env_ids] = target_pos_w + self.asset.data.root_pos_w[env_ids]
        pitch = 0. # torch.randint(-1, 2, (len(env_ids),), device=self.device) * 0.2
        self.setpoint_rpy_w_next[env_ids, 1] = pitch
        self.setpoint_rpy_w_next[env_ids, 2] = 0.
        self.virtual_mass[env_ids] = virtual_mass
    
    def sample_constant_force(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        constant_force_prob = torch.where(self.compliant[env_ids], 0.6, 0.6)
        force_ext_w = torch.zeros(len(env_ids), 3, device=self.device)
        force_ext_w[:, 0].uniform_(-30., 30.)
        force_ext_w[:, 1].uniform_(-30., 30.)
        force_ext_w[:, 2].uniform_(-10., 10.)
        constant_force_offset_b = torch.zeros(len(env_ids), 3, device=self.device)
        constant_force_offset_b[:, 0].uniform_(-0.2, 0.2)
        constant_force_offset_b[:, 1].uniform_(-0.1, 0.1)
        constant_force_offset_b[:, 2].uniform_(-0.1, 0.1)

        has_force = torch.rand(len(env_ids), 1, device=self.device) < constant_force_prob
        self.force_constant_w[env_ids] = force_ext_w * has_force
        self.constant_force_offset_b[env_ids] = constant_force_offset_b * has_force
    
    def sample_impulse_force(self, env_ids: torch.Tensor):
        if (len(env_ids) == 0) or (self.lin_vel_error_avg > 0.):
            return
        force_impulse_struct = torch.zeros(len(env_ids), 4, device=self.device)
        a = torch.zeros(len(env_ids), device=self.device).uniform_(60., 120.)
        r = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        force_impulse_struct[:, 0] = torch.sin(r) * a
        force_impulse_struct[:, 1] = torch.cos(r) * a
        force_impulse_struct[:, 3].uniform_(0.9, 0.98)
        impulse_force_offset_b = torch.zeros(len(env_ids), 3, device=self.device)
        impulse_force_offset_b[:, 0].uniform_(-0.2, 0.2)
        impulse_force_offset_b[:, 1].uniform_(-0.1, 0.1)
        impulse_force_offset_b[:, 2].uniform_(-0.1, 0.1)

        self.impulse_force_offset_b[env_ids] = impulse_force_offset_b
        self.force_impulse_struct[env_ids] = force_impulse_struct

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
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
        heading[:, 2] = - self.setpoint_rpy_w[:, 1].sin()
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            heading,
            color=(0., 0., 1., 1.)
        )

        # draw external forces (orange)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + quat_rotate(self.asset.data.root_quat_w, self.constant_force_offset_b),
            self.force_ext_w / (self.virtual_mass * 9.81),
            color=(1., 0.5, 0., 1.),
            size=4.0
        )

        self.env.debug_draw.vector(
            self.asset.data.root_pos_w + quat_rotate(self.asset.data.root_quat_w, self.impulse_force_offset_b),
            self.force_impulse_w / (self.virtual_mass * 9.81),
            color=(1., 0.0, 0.5, 1.),
            size=4.0
        )

        self.env.debug_draw.point(
            self.des_pos_w,
            size=40,
            color=(1., 1., 0., 1.)
        )

        # x, y = self.terrain_idx
        # terrain_centers = self.terrain_centers[x, y]
        # self.env.debug_draw.vector(
        #     self.asset.data.root_pos_w,
        #     terrain_centers - self.asset.data.root_pos_w,
        #     color=(1., 1., 1.0, 1.0),
        #     size=2
        # )
