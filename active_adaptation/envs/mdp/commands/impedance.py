import torch
import torch.nn.functional as F
import einops

from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.envs.mdp.commands.locomotion import Command2
import isaaclab.utils.math as math_utils
from active_adaptation.utils.math import (
    yaw_rotate, clamp_along, clamp_norm,
    quat_rotate, quat_rotate_inverse,
    quat_from_euler_xyz,
    quat_from_view,
    yaw_quat,
    EMA
)
from active_adaptation.envs.mdp import reward
from active_adaptation.envs.mdp.utils.forces import ConstantForce, ImpulseForce, SpringForce

def saturate(x: torch.Tensor, a: float):
    norm = x.norm(dim=-1, keepdim=True)
    return (x /norm.clamp_min(1e-6)) * torch.log1p(norm / a) * a


class Impedance(Command):

    CMD_COMPLIANT = 0
    CMD_LINVEL = 1
    CMD_POSITION = 2
    CMD_LARGE_FORCE = 3

    def __init__(
        self,
        env,
        body_names: str="base",
        virtual_mass_range=None, # always set to 1 if `None`
        force_saturate=80.0,
        linear_kp_range=(4.0, 24.0),
        angular_kp_range=(4.0, 24.0),
        impulse_force_momentum_scale=(5.0, 5.0, 1.0),
        impulse_force_duration_range=(0.1, 0.5),
        
        constant_force_scale=(50, 50, 10),
        constant_force_offset_scale=(0.0, 0.0, 0.0),

        temporal_smoothing: int = 32,
        surr_steps: list[int] = [16, 24, 32],
        max_acc_xy: float = (8.0, 4.0),
        max_vel_xy: float = (1.6, 1.0),
        teleop: bool = False,
        **kwargs
    ) -> None:
        super().__init__(env, teleop)

        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.virtual_mass_range = virtual_mass_range
        if self.virtual_mass_range is not None:
            self.virtual_mass_range = torch.tensor(self.virtual_mass_range, device=self.device).float().reshape(-1, 1)

        self.force_saturate = force_saturate

        self.constant_force_scale = constant_force_scale
        self.constant_force_offset_scale = constant_force_offset_scale

        self.resample_prob = 0.01
        self.linear_kp_range = linear_kp_range
        self.angular_kp_range = angular_kp_range
        self.temporal_smoothing = temporal_smoothing

        self.max_acc_xyz = max_acc_xy + (0.,)
        self.max_vel_xyz = max_vel_xy + (0.,)

        assert self.temporal_smoothing >= 32
        self.surr_steps = surr_steps

        with torch.device(self.device):
            if self.virtual_mass_range is not None:
                self.command = torch.zeros(self.num_envs, 10 + 5)
            else:
                self.command = torch.zeros(self.num_envs, 10)
            self.command_hidden = torch.zeros(self.num_envs, len(self.surr_steps) * 8)
            
            self.surrogate_pos_target = torch.zeros(self.num_envs, len(self.surr_steps), 3)
            self.surrogate_yaw_target = torch.zeros(self.num_envs, len(self.surr_steps), 1)
            self.surrogate_lin_vel_target = torch.zeros(self.num_envs, len(self.surr_steps), 3)
            self.surrogate_yaw_vel_target = torch.zeros(self.num_envs, len(self.surr_steps), 1)

            self.command_linvel = torch.zeros(self.num_envs, 3)
            self.command_speed = torch.zeros(self.num_envs, 1)

            # integration
            bshape = (self.num_envs, self.temporal_smoothing + 1)
            self.ref_lin_acc_w = torch.zeros(*bshape, 3)
            self.ref_lin_vel_w = torch.zeros(*bshape, 3)
            self.ref_pos_w = torch.zeros(*bshape, 3)

            self.ref_yaw_acc_w = torch.zeros(*bshape, 1)
            self.ref_yaw_vel_w = torch.zeros(*bshape, 1)
            self.ref_yaw_w = torch.zeros(*bshape, 1)

            self.command_setpos_w = torch.zeros(self.num_envs, 3)
            self.command_setrpy_w = torch.zeros(self.num_envs, 3)
            self.force_factor = torch.ones(self.num_envs, 1)

            self.set_linvel = torch.zeros(self.num_envs, 3)

            self.lin_kp = torch.zeros(self.num_envs, 1)
            self.lin_kd = torch.zeros(self.num_envs, 1)
            self.ang_kp = torch.zeros(self.num_envs, 1)
            self.ang_kd = torch.zeros(self.num_envs, 1)

            self.default_inertia = self.asset.root_physx_view.get_inertias()[
                0, 0, [0, 4, 8]
            ].to(self.device)
            self.default_inertia[2] += 1.2

            self.virtual_mass = torch.zeros(self.num_envs, 1)
            self.virtual_inertia = torch.zeros(self.num_envs, 3)

            self.force_ext_w = torch.zeros(self.num_envs, 3)
            self.torque_ext_w = torch.zeros(self.num_envs, 3)

            # force parameters
            self.impulse_force = ImpulseForce.sample(self.num_envs, device=self.device)
            self.impulse_force.duration.zero_()
            self.spring_force = SpringForce.sample(self.num_envs, device=self.device)
            self.spring_force.duration.zero_()
            self.constant_force = ConstantForce.sample(self.num_envs, device=self.device)
            self.constant_force.duration.zero_()

            # in regular mode, constant force and impulse force are randomly applied
            # in compliant mode, kp is zero
            # in large force mode, a large spring force is applied and other forces are disabled
            # the three modes are exclusive

            self.max_output_force_range = (80., 100.)
            self.max_output_force = torch.zeros(self.num_envs, 1)

            self._cum_error = torch.zeros(self.num_envs, 3)
            # debugging info
            # how much distance (percentage) is covered
            self.distance_commanded = torch.zeros(self.num_envs, 1)
            self.distance_covered = torch.zeros(self.num_envs, 1)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            self.command_time = torch.zeros(self.num_envs,)
            self.command_mode = torch.zeros(self.num_envs, 1, dtype=int)
            self.command_transition = torch.eye(4)
            self.command_transition[0] = torch.tensor([0.0, 0.4, 0.4, 0.2])
            self.command_transition[1] = torch.tensor([0.2, 0.4, 0.4, 0.0])
            self.command_transition[2] = torch.tensor([0.0, 0.0, 0.1, 0.9])
            self.command_transition[3] = torch.tensor([0.0, 0.0, 1.0, 0.0])

            # normal
            self.command_transition[0] = torch.tensor([0.0, 0.5, 0.5, 0.0])
            self.command_transition[1] = torch.tensor([0.2, 0.4, 0.4, 0.0])
            self.command_transition[2] = torch.tensor([0.2, 0.4, 0.4, 0.0])
            self.command_transition[3] = torch.tensor([0.0, 0.0, 1.0, 0.0])

            # self.command_transition[2] = torch.tensor([0.0, 0.0, 0.0, 1.0])
            self.command_transition /= self.command_transition.sum(dim=-1, keepdim=True)

        self.cnt = 0
        
        self.vis_arrow = None
        if self.env.sim.has_gui():
            from isaaclab.markers.config import (
                BLUE_ARROW_X_MARKER_CFG,
                RED_ARROW_X_MARKER_CFG,
                sim_utils
            )
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            print(BLUE_ARROW_X_MARKER_CFG.markers["arrow"].scale)

            self.vis_arrow = VisualizationMarkers(
                BLUE_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/target_yaw"
                ))
            self.vis_arrow.set_visibility(True)
            self.force_vis_arrow = VisualizationMarkers(
                RED_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/force"
                ))
            self.force_vis_arrow.set_visibility(True)
            self.setpoint_vis = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/setpoint",
                    markers={
                        "setpoint": sim_utils.SphereCfg(
                            radius=0.04,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        )
                    }
                )
            )
            self.setpoint_vis.set_visibility(True)
        self.lin_vel_ema = EMA(self.asset.data.root_lin_vel_w, [0.0, 0.5, 0.8])
        self.ang_vel_ema = EMA(self.asset.data.root_ang_vel_w, [0.0, 0.5, 0.8])
        self.dim_weights = torch.tensor([1.0, 1.0, 0.5], device=self.device)

    def get_pos_w(self): # currently only used for smoothing the rewards
        return self.asset.data.body_pos_w[:, self.body_ids].mean(1)
    
    def get_lin_vel_w(self): # currently only used for smoothing the rewards
        return self.asset.data.body_lin_vel_w[:, self.body_ids].mean(1)

    @reward
    def impedance_pos(self):
        diff = self.surrogate_pos_target - self.get_pos_w().unsqueeze(1)
        error_l2 = diff[:, :, :2].square().sum(dim=-1, keepdim=True)
        r = (- error_l2 / 0.25).exp().mean(1)
        return r
    
    @reward
    def impedance_vel(self):
        diff = einops.rearrange(self.surrogate_lin_vel_target, "n t1 d -> n t1 1 d") \
            - einops.rearrange(self.lin_vel_ema.ema, "n t2 d-> n 1 t2 d")
        error_l2 = (diff * self.dim_weights).square().sum(dim=-1, keepdim=True)
        r = ((- error_l2 / 0.25).exp() - 0.25 * error_l2).mean(1)
        return r.max(dim=1).values
    
    # evaluation metrics
    @reward
    def impedance_pos_error(self):
        diff = self.ref_pos_w[:, -1] - self.get_pos_w()
        error_l2 = diff[:, :2].square().sum(dim=-1, keepdim=True)
        return error_l2
    
    @reward
    def impedance_vel_error(self):
        diff = self.ref_lin_vel_w[:, -1] - self.get_lin_vel_w()
        error_l2 = diff[:, :2].square().sum(dim=-1, keepdim=True)
        return error_l2
    
    @reward
    def impedance_acc_error(self):
        diff = self.ref_lin_acc_w[:, 0] - self.asset.data.body_acc_w[:, 0, :3]
        error_l2 = diff[:, :2].square().sum(dim=-1, keepdim=True)
        return error_l2

    @reward
    def impedance_yaw_vel(self):
        diff = einops.rearrange(self.surrogate_yaw_vel_target, "n t1 d -> n t1 1 d") \
            - einops.rearrange(self.ang_vel_ema.ema[:, :, 2:3], "n t2 d-> n 1 t2 d")
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = ((- error_l2 / 0.25).exp() - 0.25 * error_l2).mean(1)
        return r.max(dim=1).values

    def reset(self, env_ids: torch.Tensor):
        self._sample_command(env_ids, command_mode=self.CMD_POSITION)
        self._cum_error[env_ids] = 0.0
        self.env.extra["stats/distance_commanded"] = self.distance_commanded.mean().item()
        self.env.extra["stats/distance_covered"] = self.distance_covered.mean().item()
        self.distance_covered[env_ids] = 0.0
        self.distance_commanded[env_ids] = 0.0

        self.ref_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids].unsqueeze(1)
        self.ref_lin_vel_w[env_ids] = 0.0
        self.ref_yaw_w[env_ids] = self.asset.data.heading_w[env_ids, None, None]
        self.ref_yaw_vel_w[env_ids] = 0.0

        self.spring_force.duration[env_ids] = 0.
        self.constant_force.duration[env_ids] = 0.
        self.impulse_force.duration[env_ids] = 0.

    def step(self, substep: int):
        self.lin_vel_ema.update(self.get_lin_vel_w())
        self.ang_vel_ema.update(self.asset.data.root_ang_vel_w)

        forces_b = self.asset._external_force_b
        forces_b[:, 0] += quat_rotate_inverse(self.asset.data.root_quat_w, self.force_ext_w)
        torques_b = self.asset._external_torque_b
        constant_force_b = quat_rotate_inverse(self.asset.data.root_quat_w, self.constant_force.get_force())
        torques_b[:, 0] += self.constant_force.offset.cross(constant_force_b, dim=-1)
        self.asset.has_external_wrench = True

    def _integrate(self, dt: float):
        setpos_w = torch.where(
            (self.command_mode == self.CMD_LINVEL).reshape(self.num_envs, 1, 1),
            self.ref_pos_w + (self.lin_kd / self.lin_kp * self.set_linvel).unsqueeze(1),
            self.command_setpos_w.unsqueeze(1)
        )
        ref_acc_w = (
            self.lin_kp.reshape(self.num_envs, 1, 1) * (setpos_w - self.ref_pos_w)
            + self.lin_kd.reshape(self.num_envs, 1, 1) * (0.0 - self.ref_lin_vel_w)
            + saturate(self.force_ext_w, self.force_saturate).reshape(self.num_envs, 1, 3)
        ) / self.virtual_mass.unsqueeze(1) # [n, t, 3]

        # x_b = torch.cat([self.ref_yaw_w.cos(), self.ref_yaw_w.sin(), torch.zeros_like(self.ref_yaw_w)], dim=-1)
        # y_b = torch.cat([-self.ref_yaw_w.sin(), self.ref_yaw_w.cos(), torch.zeros_like(self.ref_yaw_w)], dim=-1)
        ref_acc_w[..., 2] = 0.
        ref_acc_w = clamp_norm(ref_acc_w, 0., 80.)
        # ref_acc_w = clamp_along(ref_acc_w, x_b, -self.max_acc_xyz[0], self.max_acc_xyz[0])
        # ref_acc_w = clamp_along(ref_acc_w, y_b, -self.max_acc_xyz[1], self.max_acc_xyz[1])

        self.ref_lin_acc_w = ref_acc_w
        ref_vel_w = self.ref_lin_vel_w + self.ref_lin_acc_w * dt
        # ref_vel_w = clamp_along(ref_vel_w, x_b, -self.max_vel_xyz[0], self.max_vel_xyz[0])
        # ref_vel_w = clamp_along(ref_vel_w, y_b, -self.max_vel_xyz[1], self.max_vel_xyz[1])  
        ref_vel_w[..., 2] = 0.
        ref_vel_w = clamp_norm(ref_vel_w, 0., 2.4)
        ## Do not do this here. Small values may integrate to large values.
        ## Instead, zero-out small values when computing the command.
        # vel_low_mask = (torch.norm(ref_vel_w, dim=-1) < 0.8) & acc_low_mask
        # ref_vel_w[vel_low_mask] = 0.0
        
        self.ref_lin_vel_w = ref_vel_w
        self.ref_pos_w.add_(self.ref_lin_vel_w * dt)

        torque = torch.cross(
            yaw_rotate(self.ref_yaw_w.squeeze(-1), self.constant_force.offset.unsqueeze(1)),
            self.constant_force.get_force().unsqueeze(1),
            dim=-1
        )
        yaw_diff = self.command_setrpy_w[:, 2:3].unsqueeze(1) - self.ref_yaw_w
        ref_yaw_acc_w = (
            self.ang_kp.unsqueeze(1) * math_utils.wrap_to_pi(yaw_diff)
            + self.ang_kd.unsqueeze(1) * (0.0 - self.ref_yaw_vel_w)
            + torque[:, :, 2:3]
        ) / self.virtual_inertia[:, 2:3].unsqueeze(1)

        self.ref_yaw_acc_w[:] = ref_yaw_acc_w
        self.ref_yaw_vel_w.add_(self.ref_yaw_acc_w * dt)
        self.ref_yaw_w.add_(self.ref_yaw_vel_w * dt)

    def update(self):
        # compute cumulative errors
        # might be used for early termination
        yaw = self.asset.data.heading_w

        self.distance_commanded.add_(self.ref_lin_vel_w[:, -1].norm(dim=-1, keepdim=True))
        self.distance_covered.add_(self.asset.data.root_lin_vel_w.norm(dim=-1, keepdim=True))

        self.ref_lin_vel_w[:, :-1] = self.ref_lin_vel_w[:, :-1].roll(1, dims=1)
        self.ref_pos_w[:, :-1] = self.ref_pos_w[:, :-1].roll(1, dims=1)
        self.ref_yaw_vel_w[:, :-1] = self.ref_yaw_vel_w[:, :-1].roll(1, dims=1)
        self.ref_yaw_w[:, :-1] = self.ref_yaw_w[:, :-1].roll(1, dims=1)

        self.ref_lin_vel_w[:, 0] = self.asset.data.root_lin_vel_w
        self.ref_lin_vel_w[:, 0] = self.get_lin_vel_w()
        self.ref_pos_w[:, 0] = self.asset.data.root_pos_w
        self.ref_yaw_vel_w[:, 0] = self.asset.data.root_ang_vel_w[:, 2:3]
        self.ref_yaw_w[:, 0] = self.asset.data.heading_w.unsqueeze(1)

        self._integrate(self.env.step_dt)

        root_pos = self.asset.data.root_pos_w
        # check if here should be yaw rotate
        command_setpos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.command_setpos_w - self.asset.data.root_pos_w,
        )

        # print((self.command_linvel[:, 0].abs() - self.max_vel_xyz[0]).max())
        # print((self.command_linvel[:, 1].abs() - self.max_vel_xyz[1]).max())
        self.command_speed[:] = self.command_linvel.norm(dim=-1, keepdim=True)
        self.is_standing_env[:] = False #(self.command_speed < 0.1) & (self.command_angvel.abs() < 0.1).unsqueeze(1)

        yaw_diff = math_utils.wrap_to_pi(self.command_setrpy_w[:, 2] - yaw)

        self.command[:, :2] = command_setpos_b[:, :2]
        self.command[:, 2] = yaw_diff
        self.command[:, 3:5] = self.lin_kp * command_setpos_b[:, :2]
        self.command[:, 5:8] = (self.lin_kd)
        self.command[:, 8:9] = self.ang_kp * yaw_diff.unsqueeze(1)
        self.command[:, 9:10] = self.virtual_mass
        if self.virtual_mass_range is not None:
            self.command[:, 10:15] = self.command[:, 3:8] / self.virtual_mass
        
        self.surrogate_pos_target = self.ref_pos_w[:, self.surr_steps]
        self.surrogate_yaw_target = self.ref_yaw_w[:, self.surr_steps]
        self.surrogate_lin_vel_target = self.ref_lin_vel_w[:, self.surr_steps] 
        self.surrogate_yaw_vel_target = self.ref_yaw_vel_w[:, self.surr_steps]

        self.command_hidden = torch.cat([
            quat_rotate_inverse(
                self.asset.data.root_quat_w.unsqueeze(1),
                self.surrogate_pos_target - root_pos.unsqueeze(1)).reshape(self.num_envs, -1),
            quat_rotate_inverse(
                self.asset.data.root_quat_w.unsqueeze(1),
                self.surrogate_lin_vel_target).reshape(self.num_envs, -1),
            self.surrogate_yaw_vel_target.reshape(self.num_envs, -1),
            math_utils.wrap_to_pi(self.surrogate_yaw_target.squeeze(-1) - yaw.unsqueeze(1))
        ], dim=1)
        
        self.update_command()
        self.update_forces()

    def update_command(self):
        sample_command = (torch.rand(self.num_envs, device=self.device) < self.resample_prob) \
            & (self.command_time > 50)
        sample_command = sample_command.nonzero().squeeze(-1)
        if len(sample_command) > 0:
            self._sample_command(sample_command)
        self.command_time += 1
        
        self.command_setpos_w[:] = torch.where(
            (self.command_mode == self.CMD_LINVEL).reshape(self.num_envs, 1),
            self.lin_kd / self.lin_kp * self.set_linvel + self.asset.data.root_pos_w,
            # self.command_setpos_w + self.set_linvel * self.env.step_dt,
            self.command_setpos_w,
        )
        offset = torch.zeros(self.num_envs, 3, device=self.device)
        offset[:, 0] = (self.max_output_force / self.lin_kp).squeeze(1)
        self.command_setpos_w[:] = torch.where(
            (self.command_mode == self.CMD_LARGE_FORCE).reshape(self.num_envs, 1),
            self.asset.data.root_pos_w + offset,
            self.command_setpos_w
        )

    def update_forces(self):
        # sample constant force
        expire = self.constant_force.time > self.constant_force.duration - 1e-4
        r = (torch.rand(self.num_envs, 1, device=self.device) < self.resample_prob)
        sample = r & expire & (self.command_mode != self.CMD_LARGE_FORCE).reshape(self.num_envs, 1)
        constant_force = ConstantForce.sample(
            self.num_envs,
            self.constant_force_scale,
            self.constant_force_offset_scale,
            device=self.device,
        )
        self.constant_force.time.add_(self.env.step_dt)
        self.constant_force: ConstantForce = constant_force.where(sample, self.constant_force)

        # sample spring force
        expire = self.spring_force.time > self.spring_force.duration
        sample = expire & (self.command_mode == self.CMD_LARGE_FORCE).reshape(self.num_envs, 1)
        spring_force = SpringForce.sample(self.num_envs, self.device)
        spring_force.setpoint.add_(self.asset.data.root_pos_w)
        self.spring_force.time.add_(self.env.step_dt)
        self.spring_force: SpringForce = spring_force.where(sample, self.spring_force)

        expire = self.impulse_force.time > self.impulse_force.duration
        r = (torch.rand(self.num_envs, 1, device=self.device) < 0.005)
        sample = r & expire & (self.command_mode != self.CMD_LARGE_FORCE).reshape(self.num_envs, 1)
        impulse_force = ImpulseForce.sample(self.num_envs, self.device)
        self.impulse_force.time.add_(self.env.step_dt)
        self.impulse_force: ImpulseForce = impulse_force.where(sample, self.impulse_force)

        self._spring_force = self.spring_force.get_force(self.asset.data.root_pos_w, self.asset.data.root_lin_vel_w) * (self.command_mode == self.CMD_LARGE_FORCE).reshape(self.num_envs, 1)
        self.force_ext_w[:] = (
            self.constant_force.get_force()
            + self.impulse_force.get_force()
            + self._spring_force
        )

    def _sample_virtual_mass(self, env_ids: torch.Tensor):
        if self.virtual_mass_range is not None:
            i = torch.randint(0, self.virtual_mass_range.shape[0], (len(env_ids),), device=self.device)
            virtual_mass = self.virtual_mass_range[i]
        else:
            virtual_mass = 1.
        self.virtual_mass[env_ids] = virtual_mass
        self.virtual_inertia[env_ids] = self.default_inertia

    def _sample_command(self, env_ids: torch.Tensor, command_mode: int=None):
        scalar = torch.empty(len(env_ids), 1, device=self.device)
        if command_mode is None:
            command_mode_prob =  self.command_transition[self.command_mode[env_ids].squeeze(1)]
            command_mode = torch.multinomial(command_mode_prob, 1, replacement=True)
        else:
            command_mode = torch.full((len(env_ids), 1), command_mode, dtype=int, device=self.device)
        self.command_mode[env_ids] = command_mode
        self.command_time[env_ids] = 0
        
        self.max_output_force[env_ids] = scalar.uniform_(*self.max_output_force_range)

        lin_kp = torch.where(
            (command_mode == self.CMD_LARGE_FORCE).reshape(len(env_ids), 1),
            scalar.clone().uniform_(24., 40.),
            scalar.clone().uniform_(*self.linear_kp_range)
        )

        compliant = (command_mode == self.CMD_COMPLIANT).reshape(len(env_ids), 1)

        self.lin_kp[env_ids] = lin_kp * (~compliant)
        self.lin_kd[env_ids] = 1.8 * lin_kp.sqrt() # to make the system critically damped

        ang_kp = scalar.uniform_(*self.angular_kp_range)
        self.ang_kp[env_ids] = ang_kp * (~compliant)
        self.ang_kd[env_ids] = 1.8 * ang_kp.sqrt() # to make the system critically damped

        set_linvel = torch.zeros(len(env_ids), 3, device=self.device)
        set_linvel[:, 0].uniform_(0.4, 1.2)

        root_pos_w = self.asset.data.root_pos_w[env_ids]
        offset = torch.zeros(len(env_ids), 3, device=self.device)
        offset[:, 0].uniform_(0.6, 1.2)
        offset[:, 1].uniform_(0.6, 1.2)

        command_setpoint_w = torch.where(
            (command_mode == self.CMD_POSITION).reshape(len(env_ids), 1),
            root_pos_w + offset * torch.randn_like(offset).sign(),
            root_pos_w,
        )

        self.set_linvel[env_ids] = set_linvel
        self.command_setpos_w[env_ids] = command_setpoint_w

        # sample yaw in setpoint_rpy
        target_yaw = self.asset.data.heading_w[env_ids, None] + scalar.uniform_(-torch.pi/2, torch.pi/2)
        self.command_setrpy_w[env_ids, 2:3] = math_utils.wrap_to_pi(target_yaw)

        self._sample_virtual_mass(env_ids)    
        self.force_factor[env_ids] = 1.0


    def debug_draw(self):
        # draw command linvel (green)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w
            + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            self.ref_lin_vel_w[:, -2],
            color=(0.0, 1.0, 0.0, 1.0),
        )
        # draw vector to setpoint pos (red)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.command_setpos_w - self.asset.data.root_pos_w,
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.setpoint_vis.visualize(self.command_setpos_w)
        # self.env.debug_draw.vector(
        #     self.asset.data.root_pos_w + torch.tensor([0.0, 0.0, 0.3], device=self.device),
        #     # self.spring_force_setpoint - self.asset.data.root_pos_w,
        #     self.set_linvel * (self.command_mode == self.CMD_LINVEL).reshape(self.num_envs, 1),
        #     color=(1.0, 0.0, 1.0, 1.0),
        #     size=4.0
        # )
        self.env.debug_draw.plot(self.ref_pos_w[0, :-1])
        # draw setpoint pos (red)
        # self.env.debug_draw.point(
        #     self.command_setpos_w, color=(1.0, 0.0, 0.0, 1.0), size=40.0
        # )

        # draw external forces (orange)
        # self.env.debug_draw.vector(
        #     self.asset.data.root_pos_w + quat_rotate(self.asset.data.root_quat_w, self.constant_force.offset),
        #     self.constant_force.get_force() / (self.virtual_mass * 9.81),
        #     color=(1.0, 0.5, 0.0, 1.0),
        #     size=5.0,
        # )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.lin_kp * (self.command_setpos_w - self.asset.data.root_pos_w) / (self.virtual_mass * 9.81),
            color=(0.0, 0.5, 1.0, 1.0),
            size=5.0,
        )
        # self.env.debug_draw.vector(
        #     self.asset.data.root_pos_w + quat_rotate(self.asset.data.root_quat_w, self.constant_force.offset),
        #     self.force_ext_w / (self.virtual_mass * 9.81),
        #     color=(1.0, 0.5, 0.0, 1.0),
        #     size=5.0,
        # )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            (self.spring_force.setpoint - self.asset.data.root_pos_w) * self.spring_force.is_valid(),
            color=(1.0, 1.0, 1.0, 1.0),
            size=5.0,
        )
        # self.env.debug_draw.vector(
        #     self.asset.data.root_pos_w,
        #     self._spring_force,
        #     color=(1.0, 0.5, 0.0, 1.0),
        #     size=5.0,
        # )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.impulse_force.get_force() / (self.virtual_mass * 9.81),
            color=(1.0, 0.6, 0.0, 1.0),
            size=5.0,
        )
        ext_pred = self.env.input_tensordict.get("ext_rec", None)
        if ext_pred is not None:
            force_pred_w = quat_rotate(self.asset.data.root_quat_w, ext_pred[:, 0:3])
            # draw predicted external forces (blue)
            self.env.debug_draw.vector(
                self.asset.data.root_pos_w,
                force_pred_w / self.virtual_mass,
                color=(0.0, 0.0, 1.0, 1.0),
                size=4.0,
            )
        
        self.vis_arrow.visualize(
            self.asset.data.root_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            quat_from_euler_xyz(*self.command_setrpy_w.unbind(-1)),
            scales=torch.tensor([[4., 1., 0.1]]).expand(self.num_envs, 3),
        )
        # self.force_vis_arrow.visualize(
        #     self.asset.data.root_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
        #     quat_from_view(self.asset.data.root_pos_w, self.asset.data.root_pos_w + self.force_ext_w/10),
        #     scales=torch.tensor([[4., 1., 0.1]]).expand(self.num_envs, 3),
        # )

        # self.env.debug_draw.vector(
        #     self.asset.data.root_pos_w,
        #     self.asset.data.root_lin_vel_w,
        #     color=(1.0, 0.0, 0.0, 1.0),
        #     size=4.0,
        # )
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.lin_vel_ema.ema[:, 2],
            # self.asset.data.body_lin_vel_w[:, [0, self.torso_id]].mean(1),
            color=(1.0, 1.0, 0.0, 1.0),
            size=4.0,
        )



class ImpedanceImpulse(Impedance):
    
    X_VEL = 1.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trajs = []
        self.done = torch.zeros(self.num_envs, 1, device=self.device, dtype=bool)
        self.ep_id = torch.zeros(self.num_envs, 1, device=self.device, dtype=int)
        self.step_cnt = 0

    def sample_init(self, env_ids):
        init_root_state = self.init_root_state[env_ids]
        if self.env.scene.terrain.cfg.terrain_type == "plane":
            origins = self.env.scene.env_origins[env_ids]
        else:
            idx = torch.randint(0, self.env.num_envs, (len(env_ids),), device=self.device)
            origins = self._origins[idx % len(self._origins)]
        init_root_state[:, :3] += origins
        self.ep_id[env_ids] = self.ep_id[env_ids] + 1
        return init_root_state
    
    def _sample_command(self, env_ids, command_mode=None):
        scalar = torch.empty(len(env_ids), 1, device=self.device)
        lin_kp = scalar.uniform_(24., 32.).clone()
        lin_kd = 1.8 * lin_kp.sqrt()
        ang_kp = lin_kp.clone()
        ang_kd = lin_kd.clone()

        self.lin_kp[env_ids] = lin_kp
        self.lin_kd[env_ids] = lin_kd
        self.ang_kp[env_ids] = ang_kp
        self.ang_kd[env_ids] = ang_kd

        root_pos_w = self.asset.data.root_pos_w[env_ids]
        
        self.command_mode[env_ids] = self.CMD_LINVEL
        self.set_linvel[env_ids] = torch.tensor([self.X_VEL, 0., 0.], device=self.device)
        self.command_setpos_w[env_ids] = root_pos_w + lin_kd / lin_kp * self.set_linvel[env_ids]

        self.virtual_mass[env_ids] = 1.0
        self.virtual_inertia[env_ids] = self.default_inertia 

    def update_command(self):
        # sample_command = (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        # sample_command = sample_command.nonzero().squeeze(-1)
        # if len(sample_command) > 0:
        #     print("resample")
        #     self._sample_command(sample_command)
        self.command_setpos_w[:] = torch.where(
            (self.command_mode == self.CMD_LINVEL).reshape(self.num_envs, 1),
            self.lin_kd / self.lin_kp * self.set_linvel + self.asset.data.root_pos_w,
            # self.command_setpos_w + self.set_linvel * self.env.step_dt,
            self.command_setpos_w,
        )
        offset = torch.zeros(self.num_envs, 3, device=self.device)
        offset[:, 0] = (self.max_output_force / self.lin_kp).squeeze(1)
        self.command_setpos_w[:] = torch.where(
            (self.command_mode == self.CMD_LARGE_FORCE).reshape(self.num_envs, 1),
            self.asset.data.root_pos_w + offset,
            self.command_setpos_w
        )

    def update_forces(self):
        # sample constant force
        at_time = (self.env.episode_length_buf == 100).reshape(self.num_envs, 1)
        sample = at_time & (self.command_mode != self.CMD_LARGE_FORCE).reshape(self.num_envs, 1)
        constant_force = ConstantForce.sample(self.num_envs, device=self.device)
        constant_force.offset.zero_()
        constant_force.force.zero_()
        constant_force.force[:, 0] = 80.   
        constant_force.force[:, 1] = 0.
        constant_force.duration[:] = 4.
        self.constant_force.time.add_(self.env.step_dt)
        self.constant_force: ConstantForce = constant_force.where(sample, self.constant_force)

        at_time = ((self.env.episode_length_buf+1) % 150 == 0).reshape(self.num_envs, 1)
        sample = at_time & (self.command_mode != self.CMD_LARGE_FORCE).reshape(self.num_envs, 1)
        impulse_force = ImpulseForce.sample(self.num_envs, self.device)
        impulse_force.peak.zero_()
        # impulse_force.peak[:, 0] = -100.
        impulse_force.peak[:, 1] = 100 * (self.env.episode_length_buf / 150)
        self.impulse_force.time.add_(self.env.step_dt)
        self.impulse_force: ImpulseForce = impulse_force.where(sample, self.impulse_force)

        self.force_ext_w[:] = (
            # self.constant_force.get_force() \
            + self.impulse_force.get_force()
        )
    
    def update(self):
        super().update()
        root_pos = self.asset.data.root_pos_w - self.env.scene.env_origins
        if self.step_cnt > 0:
            root_pos = torch.where(self.ep_id==1, root_pos, self.trajs[-1])
        self.trajs.append(root_pos)
        self.step_cnt += 1
        # if self.step_cnt == 900:
        #     success_rate = (self.ep_id < 2).float().mean().item()
        #     print(f"success rate: {success_rate}")
        #     success_rate = (self.env.episode_length_buf >= 899).float().mean().item()
        #     print(f"success rate: {success_rate}")
        #     data = {
        #         "pos": torch.stack(self.trajs, dim=0).cpu(),
        #         "success": (self.ep_id < 2).cpu(),
        #     }
        #     # torch.save(data, "trajs_dic_400.pt")
        #     exit(0)


class ImpedanceCollision(Impedance):
    def __init__(self, env, virtual_mass_range=(0.5, 1), linear_kp_range=(2, 12), angular_kp_range=(2, 12), impulse_force_momentum_scale=(5, 5, 1), impulse_force_duration_range=(0.1, 0.5), constant_force_scale=(50, 50, 10), constant_force_duration_range=(1, 4), force_offset_scale=(0, 0, 0), temporal_smoothing = 5, max_acc_xy = (8, 4), max_vel_xy = (1.6, 1), teleop = False):
        super().__init__(env, virtual_mass_range, linear_kp_range, angular_kp_range, impulse_force_momentum_scale, impulse_force_duration_range, constant_force_scale, constant_force_duration_range, force_offset_scale, temporal_smoothing, max_acc_xy, max_vel_xy, teleop)
        self.origins: torch.Tensor = self.env.scene.env_origins
        self.trajs = []
        self.step_cnt = 0

    def sample_init(self, env_ids):
        init_root_state = self.init_root_state[env_ids]
        if self.env.scene.terrain.cfg.terrain_type == "plane":
            origins = self.env.scene.env_origins[env_ids]
        else:
            idx = torch.randint(0, self.env.num_envs, (len(env_ids),), device=self.device)
            origins = self._origins[idx % len(self._origins)]
        init_root_state[:, :3] += origins
        # self.ep_id[env_ids] = self.ep_id[env_ids] + 1
        return init_root_state
    
    def _sample_command(self, env_ids, command_mode):
        scalar = torch.empty(len(env_ids), 1, device=self.device)
        lin_kp = scalar.uniform_(32., 32.).clone()
        lin_kd = 2. * lin_kp.sqrt()
        ang_kp = lin_kp.clone()
        ang_kd = lin_kd.clone()

        self.lin_kp[env_ids] = lin_kp
        self.lin_kd[env_ids] = lin_kd
        self.ang_kp[env_ids] = ang_kp
        self.ang_kd[env_ids] = ang_kd

        root_pos_w = self.asset.data.root_pos_w[env_ids]
        
        self.command_mode[env_ids] = self.CMD_LINVEL
        self.set_linvel[env_ids] = torch.tensor([1.5, 0., 0.], device=self.device)
        self.command_setpos_w[env_ids] = root_pos_w + lin_kd / lin_kp * self.set_linvel[env_ids]

        self.virtual_mass[env_ids] = 1.0
        self.virtual_inertia[env_ids] = self.default_inertia 

    def update_command(self):
        sample_command = (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        sample_command = sample_command.nonzero().squeeze(-1)
        if len(sample_command) > 0:
            self._sample_command(sample_command)
        self.command_setpos_w[:] = torch.where(
            (self.command_mode == self.CMD_LINVEL).reshape(self.num_envs, 1),
            self.lin_kd / self.lin_kp * self.set_linvel + self.asset.data.root_pos_w,
            # self.command_setpos_w + self.set_linvel * self.env.step_dt,
            self.command_setpos_w,
        )
        offset = torch.zeros(self.num_envs, 3, device=self.device)
        offset[:, 0] = (self.max_output_force / self.lin_kp).squeeze(1)
        self.command_setpos_w[:] = torch.where(
            (self.command_mode == self.CMD_LARGE_FORCE).reshape(self.num_envs, 1),
            self.asset.data.root_pos_w + offset,
            self.command_setpos_w
        )
    
    def update_forces(self):
        p = 3. - (self.asset.data.root_pos_w - self.origins)[:, 0]
        self.force_ext_w[:, 0] = torch.where(p < 0, 700 * p - 4 * self.asset.data.root_vel_w[:, 0], 0)

    def update(self):
        super().update()
        self.trajs.append(self.force_ext_w.clone())
        if self.step_cnt == 800:
            torch.save(torch.stack(self.trajs, dim=0).cpu(), "trajs_dic_col24.pt")
            exit(0)
        self.step_cnt += 1


class VelocityImpulse(Command2):
    X_VEL = 1.5

    def __init__(self, env, linvel_x_range=..., linvel_y_range=..., angvel_range=..., yaw_stiffness_range=..., use_stiffness_ratio = 0.5, aux_input_range=..., resample_interval = 300, resample_prob = 0.75, stand_prob=0.2, target_yaw_range=..., adaptive = False, body_name = None, teleop = False):
        super().__init__(env, linvel_x_range, linvel_y_range, angvel_range, yaw_stiffness_range, use_stiffness_ratio, aux_input_range, resample_interval, resample_prob, stand_prob, target_yaw_range, adaptive, body_name, teleop)
        self.impulse_force = ImpulseForce.sample(self.num_envs, device=self.device)
        self.impulse_force.peak.zero_()
        self.constand_force = ConstantForce.sample(self.num_envs, device=self.device)
        self.constand_force.force.zero_()

        self.trajs = []
        self.done = torch.zeros(self.num_envs, 1, device=self.device, dtype=bool)
        self.ep_id = torch.zeros(self.num_envs, 1, device=self.device, dtype=int)
        self.step_cnt = 0

    def sample_init(self, env_ids):
        init_root_state = self.init_root_state[env_ids]
        if self.env.scene.terrain.cfg.terrain_type == "plane":
            origins = self.env.scene.env_origins[env_ids]
        else:
            idx = torch.randint(0, self.env.num_envs, (len(env_ids),), device=self.device)
            origins = self._origins[idx % len(self._origins)]
        init_root_state[:, :3] += origins
        self.ep_id[env_ids] = self.ep_id[env_ids] + 1
        return init_root_state
    
    def reset(self, env_ids, reward_stats=None):
        super().reset(env_ids, reward_stats)
        self.sample_vel_command(env_ids)
        self.sample_yaw_command(env_ids)
    
    def sample_vel_command(self, env_ids):
        next_command_linvel = torch.zeros(len(env_ids), 3, device=self.device)
        next_command_linvel[:, 0] = self.X_VEL

        self.next_command_linvel[env_ids] = next_command_linvel
        self.aux_input[env_ids] = 0.5
    
    def sample_yaw_command(self, env_ids):
        self.target_yaw[env_ids] = 0.
        self.yaw_stiffness[env_ids] = 1.1
        self.use_stiffness[env_ids] = True
        self.fixed_yaw_speed[env_ids] = 0.

    def step(self, substep):
        super().step(substep)
        forces_b = self.asset._external_force_b
        impulse_force = self.impulse_force.get_force()
        forces_b[:, 0] += quat_rotate_inverse(self.asset.data.root_quat_w, impulse_force)
        # forces_b[:, 0] += quat_rotate_inverse(self.asset.data.root_quat_w, self.constand_force.get_force())
        self.asset.has_external_wrench = True

    def update(self):
        super().update()
        self.command_linvel = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_quat_w),
            torch.tensor([[self.X_VEL, 0., 0.]], device=self.device)
        )
        sample = ((self.env.episode_length_buf+1) % 150 == 0).reshape(self.num_envs, 1)
        impulse_force = ImpulseForce.sample(self.num_envs, self.device)
        impulse_force.peak.zero_()
        impulse_force.peak[:, 0] = -200.
        impulse_force.peak[:, 1] = 100 * (self.env.episode_length_buf / 150)
        self.impulse_force.time.add_(self.env.step_dt)
        self.impulse_force: ImpulseForce = impulse_force.where(sample, self.impulse_force)

        constant_force = ConstantForce.sample(self.num_envs, device=self.device)
        constant_force.offset.zero_()
        constant_force.force.zero_()
        constant_force.force[:, 1] = 160.
        constant_force.duration[:] = 3.
        self.constand_force.time.add_(self.env.step_dt)
        self.constand_force: ConstantForce = constant_force.where(sample, self.constand_force)
        
        root_pos = self.asset.data.root_pos_w - self.env.scene.env_origins
        if self.step_cnt > 0:
            root_pos = torch.where(self.ep_id==1, root_pos, self.trajs[-1])
        self.trajs.append(root_pos)
        self.step_cnt += 1
        # if self.step_cnt == 900:
        #     success_rate = (self.ep_id < 2).float().mean().item()
        #     print(f"success rate: {success_rate}")
        #     success_rate = (self.env.episode_length_buf >= 899).float().mean().item()
        #     print(f"success rate: {success_rate}")
        #     data = {
        #         "pos": torch.stack(self.trajs, dim=0).cpu(),
        #         "success": (self.ep_id < 2).cpu(),
        #     }
        #     # torch.save(data, "trajs_vel1_400.pt")
        #     exit(0)

    def debug_draw(self):
        super().debug_draw()
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.impulse_force.get_force() /  9.81,
            color=(1.0, 0.6, 0.0, 1.0),
            size=3.0,
        )
        # self.env.debug_draw.vector(
        #     self.asset.data.root_pos_w,
        #     self.constand_force.get_force() /  9.81,
        #     color=(1.0, 0.6, 0.0, 1.0),
        #     size=3.0,
        # )


class VelocityCollision(Command2):
    X_VEL = 1.5

    def __init__(self, env, linvel_x_range=..., linvel_y_range=..., angvel_range=..., yaw_stiffness_range=..., use_stiffness_ratio = 0.5, aux_input_range=..., resample_interval = 300, resample_prob = 0.75, stand_prob=0.2, target_yaw_range=..., adaptive = False, body_name = None, teleop = False):
        super().__init__(env, linvel_x_range, linvel_y_range, angvel_range, yaw_stiffness_range, use_stiffness_ratio, aux_input_range, resample_interval, resample_prob, stand_prob, target_yaw_range, adaptive, body_name, teleop)
        self.force_ext_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.origins = self.env.scene.env_origins
        self.trajs = []
        self.step_cnt = 0

    def sample_init(self, env_ids):
        init_root_state = self.init_root_state[env_ids]
        if self.env.scene.terrain.cfg.terrain_type == "plane":
            origins = self.env.scene.env_origins[env_ids]
        else:
            idx = torch.randint(0, self.env.num_envs, (len(env_ids),), device=self.device)
            origins = self._origins[idx % len(self._origins)]
        init_root_state[:, :3] += origins
        # self.ep_id[env_ids] = self.ep_id[env_ids] + 1
        return init_root_state
    
    def reset(self, env_ids, reward_stats=None):
        super().reset(env_ids, reward_stats)
        self.sample_vel_command(env_ids)
        self.sample_yaw_command(env_ids)
    
    def sample_vel_command(self, env_ids):
        next_command_linvel = torch.zeros(len(env_ids), 3, device=self.device)
        next_command_linvel[:, 0] = self.X_VEL

        self.next_command_linvel[env_ids] = next_command_linvel
        self.aux_input[env_ids] = 0.5
    
    def sample_yaw_command(self, env_ids):
        self.target_yaw[env_ids] = 0.
        self.yaw_stiffness[env_ids] = 1.1
        self.use_stiffness[env_ids] = True
        self.fixed_yaw_speed[env_ids] = 0.
    
    def update(self):
        super().update()
        p = 3. - (self.asset.data.root_pos_w - self.origins)[:, 0]
        self.force_ext_w[:, 0] = torch.where(p < 0, 700 * p - 4 * self.asset.data.root_vel_w[:, 0], 0)
        self.trajs.append(self.force_ext_w.clone())

        if self.step_cnt == 800:
            torch.save(torch.stack(self.trajs, dim=0).cpu(), "trajs_vel_col.pt")
            exit(0)
        self.step_cnt += 1

    def step(self, substep):
        self.asset._external_force_b[:, 0] += quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.force_ext_w
        )
        self.asset.has_external_wrench = True