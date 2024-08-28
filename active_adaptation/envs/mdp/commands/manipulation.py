from .base import Command
from .locomotion import clamp_norm, quat_rotate_inverse, quat_rotate
from omni.isaac.lab.assets import Articulation
import torch


class EEImpedance(Command):
    """
    Command the EE position to have impedance in the task/operational space.
    """

    def __init__(
        self,
        env,
        ee_name: str,
        ee_base_name: str,
        kp_range: tuple = (100.0, 150.0),
        damping_ratio_range: tuple = (0.7, 1.5),
        virtual_mass_range: tuple = (0.5, 1.5),
        max_force: float = 20.0,
        compliant_ratio: float = 0.2,
        ext_force_ratio: float = 0.5,
        future: int = 3,
        mix_openloop: bool = False,
    ) -> None:
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.ee_name = ee_name
        self.ee_base_name = ee_base_name
        self.ee_body_id = self.robot.find_bodies(ee_name)[0][0]
        self.ee_base_body_id = self.robot.find_bodies(ee_base_name)[0][0]

        self.kp_range = kp_range
        self.damping_ratio_range = damping_ratio_range
        self.virtual_mass_range = virtual_mass_range
        self.max_force = max_force
        

        self.compliant_ratio = compliant_ratio
        self.ext_force_ratio = ext_force_ratio
        
        self.resample_prob = 0.005
        self.future = future
        self.mix_openloop = mix_openloop

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 9)
            self.command_hidden = torch.zeros(self.num_envs, 6)

            # integration
            self.acc_spring_ee_w = torch.zeros(self.num_envs, self.future, 3)
            self.desired_linacc_ee_w = torch.zeros(self.num_envs, self.future, 3)
            self.desired_linvel_ee_w = torch.zeros(self.num_envs, self.future, 3)
            self.desired_pos_ee_w = torch.zeros(self.num_envs, self.future, 3)

            # command setpoints
            self.command_setpoint_pos_ee_b = torch.zeros(self.num_envs, 3)
            self.command_setpoint_pos_ee_diff_b = torch.zeros(self.num_envs, 3)

            self.command_pos_ee_w = torch.zeros(self.num_envs, 3)
            self.command_linvel_ee_w = torch.zeros(self.num_envs, 3)
            self.command_pos_ee_b = torch.zeros(self.num_envs, 3)
            self.command_linvel_ee_b = torch.zeros(self.num_envs, 3)
            self.command_pos_ee_diff_b = torch.zeros(self.num_envs, 3)

            self.command_kp = torch.zeros(self.num_envs, 3)
            self.command_kd = torch.zeros(self.num_envs, 3)

            self.default_mass_ee = 1.0
            self.virtual_mass_ee = torch.zeros(self.num_envs, 1)

            self.force_ext_ee_w = torch.zeros(self.num_envs, 3)

            self._cum_error = torch.zeros(self.num_envs, 2)

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.init_root_state[env_ids]

    def _sample_command(self, env_ids: torch.Tensor):
        command_setpoint_pos_ee_b = torch.empty(len(env_ids), 3, device=self.device)
        command_setpoint_pos_ee_b[:, 0].uniform_(0.2, 0.6)
        command_setpoint_pos_ee_b[:, 1].uniform_(-0.2, 0.2)
        command_setpoint_pos_ee_b[:, 2].uniform_(0.2, 0.6)
        self.command_setpoint_pos_ee_b[env_ids] = command_setpoint_pos_ee_b

        kp_ee = torch.empty(len(env_ids), 3, device=self.device).uniform_(
            *self.kp_range
        )
        compliant_ee = (
            torch.rand(len(env_ids), 1, device=self.device) < self.compliant_ratio
        )
        kd_ee = (
            2.0
            * torch.sqrt(kp_ee)
            * torch.empty(len(env_ids), 3, device=self.device).uniform_(
                *self.damping_ratio_range
            )
        )
        self.command_kp[env_ids] = kp_ee * (~compliant_ee)
        self.command_kd[env_ids] = kd_ee

        self.virtual_mass_ee[env_ids] = (
            torch.empty(len(env_ids), 1, device=self.device).uniform_(
                *self.virtual_mass_range
            )
            * self.default_mass_ee
        )

    def _sample_force(self, env_ids: torch.Tensor):
        force_ext_ee_w = torch.empty(len(env_ids), 3, device=self.device).uniform_(
            -self.max_force, self.max_force
        )
        force_ext_ee_w = clamp_norm(
            force_ext_ee_w, max=self.virtual_mass_ee[env_ids] * 2.0
        )
        apply_force = (
            torch.rand(len(env_ids), 1, device=self.device) < self.ext_force_ratio
        )
        self.force_ext_ee_w[env_ids] = force_ext_ee_w * apply_force

    def reset(self, env_ids: torch.Tensor):
        self._sample_command(env_ids)
        self._sample_force(env_ids)

        self.desired_linacc_ee_w[env_ids] = 0.0
        self.desired_linvel_ee_w[env_ids] = self.asset.data.body_lin_vel_w[
            env_ids, None, self.ee_body_id
        ]
        self.desired_pos_ee_w[env_ids] = self.asset.data.body_pos_w[
            env_ids, None, self.ee_body_id
        ]

    def step(self, substep: int):
        forces_ee_b = self.asset._external_force_b[:, [self.ee_body_id]].clone()
        forces_ee_b += quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.force_ext_ee_w,
        )[:, None, :]
        quat_rotate_inverse(self.asset.data.root_quat_w, self.force_ext_ee_w)
        torques_ee_b = self.asset._external_torque_b[:, [self.ee_body_id]].clone()
        self.asset.set_external_force_and_torque(
            forces_ee_b, torques_ee_b, [self.ee_body_id]
        )

    def _integrate(self):
        command_setpoint_pos_ee_w = (
            quat_rotate(
                self.asset.data.root_quat_w,
                self.command_setpoint_pos_ee_b,
            )
            + self.asset.data.root_pos_w
        )

        self.acc_spring_ee_w[:] = self.command_kp.unsqueeze(1) * (
            command_setpoint_pos_ee_w.unsqueeze(1) - self.desired_pos_ee_w
        ) + self.command_kd.unsqueeze(1) * (-self.desired_linvel_ee_w)
        self.desired_linacc_ee_w[:] = self.acc_spring_ee_w + (
            self.force_ext_ee_w / self.virtual_mass_ee
        ).unsqueeze(1)
        self.desired_linvel_ee_w.add_(self.desired_linacc_ee_w * self.env.physics_dt)
        self.desired_pos_ee_w.add_(self.desired_linvel_ee_w * self.env.physics_dt)

    def _compute_error(self):
        linvel_ee_error = (
            self.asset.data.body_lin_vel_w[:, self.ee_body_id]
            - self.desired_linvel_ee_w[:, 0]
        ).norm(dim=-1)
        pos_ee_error = (
            self.asset.data.body_pos_w[:, self.ee_body_id]
            - self.desired_pos_ee_w[:, 0]
        ).norm(dim=-1)
        self._cum_error[:, 0].add_(linvel_ee_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 1].add_(pos_ee_error * self.env.step_dt).mul_(0.99)

    def update(self):
        if not self.mix_openloop:
            self.desired_linvel_ee_w.roll(1, 1)
            self.desired_pos_ee_w.roll(1, 1)
        self.desired_linvel_ee_w[:, 0] = self.asset.data.body_lin_vel_w[
            :, self.ee_body_id
        ]
        self.desired_pos_ee_w[:, 0] = self.asset.data.body_pos_w[:, self.ee_body_id]

        assert int(self.env.step_dt / self.env.physics_dt) == 4
        for _ in range(int(self.env.step_dt / self.env.physics_dt)):
            self._integrate()

        ee_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.asset.data.body_pos_w[:, self.ee_body_id] - self.asset.data.root_pos_w,
        )
        self.command_setpoint_pos_ee_diff_b[:] = (
            self.command_setpoint_pos_ee_b - ee_pos_b
        )

        self.command_pos_ee_w[:] = self.desired_pos_ee_w.mean(1)
        self.command_linvel_ee_w[:] = self.desired_linvel_ee_w.mean(1)

        self.command_pos_ee_b[:] = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.command_pos_ee_w,
        )
        self.command_linvel_ee_b[:] = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.command_linvel_ee_w,
        )

        self.command_pos_ee_diff_b[:] = self.command_pos_ee_b - ee_pos_b

        self.command[:, 0:3] = self.command_setpoint_pos_ee_b
        self.command[:, 3:6] = self.command_kp
        self.command[:, 6:9] = self.command_kd

        self.command_hidden[:, 0:3] = self.command_pos_ee_diff_b
        self.command_hidden[:, 3:6] = self.command_linvel_ee_b

        self._compute_error()
        
        sample_command = torch.rand(self.num_envs, device=self.device) < self.resample_prob
        sample_command = sample_command.nonzero().squeeze(-1)
        if len(sample_command) > 0:
            self._sample_command(sample_command)
        
        sample_force = torch.rand(self.num_envs, device=self.device) < self.resample_prob
        sample_force = sample_force.nonzero().squeeze(-1)
        if len(sample_force) > 0:
            self._sample_force(sample_force)
        
        

    def debug_draw(self):
        # draw desired position for ee (green)
        self.env.debug_draw.point(
            self.command_pos_ee_w,
            color=(0.0, 1.0, 0.0, 1.0),
            size=10.0,
        )
        # command linvel for ee (green)
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.ee_body_id],
            self.command_linvel_ee_w,
            color=(0.0, 1.0, 0.0, 1.0),
            size=1.0,
        )
        # draw vector from desired to setpoint (blue)
        command_setpoint_pos_ee_w = (
            quat_rotate(
                self.asset.data.root_quat_w,
                self.command_setpoint_pos_ee_b,
            )
            + self.asset.data.root_pos_w
        )
        self.env.debug_draw.vector(
            self.command_pos_ee_w,
            command_setpoint_pos_ee_w - self.command_pos_ee_w,
            color=(0.0, 0.0, 1.0, 1.0),
            size=1.0,
        )
        # draw setpoint position for ee (red)
        self.env.debug_draw.point(
            command_setpoint_pos_ee_w,
            color=(1.0, 0.0, 0.0, 1.0),
            size=10.0,
        )
        # draw external force on desired ee (orange)
        self.env.debug_draw.vector(
            self.command_pos_ee_w,
            self.force_ext_ee_w,
            color=(1.0, 0.5, 0.0, 1.0),
            size=4.0,
        )
        
        
        
