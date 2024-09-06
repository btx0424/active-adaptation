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
        setpoint_x_range: tuple = (0.2, 0.6),
        setpoint_y_range: tuple = (-0.2, 0.2),
        setpoint_z_range: tuple = (0.2, 0.6),
        relative_setpoint: bool = False,
        kp_range: tuple = (100.0, 150.0),
        damping_ratio_range: tuple = (0.7, 1.5),
        default_mass_ee: float = 1.0,
        virtual_mass_range: tuple = (0.5, 1.5),
        max_force_acc: float = 20.0,
        compliant_ratio: float = 0.2,
        ext_force_ratio: float = 0.5,
        future: int = 3,
        mix_openloop: bool = False,
        command_acc: bool = False,
        spring_force: bool = False,
        force_kp_range: tuple = (100.0, 500.0),
    ) -> None:
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.ee_name = ee_name
        self.ee_base_name = ee_base_name
        self.ee_body_id = self.robot.find_bodies(ee_name)[0][0]
        self.ee_base_body_id = self.robot.find_bodies(ee_base_name)[0][0]

        self.setpoint_x_range = setpoint_x_range
        self.setpoint_y_range = setpoint_y_range
        self.setpoint_z_range = setpoint_z_range
        self.relative_setpoint = relative_setpoint

        self.kp_range = kp_range
        self.damping_ratio_range = damping_ratio_range
        self.virtual_mass_range = virtual_mass_range
        self.max_force_acc = max_force_acc

        self.compliant_ratio = compliant_ratio
        self.ext_force_ratio = ext_force_ratio

        self.resample_prob = 0.005
        self.future = future
        self.mix_openloop = mix_openloop
        self.command_acc = command_acc
        self.spring_force = spring_force
        self.force_kp_range = force_kp_range

        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 10)
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

            self.compliant_ee = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.command_kp = torch.zeros(self.num_envs, 3)
            self.command_kd = torch.zeros(self.num_envs, 3)

            self.default_mass_ee = default_mass_ee
            self.virtual_mass_ee = torch.zeros(self.num_envs, 1)

            self.apply_force = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            if not self.spring_force:
                self.force_ext_ee_w = torch.zeros(self.num_envs, 3)
            else:
                self.force_ext_ee_setpoint_w = torch.zeros(self.num_envs, 3)
                self.force_ext_ee_kp = torch.zeros(self.num_envs, 3)

            self._cum_error = torch.zeros(self.num_envs, 2)

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.init_root_state[env_ids]

    def _sample_command(self, env_ids: torch.Tensor):
        command_setpoint_pos_ee_b = torch.empty(len(env_ids), 3, device=self.device)
        command_setpoint_pos_ee_b[:, 0].uniform_(*self.setpoint_x_range)
        command_setpoint_pos_ee_b[:, 1].uniform_(*self.setpoint_y_range)
        command_setpoint_pos_ee_b[:, 2].uniform_(*self.setpoint_z_range)
        if self.relative_setpoint:
            ee_pos_b = quat_rotate_inverse(
                self.asset.data.root_quat_w[env_ids],
                self.asset.data.body_pos_w[env_ids, self.ee_body_id] - self.asset.data.root_pos_w[env_ids],
            )
            command_setpoint_pos_ee_b += ee_pos_b
            command_setpoint_pos_ee_b[:, 0].clamp_(0.0, 0.8)
            command_setpoint_pos_ee_b[:, 1].clamp_(-0.4, 0.4)
            command_setpoint_pos_ee_b[:, 2].clamp_(0.2, 0.8)
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
        self.compliant_ee[env_ids] = compliant_ee
        self.command_kp[env_ids] = kp_ee * (~compliant_ee)
        self.command_kd[env_ids] = kd_ee

        self.virtual_mass_ee[env_ids] = (
            torch.empty(len(env_ids), 1, device=self.device).uniform_(
                *self.virtual_mass_range
            )
            * self.default_mass_ee
        )

    def _sample_force(self, env_ids: torch.Tensor):
        if not self.spring_force:
            force_ext_ee_w = torch.empty(len(env_ids), 3, device=self.device).uniform_(
                -50.0, 50.0
            )
            force_ext_ee_w = clamp_norm(
                force_ext_ee_w, max=self.virtual_mass_ee[env_ids] * self.max_force_acc
            )
            self.force_ext_ee_w[env_ids] = force_ext_ee_w
        else:
            rel_force_ext_ee_setpoint_w = torch.empty(len(env_ids), 3, device=self.device).uniform_(
                -0.2, 0.2
            )
            ee_pos_w = self.asset.data.body_pos_w[env_ids, self.ee_body_id]
            force_ext_ee_kp = torch.empty(len(env_ids), 3, device=self.device).uniform_(
                *self.force_kp_range
            )
            self.force_ext_ee_setpoint_w[env_ids] = rel_force_ext_ee_setpoint_w + ee_pos_w
            self.force_ext_ee_kp[env_ids] = force_ext_ee_kp

        apply_force = (
            torch.rand(len(env_ids), 1, device=self.device) < self.ext_force_ratio
        )
        self.apply_force[env_ids] = apply_force
        

    def _update_command(self):
        # compute command in world/body frame
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

        # compute diff
        ee_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.asset.data.body_pos_w[:, self.ee_body_id] - self.asset.data.root_pos_w,
        )
        self.command_setpoint_pos_ee_diff_b[:] = (
            self.command_setpoint_pos_ee_b - ee_pos_b
        )
        self.command_pos_ee_diff_b[:] = self.command_pos_ee_b - ee_pos_b

        # populate command tensor
        self.command[:, 0:3] = self.command_setpoint_pos_ee_diff_b * (
            ~self.compliant_ee
        )
        self.command[:, 3:6] = self.command_kp
        self.command[:, 6:9] = self.command_kd
        if self.command_acc:
            ee_linvel_b = quat_rotate_inverse(
                self.asset.data.root_quat_w,
                self.asset.data.body_lin_vel_w[:, self.ee_body_id],
            )
            self.command[:, 3:6] *= self.command_setpoint_pos_ee_diff_b
            self.command[:, 6:9] *= - ee_linvel_b
        self.command[:, 9:10] = self.virtual_mass_ee

        self.command_hidden[:, 0:3] = self.command_pos_ee_diff_b
        self.command_hidden[:, 3:6] = self.command_linvel_ee_b

    def reset(self, env_ids: torch.Tensor):
        self._sample_command(env_ids)
        self._sample_force(env_ids)

        self._cum_error[env_ids] = 0.0

        self.desired_linacc_ee_w[env_ids] = 0.0
        self.desired_linvel_ee_w[env_ids] = self.asset.data.body_lin_vel_w[
            env_ids, None, self.ee_body_id
        ]
        self.desired_pos_ee_w[env_ids] = self.asset.data.body_pos_w[
            env_ids, None, self.ee_body_id
        ]

        for _ in range(int(self.env.step_dt / self.env.physics_dt)):
            self._integrate()

        # sim reset -> command_manager.reset() -> compute obs ->  sim step -> compute reward
        self._update_command()

    def step(self, substep: int):
        if not self.spring_force:
            force_ext_ee_w = self.force_ext_ee_w
        else:
            force_ext_ee_w = (
                self.force_ext_ee_setpoint_w - self.asset.data.body_pos_w[:, self.ee_body_id]
            ) * self.force_ext_ee_kp
        
        force_ext_ee_w *= self.apply_force
        forces_ee_b = self.asset._external_force_b[:, [self.ee_body_id]].clone()
        forces_ee_b += quat_rotate_inverse(
            self.asset.data.body_quat_w[:, self.ee_body_id],
            force_ext_ee_w,
        )[:, None, :]
        torques_ee_b = self.asset._external_torque_b[:, [self.ee_body_id]].clone()
        self.asset.set_external_force_and_torque(
            forces_ee_b, torques_ee_b, [self.ee_body_id]
        )

    def _integrate(self):
        if not self.spring_force:
            force_ext_ee_w = self.force_ext_ee_w.unsqueeze(1).repeat(1, self.future, 1)
        else:
            force_ext_ee_w = (
                self.force_ext_ee_setpoint_w.unsqueeze(1) - self.desired_pos_ee_w
            ) * self.force_ext_ee_kp.unsqueeze(1)
        force_ext_ee_w *= self.apply_force.unsqueeze(1)
        
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
            force_ext_ee_w / self.virtual_mass_ee.unsqueeze(1)
        )
        self.desired_linvel_ee_w.add_(self.desired_linacc_ee_w * self.env.physics_dt)
        self.desired_pos_ee_w.add_(self.desired_linvel_ee_w * self.env.physics_dt)

    def _compute_error(self):
        linvel_ee_error = (
            self.asset.data.body_lin_vel_w[:, self.ee_body_id]
            - self.command_linvel_ee_w
        ).norm(dim=-1)
        pos_ee_error = (
            self.asset.data.body_pos_w[:, self.ee_body_id] - self.command_pos_ee_w
        ).norm(dim=-1)
        self._cum_error[:, 0].add_(linvel_ee_error * self.env.step_dt).mul_(0.99)
        self._cum_error[:, 1].add_(pos_ee_error * self.env.step_dt).mul_(0.99)

    def update(self):
        self._compute_error()

        # resample command and force
        # sim step -> compute reward -> command_manager.update() -> compute obs
        # the command/forces will be issued/applied in the next time step
        sample_command = (
            torch.rand(self.num_envs, device=self.device) < self.resample_prob
        )
        sample_command = sample_command.nonzero().squeeze(-1)
        if len(sample_command) > 0:
            self._sample_command(sample_command)

        sample_force = (
            torch.rand(self.num_envs, device=self.device) < self.resample_prob
        )
        sample_force = sample_force.nonzero().squeeze(-1)
        if len(sample_force) > 0:
            self._sample_force(sample_force)

        # update desired quantities under the current command and forces
        if not self.mix_openloop:
            self.desired_linvel_ee_w[:] = self.desired_linvel_ee_w.roll(1, 1)
            self.desired_pos_ee_w[:] = self.desired_pos_ee_w.roll(1, 1)
        self.desired_linvel_ee_w[:, 0] = self.asset.data.body_lin_vel_w[
            :, self.ee_body_id
        ]
        self.desired_pos_ee_w[:, 0] = self.asset.data.body_pos_w[:, self.ee_body_id]

        assert int(self.env.step_dt / self.env.physics_dt) == 4
        for _ in range(int(self.env.step_dt / self.env.physics_dt)):
            self._integrate()

        self._update_command()

    def _debug_draw_setpoint_boundaries(self):
        # draw the 8 setpoint boundaries
        setpoint_bounds = torch.tensor(
            [
                [
                    self.setpoint_x_range[0],
                    self.setpoint_y_range[0],
                    self.setpoint_z_range[0],
                ],
                [
                    self.setpoint_x_range[0],
                    self.setpoint_y_range[0],
                    self.setpoint_z_range[1],
                ],
                [
                    self.setpoint_x_range[0],
                    self.setpoint_y_range[1],
                    self.setpoint_z_range[0],
                ],
                [
                    self.setpoint_x_range[0],
                    self.setpoint_y_range[1],
                    self.setpoint_z_range[1],
                ],
                [
                    self.setpoint_x_range[1],
                    self.setpoint_y_range[0],
                    self.setpoint_z_range[0],
                ],
                [
                    self.setpoint_x_range[1],
                    self.setpoint_y_range[0],
                    self.setpoint_z_range[1],
                ],
                [
                    self.setpoint_x_range[1],
                    self.setpoint_y_range[1],
                    self.setpoint_z_range[0],
                ],
                [
                    self.setpoint_x_range[1],
                    self.setpoint_y_range[1],
                    self.setpoint_z_range[1],
                ],
            ],
            device=self.device,
        )
        # [8, 3]

        # Transform these points into the world frame using the base frame's orientation and position
        setpoint_bounds_world = (
            setpoint_bounds.unsqueeze(0)
            + self.asset.data.root_pos_w.unsqueeze(1)
        ).view(-1, 3)
        self.env.debug_draw.point(
            setpoint_bounds_world,
            color=(1.0, 1.0, 1.0, 1.0),
            size=15.0,
        )
        

    def debug_draw(self):
        self._debug_draw_setpoint_boundaries()
        # command position for ee (green)
        self.env.debug_draw.point(
            self.command_pos_ee_w,
            color=(0.0, 1.0, 0.0, 1.0),
            size=10.0,
        )
        # command linvel for ee (green)
        self.env.debug_draw.vector(
            self.command_pos_ee_w,
            self.command_linvel_ee_w,
            color=(0.0, 1.0, 0.0, 1.0),
            size=1.0,
        )
        # draw vector from desired to setpoint for ee (blue)
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
        # draw actual position for ee (yellow)
        self.env.debug_draw.point(
            self.asset.data.body_pos_w[:, self.ee_body_id],
            color=(1.0, 1.0, 0.0, 1.0),
            size=10.0,
        )
        if not self.spring_force:
            force_ext_ee_w = self.force_ext_ee_w
        else:
            force_ext_ee_w = (
                self.force_ext_ee_setpoint_w - self.asset.data.body_pos_w[:, self.ee_body_id]
            ) * self.force_ext_ee_kp
        force_ext_ee_w *= self.apply_force
        
        # draw external force on desired ee (orange)
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.ee_body_id],
            force_ext_ee_w,
            color=(1.0, 0.5, 0.0, 1.0),
            size=4.0,
        )
        if self.spring_force:
            # draw external force setpoint (orange)
            self.env.debug_draw.point(
                self.force_ext_ee_setpoint_w[self.apply_force.squeeze(-1)],
                color=(1.0, 0.5, 0.0, 1.0),
                size=10.0,
            )
        # draw a point to indicate if the manipulator is compliant (green for compliant, red for non-compliant)
        self.env.debug_draw.point(
            self.asset.data.root_pos_w[self.compliant_ee.squeeze(-1)]
            + torch.tensor([0.1, 0.0, 0.0], device=self.device),
            color=(0.0, 1.0, 0.0, 1.0),
            size=10.0,
        )
        self.env.debug_draw.point(
            self.asset.data.root_pos_w[~self.compliant_ee.squeeze(-1)]
            + torch.tensor([0.1, 0.0, 0.0], device=self.device),
            color=(1.0, 0.0, 0.0, 1.0),
            size=10.0,
        )

class PushWall(Command):
    """Same as above, except that the force is replaced by a wall simulated with large penetration kp and the command setpoit is programmed to reach the wall."""
    def __init__(
        self,
        env,
        ee_name: str = "arm_link6",
    ) -> None:
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.ee_body_id = self.robot.find_bodies(ee_name)[0][0]
            
        self.default_mass_ee = 1.0
        self.kp_ee_range = (100.0, 150.0)
        self.damping_ratio_range = (1.0, 1.5)
        self.virtual_mass_range = (0.5, 1.5)
        self.future = 3

        self.wall_kp = 500.0

        self.enable_teleop = True
        self.step_size = 0.005

        self.command_acc = True
        
        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 10)
            self.command_hidden = torch.zeros(self.num_envs, 6)
            
            # integration
            self.acc_spring_ee_w = torch.zeros(self.num_envs, self.future, 3)
            self.desired_linacc_ee_w = torch.zeros(self.num_envs, self.future, 3)
            self.desired_linvel_ee_w = torch.zeros(self.num_envs, self.future, 3)
            self.desired_pos_ee_w = torch.zeros(self.num_envs, self.future, 3)

            # command setpoints
            self.command_setpoint_pos_ee_wall = torch.zeros(self.num_envs, 3)
            self.command_setpoint_pos_ee_b = torch.zeros(self.num_envs, 3)
            self.command_setpoint_pos_ee_diff_b = torch.zeros(self.num_envs, 3)

            self.command_pos_ee_w = torch.zeros(self.num_envs, 3)
            self.command_linvel_ee_w = torch.zeros(self.num_envs, 3)
            self.command_pos_ee_b = torch.zeros(self.num_envs, 3)
            self.command_linvel_ee_b = torch.zeros(self.num_envs, 3)
            self.command_pos_ee_diff_b = torch.zeros(self.num_envs, 3)

            self.command_kp = torch.zeros(self.num_envs, 3)
            self.command_kd = torch.zeros(self.num_envs, 3)
            self.virtual_mass_ee = torch.zeros(self.num_envs, 1)

            self.wall_direction = torch.zeros(self.num_envs, 3)
            self.wall_quat = torch.zeros(self.num_envs, 4)
            self.wall_distance = torch.zeros(self.num_envs, 1)    
            self.wall_resistance_force_w = torch.zeros(self.num_envs, 3)
            self.wall_friction_force_w = torch.zeros(self.num_envs, 3)

            self.need_reset_mask = torch.ones(self.num_envs, dtype=torch.bool)
            self._cum_error = torch.zeros(self.num_envs, 2)
        
        self.log_file = "push_wall_force.pkl"
        self.actual_forces = []
        self.desired_forces = []
        self.desired_kp_forces = []
        self.kp_forces = []

        self.key_pressed = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
            "w": False,
            "s": False,
            "a": False,
            "d": False,
            "q": False,
            "e": False,
        }

        from pynput import keyboard
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.listener.start()
        print("[KeyboardCommandManager]: Keyboard listener started")

    def _on_press(self, key):
        try:
            if key.char.lower() in self.key_pressed:
                self.key_pressed[key.char.lower()] = True
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_pressed["up"] = True
            elif key == keyboard.Key.down:
                self.key_pressed["down"] = True
            elif key == keyboard.Key.left:
                self.key_pressed["left"] = True
            elif key == keyboard.Key.right:
                self.key_pressed["right"] = True

    def _on_release(self, key):
        try:
            if key.char.lower() in self.key_pressed:
                self.key_pressed[key.char.lower()] = False
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_pressed["up"] = False
            elif key == keyboard.Key.down:
                self.key_pressed["down"] = False
            elif key == keyboard.Key.left:
                self.key_pressed["left"] = False
            elif key == keyboard.Key.right:
                self.key_pressed["right"] = False
        
    def save_logs(self):
        print("Saving log file to", self.log_file)
        actual_forces = torch.stack(self.actual_forces, dim=0).cpu().numpy()
        desired_forces = torch.stack(self.desired_forces, dim=0).cpu().numpy()
        desired_kp_forces = torch.stack(self.desired_kp_forces, dim=0).cpu().numpy()
        kp_forces = torch.stack(self.kp_forces, dim=0).cpu().numpy()
        with open(self.log_file, "wb") as f:
            import pickle
            pickle.dump((actual_forces, desired_forces, desired_kp_forces, kp_forces), f)
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.init_root_state[env_ids]

    def _sample_command(self, env_ids: torch.Tensor):
        # sample kp kd
        kp_ee = torch.empty(len(env_ids), 3, device=self.device).uniform_(*self.kp_ee_range)
        kd_ee = (
            2.0
            * torch.sqrt(kp_ee)
            * torch.empty(len(env_ids), 3, device=self.device).uniform_(
                *self.damping_ratio_range
            )
        )
        self.command_kp[env_ids] = kp_ee
        self.command_kd[env_ids] = kd_ee

        virtual_mass_factor = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.virtual_mass_range
        )
        self.virtual_mass_ee[env_ids] = virtual_mass_factor * self.default_mass_ee
        
    def _sample_wall(self, env_ids: torch.Tensor):
        # sample wall normal direction
        wall_normal_yaw = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi / 6, torch.pi / 6)
        wall_normal_pitch = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi / 6, 0)
        # wall_direction = [1, 0, 0] rotate with yaw pitch
        wall_direction = torch.stack([
            torch.cos(wall_normal_yaw) * torch.cos(wall_normal_pitch),
            torch.sin(wall_normal_yaw) * torch.cos(wall_normal_pitch),
            -torch.sin(wall_normal_pitch)
        ], dim=-1)
        wall_quat = torch.stack([
            torch.cos(wall_normal_yaw / 2) * torch.cos(wall_normal_pitch / 2),
            -torch.sin(wall_normal_yaw / 2) * torch.sin(wall_normal_pitch / 2),
            torch.cos(wall_normal_yaw / 2) * torch.sin(wall_normal_pitch / 2),
            torch.sin(wall_normal_yaw / 2) * torch.cos(wall_normal_pitch / 2),
        ], dim=-1)
        wall_distance = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.3, 0.6)
        self.wall_direction[env_ids] = wall_direction
        self.wall_quat[env_ids] = wall_quat
        self.wall_distance[env_ids] = wall_distance
    
        self.command_setpoint_pos_ee_wall[:, 0] = self.wall_distance.squeeze(-1)
        self.command_setpoint_pos_ee_wall[:, 0].add_(0.1)
        self.command_setpoint_pos_ee_wall[:, 2] = 0.3
        
    def _update_command(self):
        if self.enable_teleop:
            delta = torch.zeros(self.num_envs, 3, device=self.device)
            if self.key_pressed["up"] or self.key_pressed["w"]:
                delta[:, 2] += self.step_size
            if self.key_pressed["down"] or self.key_pressed["s"]:
                delta[:, 2] -= self.step_size
            if self.key_pressed["right"] or self.key_pressed["d"]:
                delta[:, 1] += self.step_size
            if self.key_pressed["left"] or self.key_pressed["a"]:
                delta[:, 1] -= self.step_size
            if self.key_pressed["e"]:
                delta[:, 0] += self.step_size
            if self.key_pressed["q"]:
                delta[:, 0] -= self.step_size
            
            self.command_setpoint_pos_ee_wall.add_(delta)
                
        self.command_setpoint_pos_ee_b[:] = quat_rotate(
            self.wall_quat,
            self.command_setpoint_pos_ee_wall,
        )
        
        # compute command in world/body frame
        self.command_pos_ee_w[:] = self.desired_pos_ee_w.mean(1)
        self.command_linvel_ee_w[:] = self.desired_linvel_ee_w.mean(1)

        self.command_pos_ee_b[:] = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.command_pos_ee_w - self.asset.data.root_pos_w,
        )
        self.command_linvel_ee_b[:] = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.command_linvel_ee_w,
        )

        # compute diff
        ee_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.asset.data.body_pos_w[:, self.ee_body_id] - self.asset.data.root_pos_w,
        )
        self.command_pos_ee_diff_b[:] = self.command_pos_ee_b - ee_pos_b

        self.command_setpoint_pos_ee_diff_b[:] = self.command_setpoint_pos_ee_b - ee_pos_b
            
        # populate command tensor
        self.command[:, 0:3] = self.command_setpoint_pos_ee_diff_b
        self.command[:, 3:6] = self.command_kp
        self.command[:, 6:9] = self.command_kd
        if self.command_acc:
            ee_linvel_b = quat_rotate_inverse(
                self.asset.data.root_quat_w,
                self.asset.data.body_lin_vel_w[:, self.ee_body_id],
            )
            self.command[:, 3:6] *= self.command_setpoint_pos_ee_diff_b
            self.command[:, 6:9] *= 0.0 - ee_linvel_b
        self.command[:, 9:10] = self.virtual_mass_ee

        self.command_hidden[:, 0:3] = self.command_pos_ee_diff_b
        self.command_hidden[:, 3:6] = self.command_linvel_ee_b

    def reset(self, env_ids: torch.Tensor):
        self.need_reset_mask[env_ids] = True
        self._sample_wall(env_ids)
        self._sample_command(env_ids)

        self.command[env_ids] = 0.0
        self.command_hidden[env_ids] = 0.0

        if len(self.kp_forces):
            self.save_logs()

    def step(self, substep: int):
        # wall force
        ee_distance = (
            (self.asset.data.body_pos_w[:, self.ee_body_id] - self.asset.data.root_pos_w) * self.wall_direction
        ).sum(dim=-1, keepdim=True)
        wall_resistance = self.wall_kp * (ee_distance - self.wall_distance).clamp_min(0.0)

        body_vel = self.asset.data.body_lin_vel_w[:, self.ee_body_id]
        velocity_parallel = body_vel - (
            body_vel * self.wall_direction
        ).sum(dim=-1, keepdim=True) * self.wall_direction
        velocity_parallel[velocity_parallel.norm(dim=-1) < 1e-3] = 0.0
        velocity_parallel_dir = velocity_parallel / velocity_parallel.norm(dim=-1, keepdim=True).clamp_min(1e-3)

        self.wall_resistance_force_w[:] = -wall_resistance * self.wall_direction
        self.wall_friction_force_w[:] = -0.3 * wall_resistance * velocity_parallel_dir
        wall_force_w = self.wall_resistance_force_w + self.wall_friction_force_w

        wall_force_b = quat_rotate_inverse(
            self.asset.data.body_quat_w[:, self.ee_body_id],
            wall_force_w,
        )
        
        forces_ee_b = self.asset._external_force_b[:, [self.ee_body_id]].clone()
        forces_ee_b += wall_force_b.unsqueeze(1)
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
        wall_force_w = self.wall_resistance_force_w + self.wall_friction_force_w

        self.acc_spring_ee_w[:] = self.command_kp.unsqueeze(1) * (
            command_setpoint_pos_ee_w.unsqueeze(1) - self.desired_pos_ee_w
        ) + self.command_kd.unsqueeze(1) * (-self.desired_linvel_ee_w)
        self.desired_linacc_ee_w[:] = self.acc_spring_ee_w + (
            (wall_force_w / self.virtual_mass_ee).unsqueeze(1)
        )
        self.desired_linvel_ee_w.add_(self.desired_linacc_ee_w * self.env.physics_dt)
        self.desired_pos_ee_w.add_(self.desired_linvel_ee_w * self.env.physics_dt)

    def update(self):
        env_ids = self.need_reset_mask.squeeze(-1)
        self.desired_linacc_ee_w[env_ids] = 0.0
        self.desired_linvel_ee_w[env_ids] = self.asset.data.body_lin_vel_w[
            env_ids, None, self.ee_body_id
        ]
        self.desired_pos_ee_w[env_ids] = self.asset.data.body_pos_w[
            env_ids, None, self.ee_body_id
        ]
        self.need_reset_mask[env_ids] = False
        
        self.desired_linvel_ee_w[:] = self.desired_linvel_ee_w.roll(1, 1)
        self.desired_pos_ee_w[:] = self.desired_pos_ee_w.roll(1, 1)
        self.desired_linvel_ee_w[:, 0] = self.asset.data.body_lin_vel_w[
            :, self.ee_body_id
        ]
        self.desired_pos_ee_w[:, 0] = self.asset.data.body_pos_w[:, self.ee_body_id]
        
        for _ in range(int(self.env.step_dt / self.env.physics_dt)):
            self._integrate()
        
        self._update_command()
        
        # record forces
        actual_force = quat_rotate_inverse(
            self.wall_quat,
            self.wall_resistance_force_w + self.wall_friction_force_w,
        )
        desired_force = quat_rotate_inverse(
            self.wall_quat,
            self.virtual_mass_ee * self.acc_spring_ee_w.mean(1),
        )
        desired_kp_force = quat_rotate_inverse(
            self.wall_quat,
            self.command_kp * (self.command_setpoint_pos_ee_b - self.command_pos_ee_b),
        )
        kp_force = quat_rotate_inverse(
            self.wall_quat,
            self.command_kp * self.command_setpoint_pos_ee_diff_b,
        )
        self.actual_forces.append(actual_force)
        self.desired_forces.append(desired_force)
        self.desired_kp_forces.append(desired_kp_force)
        self.kp_forces.append(kp_force)
    
    def _debug_draw_wall(self):
        # draw a grid of 16 points on the wall
        wall_points_b = torch.tensor([
            [1.0, y, z] for y in [-0.3, -0.1, 0.1, 0.3] for z in [-0.3, -0.1, 0.1, 0.3]
        ], device=self.device).repeat(self.num_envs, 1, 1) # [envs, 16, 3]
        # tranform according to wall quat and distance
        wall_points_b[:, :, 0].mul_(self.wall_distance)
        num_points = wall_points_b.shape[1]
        wall_quat = self.wall_quat.unsqueeze(1).repeat(1, num_points, 1)
        root_quat_w = self.asset.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)

        wall_points_b = quat_rotate(wall_quat.view(-1, 4), wall_points_b.view(-1, 3)).view(-1, num_points, 3)
        wall_points_w = quat_rotate(
            root_quat_w.view(-1, 4),
            wall_points_b.view(-1, 3),
        ).view(-1, num_points, 3) + self.asset.data.root_pos_w.unsqueeze(1)
        self.env.debug_draw.point(
            wall_points_w.view(-1, 3),
            color=(1.0, 1.0, 1.0, 1.0),
            size=20.0,
        )
        # draw wall normal (white)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.wall_direction,
            color=(1.0, 1.0, 1.0, 1.0),
            size=2.0,
        )

    def _debug_draw_ee(self): 
        # draw ee setpoint, command pos/linvel, actual pos/linvel
        # command position for ee (green)
        self.env.debug_draw.point(
            self.command_pos_ee_w,
            color=(0.0, 1.0, 0.0, 1.0),
            size=10.0,
        )
        # command linvel for ee (green)
        self.env.debug_draw.vector(
            self.command_pos_ee_w,
            self.command_linvel_ee_w,
            color=(0.0, 1.0, 0.0, 1.0),
            size=1.0,
        )
        # setpoint position for ee (red)
        setpoint_pos_ee_w = (
            quat_rotate(
                self.asset.data.root_quat_w,
                self.command_setpoint_pos_ee_b,
            )
            + self.asset.data.root_pos_w
        )
        self.env.debug_draw.point(
            setpoint_pos_ee_w,
            color=(1.0, 0.0, 0.0, 1.0),
            size=10.0,
        )
        # actual position for ee (yellow)
        self.env.debug_draw.point(
            self.asset.data.body_pos_w[:, self.ee_body_id],
            color=(1.0, 1.0, 0.0, 1.0),
            size=10.0,
        )
    
    def _debug_draw_forces(self):
        # draw external force on desired ee (orange)
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.ee_body_id],
            self.wall_resistance_force_w,
            color=(1.0, 0.5, 0.0, 1.0),
            size=4.0,
        )
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.ee_body_id],
            self.wall_friction_force_w,
            color=(1.0, 0.5, 0.0, 1.0),
            size=4.0,
        )
    
    def debug_draw(self):
        self._debug_draw_wall()
        self._debug_draw_ee()
        self._debug_draw_forces()
        
        