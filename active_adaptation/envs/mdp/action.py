import torch
import einops
from typing import Dict, Literal, Tuple, Union, TYPE_CHECKING
from tensordict import TensorDictBase
import isaaclab.utils.string as string_utils
from active_adaptation.utils.math import (
    # quat_mul,
    # quat_conjugate,
    # axis_angle_from_quat,
    # quat_inv,
    quat_rotate_inverse,
    quat_rotate,
    yaw_quat,
)
import active_adaptation.utils.symmetry as symmetry_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class ActionManager:

    action_dim: int

    def __init__(self, env):
        self.env: _Env = env
        self.asset: Articulation = self.env.scene["robot"]

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device


class JointPositionNG(ActionManager):

    def __init__(
        self,
        env,
        # actuator_name: str = "",
        action_scaling: Dict[str, float] = 0.5,
        alpha: Tuple[float, float] = [0.7, 1.0],
        delay: Tuple[int, int] = [0, 4],
        clip_joint_targets: float = None,
        clip_vel_target: float = None,
        clip_tff_target: float = None,
        size: int = 4,
        use_vel: bool = False,
        use_tff: bool = False,
    ):
        super().__init__(env)
        # self.actuator = self.asset.actuators[actuator_name]
        # self.joint_ids, self.joint_names, self.action_scaling = (
        #     string_utils.resolve_matching_names_values(
        #         dict(action_scaling), self.actuator.joint_names
        #     )
        # )
        self.joint_ids, self.joint_names, self.action_scaling = (
            string_utils.resolve_matching_names_values(
                dict(action_scaling), self.asset.joint_names
            )
        )
        # self.actuator_ids = self.actuator.joint_indices
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)

        self.alpha_max = alpha[1]
        self.alpha_min = alpha[0]
        self.jit_scale = (self.alpha_max - self.alpha_min) / 10

        self.alpha = torch.empty(self.num_envs, 1, device=self.device).uniform_(
            self.alpha_min, self.alpha_max
        )
        self.alpha_jit = torch.zeros(self.num_envs, 1, device=self.device)

        self.delay_max = delay[1]
        self.delay_min = delay[0]

        self.delay = torch.randint(
            self.delay_min, self.delay_max + 1, (self.num_envs, 1), device=self.device
        )

        self.clip_joint_targets = clip_joint_targets
        self.clip_vel_target = clip_vel_target
        self.clip_tff_target = clip_tff_target

        self.use_vel = use_vel
        self.use_tff = use_tff

        self.joint_dim = len(self.joint_ids)
        # self.actuator_ids = self.actuator.joint_indices
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()

        self.joint_limits = self.asset.data.joint_limits.clone().unbind(-1)[0].clone()
        self.default_joint_vel = torch.zeros_like(self.default_joint_pos)
        self.default_joint_tff = torch.zeros_like(self.default_joint_pos)

        # Determine the number of action components based on enabled features
        num_actions = 1  # Position is always enabled
        if self.use_vel:
            num_actions += 1
        if self.use_tff:
            num_actions += 1

        self.action_dim = self.joint_dim * num_actions

        # Initialize action buffers
        self.action_buf = torch.zeros(
            self.num_envs, self.action_dim, size, device=self.device
        )
        self.applied_action = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        # self.effort_limit = self.actuator.effort_limit.clone()
        self.pos_target = self.default_joint_pos.clone()
        self.vel_target = self.default_joint_vel.clone()
        self.tff_target = self.default_joint_tff.clone()

        # Define action slices for easy access
        current_index = 0
        self.action_slices = {}
        self.action_order = []

        # Position slice
        self.action_slices["pos"] = slice(current_index, current_index + self.joint_dim)
        current_index += self.joint_dim
        self.action_order.append("pos")

        # Velocity slice
        if self.use_vel:
            self.action_slices["vel"] = slice(
                current_index, current_index + self.joint_dim
            )
            current_index += self.joint_dim
            self.action_order.append("vel")

        # Torque feedforward slice
        if self.use_tff:
            self.action_slices["tff"] = slice(
                current_index, current_index + self.joint_dim
            )
            current_index += self.joint_dim
            self.action_order.append("tff")

    def reset(self, env_ids: torch.Tensor):
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0
        self.alpha[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            self.alpha_min, self.alpha_max
        )
        self.delay[env_ids] = torch.randint(
            self.delay_min, self.delay_max + 1, (len(env_ids), 1), device=self.device
        )
        # self.actuator.reset(env_ids)

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action = tensordict["action"].clamp(-10, 10)
            # print(action)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action

            action = self.action_buf.take_along_dim(
                self.delay.unsqueeze(1), dim=-1
            ).squeeze(-1)

            self.alpha_jit.uniform_(-self.jit_scale, self.jit_scale)
            self.alpha.add_(self.alpha_jit).clamp_(self.alpha_min, self.alpha_max)
            # print(self.alpha)
            # breakpoint()
            self.applied_action.lerp_(action, self.alpha)

            # Process position action
            pos_action = self.applied_action[:, self.action_slices["pos"]]
            self.pos_target[:] = self.default_joint_pos.clone()
            self.pos_target[:, self.joint_ids] += (
                pos_action * self.action_scaling
            )  # * 0

            if self.clip_joint_targets is not None:
                j_pos = self.asset.data.joint_pos
                self.pos_target.sub_(j_pos).clamp_(
                    -self.clip_joint_targets, self.clip_joint_targets
                ).add_(j_pos)

            self.pos_target.clamp_(self.joint_limits)

            # Process velocity action
            if self.use_vel:
                vel_action = self.applied_action[:, self.action_slices["vel"]]
                self.vel_target[:] = self.default_joint_vel.clone()
                self.vel_target[:, self.joint_ids] += vel_action * self.action_scaling
                if self.clip_vel_target is not None:
                    self.vel_target.clamp_(-self.clip_vel_target, self.clip_vel_target)
            else:
                self.vel_target[:] = self.default_joint_vel.clone()

            # Process torque feedforward action
            if self.use_tff:
                tff_action = self.applied_action[:, self.action_slices["tff"]]
                self.tff_target[:] = self.default_joint_tff.clone()
                self.tff_target[:, self.joint_ids] += (
                    tff_action * self.action_scaling * 0.5
                )
                if self.clip_tff_target is not None:
                    self.tff_target.clamp_(-self.clip_tff_target, self.clip_tff_target)
                # self.tff_target *= self.effort_limit
            else:
                self.tff_target[:] = self.default_joint_tff.clone()

        self.asset.set_joint_position_target(self.pos_target)
        self.asset.set_joint_velocity_target(self.vel_target)
        self.asset.set_joint_effort_target(self.tff_target)
        self.asset.write_data_to_sim()


class JointPosition(ActionManager):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = None,  # delay in simulation steps
        fixed_delay: bool = False,
        alpha: Union[
            float, Tuple[float, float], Dict[str, float], Dict[str, Tuple[float, float]]
        ] = (0.5, 1.0),
        custom_command: Dict[str, float] = None,
        clip_joint_targets: float = None,
    ):
        super().__init__(env)
        self.joint_ids, self.joint_names, self.action_scaling = (
            string_utils.resolve_matching_names_values(
                dict(action_scaling), self.asset.joint_names
            )
        )

        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.max_delay = max_delay if max_delay is not None else self.env.decimation
        self.max_delay = min(self.max_delay, self.env.decimation)
        self.fixed_delay = fixed_delay

        self.action_dim = len(self.joint_ids)

        import omegaconf

        if isinstance(alpha, float):
            self.alpha_range = (alpha, alpha)
        elif isinstance(alpha, omegaconf.listconfig.ListConfig):
            self.alpha_range = tuple(alpha)
        else:
            raise ValueError(f"Invalid alpha type: {type(alpha)}")

        if custom_command is not None:
            self.custom_command_joint_ids, self.custom_command_joint_names, self.custom_command = self.resolve(custom_command)
            self.custom_command = torch.tensor(self.custom_command, device=self.device)
            if len(self.joint_ids) + len(self.custom_command_joint_ids) != self.asset.num_joints:
                raise ValueError(f"{set(self.asset.joint_names) - set(self.joint_names) - set(self.custom_command_joint_names)}")
        if clip_joint_targets is not None:
            self.clip_joint_targets = clip_joint_targets

        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)
        # self.joint_limits = self.asset.data.joint_limits.clone().unbind(-1)
        self.decimation = int(self.env.step_dt / self.env.physics_dt)
        self.count = 0

        with torch.device(self.device):
            action_buf_hist = max(max_delay + 1, 3) if max_delay is not None else 3
            self.action_buf = torch.zeros(
                self.num_envs, self.action_dim, action_buf_hist
            )  # at least 3 for action_rate_2_l2 reward
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

    def resolve(self, spec):
        return string_utils.resolve_matching_names_values(dict(spec), self.asset.joint_names)

    def symmetry_transforms(self):
        transform = symmetry_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform

    def reset(self, env_ids: torch.Tensor):
        self.delay[env_ids] = torch.randint(0, 2, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        default_joint_pos = self.asset.data.default_joint_pos[env_ids]
        self.default_joint_pos[env_ids] = default_joint_pos + self.offset[env_ids]

        alpha = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            self.alpha_range[0], self.alpha_range[1]
        )
        self.alpha[env_ids] = alpha

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action = tensordict["action"].clamp(-10, 10)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action
            action = self.action_buf.take_along_dim(self.delay.unsqueeze(1), dim=-1)
            self.applied_action.lerp_(action.squeeze(-1), self.alpha)

            pos_target = self.default_joint_pos.clone()
            pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
            if hasattr(self, "custom_command"):
                pos_target[:, self.custom_command_joint_ids] = self.custom_command
            if hasattr(self, "clip_joint_targets"):
                pos_target = self.asset.data.joint_pos + (
                    pos_target - self.asset.data.joint_pos
                ).clamp(-self.clip_joint_targets, self.clip_joint_targets)
            self.asset.set_joint_position_target(pos_target)


class QuadrupedWithArm(JointPosition):
    def __init__(
        self,
        env,
        joint_names: str = ".*",
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 4,
        alpha: Tuple[float, float] = (0.5, 1.0),
        arm_joint_names: str = "joint.*",
    ):
        super().__init__(env, joint_names, action_scaling, None, None, max_delay, alpha)
        self.arm_joint_ids, _ = self.asset.find_joints(arm_joint_names)
        self.arm_joint_pos = self.default_joint_pos[:, self.arm_joint_ids].clone()
        self.arm_joint_limits = self.asset.data.joint_limits[:, self.arm_joint_ids]

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        self.arm_joint_pos[env_ids] = self.default_joint_pos[
            env_ids.unsqueeze(-1), self.arm_joint_ids
        ]

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action = tensordict["action"].clamp(-10, 10)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action
            action = self.action_buf.take_along_dim(self.delay.unsqueeze(1), dim=-1)
            self.applied_action.lerp_(action.squeeze(-1), self.alpha)

            pos_target = self.default_joint_pos.clone()
            pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
            pos_target.clamp_(-torch.pi, torch.pi)

            # overwrite arm joint positions with incremental action control
            self.arm_joint_pos += (
                self.applied_action[:, self.arm_joint_ids]
                * self.action_scaling[self.arm_joint_ids]
            )
            self.arm_joint_pos.clamp_(
                self.arm_joint_limits[..., 0], self.arm_joint_limits[..., 1]
            )
            pos_target[:, self.arm_joint_ids] = self.arm_joint_pos
            self.asset.set_joint_position_target(pos_target)
        # self.asset.write_data_to_sim()


class HumanoidWithArm(ActionManager):

    def __init__(self, env, joint_names: str = ".*", action_scaling: float = 0.5):
        super().__init__(env)
        self.action_scaling = action_scaling
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.action_dim = len(self.joint_ids) + 6

        self.default_joint_pos = self.asset.data.default_joint_pos.clone()

        with torch.device(self.device):
            self.command_arm_linvel = torch.zeros(self.num_envs, 2, 3)
            self.action_buf = torch.zeros(self.num_envs, len(self.joint_ids), 4)
            self.applied_action = torch.zeros(self.num_envs, len(self.joint_ids))

    def reset(self, env_ids: torch.Tensor):
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0
        self.command_arm_linvel[env_ids] = 0

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action_joint, action_arm_cmd = tensordict["action"].split(
                [len(self.joint_ids), 6], dim=-1
            )
            action_arm_cmd = action_arm_cmd.reshape(self.num_envs, 2, 3)
            self.command_arm_linvel.lerp_(action_arm_cmd, 0.75)

            action_joint = action_joint.clamp(-10, 10)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action_joint
            self.applied_action.lerp_(action_joint, 0.8)

            pos_target = self.default_joint_pos.clone()
            pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
            pos_target.clamp_(-torch.pi, torch.pi)
            self.asset.set_joint_position_target(pos_target)
        self.asset.write_data_to_sim()


class QuadrupedAndArm(ActionManager):
    def __init__(
        self,
        env,
        regular_joints: str,
        action_scaling: Dict[str, float],
        arm_joints: Union[str, None] = None,
        gripper_joints: Union[str, None] = None,
        max_delay: int = 1,
        alpha: float = 0.8,
    ):
        super().__init__(env)

        self.regular_joint_ids = self.asset.find_joints(regular_joints)[0]
        self.action_dim = len(self.regular_joint_ids)

        if arm_joints is not None:
            self.arm_joint_ids = self.asset.find_joints(arm_joints)[0]
            self.action_dim += len(self.arm_joint_ids)
        else:
            self.arm_joint_ids = None

        if gripper_joints is not None:
            self.gripper_joint_ids = self.asset.find_joints(gripper_joints)[0]
        else:
            self.gripper_joint_ids = None

        self.action_scaling = torch.tensor(
            string_utils.resolve_matching_names_values(
                dict(action_scaling), self.asset.joint_names
            )[2],
            device=self.device,
        )

        assert len(self.action_scaling) == self.action_dim

        with torch.device(self.device):
            self.action_buf = torch.zeros(self.num_envs, self.action_dim, 4)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1) * alpha
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

        self.jpos_default = self.asset.data.default_joint_pos.clone()
        self.jpos_targets = self.asset.data.default_joint_pos.clone()
        self.jpos_limit = self.asset.data.default_joint_limits.clone().unbind(-1)

    def reset(self, env_ids: torch.Tensor):
        self.action_buf[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        self.jpos_targets[env_ids] = 0.0

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action = tensordict["action"].clamp(-10, 10)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action
            action = self.action_buf.take_along_dim(self.delay.unsqueeze(1), dim=-1)
            self.applied_action.lerp_(action.squeeze(-1), self.alpha)

            action_scaled = self.action_scaling * self.applied_action
            self.jpos_targets[:, self.regular_joint_ids] = (
                action_scaled[:, self.regular_joint_ids]
                + self.jpos_default[:, self.regular_joint_ids]
            )

            if self.arm_joint_ids is not None:
                self.jpos_targets[:, self.arm_joint_ids] = (
                    action_scaled[:, self.arm_joint_ids]
                    + self.jpos_targets[:, self.arm_joint_ids]
                )

            if self.gripper_joint_ids is not None:
                self.jpos_targets[:, self.gripper_joint_ids] = self.jpos_default[
                    :, self.gripper_joint_ids
                ]

            self.jpos_targets.clamp_(*self.jpos_limit)
            self.asset.set_joint_position_target(self.jpos_targets)

        self.asset.write_data_to_sim()


class QuadrupedJointForce(ActionManager):
    def __init__(self, env, action_scaling: float = 30.0):
        super().__init__(env)
        self.action_dim = 16
        self.action_buf = torch.zeros(
            self.num_envs, self.action_dim, 4, device=self.device
        )
        self.applied_action = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self.feet_ids, self.feet_names = self.asset.find_bodies(".*foot")
        self.joint_ids = torch.as_tensor(
            [
                self.asset.find_joints("FL_.*_joint")[0],
                self.asset.find_joints("FR_.*_joint")[0],
                self.asset.find_joints("RL_.*_joint")[0],
                self.asset.find_joints("RR_.*_joint")[0],
            ],
            device=self.device,
        )
        print(self.asset.find_joints("FL_.*_joint")[1])

        self.feet_ids = torch.as_tensor(self.feet_ids, device=self.device)

        self.action_scaling = torch.tensor([0.2, 0.0, 0.2], device=self.device)
        self.default_feet_pos = (
            self.asset.data.body_pos_w[0, self.feet_ids] - self.asset.data.root_pos_w[0]
        )
        self.default_feet_pos[:, 2] = -0.40
        self.kp = 200
        self.kd = 10.0

        self.asset.actuators["base_legs"].stiffness[:] = 0.0
        self.asset.actuators["base_legs"].damping[:] = 0.1

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action = tensordict["action"].clamp(-10, 10)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action
            self.applied_action = self.action_buf[:, :, 0]
        self.feet_pos_w = self.asset.data.body_pos_w[:, self.feet_ids]
        self.feet_vel_w = self.asset.data.body_lin_vel_w[:, self.feet_ids]
        self.root_quat_w = self.asset.data.root_quat_w
        self.root_pos_w = self.asset.data.root_pos_w
        self.root_lin_vel_w = self.asset.data.root_lin_vel_w
        self.jacobian = self.asset.root_physx_view.get_jacobians()[:, :, :3, 6:]

        effort = self.compute_effort()
        hip_pos = self.asset.data.joint_pos.reshape(-1, 3, 4)[:, 0]
        hip_vel = self.asset.data.joint_vel.reshape(-1, 3, 4)[:, 0]
        hip_pos_target = self.asset.data.default_joint_pos.reshape(-1, 3, 4)[
            :, 0
        ] + 0.5 * self.applied_action[:, 12:].reshape(-1, 4)
        effort[:, :, 0] = 20 * (hip_pos_target - hip_pos) + 0.5 * -hip_vel

        self.asset.set_joint_effort_target(effort.flatten(1), self.joint_ids.flatten())
        self.asset.write_data_to_sim()

    def compute_effort(self):
        feet_pos_b = quat_rotate_inverse(
            self.root_quat_w.unsqueeze(1),
            self.feet_pos_w - self.root_pos_w.unsqueeze(1),
        )
        feet_vel_b = quat_rotate_inverse(
            self.root_quat_w.unsqueeze(1),
            self.feet_vel_w - self.root_lin_vel_w.unsqueeze(1),
        )
        pos_error = (
            self.action_scaling * self.applied_action[:, :12].reshape(-1, 4, 3)
            + self.default_feet_pos
            - feet_pos_b
        )

        feet_torque_b = self.kp * pos_error + self.kd * -feet_vel_b
        feet_torque_w = quat_rotate(self.root_quat_w.unsqueeze(1), feet_torque_b)
        jacobian = einops.rearrange(self.jacobian, "n b c j -> n b j c")
        jacobian = jacobian[:, self.feet_ids.unsqueeze(1), self.joint_ids]
        effort = (jacobian @ feet_torque_w.unsqueeze(-1)).squeeze(
            -1
        )  # [n, 4, j, 3] -> [n, 4, j]
        return effort

    def debug_draw(self):
        feet_setpos_w = quat_rotate(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.action_scaling * self.applied_action[:, :12].reshape(-1, 4, 3)
            + self.default_feet_pos,
        ) + self.asset.data.root_pos_w.unsqueeze(1)
        self.env.debug_draw.point(
            feet_setpos_w.reshape(-1, 3), size=10.0, color=(0.0, 1.0, 0.0, 1.0)
        )


class LegWheel(ActionManager):
    def __init__(
        self,
        env,
        leg_scaling: Dict[str, float],
        wheel_scaling: Dict[str, float]
    ):
        super().__init__(env)
        self.leg_ids, self.leg_names, self.leg_scaling = (
            string_utils.resolve_matching_names_values(
                dict(leg_scaling), self.asset.joint_names
            )
        )
        self.wheel_ids, self.wheel_names, self.wheel_scaling = (
            string_utils.resolve_matching_names_values(
                dict(wheel_scaling), self.asset.joint_names
            )
        )
        self.leg_scaling = torch.tensor(self.leg_scaling, device=self.device)
        self.wheel_scaling = torch.tensor(self.wheel_scaling, device=self.device)

        self.leg_action_dim = len(self.leg_ids)
        self.wheel_action_dim = len(self.wheel_ids)
        self.action_dim = len(self.leg_ids) + len(self.wheel_ids)
        
        self.applied_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.action_buf = torch.zeros(self.num_envs, self.action_dim, 4, device=self.device)

    def __call__(self, tensordict: TensorDictBase, substep: int):
        action = tensordict["action"].clamp(-10., 10.)
        if substep == 0:
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action
        self.applied_action = self.applied_action.lerp(action, 0.8)
        leg_action, wheel_action = self.applied_action.split([self.leg_action_dim, self.wheel_action_dim], dim=-1)
        leg_pos_target = self.asset.data.default_joint_pos[:, self.leg_ids] + self.leg_scaling * leg_action
        self.asset.set_joint_position_target(leg_pos_target, self.leg_ids)
        wheel_vel_target = self.wheel_scaling * wheel_action
        self.asset.set_joint_velocity_target(wheel_vel_target, self.wheel_ids)

    def symmetry_transforms(self):
        return symmetry_utils.SymmetryTransform.cat([
            symmetry_utils.joint_space_symmetry(self.asset, self.leg_names),
            symmetry_utils.joint_space_symmetry(self.asset, self.wheel_names),
        ])
    
def clamp_norm(x: torch.Tensor, max_norm: float):
    norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / norm.clamp(min=1e-6)).clamp(max=1.0)
