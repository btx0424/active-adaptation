import torch
import einops
from typing import Dict, Literal, Tuple, Union, TYPE_CHECKING, List
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
        self.action_buf: torch.Tensor

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


class ConcatenatedAction(ActionManager):
    def __init__(self, env, action_managers: List[ActionManager]):
        super().__init__(env)
        self.action_managers = action_managers
        self.action_dims = [action_manager.action_dim for action_manager in self.action_managers]
        self.action_dim = sum(self.action_dims)
    
    @property
    def action_buf(self):
        return torch.cat([action_manager.action_buf for action_manager in self.action_managers], dim=1)

    def __call__(self, tensordict: TensorDictBase, substep: int):
        actions = torch.split(tensordict["action"], self.action_dims, dim=-1)
        for action_manager, action in zip(self.action_managers, actions):
            action_manager(action, substep)
    
    def symmetry_transforms(self):
        return symmetry_utils.SymmetryTransform.cat(
            [action_manager.symmetry_transforms() for action_manager in self.action_managers]
        )


class JointPosition(ActionManager):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = None,  # delay in simulation steps
        fixed_delay: bool = False,
        alpha: Union[float, Tuple[float, float]] = (0.5, 1.0),
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

    def __call__(self, action: torch.Tensor, substep: int):
        if substep == 0:
            if isinstance(action, TensorDictBase):
                action = action["action"]
            action = action.clamp(-10, 10)
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


class DecapAction(ActionManager):
    """
    Decaying Action Prior as described in https://arxiv.org/pdf/2310.05714
    """
    def __init__(self, env, action_scaling: Dict[str, float] = 8.0):
        super().__init__(env)
        self.joint_ids, self.joint_names, self.action_scaling = (
            string_utils.resolve_matching_names_values(
                dict(action_scaling), self.asset.joint_names
            )
        )
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.action_dim = len(self.joint_ids)

        self.action_buf = torch.zeros(self.num_envs, self.action_dim, 4, device=self.device)
        self.applied_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    
    def symmetry_transforms(self):
        transform = symmetry_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform
    
    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            if isinstance(tensordict, TensorDictBase):
                action = tensordict["action"]
            action = action.clamp(-10, 10)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action
            self.applied_action = self.applied_action.lerp(action, 0.8)
        torque = self.applied_action * self.action_scaling
        self.asset.set_joint_effort_target(torque, self.joint_ids)
    

def clamp_norm(x: torch.Tensor, max_norm: float):
    norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / norm.clamp(min=1e-6)).clamp(max=1.0)
