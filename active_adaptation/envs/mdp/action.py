import torch
import einops
from typing import Dict, Literal, Tuple, Union, TYPE_CHECKING
from tensordict import TensorDictBase
from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_mul, quat_conjugate, axis_angle_from_quat, quat_inv, quat_rotate_inverse, quat_rotate, yaw_quat
from active_adaptation.utils.helpers import batchify

quat_rotate = batchify(quat_rotate)

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env

class ActionManager:
    
    action_dim: int

    def __init__(self, env):
        self.env: Env = env
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


class IKResidual(ActionManager):
    def __init__(
        self, 
        env, 
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 4,
        alpha: Tuple[float, float] = (0.5, 1.0),
        fwd: bool = False,
    ):
        super().__init__(env)
        self.joint_ids, self.joint_names, self.action_scaling = string_utils.resolve_matching_names_values(dict(action_scaling), self.asset.joint_names)
        
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.action_dim = len(self.joint_ids)
        
        self.fwd = fwd
        if self.fwd:
            self.target_quat = torch.tensor([0.5, 0.5, 0.5, 0.5], device=self.device).repeat(self.num_envs, 1)

        self.max_delay = max_delay
        if isinstance(alpha, float):
            self.alpha_range = (alpha, alpha)
        else:
            self.alpha_range = tuple(alpha)
        with torch.device(self.device):
            self.action_buf = torch.zeros(self.num_envs, self.action_dim, max(max_delay + 1, 3)) # at least 3 for action_rate_2_l2 reward
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)
        
        from active_adaptation.envs.mdp.commands import EEImpedance
        self.command_manager: EEImpedance = self.env.command_manager
        self.ee_body_id = self.command_manager.ee_body_id
        
        self.asset.write_joint_stiffness_to_sim(40., self.joint_ids)
        self.asset.write_joint_damping_to_sim(40., self.joint_ids)
    
    def reset(self, env_ids: torch.Tensor):
        self.delay[env_ids] = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        alpha = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.alpha_range)
        self.alpha[env_ids] = alpha

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            # update action buf
            action = tensordict["action"].clamp(-10, 10)
            self.action_buf[:] = self.action_buf.roll(1, dims=-1)
            self.action_buf[:, :, 0] = action
            delayed_action = self.action_buf.take_along_dim(self.delay.unsqueeze(1), dim=-1)
            self.applied_action.lerp_(delayed_action.squeeze(-1), self.alpha)

        ee_vel_w = self.asset.data.body_lin_vel_w[:, self.ee_body_id]
        pos_error = self.command_manager.command_pos_ee_diff_b
        vel_error = self.command_manager.command_linvel_ee_w - ee_vel_w
        desired_vel = 40 * pos_error + 10 * vel_error
        nomial_error = self.asset.data.default_joint_pos - self.asset.data.joint_pos

        J = self.asset.root_physx_view.get_jacobians()[:, self.ee_body_id, :3]
        J_T = J.transpose(1, 2)

        # lambda_matrix = torch.eye(3, device=J.device) * 1.0
        # J_inv = J_T @ torch.inverse(J @ J_T + lambda_matrix) # [N, 3, 3]
        # delta_q = (J_inv @ desired_vel.unsqueeze(-1)).squeeze(-1) + 0.1 * nomial_error
        
        # use `torch.linalg.lstsq`` for performance and stability
        lambda_matrix = torch.eye(J.shape[2], device=J.device) * 1.0
        A = J_T @ J + lambda_matrix
        b = J_T @ desired_vel.unsqueeze(-1)
        delta_q = torch.linalg.lstsq(A, b).solution.squeeze(-1) + 0.1 * nomial_error
        delta_q[:, self.joint_ids] += self.action_scaling * self.applied_action
        
        self.asset.set_joint_position_target(self.asset.data.joint_pos + delta_q * self.env.physics_dt)
        self.asset.set_joint_velocity_target(delta_q)
        self.asset.write_data_to_sim()
    
    def _compute_axis_angle_error(
        self,
        q01: torch.Tensor,
        q02: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute quaternion error (i.e., difference quaternion)
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quaternion.html
        # q_current_norm = q_current * q_current_conj
        source_quat_norm = quat_mul(q01, quat_conjugate(q01))[:, 0]
        # q_current_inv = q_current_conj / q_current_norm
        source_quat_inv = quat_conjugate(q01) / source_quat_norm.unsqueeze(-1)
        # q_error = q_target * q_current_inv
        quat_error = quat_mul(q02, source_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)
        return axis_angle_error
    
    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Computes the change in joint position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired
        end-effector position.

        Args:
            delta_pose: The desired delta pose in shape (N, 3) or (N, 6).
            jacobian: The geometric jacobian matrix in shape (N, 3, num_joints) or (N, 6, num_joints).

        Returns:
            The desired delta in joint space. Shape is (N, num-jointsß).
        """
        if self.ik_params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{self.ik_method}' is not defined!")
        # compute the delta in joint-space
        if self.ik_method == "pinv":  # Jacobian pseudo-inverse
            # parameters
            k_val = self.ik_params["k_val"]
            # computation
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.ik_method == "svd":  # adaptive SVD
            # parameters
            k_val = self.ik_params["k_val"]
            min_singular_value = self.ik_params["min_singular_value"]
            # computation
            # U: 6xd, S: dxd, V: d x num-joint
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.ik_method == "trans":  # Jacobian transpose
            # parameters
            k_val = self.ik_params["k_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_joint_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.ik_method == "dls":  # damped least squares
            # parameters
            lambda_val = 0.5 # self.ik_params["lambda_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2) # [N, num_joints, 3/6]
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[2], device=self.device) # [num_joints, num_joints]
            delta_joint_pos = (
                # jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
                torch.inverse(jacobian_T @ jacobian + lambda_matrix) @ jacobian_T @ delta_pose.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.ik_method}")

        return delta_joint_pos

class IKResidualOnBase(ActionManager):
    def __init__(
        self, 
        env, 
        action_scaling: Dict[str, float] = 0.5,
        arm_joint_names: str = "arm_link[1-6]",
        max_delay: int = 4,
        alpha: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__(env)
        self.joint_ids, self.joint_names, self.action_scaling = string_utils.resolve_matching_names_values(dict(action_scaling), self.asset.joint_names)
        
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.action_dim = len(self.joint_ids)

        self.max_delay = max_delay
        if isinstance(alpha, float):
            self.alpha_range = (alpha, alpha)
        else:
            self.alpha_range = tuple(alpha)
        with torch.device(self.device):
            self.action_buf = torch.zeros(self.num_envs, self.action_dim, max(max_delay + 1, 3))
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1) * alpha
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

        from active_adaptation.envs.mdp.commands import BaseEEImpedance
        self.command_manager: BaseEEImpedance = self.env.command_manager
        self.ee_body_id = self.command_manager.ee_body_id

        self.arm_joint_ids, self.arm_joint_names = self.asset.find_joints(arm_joint_names)
        self.arm_joint_ids_jacobians = torch.tensor(self.arm_joint_ids, device=self.device)
        self.arm_joint_ids_jacobians += 6
        self.nominal_error_weight = torch.ones(self.num_envs, 1, device=self.device) * 0.1
        self.nominal_error_weight[0] = 0.5
        
        self.asset.write_joint_stiffness_to_sim(40., self.arm_joint_ids)
        self.asset.write_joint_damping_to_sim(40., self.arm_joint_ids)
    
    def reset(self, env_ids: torch.Tensor):
        self.delay[env_ids] = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0
    
    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action = tensordict["action"].clamp(-10, 10)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action
            action = self.action_buf.take_along_dim(self.delay.unsqueeze(1), dim=-1)
            self.applied_action.lerp_(action.squeeze(-1), self.alpha)
        
        ee_pos_to_base_w = self.asset.data.body_pos_w[:, self.ee_body_id] - self.asset.data.root_pos_w
        ee_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            ee_pos_to_base_w,
        )
        ee_vel_w = self.asset.data.body_lin_vel_w[:, self.ee_body_id]
        ee_vel_coriolis_w = self.asset.data.root_lin_vel_w + torch.cross(self.asset.data.root_ang_vel_w, ee_pos_to_base_w, dim=-1)
        ee_vel_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            ee_vel_w - ee_vel_coriolis_w
        )
        command_pos_ee_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            quat_rotate(
                yaw_quat(self.asset.data.root_quat_w),
                self.command_manager.command_pos_ee_b,
            ),
        )
        pos_error = command_pos_ee_b - ee_pos_b
        vel_error = self.command_manager.command_linvel_ee_b - ee_vel_b
        desired_vel = 40 * pos_error + 10 * vel_error
        nominal_error = (self.asset.data.default_joint_pos - self.asset.data.joint_pos)[:, self.arm_joint_ids]
        
        J = self.asset.root_physx_view.get_jacobians()[:, self.ee_body_id, :3, self.arm_joint_ids_jacobians]
        J_T = J.transpose(1, 2)
        
        lambda_matrix = torch.eye(J.shape[2], device=J.device) * 1.0
        A = J_T @ J + lambda_matrix
        b = J_T @ desired_vel.unsqueeze(-1)
        delta_q = torch.linalg.lstsq(A, b).solution.squeeze(-1) + self.nominal_error_weight * nominal_error
        delta_q += (self.action_scaling * self.applied_action)[:, self.arm_joint_ids]
        
        # set joint position target for legs
        leg_pos_target = self.asset.data.default_joint_pos.clone()
        leg_pos_target[:, self.joint_ids] = self.action_scaling * self.applied_action
        self.asset.set_joint_position_target(leg_pos_target)

        # overwrite arm joint positions with incremental action control
        arm_pos_target = self.asset.data.joint_pos[:, self.arm_joint_ids] + delta_q * self.env.physics_dt
        self.asset.set_joint_position_target(arm_pos_target, self.arm_joint_ids)
        self.asset.set_joint_velocity_target(delta_q, self.arm_joint_ids)
        
        self.asset.write_data_to_sim()
        

class JointPosition(ActionManager):
    def __init__(
        self, 
        env,
        joint_names: str = ".*",
        action_scaling: Dict[str, float] = 0.5,
        left_joints = None,
        right_joints = None,
        asym_joints = None,
        max_delay: int = None, # delay in simulation steps
        fixed_delay: bool = False,
        alpha: Union[float, Tuple[float, float], Dict[str, float], Dict[str, Tuple[float, float]]] = (0.5, 1.0),
        cumstom_command: Dict[str, float] = None,
        clip_joint_targets: float = None
    ):
        super().__init__(env)
        self.joint_ids, self.joint_names, self.action_scaling = string_utils.resolve_matching_names_values(
            dict(action_scaling), self.asset.joint_names)
        if left_joints is not None:
            self.left_joint_ids = string_utils.resolve_matching_names(left_joints, self.joint_names)[0]
            self.right_joint_ids = string_utils.resolve_matching_names(right_joints, self.joint_names)[0]
            assert len(self.left_joint_ids) == len(self.right_joint_ids), "Left and right joints must have the same length."
        else:
            self.left_joint_ids = None
            self.right_joint_ids = None
        
        self.signs = torch.ones(len(self.joint_ids), device=self.device)
        if asym_joints is not None:
            self.asym_joint_ids = string_utils.resolve_matching_names(asym_joints, self.joint_names)[0]
            self.signs[self.asym_joint_ids] = -1
        
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.max_delay = max_delay if max_delay is not None else self.env.cfg.decimation
        self.max_delay = min(self.max_delay, self.env.cfg.decimation)
        self.fixed_delay = fixed_delay
        
        self.action_dim = len(self.joint_ids)
        
        import omegaconf
        if isinstance(alpha, float):
            self.alpha_range = (alpha, alpha)
        elif isinstance(alpha, omegaconf.listconfig.ListConfig):
            self.alpha_range = tuple(alpha)
        else:
            raise ValueError(f"Invalid alpha type: {type(alpha)}")

        if cumstom_command is not None:
            cumstom_command = dict(cumstom_command)
            self.cumstom_command_joint_ids, _, self.cumstom_command = string_utils.resolve_matching_names_values(cumstom_command, self.asset.joint_names)
            self.cumstom_command = torch.tensor(self.cumstom_command, device=self.device)
        if clip_joint_targets is not None:
            self.clip_joint_targets = clip_joint_targets
            
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)
        self.joint_limits = self.asset.data.joint_limits.clone().unbind(-1)
        self.decimation = int(self.env.step_dt / self.env.physics_dt)

        with torch.device(self.device):
            action_buf_hist = max(max_delay + 1, 3) if max_delay is not None else 3
            self.action_buf = torch.zeros(self.num_envs, self.action_dim, action_buf_hist) # at least 3 for action_rate_2_l2 reward
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

    
    def fliplr(self, action: torch.Tensor):
        """
        Used for flipping the `action` and `prev_action`.
        """
        if self.left_joint_ids is None:
            raise ValueError("Left and right joint names must be provided to flip the action.")
        action_flipped = action.reshape(self.num_envs, self.action_dim, -1).clone()
        left = action_flipped[:, self.left_joint_ids]
        right = action_flipped[:, self.right_joint_ids]
        action_flipped[:, self.left_joint_ids] = right
        action_flipped[:, self.right_joint_ids] = left
        return (action_flipped * self.signs.unsqueeze(-1)).reshape(action.shape)

    def reset(self, env_ids: torch.Tensor):
        self.delay[env_ids] = torch.randint(0, 2, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        self.default_joint_pos[env_ids] = self.asset.data.default_joint_pos[env_ids].clone()
        self.default_joint_pos[env_ids] += self.offset[env_ids]

        alpha = torch.empty(len(env_ids), 1, device=self.device).uniform_(self.alpha_range[0], self.alpha_range[1])
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
            if hasattr(self, "cumstom_command"):
                pos_target[:, self.cumstom_command_joint_ids] = self.cumstom_command
            if hasattr(self, "clip_joint_targets"):
                pos_target = self.asset.data.joint_pos + (pos_target - self.asset.data.joint_pos).clamp(-self.clip_joint_targets, self.clip_joint_targets)
            self.asset.set_joint_position_target(pos_target.clamp(*self.joint_limits))
            self.asset.write_data_to_sim()

class QuadrupedWithArm(JointPosition):
    def __init__(
        self, 
        env, 
        joint_names: str=".*", 
        action_scaling: Dict[str, float]=0.5,
        max_delay: int=4,
        alpha: Tuple[float, float]=(0.5, 1.0),
        arm_joint_names: str = "joint.*",
    ):
        super().__init__(env, joint_names, action_scaling, None, None, max_delay, alpha)
        self.arm_joint_ids, _ = self.asset.find_joints(arm_joint_names)
        self.arm_joint_pos = self.default_joint_pos[:, self.arm_joint_ids].clone()
        self.arm_joint_limits = self.asset.data.joint_limits[:, self.arm_joint_ids]
    
    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        self.arm_joint_pos[env_ids] = self.default_joint_pos[env_ids.unsqueeze(-1), self.arm_joint_ids]

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
            self.arm_joint_pos += self.applied_action[:, self.arm_joint_ids] * self.action_scaling[self.arm_joint_ids]
            self.arm_joint_pos.clamp_(self.arm_joint_limits[..., 0], self.arm_joint_limits[..., 1])
            pos_target[:, self.arm_joint_ids] = self.arm_joint_pos
            self.asset.set_joint_position_target(pos_target)
        # self.asset.write_data_to_sim()

class HumanoidWithArm(ActionManager):

    def __init__(
        self, 
        env, 
        joint_names: str=".*", 
        action_scaling: float=0.5
    ):
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
            action_joint, action_arm_cmd = tensordict["action"].split([len(self.joint_ids), 6], dim=-1)
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
        arm_joints: Union[str, None]=None,
        gripper_joints: Union[str, None]=None,
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
                dict(action_scaling), 
                self.asset.joint_names
            )[2],
            device=self.device
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
        self.action_buf[env_ids] = 0.
        self.applied_action[env_ids] = 0.
        self.jpos_targets[env_ids] = 0.

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
                self.jpos_targets[:, self.gripper_joint_ids] = self.jpos_default[:, self.gripper_joint_ids]
            
            self.jpos_targets.clamp_(*self.jpos_limit)
            self.asset.set_joint_position_target(self.jpos_targets)
        
        self.asset.write_data_to_sim()


class QuadrupedJointForce(ActionManager):
    def __init__(self, env, action_scaling: float=30.):
        super().__init__(env)
        self.action_dim = 16
        self.action_buf = torch.zeros(self.num_envs, self.action_dim, 4, device=self.device)
        self.applied_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.feet_ids, self.feet_names = self.asset.find_bodies(".*foot")
        self.joint_ids = torch.as_tensor([
            self.asset.find_joints("FL_.*_joint")[0],
            self.asset.find_joints("FR_.*_joint")[0],
            self.asset.find_joints("RL_.*_joint")[0],
            self.asset.find_joints("RR_.*_joint")[0],
        ], device=self.device)
        print(self.asset.find_joints("FL_.*_joint")[1])

        self.feet_ids = torch.as_tensor(self.feet_ids, device=self.device)
        
        self.action_scaling = torch.tensor([0.2, 0.0, 0.2], device=self.device)
        self.default_feet_pos = self.asset.data.body_pos_w[0, self.feet_ids] - self.asset.data.root_pos_w[0]
        self.default_feet_pos[:, 2] = -0.40
        self.kp = 200
        self.kd = 10.

        self.asset.actuators["base_legs"].stiffness[:] = 0.
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
        hip_pos_target = self.asset.data.default_joint_pos.reshape(-1, 3, 4)[:, 0] + 0.5 * self.applied_action[:, 12:].reshape(-1, 4)
        effort[:, :, 0] = 20 * (hip_pos_target - hip_pos) + 0.5 * - hip_vel
        
        self.asset.set_joint_effort_target(effort.flatten(1), self.joint_ids.flatten())
        self.asset.write_data_to_sim()
    
    def compute_effort(self):
        feet_pos_b = quat_rotate_inverse(
            self.root_quat_w.unsqueeze(1),
            self.feet_pos_w - self.root_pos_w.unsqueeze(1)
        )
        feet_vel_b = quat_rotate_inverse(
            self.root_quat_w.unsqueeze(1),
            self.feet_vel_w - self.root_lin_vel_w.unsqueeze(1)
        )
        pos_error = self.action_scaling * self.applied_action[:, :12].reshape(-1, 4, 3) + self.default_feet_pos - feet_pos_b
        
        feet_torque_b = self.kp * pos_error + self.kd * - feet_vel_b
        feet_torque_w = quat_rotate(self.root_quat_w.unsqueeze(1), feet_torque_b)
        jacobian = einops.rearrange(self.jacobian, "n b c j -> n b j c")
        jacobian = jacobian[:, self.feet_ids.unsqueeze(1), self.joint_ids]
        effort = (jacobian @ feet_torque_w.unsqueeze(-1)).squeeze(-1) # [n, 4, j, 3] -> [n, 4, j] 
        return effort
    
    def debug_draw(self):
        feet_setpos_w = quat_rotate(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.action_scaling * self.applied_action[:, :12].reshape(-1, 4, 3) + self.default_feet_pos
        ) + self.asset.data.root_pos_w.unsqueeze(1)
        self.env.debug_draw.point(
            feet_setpos_w.reshape(-1, 3),
            size=10.,
            color=(0., 1., 0., 1.)
        )

def clamp_norm(x: torch.Tensor, max_norm: float):
    norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / norm.clamp(min=1e-6)).clamp(max=1.0)