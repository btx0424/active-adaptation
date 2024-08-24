import torch
from typing import Dict, Tuple, Union, TYPE_CHECKING
from tensordict import TensorDictBase
from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.utils.string as string_utils

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env

class ActionManager:
    
    action_dim: int

    def __init__(self, env):
        self.env: Env = env
        self.asset: Articulation = self.env.scene["robot"]
    
    def reset(self, env_ids: torch.Tensor):
        pass
    
    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device


class JointPosition(ActionManager):
    def __init__(
        self, 
        env,
        joint_names: str = ".*",
        action_scaling: Dict[str, float] = 0.5,
        left_names = None,
        right_names = None,
        max_delay: int = 4,
        alpha: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__(env)
        self.joint_ids, self.joint_names, self.action_scaling = string_utils.resolve_matching_names_values(
            dict(action_scaling), self.asset.joint_names)
        if left_names is not None:
            self.left_joint_ids = string_utils.resolve_matching_names(left_names, self.joint_names)[0]
            self.right_joint_ids = string_utils.resolve_matching_names(right_names, self.joint_names)[0]
            assert len(self.left_joint_ids) == len(self.right_joint_ids), "Left and right joints must have the same length."
        else:
            self.left_joint_ids = None
            self.right_joint_ids = None
        
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.max_delay = max_delay
        
        if isinstance(alpha, float):
            self.alpha_range = (alpha, alpha)
        else:
            self.alpha_range = tuple(alpha)

        self.action_dim = len(self.joint_ids)
        
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()

        with torch.device(self.device):
            self.action_buf = torch.zeros(self.num_envs, self.action_dim, 4)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)
            self.offset = torch.zeros_like(self.default_joint_pos)
    
    def fliplr(self, action: torch.Tensor):
        """
        Used for flipping the `action` and `prev_action`.
        """
        if self.left_joint_ids is None:
            raise ValueError("Left and right joint names must be provided to flip the action.")
        action_flipped = action.reshape(self.num_envs, self.action_dim, -1).clone()
        left = action_flipped[:, self.left_joint_ids]
        right = action_flipped[:, self.right_joint_ids]
        action_flipped[:, self.left_joint_ids] = left
        action_flipped[:, self.right_joint_ids] = right
        return action_flipped.reshape(action.shape)

    def reset(self, env_ids: torch.Tensor):
        self.delay[env_ids] = torch.randint(0, self.max_delay, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        alpha = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.alpha_range)
        self.alpha[env_ids] = alpha

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action = tensordict["action"].clamp(-10, 10)
            if self.env.use_flipping:
                action = torch.where(self.env.fliplr.unsqueeze(1), self.fliplr(action), action)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action
            action = self.action_buf.take_along_dim(self.delay.unsqueeze(1), dim=-1)
            self.applied_action.lerp_(action.squeeze(-1), self.alpha)

            pos_target = self.default_joint_pos + self.offset
            pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
            pos_target.clamp_(-torch.pi, torch.pi)
            self.asset.set_joint_position_target(pos_target)
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
        super().__init__(env, joint_names, action_scaling, max_delay, alpha)
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



def clamp_norm(x: torch.Tensor, max_norm: float):
    norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / norm.clamp(min=1e-6)).clamp(max=1.0)