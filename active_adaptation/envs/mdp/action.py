import torch
from tensordict import TensorDictBase
from omni.isaac.lab.assets import Articulation

class ActionManager:
    
    action_dim: int

    def __init__(self, env):
        self.env = env
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
        joint_names: str, 
        action_scaling: float = 0.5,
        max_delay: int = 4,
        alpha: float = 0.8
    ):
        super().__init__(env)
        self.action_scaling = action_scaling
        self.max_delay = max_delay

        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.action_dim = len(self.joint_ids)
        
        with torch.device(self.device):
            self.action_buf = torch.zeros(self.num_envs, self.action_dim, 4)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1) * alpha
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)
        
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()

    def reset(self, env_ids: torch.Tensor):
        self.delay[env_ids] = torch.randint(0, self.max_delay, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

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
            self.asset.set_joint_position_target(pos_target)
        self.asset.write_data_to_sim()


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
    
    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action_joint, action_arm_cmd = tensordict["action"].split([len(self.joint_ids), 6], dim=-1)
            action_arm_cmd = clamp_norm(action_arm_cmd.reshape(self.num_envs, 2, 3), 3.0)
            self.command_arm_linvel[:] = action_arm_cmd

            action_joint = action_joint.clamp(-10, 10)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action_joint
            self.applied_action.lerp_(action_joint, 0.8)
            
            pos_target = self.default_joint_pos.clone()
            pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
            pos_target.clamp_(-torch.pi, torch.pi)
            self.asset.set_joint_position_target(pos_target)
        self.asset.write_data_to_sim()


def clamp_norm(x: torch.Tensor, max_norm: float):
    norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / norm.clamp(min=1e-6)).clamp(max=1.0)