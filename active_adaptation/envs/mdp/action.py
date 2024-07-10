import torch
from tensordict import TensorDictBase
from omni.isaac.lab.assets import Articulation

class ActionManager:
    
    action_dim: int

    def __init__(self, env):
        self.env = env
    
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

        self.asset: Articulation = self.env.scene["robot"]
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