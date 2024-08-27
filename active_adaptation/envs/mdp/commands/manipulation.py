from .base import Command
from omni.isaac.lab.assets import Articulation
import torch

class EEImpedance(Command):
    def __init__(
        self,
        env,
        ee_name: str,
        ee_base_name: str,
    ) -> None:
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.ee_name = ee_name
        self.ee_base_name = ee_base_name
        self.ee_body_id = self.robot.find_bodies(ee_name)[0][0]
        self.ee_base_body_id = self.robot.find_bodies(ee_base_name)[0][0]
        
        with torch.device(self.device):
            self.command = torch.zeros(self.num_envs, 6, device=self.device)
        
            self.command_pos = self.command[:, :3]
            self.command_fwd = self.command[:, 3:]
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.init_root_state[env_ids]
    
    def reset(self, env_ids: torch.Tensor):
        pass
    
    def step(self, substep: int):
        pass

    def update(self):
        pass
        