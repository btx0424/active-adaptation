from math import inf
import torch

from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.actuators import DCMotor
from omni.isaac.lab.assets import Articulation
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

from active_adaptation.envs.locomotion import Env, LocomotionEnv

import active_adaptation.envs.mdp as mdp

class Humanoid(LocomotionEnv):
    
    feet_name_expr = ".*ankle_link"

    @property
    def action_dim(self):
        robot = self.scene["robot"]
        return robot.num_joints
    

    class feet_too_close(mdp.Termination):
        def __init__(self, env, feet_names: str, thres: float=0.1):
            super().__init__(env)
            self.threshold = thres
            self.asset: Articulation = self.env.scene["robot"]
            self.body_ids = self.asset.find_bodies(feet_names)[0]
            assert len(self.body_ids) == 2, "Only support two feet"

        def __call__(self):
            feet_pos = self.asset.data.body_pos_w[:, self.body_ids]
            distance_xy = (feet_pos[:, 0, :2] - feet_pos[:, 1, :2]).norm(dim=-1)
            return (distance_xy < self.threshold).reshape(-1, 1)
    

    class step_up(mdp.Reward):
        
        env: "Humanoid"

        def __init__(self, env, feet_names: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.body_ids = self.asset.find_bodies(feet_names)[0]
            assert len(self.body_ids) == 2, "Only support two feet"

            self.height_scan: torch.Tensor = self.asset.data.height_scan
            self.phase: torch.Tensor = self.asset.data.phase
            self.scan_size = self.height_scan.shape[-2:]
        
        def update(self):
            self.feet_pos = self.asset.data.body_pos_w[:, self.body_ids]
            self.feet_height = self.feet_pos[:, :, 2]
            height_scan = (
                self.height_scan 
                - self.asset.data.root_pos_w[:, 2].reshape(-1, 1, 1)
                + self.feet_height.min(dim=1).values.reshape(-1, 1, 1)
            )
            height_front = height_scan[:, :, self.scan_size[1]//2:].mean(dim=(1, 2))
            self.stairs_front = (height_front < -0.0).unsqueeze(1)
            
        def compute(self) -> torch.Tensor:
            phase_sin = self.phase.sin().unsqueeze(1)
            feet_height_diff = (self.feet_height[:, 0] - self.feet_height[:, 1]).unsqueeze(1)
            feet_height_diff = torch.where(phase_sin > 0, feet_height_diff, -feet_height_diff)
            r = (feet_height_diff.clamp(0, 0.15) / 0.15).sqrt()
            r = (self.stairs_front & (phase_sin.abs() > 0.1)) * r
            return r.reshape(self.num_envs, 1)

        def debug_draw(self):
            phase_sin = self.phase.sin().unsqueeze(1)
            with torch.device(self.device):
                feet_pos = self.asset.data.body_pos_w[:, self.body_ids]
                lift_foot = torch.where(phase_sin > 0, feet_pos[:, 0], feet_pos[:, 1])
                lift = torch.tensor([0, 0, 1.5]).expand_as(lift_foot)
            self.env.debug_draw.vector(
                lift_foot, 
                lift * self.stairs_front,
                size=5,
                color=(1, 0, 0, 1)
            )

