import torch

from omni.isaac.orbit.sensors import ContactSensor, RayCaster
from omni.isaac.orbit.actuators import DCMotor
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

from active_adaptation.envs.locomotion import Env, LocomotionEnv

import active_adaptation.envs.mdp as mdp

class Humanoid(LocomotionEnv):
    
    feet_name_expr = ".*ankle_link"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.action_scaling = 0.5
        self.robot = self.scene.articulations["robot"]
        self.mass_distibution: torch.Tensor = self.default_masses / self.default_mass_total
        
        self.motor_joint_indices = slice(None)

    @property
    def action_dim(self):
        return 19
    
