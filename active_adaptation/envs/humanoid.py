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

