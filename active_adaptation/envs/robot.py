import torch
from pxr import PhysxSchema
from omni.isaac.orbit.robots.legged_robot import LeggedRobot as _LeggedRobot
from omni.isaac.orbit.robots.legged_robot.legged_robot_cfg import LeggedRobotCfg
import omni.isaac.core.utils.prims as prim_utils

from typing import List, Optional, Sequence
from omni_drones.views import RigidPrimView, disable_warnings
from omni_drones.utils.torch import quat_rotate_inverse, quat_axis

class LeggedRobot(_LeggedRobot):

    force_sensor_forces: torch.Tensor
    # force_sensor_torques: torch.Tensor
    feet_pos_b: torch.Tensor
    feet_vel_b: torch.Tensor

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        super().spawn(prim_path, translation, orientation)

        base_prim = prim_utils.get_prim_at_path(prim_path + "/base")
        PhysxSchema.PhysxArticulationForceSensorAPI.Apply(base_prim)
        
        for path in ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
            calf_prim = prim_utils.get_prim_at_path(prim_path + "/" + path)
            PhysxSchema.PhysxArticulationForceSensorAPI.Apply(calf_prim)
        
        for path in ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]:
            thigh_prim = prim_utils.get_prim_at_path(prim_path + "/" + path)
            PhysxSchema.PhysxArticulationForceSensorAPI.Apply(thigh_prim)

        
    def initialize(self, prim_paths_expr: str = None):
        super().initialize(prim_paths_expr)
        self.base = RigidPrimView(
            f"{self._prim_paths_expr}/base",
            reset_xform_properties=False,
        )
        self.legs = RigidPrimView(
            f"{self._prim_paths_expr}/.*_(calf|thigh)",
            reset_xform_properties=False,
            shape=(self.count, 8)
        )
        self.base.initialize()
        self.legs.initialize()
        self.heading = torch.zeros(self.count, 3, device=self.device)
        self.feet_pos_b = torch.zeros(self.count, 4, 3, device=self.device)
        self.feet_vel_b = torch.zeros(self.count, 4, 3, device=self.device)
        self.force_sensor_forces = torch.zeros(self.count, 9, 3, device=self.device)
    
    def update_buffers(self, dt: float):
        super().update_buffers(dt)
        self.heading[:] = quat_axis(self.data.root_quat_w, 0)

        with disable_warnings(self.articulations._physics_sim_view):
            force, torque = (
                self.articulations._physics_view
                .get_force_sensor_forces()
                .clone()
                .split([3, 3], dim=-1)
            )
        self.force_sensor_forces[:] = force

        feet_pos = []
        feet_vel = []
        for body, view in self.feet_bodies.items():
            feet_vel.append(view.get_velocities()[..., :3])
            feet_pos_w, feet_quat_w = view.get_world_poses()
            feet_pos.append(feet_pos_w - self.data.root_pos_w)
        
        self.feet_vel_b[:] = quat_rotate_inverse(
            self.data.root_quat_w.unsqueeze(1).expand(-1, 4, -1),
            torch.stack(feet_vel, dim=-2)
        )
        self.feet_pos_b[:] = quat_rotate_inverse(
            self.data.root_quat_w.unsqueeze(1).expand(-1, 4, -1),
            torch.stack(feet_pos, dim=-2)
        )


