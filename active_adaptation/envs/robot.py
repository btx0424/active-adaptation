import torch
from pxr import PhysxSchema
from omni.isaac.orbit.robots.legged_robot import LeggedRobot as _LeggedRobot
from omni.isaac.orbit.robots.legged_robot.legged_robot_cfg import LeggedRobotCfg
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.prims import RigidContactView

from typing import List, Optional, Sequence, Dict
from omni_drones.views import RigidPrimView, _RigidPrimView, disable_warnings
from omni_drones.utils.torch import quat_rotate_inverse, quat_axis

from torchrl.data import CompositeSpec

class LeggedRobot(_LeggedRobot):

    force_sensor_forces: torch.Tensor
    # force_sensor_torques: torch.Tensor
    feet_pos_b: torch.Tensor
    feet_vel_b: torch.Tensor

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        super().spawn(prim_path, translation, orientation)

        base_prim = prim_utils.get_prim_at_path(prim_path + "/base")
        PhysxSchema.PhysxArticulationForceSensorAPI.Apply(base_prim)
        
        # for path in ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
        #     prim = prim_utils.get_prim_at_path(prim_path + "/" + path)
        #     cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        #     cr_api.CreateThresholdAttr().Set(0)
        
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
        self.feet_pos_b = self.data.feet_state_b[..., :3]
        self.feet_vel_b = self.data.feet_state_b[..., 7:10]
        self.feet_pos_w = self.data.feet_state_w[..., :3]
        
        n_sensors = self.articulations._physics_view.max_force_sensors
        self.force_sensor_forces = torch.zeros(self.count, n_sensors, 3, device=self.device)
        # self.contact_view = RigidContactView(
        #     f"{self._prim_paths_expr}/.*_calf",
        #     filter_paths_expr=[],
        #     prepare_contact_sensors=False
        # )
        # self.contact_view.initialize()
        self.contact_forces = torch.zeros(self.count, 4, 3, device=self.device)
    
    def update_buffers(self, dt: float):
        super().update_buffers(dt)
        self.heading[:] = quat_axis(self.data.root_quat_w, 0)

        # with disable_warnings(self.articulations._physics_sim_view):
        #     force, torque = (
        #         self.articulations._physics_view
        #         .get_force_sensor_forces()
        #         .clone()
        #         .split([3, 3], dim=-1)
        #     )
        # self.force_sensor_forces.lerp_(force, 0.5)
        # self.contact_forces[:] = (
        #     self.contact_view.get_net_contact_forces()
        #     .reshape(self.count, 4, 3)
        # )


    def reset_buffers(self, env_ids):
        super().reset_buffers(env_ids)
        self.force_sensor_forces[env_ids] = 0.
