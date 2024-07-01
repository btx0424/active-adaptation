import re
import torch
from typing import Sequence

from omni.isaac.lab.sensors import (
    RayCaster as _RayCaster,
    RayCasterData,
    RayCasterCfg,
)

import omni.isaac.lab.sim as sim_utils
import omni.physics.tensors.impl.api as physx
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, UsdPhysics

class RayCaster(_RayCaster):
    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        # Initialize base class
        super(_RayCaster, self).__init__(cfg)
        # Create empty variables for storing output data
        self._data = RayCasterData()

    def _initialize_impl(self):
        super()._initialize_impl()
        self._ALL_INDICES = torch.arange(self.num_instances, device=self.device)
    
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        env_ids = self._ALL_INDICES.reshape(self._num_envs, -1)[env_ids].flatten()
        super()._update_buffers_impl(env_ids)