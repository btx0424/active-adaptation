import os
import copy
import torch

from isaaclab_assets import (
    ArticulationCfg,
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG,
    UNITREE_GO2_CFG,
    ANYMAL_C_CFG,
)

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file, spawn_from_usd
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg

from .spawn import clone
from .quadruped import *
from .humanoid import *
from .scene import *
from .arm import *
from .sirius import *


ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "a1": UNITREE_A1_CFG,
    "go1": UNITREE_GO1_CFG,
    "go1m": UNITREE_GO1M_CFG,
    "go2": UNITREE_GO2_CFG,
    "go2m": UNITREE_GO2M_CFG,
    "go2abp": UNITREE_GO2ABP_CFG,
    "go2arx": UNITREE_GO2ARX_CFG,
    "aliengo": UNITREE_ALIENGO_CFG,
    "aliengo-a1": UNITREE_ALIENGO_A1_CFG,
    "aliengo-a1-fix": UNITREE_ALIENGO_A1_FIX_CFG,
    "h1": H1_CFG,
    "cy1": CY1_CFG,
    "cyberdog": CYBERDOG_CFG,
    "abp": ABP_CFG,
    "a1-arm": A1_CFG,
    "sirius": SIRIUS_CFG
}

