import os
import copy
import torch

from .quadruped import *
from .humanoid import *
from .scene import *
from .arm import *
from .sirius import *


ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "a1": UNITREE_A1_CFG,
    "go2": UNITREE_GO2_CFG,
    "go2m": UNITREE_GO2M_CFG,
    "go2abp": UNITREE_GO2ABP_CFG,
    "go2arx": UNITREE_GO2ARX_CFG,
    "aliengo": UNITREE_ALIENGO_CFG,
    "aliengo-a1": UNITREE_ALIENGO_A1_CFG,
    "aliengo-a1-fix": UNITREE_ALIENGO_A1_FIX_CFG,
    "h1": H1_CFG,
    "cy1": CY1_CFG,
    "abp": ABP_CFG,
    "a1-arm": A1_CFG,
    "sirius_wheel": SIRIUS_WHEEL_CFG,
    "g1_27dof": G1_CFG,
    "h2": H2_CFG,
}


def get_asset_meta(asset: Articulation):
    if not asset.is_initialized:
        raise RuntimeError("Articulation is not initialized. Please wait until `sim.reset` is called.")
    meta = {
        "init_state": asset.cfg.init_state.to_dict(),
        "body_names_isaac": asset.body_names,
        "joint_names_isaac": asset.joint_names,
        "actuators": {},
        "default_joint_pos": asset.data.default_joint_pos[0].tolist(),
    }
    for actuator_name, actuator in asset.actuators.items():
        meta["actuators"][actuator_name] = actuator.cfg.to_dict()
    return meta

