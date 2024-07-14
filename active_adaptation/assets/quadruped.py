import os
import copy

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import UNITREE_GO2_CFG, UNITREE_A1_CFG, ArticulationCfg
from omni.isaac.lab.actuators import DCMotorCfg

ASSET_PATH = os.path.dirname(__file__)

UNITREE_GO2_CFG = copy.deepcopy(UNITREE_GO2_CFG)
UNITREE_GO2_CFG.spawn.usd_path = f"{ASSET_PATH}/Go2/go2.usd"
UNITREE_GO2_CFG.init_state.pos = (0., 0., 0.35)
UNITREE_GO2_CFG.actuators["base_legs"].effort_limit = {
    "(?!.*_calf_joint).*": 23.5,
    ".*_calf_joint": 35.5,
}
UNITREE_GO2_CFG.actuators["base_legs"].saturation_effort = 35.5

UNITREE_GO2M_CFG = copy.deepcopy(UNITREE_GO2_CFG)
UNITREE_GO2M_CFG.spawn.usd_path = f"{ASSET_PATH}/go2m.usd"
UNITREE_GO2M_CFG.actuators["arm"] = DCMotorCfg(
    joint_names_expr=["joint.*"],
    effort_limit=200.,
    saturation_effort=200.,
    velocity_limit=5.0,
    # stiffness=30.0,
    # damping=1.0,
    # friction=0.0,
    stiffness={
        "joint[1-3]": 20.0,
        "joint[4-6]": 15.0,
    },
    damping={
        "joint[1-3]": 1.0,
        "joint[4-6]": 0.5,
    },
    friction=0.001,
)
UNITREE_GO2M_CFG.init_state.joint_pos["joint[1,2]"] = 0.3

CYBERDOG_CFG = copy.deepcopy(UNITREE_A1_CFG)
CYBERDOG_CFG.spawn.usd_path = f"{ASSET_PATH}/cyberdog2_v3.usd"
CYBERDOG_CFG.actuators["base_legs"].stiffness = 20.
CYBERDOG_CFG.actuators["base_legs"].damping = 0.5
CYBERDOG_CFG.actuators["base_legs"].effort_limit = 12.
CYBERDOG_CFG.actuators["base_legs"].saturation_effort = 12.
CYBERDOG_CFG.actuators["base_legs"].friction = 0.02
CYBERDOG_CFG.init_state.pos = (0., 0., 0.33)
CYBERDOG_CFG.init_state.joint_pos = {
    ".*_hip_joint": 0.0,
    ".*thigh_joint": 0.78,
    ".*calf_joint": -1.22,
}
CYBERDOG_CFG.spawn.collision_props=sim_utils.CollisionPropertiesCfg(
    contact_offset=0.05,
    rest_offset=0.0,
)

UNITREE_GO1M_CFG: ArticulationCfg = UNITREE_A1_CFG.replace()
UNITREE_GO1M_CFG.spawn.usd_path = f"{ASSET_PATH}/widowGo1.usd"
UNITREE_GO1M_CFG.actuators["arm"] = DCMotorCfg(
    joint_names_expr=[".*widow_(waist|shoulder|elbow)"],
    stiffness=30.0,
    velocity_limit=2.0,
    damping=1.0,
    saturation_effort=200
)