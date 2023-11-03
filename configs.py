from active_adaptation.assets import (
    ArticulationCfg,
    UNITREE_A1_CFG, 
    CASSIE_CFG, 
    ANYMAL_C_CFG
)
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.envs import ViewerCfg
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.sensors import ContactSensorCfg
import omni.isaac.orbit.sim as sim_utils

from dataclasses import MISSING
from typing import Dict, List

@configclass
class LocomotionSceneCfg(InteractiveSceneCfg):
    
    num_envs: int = 4096
    env_spacing: float = 4.

    robot: ArticulationCfg = MISSING
    contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=3000.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="max",
            static_friction=1.0,
            dynamic_friction=1.0,
            improve_patch_friction=True
        ),
    )


@configclass
class EnvCfg:

    max_episode_length: int = 800
    decimation: int  = 4
    target_base_height: float = MISSING

    viewer: ViewerCfg = ViewerCfg()
    scene: InteractiveSceneCfg = MISSING

    sim = sim_utils.SimulationCfg(dt=0.005, disable_contact_processing=True)
    
    reward: Dict[str, float] = MISSING
    observation: Dict[str, List] = MISSING
    termination: List = [
        "crash"
    ]


UNITREE_A1_ENV = EnvCfg(
    target_base_height=0.3,
    scene = LocomotionSceneCfg(
        robot=UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    ),
    reward = {
        "linvel": 2.0,
        "heading": 0.5,
        "base_height": 0.5,
        "energy": 0.0005,
        "joint_acc_l2": 2.5e-7,
        "joint_torques_l2": 2.5e-6,
    },
    observation = {
        ("agents", "observation"): [
            "command",
            "root_quat_w",
            "root_angvel_b",
            "projected_gravity_b",
            "joint_pos",
            "joint_vel",
        ],
        ("agents", "observation_priv"): [
            "root_linvel_b",
            "applied_torques",
        ]
    }
)


CASSIE_ENV = EnvCfg(
    target_base_height=0.7,
    scene = LocomotionSceneCfg(
        robot=CASSIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    ),
    reward = {
        "linvel": 2.0,
        "heading": 0.5,
        "base_height": 0.5,
        "energy": 0.0005,
        "joint_acc_l2": 2.5e-7,
        "joint_torques_l2": 2.5e-6,
        "survive": 0.5,
    },
    observation = {
        ("agents", "observation"): [
            "command",
            "root_quat_w",
            "root_angvel_b",
            "projected_gravity_b",
            "joint_pos",
            "joint_vel",
        ],
        ("agents", "observation_priv"): [
            "root_linvel_b",
            "applied_torques",
        ]
    },
)