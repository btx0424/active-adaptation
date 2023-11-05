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
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg
import omni.isaac.orbit.sim as sim_utils

from dataclasses import MISSING
from typing import Dict, List

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=20,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_choices=(0.5, 0.75, 0.9),
    use_cache=False,
    sub_terrains={
        "random_rough_hard": HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.4
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.4
        ),
    },
)

@configclass
class LocomotionSceneCfg(InteractiveSceneCfg):
    
    num_envs: int = 4096
    env_spacing: float = 2.5

    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=3000.0,
        ),
    )

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     physics_material = sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         improve_patch_friction=True
    #     ),
    # )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    #     history_length=1
    # )


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
        "action_rate_l2": 0.01,
        "orientation": 0.1
    },
    observation = {
        ("agents", "observation"): [
            "command",
            "root_quat_w",
            "root_angvel_b",
            "projected_gravity_b",
            "joint_pos",
            "prev_actions",
            # privileged
            # "joint_vel",
            # "root_linvel_b",
            # "feet_pos_b",
        ],
        ("agents", "observation_priv"): [
            "joint_vel",
            "root_linvel_b",
            "feet_pos_b",
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