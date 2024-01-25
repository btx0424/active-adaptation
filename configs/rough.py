from active_adaptation.assets import (
    ArticulationCfg,
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG,
    UNITREE_GO2_CFG,
    CASSIE_CFG, 
    ANYMAL_C_CFG,
    spawn_with_payload
)
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.envs import ViewerCfg
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.terrains import (
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    TerrainGeneratorCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPyramidStairsTerrainCfg
)
import omni.isaac.orbit.sim as sim_utils

from dataclasses import MISSING
from typing import Dict, List

from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG as ROUGH_HARD

ROUGH_EASY = TerrainGeneratorCfg(
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
            proportion=0.35, noise_range=(0.02, 0.05), noise_step=0.01, border_width=0.4
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.35, noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.4
        ),
         "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.3), platform_width=1.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.3), platform_width=1.0, border_width=0.25
        ),
    },
)

ROUGH_MEDIUM = TerrainGeneratorCfg(
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
            proportion=0.35, noise_range=(0.02, 0.05), noise_step=0.01, border_width=0.4
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.35, noise_range=(0.01, 0.02), noise_step=0.01, border_width=0.4
        ),
         "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.3), platform_width=1.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.3), platform_width=1.0, border_width=0.25
        ),
        # "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.15,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


ROUGH_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=MISSING,
    max_init_terrain_level=None,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
        project_uvw=True,
    ),
    debug_vis=True,
)

FLAT_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        improve_patch_friction=True
    ),
)


@configclass
class LocomotionSceneCfg(InteractiveSceneCfg):
    
    num_envs: int = 4096
    env_spacing: float = 2.5

    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=2, track_air_time=True)
    
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=3000.0,
        ),
    )

    terrain: TerrainImporterCfg = MISSING

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

    max_episode_length: int = 1000
    decimation: int  = 2
    target_base_height: float = MISSING
    payload: bool = False

    history_length: int = 32

    viewer: ViewerCfg = ViewerCfg()
    scene: LocomotionSceneCfg = MISSING

    sim = sim_utils.SimulationCfg(dt=0.01, disable_contact_processing=True)
    
    reward: Dict[str, float] = MISSING
    observation: Dict[str, List] = MISSING
    termination: List = [
        "crash"
    ]

    def __post_init__(self):
        if self.payload:
            self.scene.robot.spawn.func = spawn_with_payload

REWARD_RECOVER = {
    "orientation": 1.0,
    "base_height": 0.2,
    "stand_on_feet": 0.2,
    "action_rate_l2": 0.01,
}

def LocomotionEnvCfg(task_cfg):

    robot_cfg = {
        "a1": UNITREE_A1_CFG,
        "go2": UNITREE_GO2_CFG,
        "cassie": CASSIE_CFG,
    }[task_cfg.robot.lower()]

    for key, actuator in robot_cfg.actuators.items():
        actuator.friction = 0.02

    if task_cfg.terrain == "plane":
        terrain_cfg = FLAT_TERRAIN_CFG
    elif task_cfg.terrain == "easy":
        terrain_cfg = ROUGH_TERRAIN_CFG
        terrain_cfg.terrain_generator = ROUGH_EASY
    elif task_cfg.terrain == "medium":
        terrain_cfg = ROUGH_TERRAIN_CFG
        terrain_cfg.terrain_generator = ROUGH_MEDIUM
    else:
        raise ValueError(task_cfg.terrain)
    
    env_cfg = EnvCfg(
        max_episode_length=task_cfg.max_episode_length,
        payload=task_cfg.payload,
        target_base_height=0.3,
        scene = LocomotionSceneCfg(
            num_envs=task_cfg.num_envs,
            robot=robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot"),
            terrain=terrain_cfg,
        ),
        reward = task_cfg.reward,
        observation = task_cfg.observation
    )
    return env_cfg

