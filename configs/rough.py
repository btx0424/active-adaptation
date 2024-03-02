from active_adaptation.assets import (
    ArticulationCfg,
    ROBOTS,
    spawn_with_payload
)
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.envs import ViewerCfg
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
import omni.isaac.orbit.sim as sim_utils

from dataclasses import MISSING
from typing import Dict, List

from .terrain import *

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

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
        history_length=1
    )


@configclass
class EnvCfg:

    max_episode_length: int = 1000
    target_base_height: float = MISSING
    payload: bool = False

    history_length: int = 32

    viewer: ViewerCfg = ViewerCfg()
    scene: LocomotionSceneCfg = MISSING

    decimation: int  = 2
    sim = sim_utils.SimulationCfg(dt=0.01, disable_contact_processing=True)

    # decimation: int  = 4
    # sim = sim_utils.SimulationCfg(dt=0.005, disable_contact_processing=True)
    
    reward: Dict[str, float] = MISSING
    observation: Dict[str, List] = MISSING
    termination: List = MISSING
    randomization: List = MISSING

    def __post_init__(self):
        if self.payload:
            self.scene.robot.spawn.func = spawn_with_payload


def LocomotionEnvCfg(task_cfg):

    robot_cfg = ROBOTS[task_cfg.robot.lower()]

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
        observation = task_cfg.observation,
        termination = task_cfg.termination,
        randomization = task_cfg.get("randomization", {})
    )
    if "height_scan" not in task_cfg.observation.keys():
        env_cfg.scene.height_scanner = None
    else:
        prim_path = "{ENV_REGEX_NS}/Robot/" + task_cfg.observation["height_scan"]["height_scan"]["prim_path"]
        env_cfg.scene.height_scanner.prim_path = prim_path
    
    # slightly reduces GPU memory usage
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**21
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**21
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**20
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**22
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**19
    env_cfg.sim.physx.gpu_collision_stack_size = 2**25
    env_cfg.sim.physx.gpu_heap_capacity = 2**24
    
    return env_cfg

