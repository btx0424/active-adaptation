from active_adaptation.assets import (
    ArticulationCfg,
    ROBOTS,
    spawn_with_payload
)
from active_adaptation.utils.orbit import RayCaster
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.envs import ViewerCfg
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns, TiledCameraCfg
import omni.isaac.lab.sim as sim_utils

from dataclasses import MISSING
from typing import Dict, List

from .terrain import *

@configclass
class LocomotionSceneCfg(InteractiveSceneCfg):
    
    num_envs: int = 4096
    env_spacing: float = 2.5

    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    
    light_0: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light_0",
        spawn=sim_utils.DistantLightCfg(
            color=(0.4, 0.7, 0.9),
            intensity=3000.0,
            angle=10,
            exposure=0.2,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            rot=(0.9330127,  0.25     ,  0.25     , -0.0669873)
        )
    )
    light_1: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light_1",
        spawn=sim_utils.DistantLightCfg(
            color=(0.8, 0.5, 0.5),
            intensity=3000.0,
            angle=20,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            rot=(0.78201786,  0.3512424 ,  0.50162613, -0.11596581)
        )
    )
    light_2: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light_2",
        spawn=sim_utils.DistantLightCfg(
            color=(0.8, 0.5, 0.4),
            intensity=3000.0,
            angle=20,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            rot=(7.07106781e-01, 5.55111512e-17, 6.12372436e-01, 3.53553391e-01)
        )
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
    payload: bool = False
    action_scaling: float = 0.5

    history_length: int = 32

    viewer: ViewerCfg = ViewerCfg(
        eye=(4., 4., 4.)
    )
    scene: LocomotionSceneCfg = MISSING

    # decimation: int  = 2
    # sim = sim_utils.SimulationCfg(dt=0.01, disable_contact_processing=True)

    decimation: int  = 4
    sim = sim_utils.SimulationCfg(dt=0.005, disable_contact_processing=True)
    
    command: Dict = MISSING
    reward: Dict[str, float] = MISSING
    observation: Dict[str, List] = MISSING
    termination: List = MISSING
    randomization: List = MISSING

    def __post_init__(self):
        if self.payload:
            self.scene.robot.spawn.func = spawn_with_payload


def LocomotionEnvCfg(task_cfg):

    robot_cfg = ROBOTS[task_cfg.robot.lower()]

    terrain = task_cfg.get("terrain", "plane")
    if not terrain == "plane":
        terrain_cfg = ROUGH_TERRAIN_CFG
        terrain_cfg.terrain_generator = TERRAINS[task_cfg.terrain]
    else:
        terrain_cfg = FLAT_TERRAIN_CFG
    
    randomizations = dict(task_cfg.get("randomization", {}))
    scale_range = randomizations.pop("random_scale", (1.0, 1.0))
    robot_cfg.spawn.scale_range = scale_range

    env_cfg = EnvCfg(
        max_episode_length=task_cfg.max_episode_length,
        action_scaling=task_cfg.action_scaling,
        payload=task_cfg.payload,
        scene = LocomotionSceneCfg(
            num_envs=task_cfg.num_envs,                                                                                                                                                         
            robot=robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot"),
            terrain=terrain_cfg,
            replicate_physics=False,
        ),
        command = task_cfg.command,
        reward = task_cfg.reward,
        observation = task_cfg.observation,
        termination = task_cfg.termination,
        randomization = randomizations
    )
    use_height_scan = False
    for group in task_cfg.observation.values():
        if "height_scan" in group.keys():
            prim_path = "{ENV_REGEX_NS}/Robot/" + group["height_scan"]["prim_path"]
            env_cfg.scene.height_scanner.prim_path = prim_path
            use_height_scan = True
    if not use_height_scan:
        env_cfg.scene.height_scanner = None
    
    # slightly reduces GPU memory usage
    # env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**21
    # env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**21
    # env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**20
    # env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
    # env_cfg.sim.physx.gpu_collision_stack_size = 2**25
    # env_cfg.sim.physx.gpu_heap_capacity = 2**24
    
    return env_cfg

