from omni.isaac.orbit.terrains import (
    TerrainImporterCfg,
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg,
    HfPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPyramidStairsTerrainCfg
)
from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG as ROUGH_HARD
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils


ROUGH_LEGACY = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=20,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough_hard": HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.0, 0.06), noise_step=0.03, border_width=0.4
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.0, 0.05), noise_step=0.01, border_width=0.4
        ),
    },
)

ROUGH_EASY = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough_hard": HfRandomUniformTerrainCfg(
            proportion=0.35, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.5
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.35, noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.5
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.1, 
            noise_range=(0.01, 0.1), 
            noise_step=0.02, 
            border_width=0.5,
            downsampled_scale=0.3,
            slope_threshold=None,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, 
            slope_range=(0.0, 0.3), 
            platform_width=1.0, 
            border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, 
            slope_range=(0.0, 0.3), 
            platform_width=1.0, 
            border_width=0.25
        ),
    },
)

ROUGH_MEDIUM = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=40.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough_hard": HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.0, 0.1),
            noise_step=0.02,
            border_width=0.5,
            downsampled_scale=0.2
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.0, 0.05),
            noise_step=0.02,
            border_width=0.5
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.25),
            platform_width=1.0,
            border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.25),
            platform_width=1.0,
            border_width=0.25
        ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.2),
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
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #     project_uvw=True,
    # ),
    debug_vis=False,
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