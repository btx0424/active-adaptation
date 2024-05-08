from omni.isaac.orbit.terrains import (
    TerrainImporterCfg,
    HfTerrainBaseCfg,
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    height_field
)
from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG as ROUGH_HARD
from omni.isaac.orbit.utils import configclass
from dataclasses import MISSING
import numpy as np

import omni.isaac.orbit.sim as sim_utils


@height_field.utils.height_field_to_mesh
def random_grid_terrain(difficulty: float, cfg: "HfRandomGridTerrainCfg"):
    
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    hf = np.random.uniform(
        cfg.grid_height_range[0] / cfg.vertical_scale,
        cfg.grid_height_range[1] / cfg.vertical_scale,
        (int(cfg.size[0] / cfg.grid_width), int(cfg.size[1] / cfg.grid_width))
    )
    x = np.linspace(0, hf.shape[0], width_pixels, endpoint=False).astype(int)
    y = np.linspace(0, hf.shape[1], length_pixels, endpoint=False).astype(int)
    hf = hf[x.reshape(-1, 1), y]    
    return np.rint(hf).astype(np.int16)


@configclass
class HfRandomGridTerrainCfg(HfTerrainBaseCfg):
    
    function = random_grid_terrain

    grid_width: float = MISSING
    """The width of the grid cells (in m)."""
    grid_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the grid cells (in m)."""


ROUGH_MEDIUM = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=40.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.15,
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.0, 0.05),
            noise_step=0.02,
            border_width=0.5
        ),
        # "boxes": MeshRandomGridTerrainCfg(
        #     proportion=0.20, 
        #     grid_width=0.45, 
        #     grid_height_range=(0.02, 0.05), 
        #     platform_width=2.0
        # ),
        "boxes": HfRandomGridTerrainCfg(
            proportion=0.20, 
            grid_width=0.45, 
            grid_height_range=(0.0, 0.05), 
            slope_threshold=0.1,
            border_width=0.5
        ),
        # "box": MeshRepeatedBoxesTerrainCfg(
        #     proportion=0.20,
        #     object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=36, height=0.1, size=(0.5, 0.5), max_yx_angle=15),
        #     object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=36, height=0.1, size=(0.5, 0.5), max_yx_angle=15),
        #     platform_width=2.0
        # ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.15),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.15),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.15, 0.25),
            platform_width=1.0,
            border_width=0.25
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.15, 0.25),
            platform_width=1.0,
            border_width=0.25
        ),
    },
)

ROUGH_EASY = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=40.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.30,
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.30,
            noise_range=(0.0, 0.05),
            noise_step=0.01,
            border_width=0.5
        ),
        # "boxes": MeshRandomGridTerrainCfg(
        #     proportion=0.20, 
        #     grid_width=0.45, 
        #     grid_height_range=(0.02, 0.05), 
        #     platform_width=2.0
        # ),
        "pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
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

TERRAINS = {
    "medium": ROUGH_MEDIUM,
    "easy": ROUGH_EASY,
}