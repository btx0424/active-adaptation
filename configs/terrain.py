from omni.isaac.lab.terrains import (
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
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG as ROUGH_HARD
from omni.isaac.lab.utils import configclass
from dataclasses import MISSING
import numpy as np

import omni.isaac.lab.sim as sim_utils


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


FLAT = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.5,
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0.0, 0.06),
            noise_step=0.02,
            border_width=0.5
        ),
    },
)

ROUGH_MEDIUM = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.0, 0.06),
            noise_step=0.02,
            border_width=0.5
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45, 
            grid_height_range=(0.02, 0.05), 
            platform_width=2.0
        ),
        # "box": MeshRepeatedBoxesTerrainCfg(
        #     proportion=0.20,
        #     object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=36, height=0.15, size=(0.6, 0.6), max_yx_angle=15),
        #     object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=36, height=0.15, size=(0.6, 0.6), max_yx_angle=15),
        #     platform_width=2.0
        # ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.15),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.20),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.15, 0.25),
            platform_width=1.0,
            border_width=0.25
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.15, 0.25),
            platform_width=1.0,
            border_width=0.25
        ),
    },
)

ROUGH_EASY = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
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
            noise_range=(0.0, 0.06),
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

# scale down the terrains because the robot is small
ROUGH_HARD.sub_terrains["boxes"].grid_height_range = (0.025, 0.025)
ROUGH_HARD.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
ROUGH_HARD.sub_terrains["random_rough"].noise_step = 0.01


STAIRS = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.0, 0.06),
            noise_step=0.02,
            border_width=0.5
        ),
        "box": MeshRepeatedBoxesTerrainCfg(
            proportion=0.20,
            object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=36, height=0.15, size=(0.6, 0.6), max_yx_angle=15),
            object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=36, height=0.15, size=(0.6, 0.6), max_yx_angle=15),
            platform_width=2.0
        ),
        "pyramid_stairs_inv_a": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.20),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_b": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.20),
            step_width=0.50,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
    },
)

STAIRS_TEST = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_inv_a": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.10, 0.20),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_b": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.10, 0.20),
            step_width=0.50,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
    },
)


SLOPES_AND_CURBS = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.25,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.1),
            step_width=0.40,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.25, 
            grid_width=0.60, 
            grid_height_range=(0.02, 0.05), 
            platform_width=2.0
        ),
        "pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
        ),
    },
)

STAIRS_EASY = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.0, 0.04),
            noise_step=0.02,
            border_width=0.5
        ),
        "pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
        ),
        "pyramid_stairs_inv_a": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.15),
            step_width=0.40,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
    },
)

SLOPES_AND_BOXES = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.1,
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.5, 
            grid_width=0.60, 
            grid_height_range=(0.02, 0.05), 
            platform_width=2.0
        ),
        "pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
        ),
        "pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
        ),
    },
)

DIC = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.2,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.25,
        ),
        # "pyramid_stairs_inv_a": MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.25,
        #     step_height_range=(0.08, 0.15),
        #     step_width=0.80,
        #     platform_width=3.5,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "random_rough_easy": HfRandomUniformTerrainCfg(
        #     proportion=0.25,
        #     noise_range=(0.0, 0.10),
        #     noise_step=0.02,
        #     border_width=0.5,
        #     # downsampled_scale=0.3
        # ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.25, 
            grid_width=0.45, 
            grid_height_range=(0.02, 0.05), 
            platform_width=2.0
        ),
    },
)


TERRAINS = {
    "medium": ROUGH_MEDIUM,
    "easy": ROUGH_EASY,
    "hard": ROUGH_HARD,
    "flat": FLAT,
    "stairs": STAIRS,
    "snc": SLOPES_AND_CURBS,
    "stairs_test": STAIRS_TEST,
    "stairs_easy": STAIRS_EASY,
    "slopes_boxes": SLOPES_AND_BOXES,
    "dic": DIC
}

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
