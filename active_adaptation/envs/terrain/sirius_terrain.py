from isaaclab.terrains import (
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
    MeshBoxTerrainCfg,
    MeshFloatingRingTerrainCfg,
    MeshGapTerrainCfg,
    MeshPitTerrainCfg,
    MeshRailsTerrainCfg,
    MeshStarTerrainCfg,
    SubTerrainBaseCfg,
    FlatPatchSamplingCfg
)
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG as ROUGH_HARD
from isaaclab.utils import configclass
from dataclasses import MISSING
import numpy as np
import trimesh

import isaaclab.sim as sim_utils


def ramp_terrain(difficulty: float, cfg: "RampTerrainCfg"):
    height = cfg.height_range[0] + (cfg.height_range[1] - cfg.height_range[0]) * difficulty
    
    mesh = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], height))

    up = np.random.rand() > 0.5
    if up:
        # remove the bottom face
        bottom_faces = np.where(mesh.triangles_center[:, 2] <= -height / 2)
        mesh.faces = np.delete(mesh.faces, bottom_faces, axis=0)
        # bevel top edges
        top_vertices = mesh.vertices[mesh.vertices[:, 2] >= height / 2]
        top_vertices[:, 0] *= 0.5
        mesh.vertices[mesh.vertices[:, 2] >= height / 2] = top_vertices
        mesh.vertices[:, 2] += height / 2
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, height])
    else:
        # remove the top face
        top_faces = np.where(mesh.triangles_center[:, 2] >= height / 2)
        mesh.faces = np.delete(mesh.faces, top_faces, axis=0)
        # bevel bottom edges
        bottom_vertices = mesh.vertices[mesh.vertices[:, 2] <= -height / 2]
        bottom_vertices[:, 0] *= 0.5
        mesh.vertices[mesh.vertices[:, 2] <= -height / 2] = bottom_vertices
        mesh.vertices[:, 2] -= height / 2
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, -height])
        # flip the normals for correct collision
        mesh.faces = np.fliplr(mesh.faces)
    # center the mesh
    mesh.vertices += np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    return [mesh], origin


@configclass
class RampTerrainCfg(SubTerrainBaseCfg):
    function = ramp_terrain
    height_range: tuple[float, float] = MISSING


ROUGH_EASY = TerrainGeneratorCfg(
    seed=0,
    size=(9.0, 9.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
        # "grid": MeshRandomGridTerrainCfg(
        #     proportion=0.20, 
        #     grid_width=0.45, 
        #     grid_height_range=(0.02, 0.05), 
        #     platform_width=2.0
        # ),
        "boxes": MeshBoxTerrainCfg(
            proportion=0.20,
            box_height_range=(0.1, 0.2),
            platform_width=2.0,
            double_box=True
        ),
        "ramp": RampTerrainCfg(
            proportion=0.20,
            height_range=(0.2, 0.4),
        ),
        "rails": MeshRailsTerrainCfg(
            proportion=0.20,
            rail_height_range=(0.05, 0.25),
            rail_thickness_range=(0.2, 0.4),
            platform_width=2.0,
        ),
        "star": MeshStarTerrainCfg(
            proportion=0.20,
            num_bars=5,
            bar_width_range=(0.8, 1.0),
            bar_height_range=(1.0, 1.0),
            platform_width=4.0,
            flat_patch_sampling={
                "star": FlatPatchSamplingCfg(
                    num_patches=5,
                    patch_radius=0.5,
                    max_height_diff=0.2,
                )
            }
        ),
    },
)

PLANE_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
        improve_patch_friction=True
    ),
)

ROUGH_TERRAIN_BASE_CFG = TerrainImporterCfg(
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
        restitution=1.0,
    ),
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #     project_uvw=True,
    # ),
    debug_vis=False,
)


TERRAINS = {
    "sirius_easy": ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_EASY),
    "sirius_hard": ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_HARD),
}


