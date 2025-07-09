import torch
import numpy as np
from typing import Tuple, TYPE_CHECKING

import active_adaptation
from active_adaptation.envs.mdp.base import Observation
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat
from active_adaptation.utils.symmetry import SymmetryTransform

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

if active_adaptation.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom, UsdPhysics
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, sim_utils


MESHES = {}


class height_scan(Observation):
    def __init__(
        self,
        env,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        x_resolution: float,
        y_resolution: float,
        flatten: bool=False,
        noise_scale = 0.005
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.flatten = flatten
        self.noise_scale = noise_scale
        
        x = torch.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0]) / x_resolution)+1)
        y = torch.linspace(y_range[0], y_range[1], int((y_range[1] - y_range[0]) / y_resolution)+1)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        self.pos = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1).to(self.device)
        self.shape = self.pos.shape[:2]
        
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/Command/height_scan",
                    markers={
                        "scandot": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
                        ),
                    }
                )
            )
            self.marker.set_visibility(True)

    def compute(self):
        root_pos_w = self.asset.data.root_pos_w.reshape(self.num_envs, 1, 1, 3)
        root_quat = yaw_quat(self.asset.data.root_quat_w).reshape(self.num_envs, 1, 1, 4)
        self.offset = quat_rotate(root_quat, self.pos.unsqueeze(0))
        self.height_map_w = self.env.get_ground_height_at(root_pos_w + self.offset)
        height_map = (self.height_map_w - root_pos_w[:, :, :, 2]).clamp(-1., 1.)
        if self.flatten:
            return height_map.reshape(self.num_envs, -1)
        else:
            return height_map.reshape(self.num_envs, 1, *self.shape)
    
    def debug_draw(self):
        if self.env.backend == "isaac":
            pos = self.asset.data.root_pos_w.reshape(self.num_envs, 1, 1, 3) + self.offset
            pos[:, :, :, 2] = self.height_map_w
            self.marker.visualize(pos.reshape(-1, 3))

    def symmetry_transforms(self):
        assert not self.flatten
        return SymmetryTransform(
            perm=torch.arange(self.shape[1]).flip(0), # (1, H, W), flip W
            signs=torch.ones(self.shape[1])
        )



class forward_scan(Observation):
    def __init__(
        self,
        env,
        hfov: Tuple[float, float],
        vfov: Tuple[float, float],
        resolution: Tuple[int, int],
        max_range: float = 5.0,
        flatten: bool=False,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.ground_mesh = self.env.ground_mesh
        self.max_range = max_range
        self.flatten = flatten
        
        hangles = torch.linspace(hfov[0], hfov[1], resolution[0])
        vangles = torch.linspace(vfov[0], vfov[1], resolution[1])
        vv, hh = torch.meshgrid(vangles, hangles, indexing="ij")
        directions = torch.stack([
            torch.cos(hh) * torch.cos(vv),
            torch.sin(hh) * torch.cos(vv),
            torch.sin(vv),
        ], dim=-1)
        self.shape = directions.shape[:2]
        self.directions = directions.reshape(-1, 3).to(self.device)
        self.num_rays = self.directions.shape[0]

        if self.env.backend == "isaac" and self.env.sim.has_gui():
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/Command/forward_scan",
                    markers={
                        "scandot": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.8)),
                        ),
                    }
                )
            )
            self.marker.set_visibility(True)
    
    def compute(self) -> torch.Tensor:
        directions = quat_rotate(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.directions.expand(self.num_envs, self.num_rays, 3)
        )
        ray_starts = self.asset.data.root_pos_w.unsqueeze(1).expand_as(directions)
        ray_hits = raycast_mesh(
            ray_starts=ray_starts.reshape(-1, 3),
            ray_directions=directions.reshape(-1, 3),
            max_dist=self.max_range,
            mesh=self.ground_mesh,
            return_distance=False,
        )[0].reshape(ray_starts.shape)
        ray_distance = (ray_hits - ray_starts).norm(dim=-1)
        ray_distance = ray_distance.nan_to_num(posinf=self.max_range)
        self.ray_hits = ray_starts + ray_distance.unsqueeze(-1) * directions
        if self.flatten:
            return ray_distance.reshape(self.num_envs, -1)
        else:
            return ray_distance.reshape(self.num_envs, 1, *self.shape)
    
    def symmetry_transforms(self):
        if self.flatten:
            perm = torch.arange(self.shape.numel())
            perm = perm.reshape(self.shape).flip(1)
            return SymmetryTransform(
                perm=perm.reshape(-1),
                signs=torch.ones(perm.numel())
            )
        else:
            return SymmetryTransform(
                perm=torch.arange(self.shape[1]).flip(0), # (1, H, W), flip W
                signs=torch.ones(self.shape[1])
            )

    def debug_draw(self):
        if self.env.backend == "isaac":
            pos = self.ray_hits.reshape(-1, 3)
            self.marker.visualize(pos)
