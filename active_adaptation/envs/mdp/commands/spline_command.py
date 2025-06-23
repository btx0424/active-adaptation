import torch

from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.envs.mdp.base import Reward
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat
from active_adaptation.utils.symmetry import SymmetryTransform
import active_adaptation.utils.spline as spline


class SplineCommand(Command):
    """
    
    """
    def __init__(self, env) -> None:
        super().__init__(env)

        with torch.device(self.device):
            self.spline_ps = torch.zeros(self.num_envs, 4, 2)
            self.spline_t = torch.zeros(self.num_envs, 1)
            self.spline_time_scale = torch.ones(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

        if self.env.sim.has_gui() and self.env.backend == "isaac":
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, CONTACT_SENSOR_MARKER_CFG
            import isaaclab.sim as sim_utils
            self.control_points_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/control_points",
                    markers={
                        "control_points": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                        "waypoints": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                        ),
                    },
                )
            )
            self.control_points_marker.set_visibility(True)
    
    def reset(self, env_ids):
        self.spline_t[env_ids] = 0.0
        self.spline_ps[env_ids] = spline.create_from(
            self.asset.data.root_pos_w[env_ids, :2],
            torch.zeros(len(env_ids), 2, device=self.device),
        )
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            t = torch.linspace(0, 1, 25, device=self.device)
            x, v = spline.cubic_bezier(t.unsqueeze(0), self.spline_ps[0:1])
            self.traj_vis = x[0].cpu()

    @property
    def command(self):
        return torch.cat(
            quat_rotate_inverse(self.asset.data.root_quat_w, self.target_pos_w - self.asset.data.root_pos_w),
            quat_rotate_inverse(self.asset.data.root_quat_w, self.target_lin_vel_w),
            # quat_rotate_inverse(self.asset.data.root_quat_w, self.target_ang_vel_w),
        )
    
    def symmetry_transforms(self):
        return SymmetryTransform(
            perm=torch.arange(6),
            signs=torch.tensor([1, -1, 1, 1, -1, 1]),
        )

    def update(self):
        x, v = spline.cubic_bezier(self.spline_t, self.spline_ps)
        v = v * self.spline_time_scale.reshape(self.num_envs, 1, 1)

        self.target_pos_w = x
        self.target_lin_vel_w = v

        self._cum_error = (self.target_pos_w - self.asset.data.root_pos_w)[:, :2].norm(dim=-1, keepdim=True)
        self.spline_t += self.env.step_dt * self.spline_time_scale
    
    def debug_draw(self):
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            ctps = torch.cat([self.spline_ps.cpu(), 0.5 * torch.ones(*self.spline_ps.shape[:2], 1)], 2)
            ctps = ctps.reshape(-1, 3)
            wps = torch.cat([self.target_pos_w.cpu(), 0.5 * torch.ones(*self.target_pos_w.shape[:2], 1)], 2)
            wps = wps.reshape(-1, 3)
            self.control_points_marker.visualize(
                translations=torch.cat([ctps, wps]),
                marker_indices=[0] * ctps.shape[0] + [1] * wps.shape[0],
            )
            self.env.debug_draw.plot(
                torch.cat([self.traj_vis, torch.ones(self.traj_vis.shape[0], 1) * 0.5], dim=-1),
                color=(1., 1., 1., 1.),
            )

