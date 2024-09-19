from typing import Sequence
import torch
import itertools
import math

from omni.isaac.lab.app import AppLauncher

def main():

    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.sim import SimulationContext, SimulationCfg
    from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
    from omni.isaac.lab.terrains import TerrainImporterCfg
    from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, Articulation, RigidObjectCfg, RigidObject
    from omni.isaac.lab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg, DCMotorCfg
    from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
    from omni.isaac.lab.utils.math import (
        quat_rotate_inverse, 
        quat_rotate, 
        quat_conjugate, 
        quat_mul,
        random_yaw_orientation,
        quat_from_angle_axis,
        quat_from_euler_xyz,
    )
    from omni.isaac.lab.markers import VisualizationMarkers
    from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

    from active_adaptation.assets.arm import A1_CFG

    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
        )
        # lights
        # sky_light = AssetBaseCfg(
        #     prim_path="/World/skyLight",
        #     spawn=sim_utils.DomeLightCfg(
        #         intensity=750.0,
        #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        #     ),
        # )

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
        
        arm = A1_CFG
        arm.actuators["arm"].stiffness = 0
        arm.actuators["arm"].damping = 40
    
    sim = SimulationContext(SimulationCfg())
    scene = InteractiveScene(SceneCfg(num_envs=4, env_spacing=4))
    sim.reset()
    for _ in range(4):
        sim.step(render=True)

    arm: Articulation = scene["arm"]
    ee_body_id = arm.find_bodies("arm_link6")[0][0]
    base_pos_w = arm.data.root_pos_w.clone()
    center = torch.tensor([0.5, 0.0, 0.3], device=base_pos_w.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    for i in itertools.count():
        t = i * scene.physics_dt

        offset = torch.tensor([0., math.cos(t), math.sin(t)], device=scene.device) * 0.25
        target_pos_b = center + offset
        target_vel_b = torch.tensor([0., -math.sin(t), math.cos(t)], device=scene.device) * 0.2

        ee_pos_b = arm.data.body_pos_w[:, ee_body_id] - arm.data.root_pos_w
        delta_pos_b = (target_pos_b - ee_pos_b)
        delta_vel_b = (target_vel_b - arm.data.body_lin_vel_w[:, ee_body_id])
        desired_vel = 20 * delta_pos_b + 5 * delta_vel_b
        
        jacobian: torch.Tensor = arm.root_physx_view.get_jacobians()[:, ee_body_id, :3]
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = torch.eye(jacobian.shape[2], device=jacobian.device) * 1.
        # jacobian_pinv = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix)
        # delta_q = (jacobian_pinv @ desired_vel.unsqueeze(-1)).squeeze(-1)
        A = jacobian_T @ jacobian + lambda_matrix
        b = jacobian_T @ desired_vel.unsqueeze(-1)
        delta_q = torch.linalg.lstsq(A, b).solution.squeeze(-1)
        
        arm.set_joint_position_target(arm.data.joint_pos + delta_q * scene.physics_dt)
        arm.set_joint_velocity_target(delta_q)

        scene.write_data_to_sim()

        sim.step(render=True)
        scene.update(sim.cfg.dt)

        if i % 4 == 0:
            print(delta_pos_b[0])
        
        goal_marker.visualize(target_pos_b + base_pos_w)



def clamp_norm(x: torch.Tensor, max: float):
    norm = x.norm(dim=-1, keepdim=True)
    return (x / norm) * norm.clamp_max(max)

if __name__ == "__main__":
    main()