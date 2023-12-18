import hydra
import yaml
import torch
import numpy as np
from tqdm import tqdm
from omni.isaac.orbit.app import AppLauncher

from omegaconf import OmegaConf

def main():
    # launch omniverse app
    app_launcher = AppLauncher({"headless": False})
    simulation_app = app_launcher.app

    from omni.isaac.orbit.sim import SimulationContext
    import omni.isaac.orbit.sim as sim_utils
    from omni.isaac.orbit.assets import AssetBaseCfg
    from omni.isaac.orbit.terrains import TerrainImporterCfg
    from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
    from omni.isaac.orbit.utils import configclass, class_to_dict
    from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor, RayCasterCfg, patterns
    from omni.isaac.orbit.sim import schemas
    from omni.isaac.core.utils.torch.rotations import quat_rotate


    from active_adaptation.assets import (
        UNITREE_A1_CFG,
        UNITREE_GO2_CFG,
        CASSIE_CFG, 
        spawn_with_payload
    )
    from active_adaptation.envs.utils import attach_payload
    from active_adaptation.utils.helpers import batchify

    quat_rotate = batchify(quat_rotate)

    from configs import ROUGH_TERRAIN_CFG
    from omni_drones.envs.isaac_env import DebugDraw

    robot_cfg = UNITREE_GO2_CFG

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        lazy_sensor_update: bool = False

        # terrain - flat terrain plane
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
        )
        # terrain - rough terrain
        # terrain = TerrainImporterCfg(
        #     prim_path="/World/ground",
        #     terrain_type="generator",
        #     terrain_generator=ROUGH_TERRAINS_CFG,
        #     max_init_terrain_level=5,
        #     collision_group=-1,
        #     physics_material=sim_utils.RigidBodyMaterialCfg(
        #         friction_combine_mode="multiply",
        #         restitution_combine_mode="multiply",
        #         static_friction=1.0,
        #         dynamic_friction=1.0,
        #     ),
        #     debug_vis=False,
        # )
        
        robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # robot.spawn.func = spawn_with_payload
        # robot.spawn.activate_contact_sensors = ".*_calf"
        # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_calf", debug_vis=False)

        robot.spawn.activate_contact_sensors = True
        contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", debug_vis=False)

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            # init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
        )

        # height_scanner = RayCasterCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/base",
        #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        #     attach_yaw_only=True,
        #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        #     debug_vis=True,
        #     mesh_prim_paths=["/World/ground"],
        # )

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005, 
        disable_contact_processing=True
    )
    sim_cfg.physx.gpu_max_rigid_contact_count = 2**21
    sim_cfg.physx.gpu_max_rigid_patch_count = 2**15
    sim_cfg.physx.gpu_found_lost_pairs_capacity = 2**20
    sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 2**22
    sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**19
    sim_cfg.physx.gpu_collision_stack_size = 2**24
    sim_cfg.physx.gpu_max_soft_body_contacts = 0
    sim_cfg.physx.gpu_max_particle_contacts = 0
    sim_cfg.physx.gpu_heap_capacity = 2**24
    sim_cfg.physx.gpu_temp_buffer_capacity = 2**22
    sim = SimulationContext(sim_cfg)
    
    # use viewport camera for rendering
    import omni.replicator.core as rep
    render_product = rep.create.render_product("/OmniverseKit_Persp", (960, 720))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annotator.attach([render_product])

    scene_cfg = SceneCfg(num_envs=1024, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    SimulationContext.set_camera_view(
        eye=torch.tensor([5., 5., 5.]),
        target=torch.tensor([0.0, 0.0, 0.0]),
    )
    
    sim.reset()
    for _ in range(4):
        sim.step(render=True)
    debug_draw = DebugDraw()
    
    sim_dt = sim.get_physics_dt()
    robot = scene.articulations["robot"]
    foot_indices = [i for i, name in enumerate(robot.body_names) if "foot" in name]
    calf_indices = [i for i, name in enumerate(robot.body_names) if "calf" in name]

    contact_forces: ContactSensor = scene.sensors["contact_forces"]
    feet_offset = (
        torch.tensor([0., 0., -0.2], device=sim.device)
        .reshape(1, 1, 3)
    )

    init_root_state = robot.data.default_root_state.clone()
    init_root_state[..., :3] += scene._default_env_origins
    init_joint_pos = robot.data.default_joint_pos.clone()
    init_joint_vel = robot.data.default_joint_vel.clone()

    def reset():
        scene.reset()
        scene.write_data_to_sim()
        robot.write_root_state_to_sim(init_root_state)
        robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel)
        sim.physics_sim_view.flush()

    reset()

    frames = []
    for t in tqdm(range(2000)):
        should_render = t % 2 == 0
        robot.set_joint_position_target(init_joint_pos)
        scene.write_data_to_sim()

        sim.step(render=should_render)
        scene.update(sim_dt)
        # feet_pos = (
        #     robot.data.body_pos_w[:, calf_indices]
        #     + quat_rotate(robot.data.body_quat_w[:, calf_indices], feet_offset)
        # )
        feet_pos = robot.data.body_pos_w[:, foot_indices]
        contact_forces.update(sim_dt, force_recompute=True)

        if should_render:
            debug_draw.clear()
            # debug_draw.vector(
            #     robot.data.root_pos_w.cpu() + torch.tensor([0.0, 0.0, 0.3]),
            #     robot.data.root_lin_vel_w,
            # )
            debug_draw.vector(
                feet_pos.reshape(-1, 3),
                contact_forces.data.net_forces_w[:, foot_indices].reshape(-1, 3) * sim_dt,
            )
            # rgb_data = rgb_annotator.get_data()
            # rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(rgb_data.shape)
            # frames.append(rgb_data[:, :, :3])
        
        if (t + 1) % 500 == 0:
           reset()

    from torchvision.io import write_video
    if len(frames) > 0:
        frames = np.stack(frames)
        write_video("test.mp4", frames, 1 / (sim_dt * 2))

    simulation_app.close()


if __name__ == "__main__":
    main()