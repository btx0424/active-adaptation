from typing import Sequence
import torch

from isaaclab.app import AppLauncher

def main():

    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation, RigidObjectCfg, RigidObject
    from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg, DCMotorCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
    from isaaclab.utils.math import (
        quat_rotate_inverse, 
        quat_rotate, 
        quat_conjugate, 
        quat_mul,
        random_yaw_orientation,
        quat_from_angle_axis,
        quat_from_euler_xyz,
    )

    from active_adaptation.assets.scene import DoorArticulation, DOOR_CFG

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
        
        door = DOOR_CFG

        gripper = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Gripper",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/btx0424/isaac_lab/active-adaptation/active_adaptation/assets/gripper.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                )
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
            ),
            actuators={
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=".*",
                    stiffness=5.0,
                    damping=0.2
                )
            },
        )
    
    sim = SimulationContext(SimulationCfg())
    scene = InteractiveScene(SceneCfg(num_envs=4, env_spacing=4))
    sim.reset()
    for _ in range(4):
        sim.step(render=True)

    door: DoorArticulation = scene["door"]
    handle_id = door.find_bodies("Handle")[0][0]

    gripper: Articulation = scene["gripper"]

    forces = torch.zeros_like(gripper._external_force_b)
    torques = torch.zeros_like(gripper._external_torque_b)

    init_root_state = gripper.data.default_root_state.clone()

    target_pos = torch.tensor([0.0, 0.0, 1.0], device=sim.device) + scene.env_origins
    target_quat = init_root_state[:, 3:7].clone()

    gripper_rest = gripper.data.default_joint_pos.clone()
    gripper_close = gripper_rest.clone().fill_(1.)

    def reset(env_ids: torch.Tensor):
        state = init_root_state[env_ids]
        state[:, :3] += scene.env_origins[env_ids]
        # state[:, :3] += torch.randn_like(state[:, :3]) * 0.1
        # state[:, 3:7] = random_yaw_orientation(len(env_ids), "cuda")
        gripper.write_root_state_to_sim(state, env_ids)

        # state = door.data.default_root_state[env_ids]
        # state[:, :3] += scene.env_origins[env_ids]
        # door.write_root_state_to_sim(state, env_ids)
        door.write_joint_state_to_sim(door.data.default_joint_pos, door.data.default_joint_vel)

    reset(torch.arange(4, device="cuda"))
    scene.update(sim.cfg.dt)
    
    i = 0
    
    def target_step_0():
        target_pos = door.data.body_pos_w[:, handle_id]
        handle_quat = door.data.body_quat_w[:, handle_id]
        axis = torch.tensor([-1., 0., 0.], device="cuda").expand_as(target_pos)
        axis = quat_rotate(handle_quat, axis)

        offset = torch.tensor([-0.15, 0.03, 0.], device="cuda").expand_as(target_pos)
        offset = quat_rotate(handle_quat, offset)
        target_pos = target_pos + offset

        angle = torch.tensor([torch.pi/2], device="cuda").expand(axis.shape[0])
        target_quat = quat_from_angle_axis(angle, axis)
        return target_pos, target_quat
    
    def target_step_1():
        target_pos = door.data.body_pos_w[:, handle_id]
        handle_quat = door.data.body_quat_w[:, handle_id]
        axis = torch.tensor([-1., 0., 0.], device="cuda").expand_as(target_pos)
        axis = quat_rotate(handle_quat, axis)

        offset = torch.tensor([-0.5, 0.03, 0.], device="cuda").expand_as(target_pos)
        offset = quat_rotate(handle_quat, offset)
        target_pos = target_pos + offset

        angle = torch.tensor([torch.pi], device="cuda").expand(axis.shape[0])
        target_quat = quat_from_angle_axis(angle, axis)
        target_quat = quat_mul(target_quat, handle_quat)
        return target_pos, target_quat
    
    state = torch.zeros(scene.num_envs, dtype=int, device="cuda")

    while True:
        
        target_pos_0, target_quat_0 = target_step_0()
        target_pos_1, target_quat_1 = target_step_1()

        target_pos = torch.where((state == 0).unsqueeze(1), target_pos_0, target_pos_1)
        target_quat = torch.where((state == 0).unsqueeze(1), target_quat_0, target_quat_1)

        pos_error = target_pos - gripper.data.root_pos_w
        kp = torch.where((state==1), torch.ones(4, device="cuda"), torch.ones(4, device="cuda") * 2).unsqueeze(1)
        force = clamp_norm(kp * pos_error - 1 * gripper.data.root_lin_vel_w, 60)

        forces[:, 0] = quat_rotate_inverse(gripper.data.root_quat_w, force)
        ori_error = quat_mul(target_quat, quat_conjugate(gripper.data.root_quat_w))
        ori_error = 2 * ori_error[:, 1:]
        torque = 0.2 * ori_error - 0.04 * gripper.data.root_ang_vel_w
        torques[:, 0] = quat_rotate_inverse(gripper.data.root_quat_w, torque)

        gripper.set_external_force_and_torque(forces, torques)
        pos_target = torch.where((state == 1).unsqueeze(1), gripper_close, gripper_rest)
        gripper.set_joint_position_target(pos_target)
        
        door.set_joint_position_target(door.data.default_joint_pos)

        scene.write_data_to_sim()

        sim.step(render=True)
        scene.update(sim.cfg.dt)

        success = (pos_error.norm(dim=-1) < 0.05) & (ori_error.norm(dim=-1) < 0.05)
        state[success] = 1

        if i % 20 == 0:
            # print(success)
            # print(door.data.applied_torque)
            print(box.data.body_pos_w)

        if i % 1000 == 0:
            reset(torch.arange(4, device="cuda"))
            state[:] = 0

        i += 1

    # Add your code here

def clamp_norm(x: torch.Tensor, max: float):
    norm = x.norm(dim=-1, keepdim=True)
    return (x / norm) * norm.clamp_max(max)

if __name__ == "__main__":
    main()