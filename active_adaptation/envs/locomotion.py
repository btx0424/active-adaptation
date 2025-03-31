import os
import json
import torch
from isaaclab.utils import configclass

import active_adaptation
import active_adaptation.envs.mdp as mdp
from active_adaptation.envs.base import _Env

class SimpleEnv(_Env):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.robot = self.scene.articulations["robot"]
        self.init_root_state = self.robot.data.default_root_state.clone()
        self.init_joint_pos = self.robot.data.default_joint_pos.clone()
        self.init_joint_vel = self.robot.data.default_joint_vel.clone()
        self.default_joint_pos = self.init_joint_pos.clone()
        
        if self.backend == "isaac":
            self.lookat_env_i = (
                self.scene._default_env_origins.cpu() 
                - torch.tensor(self.cfg.viewer.lookat)
            ).norm(dim=-1).argmin()

        self.action_buf: torch.Tensor = self.action_manager.action_buf
        self.last_action: torch.Tensor = self.action_manager.applied_action

    def setup_scene(self):
        import active_adaptation.envs.scene as scene

        if active_adaptation.get_backend() == "isaac":
            import isaaclab.sim as sim_utils
            from isaaclab.scene import InteractiveSceneCfg
            from isaaclab.assets import AssetBaseCfg
            from isaaclab.sensors import ContactSensorCfg
            from active_adaptation.assets import ROBOTS, get_asset_meta
            from active_adaptation.envs.terrain import TERRAINS
            
            scene_cfg = InteractiveSceneCfg(num_envs=self.cfg.num_envs, env_spacing=2.5)
            scene_cfg.light_0 = AssetBaseCfg(
                prim_path="/World/light_0",
                spawn=sim_utils.DistantLightCfg(
                    color=(0.4, 0.7, 0.9),
                    intensity=3000.0,
                    angle=10,
                    exposure=0.2,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    rot=(0.9330127, 0.25, 0.25, -0.0669873)
                ),
            )
            scene_cfg.light_1 = AssetBaseCfg(
                prim_path="/World/light_1",
                spawn=sim_utils.DistantLightCfg(
                    color=(0.8, 0.5, 0.5),
                    intensity=3000.0,
                    angle=20,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    rot=(0.78201786, 0.3512424, 0.50162613, -0.11596581)
                ),
            )
            scene_cfg.robot = ROBOTS[self.cfg.robot.name]
            scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
            scene_cfg.terrain = TERRAINS[self.cfg.terrain]
            scene_cfg.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*", 
                history_length=3,
                track_air_time=True
            )
            sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.sim.isaac_physics_dt)
            
            # slightly reduces GPU memory usage
            # sim_cfg.physx.gpu_max_rigid_contact_count = 2**21
            # sim_cfg.physx.gpu_max_rigid_patch_count = 2**21
            sim_cfg.physx.gpu_found_lost_pairs_capacity = 2538320 # 2**20
            sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 61999079 # 2**26
            sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**23
            # sim_cfg.physx.gpu_collision_stack_size = 2**25
            # sim_cfg.physx.gpu_heap_capacity = 2**24
            
            self.sim, self.scene = scene.create_isaaclab_sim_and_scene(sim_cfg, scene_cfg)

            # set camera view for "/OmniverseKit_Persp" camera
            self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
            try:
                import omni.replicator.core as rep
                # create render product
                self._render_product = rep.create.render_product(
                    "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            except ModuleNotFoundError as e:
                print("Set enable_cameras=true to use cameras.")
            
            try:
                from active_adaptation.utils.debug import DebugDraw
                self.debug_draw = DebugDraw()
                print("[INFO] Debug Draw API enabled.")
            except ModuleNotFoundError:
                print()
            
            asset_meta = get_asset_meta(self.scene["robot"])
            path = os.path.join(os.getcwd(), "asset_meta.json")
            print(f"Saving asset meta to {path}")
            with open(path, "w") as f:
                json.dump(asset_meta, f, indent=4)
        else:
            from active_adaptation.envs.mujoco import MJScene, MJSim
            from active_adaptation.assets_mjcf import ROBOTS

            @configclass
            class SceneCfg:
                robot = ROBOTS[self.cfg.robot.name]
            
            self.scene = MJScene(SceneCfg())
            self.sim = MJSim(self.scene)

        
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.
        self.scene.reset(env_ids)

    def render(self, mode: str = "human"):
        robot_pos = self.robot.data.root_pos_w[self.lookat_env_i].cpu()
        if mode == "rgb_array":
            eye = torch.tensor(self.cfg.viewer.eye) + robot_pos
            lookat = torch.tensor(self.cfg.viewer.lookat) + robot_pos
            self.sim.set_camera_view(eye, lookat)
        return super().render(mode)


    class feet_pos_b(mdp.body_pos):
        def __init__(self, env, feet_names, yaw_only: bool=False):
            super().__init__(env, feet_names, yaw_only=yaw_only)
            self.asset.data.feet_pos_b = self.body_pos_b
        
        def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
            obs = obs.reshape(self.num_envs, 4, 3)[:, [1, 0, 3, 2]] * torch.tensor([1., -1., 1.])
            return obs.reshape(self.num_envs, -1)
    
    class feet_vel_b(mdp.body_vel):
        def __init__(self, env, feet_names, yaw_only: bool=False):
            super().__init__(env, feet_names, yaw_only=yaw_only)
            self.asset.data.feet_vel_b = self.body_vel_b
        
        def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
            obs = obs.reshape(self.num_envs, 4, 3)[:, [1, 0, 3, 2]] * torch.tensor([1., -1., 1.])
            return obs.reshape(self.num_envs, -1)

