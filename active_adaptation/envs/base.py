import torch
import numpy as np

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    CompositeSpec, 
    BinaryDiscreteTensorSpec, 
    UnboundedContinuousTensorSpec
)

from omni.isaac.orbit.scene import InteractiveScene
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils.timer import Timer
from collections import OrderedDict

from abc import abstractmethod

class Env(EnvBase):

    def __init__(self, cfg):
        super().__init__(
            device=cfg.sim.device,
            batch_size=[cfg.scene.num_envs],
            run_type_checks=False,
        )
        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False
        self.enable_render = False

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim = SimulationContext(self.cfg.sim)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")
        # set camera view for "/OmniverseKit_Persp" camera
        self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
        import omni.replicator.core as rep
        # create render product
        self._render_product = rep.create.render_product(
            "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
        )
        # create rgb annotator -- used to read data from the render product
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._rgb_annotator.attach([self._render_product])

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        # print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.substeps}")
        # print(f"\tEnvironment step-size : {self.step_dt}")
        print(f"\tPhysics GPU pipeline  : {self.cfg.sim.use_gpu_pipeline}")
        print(f"\tPhysics GPU simulation: {self.cfg.sim.physx.use_gpu}")

        # generate scene
        with Timer("[INFO]: Time taken for scene creation"):
            self.scene = InteractiveScene(self.cfg.scene)
        print("[INFO]: Scene manager: ", self.scene)

        with Timer("[INFO]: Time taken for simulation reset"):
            self.sim.reset()
        for _ in range(4):
            self.sim.step(render=True)
        
        self.max_episode_length = self.cfg.max_episode_length
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=int, device=self.device)

        observation_funcs = {}
        reward_funcs = {}
        termination_funcs = {}
        for name, func in self.__class__.__dict__.items():
            if hasattr(func, "_is_obs"):
                observation_funcs[name] = func
            if hasattr(func, "_is_reward"):
                reward_funcs[name] = func
            if hasattr(func, "_is_termination"):
                termination_funcs[name] = func
        
        self.observation_funcs = {}
        self.reward_funcs = {}
        for group, funcs in self.cfg.observation.items():
            self.observation_funcs[group] = OrderedDict(
                {
                    key: observation_funcs[key] 
                    for key in funcs
                }
            )
        self.observation_spec = CompositeSpec(
            {}, 
            shape=[self.num_envs],
            device=self.device
        )
        stats_spec = CompositeSpec(
            {
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
            }
        )
        for key, weight in self.cfg.reward.items():
            self.reward_funcs[key] = (reward_funcs[key], weight)
            stats_spec[key] = UnboundedContinuousTensorSpec(1)

        self.observation_spec["stats"] = stats_spec.expand(self.num_envs).to(self.device)

        self.termination_funcs = OrderedDict(
            {
                key: termination_funcs[key] 
                for key in self.cfg.termination
            }
        )

        self.done_spec = (
            CompositeSpec(
                {
                    "done": BinaryDiscreteTensorSpec(1, dtype=bool),
                    "terminated": BinaryDiscreteTensorSpec(1, dtype=bool),
                    "truncated": BinaryDiscreteTensorSpec(1, dtype=bool),
                }, 
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.stats = self.observation_spec["stats"].zero()

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs
    
    @property
    def physics_dt(self) -> float:
        return self.sim.get_physics_dt()
    
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
        else:
            env_mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        env_ids = env_mask.nonzero().squeeze(-1)
        self._reset_idx(env_ids)
        self.sim._physics_sim_view.flush()
        self.episode_length_buf[env_ids] = 0
        tensordict = TensorDict(
            self._compute_observation(), 
            self.num_envs,
            device=self.device
        )
        return tensordict

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def apply_action(self, tensordict: TensorDictBase, substep: int):
        raise NotImplementedError

    def _compute_observation(self) -> TensorDictBase:
        observation = TensorDict({
            group: torch.cat([func(self) for func in funcs.values()], dim=-1)
            for group, funcs in self.observation_funcs.items()
        }, self.num_envs)
        observation["stats"] = self.stats.clone()
        return observation
    
    def _compute_reward(self) -> TensorDictBase:
        rewards = []
        for key, (func, weight) in self.reward_funcs.items():
            reward = weight * func(self)
            self.stats[key].add_(reward)
            rewards.append(reward)
        reward = sum(rewards)
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
        return {("agents", "reward"): reward}
    
    def _compute_termination(self) -> TensorDictBase:
        flags = torch.cat([func(self) for func in self.termination_funcs.values()], dim=-1)
        return flags.any(dim=-1, keepdim=True)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for substep in range(self.cfg.decimation):
            self.apply_action(tensordict, substep)
            self.scene.write_data_to_sim()
            self.sim.step(self.enable_render and substep == self.cfg.decimation - 1)
        self.scene.update(self.physics_dt * self.cfg.decimation)
        self.episode_length_buf.add_(1)
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self._compute_observation())
        tensordict.update(self._compute_reward())
        terminated = self._compute_termination()
        truncated = (self.episode_length_buf >= self.max_episode_length).unsqueeze(1)
        tensordict.set("terminated", terminated)
        tensordict.set("truncated", truncated)
        tensordict.set("done", terminated | truncated)
        return tensordict
    
    def _set_seed(self, seed: int = -1):
        import omni.replicator.core as rep
        rep.set_global_seed(seed)
        torch.manual_seed(seed)

    def render(self, mode: str = "human"):
        self.sim.render()
        if mode == "human":
            return None
        elif mode == "rgb_array":
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            return rgb_data[:, :, :3]
        else:
            raise NotImplementedError


def observation_func(func):
    func._is_obs = True
    return func

def reward_func(func):
    func._is_reward = True
    return func

def termination_func(func):
    func._is_termination = True
    return func

from omni_drones.envs.isaac_env import DebugDraw

class LocomotionEnv(Env):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.robot = self.scene.articulations["robot"]
        self.init_root_state = self.robot.data.default_root_state_w.clone()
        self.init_root_state[..., :3] += self.scene.env_origins
        self.init_joint_pos = self.robot.data.default_joint_pos.clone()
        self.init_joint_vel = self.robot.data.default_joint_vel.clone()
        self.debug_draw = DebugDraw()
        self.lookat_env_i = (
            self.scene.env_origins.cpu() - torch.tensor(self.cfg.viewer.lookat)
        ).norm(dim=-1).argmin()

        self.target_base_height = 0.3 # self.cfg.target_base_height

        with torch.device(self.device):
            self._command = torch.zeros(self.num_envs, 3)
            self.target_pos = torch.zeros(self.num_envs, 4, 3)
            self.offset = torch.tensor([
                [-1., -1.],
                [-1., 0.],
                [0., -1.],
                [0., 0.]
            ])
        
        obs = self._compute_observation()
        reward = self._compute_reward()

        self.action_spec = CompositeSpec(
            {
                ("agents", "action"): UnboundedContinuousTensorSpec((self.num_envs, 12))
            }, 
            shape=[self.num_envs]
        ).to(self.device)

        for key, value in obs.items(True, True):
            if key not in self.observation_spec.keys(True, True):
                self.observation_spec[key] = UnboundedContinuousTensorSpec(value.shape, device=self.device)
        
        self.reward_spec = CompositeSpec(
            {
                key: UnboundedContinuousTensorSpec(value.shape)
                for key, value in reward.items()
            },
            shape=[self.num_envs]
        ).to(self.device)
    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.robot.reset(env_ids)
        self.robot.write_root_state_to_sim(self.init_root_state[env_ids], env_ids)
        self.robot.write_joint_state_to_sim(
            self.init_joint_pos[env_ids],
            self.init_joint_vel[env_ids],
            env_ids=env_ids
        )
        self.stats[env_ids] = 0.
        self.target_pos[env_ids, :, :2] = torch.gather(
            torch.rand(len(env_ids), 4, 2, device=self.device) + self.offset,
            dim=1,
            index=torch.rand(len(env_ids), 4, 1, device=self.device).argsort(dim=1).expand(-1, -1, 2)
        )
        self.scene.reset(env_ids)
        self.scene.update(dt=self.physics_dt)

    def apply_action(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:    
            actions = tensordict[("agents", "action")]
            actions = actions + self.init_joint_pos
            self.robot.set_joint_position_target(actions)
        self.robot.write_data_to_sim()
    
    def _compute_observation(self) -> TensorDictBase:
        index = (self.episode_length_buf) // 300
        current_target_pos = self.target_pos.take_along_dim(
            index.reshape(-1, 1, 1),
            dim=1
        ).squeeze(1)
        self._command[:, :2] = noarmalize(
            current_target_pos[:, :2] 
            - self.robot.data.root_pos_w[:, :2]
            + self.scene.env_origins[:, :2]
        )
        return super()._compute_observation()
    
    def render(self, mode: str = "human"):
        robot_pos = (
            self.robot.data.root_pos_w[self.lookat_env_i].cpu()
            + torch.tensor([0., 0., 0.2])
        )
        self.debug_draw.clear()
        self.debug_draw.vector(
            robot_pos, 
            self._command[self.lookat_env_i],
            color=(1., 1., 1., 0)
        )
        self.debug_draw.vector(
            robot_pos, 
            self.robot.data.root_lin_vel_w[self.lookat_env_i],
            color=(1., .5, .5, 0)
        )
        return super().render(mode)
    
    @observation_func
    def command(self):
        return self._command
    
    @observation_func
    def root_quat_w(self):
        return self.robot.data.root_quat_w
    
    @observation_func
    def root_angvel_b(self):
        return self.robot.data.root_ang_vel_b
    
    @observation_func
    def joint_pos(self):
        return self.robot.data.joint_pos
    
    @observation_func
    def joint_vel(self):
        return self.robot.data.joint_vel
    
    @observation_func
    def projected_gravity_b(self):
        return self.robot.data.projected_gravity_b
    
    @observation_func
    def root_linvel_b(self):
        return self.robot.data.root_lin_vel_b
    
    # @observation_func
    # def prev_actions(self):
    #     return self.actions
    
    @observation_func
    def applied_torques(self):
        return self.robot.data.applied_torque
    
    @reward_func
    def linvel(self):
        linvel_w = self.robot.data.root_lin_vel_w
        linvel_error = square_norm(linvel_w - self._command)
        return 1. / (1. + linvel_error / 0.25)
    
    @reward_func
    def heading(self):
        return noarmalize(self.robot.data.root_lin_vel_b)[:, [0]]
    
    @reward_func
    def base_height(self):
        return self.robot.data.root_pos_w[:, [2]] - self.target_base_height

    @reward_func
    def energy(self):
        energy = (
            (self.robot.data.joint_vel * self.robot.data.applied_torque)
            .abs()
            .sum(dim=-1, keepdim=True)
        )
        return - energy
    
    @reward_func
    def joint_acc_l2(self):
        return -square_norm(self.robot.data.joint_acc)
    
    @reward_func
    def joint_torques_l2(self):
        return -square_norm(self.robot.data.applied_torque)

    @termination_func
    def crash(self):
        terminated = (
            (self.robot.data.root_pos_w[:, 2] <= self.target_base_height * 0.5)
            | (self.robot.data.projected_gravity_b[:, 2] >= -0.3)
        ).unsqueeze(1)
        return terminated


def square_norm(x: torch.Tensor):
    return x.square().sum(dim=-1, keepdim=True)

def noarmalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(dim=-1, keepdim=True)