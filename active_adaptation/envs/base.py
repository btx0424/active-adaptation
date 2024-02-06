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
import time

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
        self.step_dt = self.physics_dt * self.cfg.decimation

        observation_funcs = {}
        reward_funcs = {}
        termination_funcs = {}
        for name in dir(self):
            try:
                item = getattr(self, name)
                if hasattr(item, "_is_obs"):
                    observation_funcs[name] = item
                if hasattr(item, "_is_reward"):
                    reward_funcs[name] = item
                if hasattr(item, "_is_termination"):
                    termination_funcs[name] = item
            except:
                pass
        
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
        self.scene.update(self.step_dt)
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
            group: torch.cat([func() for func in funcs.values()], dim=-1)
            for group, funcs in self.observation_funcs.items()
        }, [self.num_envs])
        observation["stats"] = self.stats.clone()
        return observation
    
    def _compute_reward(self) -> TensorDictBase:
        rewards = []
        for key, (func, weight) in self.reward_funcs.items():
            reward = weight * func()
            self.stats[key].add_(reward)
            rewards.append(reward)
        reward = sum(rewards).clip(0.)
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
        return {"reward": reward}
    
    def _compute_termination(self) -> TensorDictBase:
        flags = torch.cat([func() for func in self.termination_funcs.values()], dim=-1)
        return flags.any(dim=-1, keepdim=True)

    def _update(self):
        # self.scene.update(self.step_dt)
        if self.sim.has_gui():
            self.sim.render()
        self.episode_length_buf.add_(1)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # start = time.perf_counter()
        for substep in range(self.cfg.decimation):
            self.apply_action(tensordict, substep)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        # end = time.perf_counter()
        # print(end - start, self.cfg.decimation)
        self._update()
        
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
    
    def close(self):
        super().close()
        if not self._is_closed:
            # destructor is order-sensitive
            del self.scene
            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            # update closing status
            self._is_closed = True


def observation_func(func):
    func._is_obs = True
    return func

def reward_func(func):
    func._is_reward = True
    return func

def termination_func(func):
    func._is_termination = True
    return func

