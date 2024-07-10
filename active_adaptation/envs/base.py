import torch
import numpy as np
import hydra

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    CompositeSpec, 
    BinaryDiscreteTensorSpec, 
    UnboundedContinuousTensorSpec
)
import builtins

from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.timer import Timer
from collections import OrderedDict

from abc import abstractmethod
import time

import active_adaptation.envs.mdp as mdp

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
            print(e)
            print("Set enable_cameras=true to use cameras.")

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

        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start"):
                self.sim.reset()
        for _ in range(4):
            self.sim.step(render=True)
        
        self.max_episode_length = self.cfg.max_episode_length
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.step_dt = self.physics_dt * self.cfg.decimation

        # parse obs and reward functions
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

        import inspect
        members = dict(inspect.getmembers(self.__class__, inspect.isclass))

        RAND_FUNCS = mdp.RAND_FUNCS
        OBS_FUNCS = mdp.OBS_FUNCS
        OBS_FUNCS.update(mdp.get_obj_by_class(members, mdp.Observation))
        REW_FUNCS = mdp.REW_FUNCS
        REW_FUNCS.update(mdp.get_obj_by_class(members, mdp.Reward))
        TERM_FUNCS = mdp.TERM_FUNCS
        TERM_FUNCS.update(mdp.get_obj_by_class(members, mdp.Termination))

        self.randomizations = OrderedDict()
        self.observation_funcs = OrderedDict()
        self.reward_funcs = OrderedDict()
        self._startup_callbacks = []
        self._update_callbacks = []
        self._reset_callbacks = []
        self._debug_draw_callbacks = []
        self._step_callbacks = []
        self.command_manager: mdp.Command = hydra.utils.instantiate(self.cfg.command, env=self)
        self.action_manager: mdp.ActionManager = hydra.utils.instantiate(self.cfg.action, env=self)
        self._reset_callbacks.append(self.action_manager.reset)
        
        self.action_spec = CompositeSpec(
            {
                "action": UnboundedContinuousTensorSpec((self.num_envs, self.action_dim))
            },
            shape=[self.num_envs]
        ).to(self.device)

        self._debug_draw_callbacks.append(self.command_manager.debug_draw)

        for key, params in self.cfg.randomization.items():
            rand = RAND_FUNCS[key](self, **params if params is not None else {})
            self.randomizations[key] = rand
            self._startup_callbacks.append(rand.startup)
            self._reset_callbacks.append(rand.reset)
            self._debug_draw_callbacks.append(rand.debug_draw)
            self._step_callbacks.append(rand.step)

        for group, funcs in self.cfg.observation.items():
            self.observation_funcs[group] = OrderedDict()
            for key, params in funcs.items():
                obs = OBS_FUNCS[key](self, **(params if params is not None else {}))
                self.observation_funcs[group][key] = obs
                self._startup_callbacks.append(obs.startup)
                self._update_callbacks.append(obs.update)
                self._reset_callbacks.append(obs.reset)
                self._debug_draw_callbacks.append(obs.debug_draw)
        
        for callback in self._startup_callbacks:
            callback()
        
        # self.sim.physics_sim_view.flush()
        
        reward_spec = CompositeSpec({
            "reward": UnboundedContinuousTensorSpec(1),
            "stats": {
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "success": UnboundedContinuousTensorSpec(1),
                "reward_clip_ratio": UnboundedContinuousTensorSpec(1),
            }
        })
        enabled_rewards = 0
        for key, params in self.cfg.reward.items():
            reward = REW_FUNCS[key](self, **params)
            if reward.enabled:
                enabled_rewards += 1
            self.reward_funcs[key] = reward
            self._update_callbacks.append(reward.update)
            self._reset_callbacks.append(reward.reset)
            self._debug_draw_callbacks.append(reward.debug_draw)
            self._step_callbacks.append(reward.step)
            reward_spec["stats", key] = UnboundedContinuousTensorSpec(1, device=self.device)
        self._reward_buf = torch.zeros(self.num_envs, enabled_rewards, device=self.device)
        self.reward_spec = reward_spec.expand(self.num_envs).to(self.device)
        self.stats = self.reward_spec["stats"].zero()

        observation_spec = {}
        for group, funcs in self.observation_funcs.items():
            print(f"Observation group: {group}")
            tensors = []
            for obs_name, func in funcs.items():
                tensor, mask = func()
                tensors.append(tensor)
                print(f"\t{obs_name}: \t{tensor.shape}, \tmask_ratio {func.mask_ratio:.2f}")
            tensor = torch.cat(tensors, -1)
            observation_spec[group] = UnboundedContinuousTensorSpec(tensor.shape, device=self.device)
            observation_spec[group + "_mask_"] = BinaryDiscreteTensorSpec(
                len(funcs), 
                (self.num_envs, len(funcs)), 
                device=self.device, dtype=bool
            )

        self.observation_spec = CompositeSpec(
            observation_spec, 
            shape=[self.num_envs],
            device=self.device
        )

        self.termination_funcs = OrderedDict(
            {
                key: TERM_FUNCS[key](self, **params) 
                for key, params in self.cfg.termination.items()
            }
        )

        
        self.time_stamp = 0
    
    @property
    def action_dim(self) -> int:
        return self.action_manager.action_dim

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
        self._reward_buf[env_ids] = 0.
        for callback in self._reset_callbacks:
            callback(env_ids)
        # self.sim._physics_sim_view.flush()
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
    
    def apply_action(self, tensordict: TensorDictBase, substep: int):
        self.action_manager(tensordict, substep)

    def _compute_observation(self) -> TensorDictBase:
        observation = TensorDict({}, [self.num_envs])
        try:
            for group, funcs in self.observation_funcs.items():
                tensors = []
                masks = []
                for obs_name, func in funcs.items():
                    tensor, mask = func()
                    tensors.append(tensor)
                    masks.append(mask)
                observation[group] = torch.cat(tensors, dim=-1)
                observation[group + "_mask_"] = torch.stack(masks, dim=-1)
        except Exception as e:
            print(f"Error in computing observation for {group}.{obs_name}: {e}")
            raise e
        return observation
    
    def _compute_reward(self) -> TensorDictBase:
        rewards = []
        for key, reward_func in self.reward_funcs.items():
            reward = reward_func()
            self.stats[key].add_(reward)
            if reward_func.enabled:
                rewards.append(reward)
        self._reward_buf[:] = torch.cat(rewards, 1)
        reward = self._reward_buf.sum(1, True)
        neg_rewar = reward < 0.
        reward = reward.clamp(min=0.)
        self.stats["return"].add_(reward)
        self.stats["reward_clip_ratio"].add_(neg_rewar.float())
        self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
        self.stats["success"][:] = (self.episode_length_buf >= self.max_episode_length * 0.9).unsqueeze(1).float()
        return {"reward": reward, "stats": self.stats.clone()}
    
    def _compute_termination(self) -> TensorDictBase:
        flags = torch.cat([func() for func in self.termination_funcs.values()], dim=-1)
        return flags.any(dim=-1, keepdim=True)

    def _update(self):
        for callback in self._update_callbacks:
            callback()
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()
        self.episode_length_buf.add_(1)
        self.time_stamp += 1

        if self.sim.has_gui() and hasattr(self, "debug_draw"):
            self.debug_draw.clear()
            for callback in self._debug_draw_callbacks:
                callback()
            self.debug_vis()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # start = time.perf_counter()
        for substep in range(self.cfg.decimation):
            self.apply_action(tensordict, substep)
            # self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            for callback in self._step_callbacks:
                callback(substep)
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
        # import omni.replicator.core as rep
        # rep.set_global_seed(seed)
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
    
    def debug_vis(self):
        pass
    
    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.scene
            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            # update closing status
            super().close()


def generate_mask(size: int, split: torch.Tensor, device: str):
    if isinstance(size, int):
        size = (size,)
    repeats = torch.as_tensor(split, device=device)
    masks = torch.zeros(*size, len(split), dtype=torch.bool, device=device)
    masks = masks.scatter(-1, torch.randint(len(split), (*size, 1), device=device), 1)
    masks = torch.repeat_interleave(masks, repeats, -1)
    return masks

