import torch
import numpy as np
import hydra
import cv2

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite, 
    Binary,
    UnboundedContinuous,
)
import builtins

from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.timer import Timer
from collections import OrderedDict

from abc import abstractmethod
from typing import NamedTuple, Dict
import time

import active_adaptation.envs.mdp as mdp


class ObsGroup:
    
    def __init__(
        self,
        name: str,
        funcs: Dict[str, mdp.Observation],
        max_delay: int,
        use_flip: bool = False,
    ):
        self.name = name
        self.funcs = funcs
        self.max_delay = max_delay
        self.use_flip = use_flip
        self.raw_obs_t = OrderedDict()
        self.raw_obs_tm1: OrderedDict = None
        self.buf_obs = OrderedDict()
        self.buf_mask = OrderedDict()
        self.timestamp = -1

    @property
    def keys(self):
        return self.funcs.keys()

    @property
    def spec(self):
        if not hasattr(self, "_spec"):
            foo = self.compute({}, 0)
            spec = {}
            spec[self.name] = UnboundedContinuous(foo[self.name].shape, dtype=foo[self.name].dtype)
            spec[self.name + "_mask_"] = Binary(len(self.keys), foo[self.name + "_mask_"].shape, dtype=bool)
            if self.use_flip:
                spec[self.name + "_flipped"] = spec[self.name]
            self._spec = Composite(spec, shape=[foo[self.name].shape[0]]).to(foo[self.name].device)
        return self._spec

    def compute(self, tensordict: TensorDictBase, timestamp: int) -> torch.Tensor:
        # update only if outdated
        if timestamp > self.timestamp:
            self.raw_obs_tm1 = self.raw_obs_t
            self.raw_obs_t = OrderedDict()
            for obs_key, func in self.funcs.items():
                tensor, mask = func()
                self.raw_obs_tm1[obs_key] = self.raw_obs_t.get(obs_key, tensor)
                self.raw_obs_t[obs_key] = tensor
                self.buf_mask[obs_key] = mask
                if self.max_delay > 0:
                    shape = tensor.shape[0:1] + (1,) * (tensor.ndim - 1)
                    delay = torch.rand(shape, device=tensor.device) * self.max_delay
                    self.buf_obs[obs_key] = func.lerp(self.raw_obs_tm1[obs_key], self.raw_obs_t[obs_key], 1 - delay)
                else:
                    self.buf_obs[obs_key] = self.raw_obs_t[obs_key]
        self.timestamp = timestamp
        
        tensors = torch.cat([self.buf_obs[key] for key in self.funcs.keys()], dim=-1)
        masks = torch.stack([self.buf_mask[key] for key in self.funcs.keys()], dim=-1)
        tensordict[self.name] = tensors
        tensordict[self.name + "_mask_"] = masks

        if self.use_flip:
            tensors = torch.cat([func.fliplr(self.buf_obs[key]) for key, func in self.funcs.items()], dim=-1)
            tensordict[self.name + "_flipped"] = tensors
        
        return tensordict


class Env(EnvBase):
    """
    
    2024.10.10
    - disable delay
    - refactor flipping
    - no longer recompute observation upon reset

    """
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

        # generate scene
        with Timer("[INFO]: Time taken for scene creation"):
            self.scene = InteractiveScene(self.cfg.scene)
            for k, v in self.scene.articulations.items():
                v._env = self
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
        self.fix_root_link = self.scene.articulations["robot"].cfg.spawn.articulation_props.fix_root_link

        # parse obs and reward functions
        self.done_spec = (
            Composite(
                {
                    "done": Binary(1, dtype=bool),
                    "terminated": Binary(1, dtype=bool),
                    "truncated": Binary(1, dtype=bool),
                },
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        self.reward_spec = Composite(
            {
                "stats": {
                    "episode_len": UnboundedContinuous([self.num_envs, 1]),
                    "success": UnboundedContinuous([self.num_envs, 1]),
                },
            },
            shape=[self.num_envs]
        ).to(self.device)

        import inspect
        members = dict(inspect.getmembers(self.__class__, inspect.isclass))

        RAND_FUNCS = mdp.RAND_FUNCS
        RAND_FUNCS.update(mdp.get_obj_by_class(members, mdp.Randomization))
        OBS_FUNCS = mdp.OBS_FUNCS
        OBS_FUNCS.update(mdp.get_obj_by_class(members, mdp.Observation))
        REW_FUNCS = mdp.REW_FUNCS
        REW_FUNCS.update(mdp.get_obj_by_class(members, mdp.Reward))
        TERM_FUNCS = mdp.TERM_FUNCS
        TERM_FUNCS.update(mdp.get_obj_by_class(members, mdp.Termination))

        self.randomizations = OrderedDict()
        self.observation_funcs: Dict[str, ObsGroup] = OrderedDict()
        self.reward_funcs = OrderedDict()
        self._startup_callbacks = []
        self._update_callbacks = []
        self._reset_callbacks = []
        self._debug_draw_callbacks = []
        self._pre_step_callbacks = []
        self._post_step_callbacks = []

        self.command_manager: mdp.Command = hydra.utils.instantiate(self.cfg.command, env=self)
        self._pre_step_callbacks.append(self.command_manager.step)
        # self._update_callbacks.append(self.command_manager.update)
        self._reset_callbacks.append(self.command_manager.reset)
        self._debug_draw_callbacks.append(self.command_manager.debug_draw)
        
        self.action_manager: mdp.ActionManager = hydra.utils.instantiate(self.cfg.action, env=self)
        self._reset_callbacks.append(self.action_manager.reset)
        self._debug_draw_callbacks.append(self.action_manager.debug_draw)
        
        self.action_spec = Composite(
            {
                "action": UnboundedContinuous((self.num_envs, self.action_dim))
            },
            shape=[self.num_envs]
        ).to(self.device)


        for key, params in self.cfg.randomization.items():
            rand = RAND_FUNCS[key](self, **params if params is not None else {})
            self.randomizations[key] = rand
            self._startup_callbacks.append(rand.startup)
            self._reset_callbacks.append(rand.reset)
            self._debug_draw_callbacks.append(rand.debug_draw)
            self._pre_step_callbacks.append(rand.step)
            self._update_callbacks.append(rand.update)

        for group_key, params in self.cfg.observation.items():
            max_delay = params.pop("_max_delay_", 0)
            use_flip = params.pop("_use_flip_", False)
            if max_delay > self.cfg.decimation:
                raise ValueError("Max delay cannot be greater than decimation.")
            max_delay = max_delay / self.cfg.decimation
            funcs = OrderedDict()
            
            for key, kwargs in params.items():
                obs = OBS_FUNCS[key](self, **(kwargs if kwargs is not None else {}))
                funcs[key] = obs

                self._startup_callbacks.append(obs.startup)
                self._update_callbacks.append(obs.update)
                self._reset_callbacks.append(obs.reset)
                self._debug_draw_callbacks.append(obs.debug_draw)
                self._post_step_callbacks.append(obs.post_step)
            
            self.observation_funcs[group_key] = ObsGroup(group_key, funcs, max_delay=max_delay, use_flip=use_flip)
        
        for callback in self._startup_callbacks:
            callback()        
       
        reward_spec = Composite({})

        # parse rewards
        self.clip_rewards = self.cfg.reward.pop("_clip_", True)
        self.mult_dt = self.cfg.reward.pop("_mult_dt_", True)

        self.reward_groups = OrderedDict()
        for group_name, func_specs in self.cfg.reward.items():
            print(f"Reward group: {group_name}")
            funcs = OrderedDict()
            for key, params in func_specs.items():
                reward: mdp.Reward = REW_FUNCS[key](self, **params)
                funcs[key] = reward
                reward_spec["stats", group_name, key] = UnboundedContinuous(1, device=self.device)
                self._update_callbacks.append(reward.update)
                self._reset_callbacks.append(reward.reset)
                self._debug_draw_callbacks.append(reward.debug_draw)
                self._pre_step_callbacks.append(reward.step)
                self._post_step_callbacks.append(reward.post_step)
                print(f"\t{key}: \t{reward.weight:.2f}, \t{reward.enabled}")
            self.reward_groups[group_name] = RewardGroup(self, group_name, funcs)
            reward_spec["stats", group_name, "return"] = UnboundedContinuous(1, device=self.device)
            reward_spec["stats", group_name, "reward_clip_ratio"] = UnboundedContinuous(1, device=self.device)

        reward_spec["reward"] = UnboundedContinuous(len(self.reward_groups), device=self.device)
        # reward_spec["discount"] = UnboundedContinuous(1, device=self.device)
        self.reward_spec.update(reward_spec.expand(self.num_envs).to(self.device))

        observation_spec = {}
        for group_key, group in self.observation_funcs.items():
            observation_spec.update(group.spec)

        self.observation_spec = Composite(
            observation_spec, 
            shape=[self.num_envs],
            device=self.device
        )

        self.termination_funcs = OrderedDict()
        for key, params in self.cfg.termination.items():
            term_func = TERM_FUNCS[key](self, **params)
            self.termination_funcs[key] = term_func
            self._update_callbacks.append(term_func.update)
            self._reset_callbacks.append(term_func.reset)
            self.reward_spec["stats", "termination", key] = UnboundedContinuous((self.num_envs, 1), device=self.device)
        
        self.timestamp = 0

        self.stats = self.reward_spec["stats"].zero()
    
        # Video recording variables
        self.video_frames = []
        self.complete_video_frames = None
        self.record_now = False
        self.render_decimation = 1
        self.last_recording_it = 0
        self.input_tensordict = None

        self.lookat_env_i = 0

        self.extra = {}
        self.observation_prev = TensorDict({}, [self.num_envs])

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
        if len(env_ids):
            self._reset_idx(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.scene.update(self.step_dt)
        for callback in self._reset_callbacks:
            callback(env_ids)

        tensordict = TensorDict({}, self.num_envs, device=self.device)
        # tensordict.update(self.observation_spec.zero())
        self._compute_observation(tensordict)
        
        if self.record_now and env_mask[self.lookat_env_i]:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames.extend(self.video_frames)
            self.video_frames = []
        return tensordict

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        raise NotImplementedError
    
    def apply_action(self, tensordict: TensorDictBase, substep: int):
        self.input_tensordict = tensordict
        self.action_manager(tensordict, substep)

    def _compute_observation(self, tensordict: TensorDictBase):
        observation_this = TensorDict({}, [self.num_envs])
        try:
            for group_key, obs_group in self.observation_funcs.items():
                obs_group.compute(tensordict, self.timestamp)
        except Exception as e:
            print(f"Error in computing observation for {group_key}: {e}")
            raise e
        self.observation_prev = observation_this.clone()
    
    def _compute_reward(self) -> TensorDictBase:
        rewards = []
        for group, reward_group in self.reward_groups.items():
            reward = reward_group.compute()
            if self.mult_dt:
                reward *= self.step_dt
            rewards.append(reward)
            self.stats[group, "return"].add_(reward)

            neg_rewar = reward < 0.
            self.stats[group, "reward_clip_ratio"].add_(neg_rewar.float())

        rewards = torch.cat(rewards, 1)
        if self.clip_rewards:
            rewards = rewards.clamp(min=0.)

        self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
        self.stats["success"][:] = (self.episode_length_buf >= self.max_episode_length * 0.9).unsqueeze(1).float()
        return {"reward": rewards}
    
    def _compute_termination(self) -> TensorDictBase:
        flags = []
        for key, func in self.termination_funcs.items():
            flag = func()
            self.stats["termination", key][:] = flag.float()
            flags.append(flag)
        flags = torch.cat(flags, dim=-1)
        return flags.any(dim=-1, keepdim=True)

    def _update(self):
        for callback in self._update_callbacks:
            callback()
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()
        self.episode_length_buf.add_(1)
        self.timestamp += 1

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # start = time.perf_counter()
        for substep in range(self.cfg.decimation):
            for asset in self.scene.articulations.values():
                if asset.has_external_wrench:
                    asset._external_force_b.zero_()
                    asset._external_torque_b.zero_()
                    asset.has_external_wrench = False
            self.apply_action(tensordict, substep)
            for callback in self._pre_step_callbacks:
                callback(substep)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            for callback in self._post_step_callbacks:
                callback(substep)
        # end = time.perf_counter()
        # print(end - start, self.cfg.decimation)
        self._update()
        
        if self.timestamp % self.render_decimation == 0:
            self._render_headless()
        
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self._compute_reward())
        self.command_manager.update()
        self._compute_observation(tensordict)
        terminated = self._compute_termination()
        truncated = (self.episode_length_buf >= self.max_episode_length).unsqueeze(1)
        tensordict.set("terminated", terminated)
        tensordict.set("truncated", truncated)
        tensordict.set("done", terminated | truncated)
        tensordict["stats"] = self.stats.clone()

        if self.sim.has_gui() and hasattr(self, "debug_draw"):
            self.debug_draw.clear()
            for callback in self._debug_draw_callbacks:
                callback()
            self.debug_vis()
            
        return tensordict
    
    def _set_seed(self, seed: int = -1):
        # import omni.replicator.core as rep
        # rep.set_global_seed(seed)
        torch.manual_seed(seed)

    def start_recording(self, render_decimation: int = 1):
        if not self.record_now:
            self.complete_video_frames = None
            self.render_decimation = render_decimation
            self.record_now = True

    def pause_recording(self):
        self.complete_video_frames = self.video_frames[:]
        self.video_frames = []
        self.record_now = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            frame = self.render(mode="rgb_array")
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(frame, f'Timestamp: {self.timestamp}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            self.video_frames.append(frame)

    def render(self, mode: str = "human"):
        if mode == "rgb_array":
            robot_pos = self.robot.data.root_pos_w[self.lookat_env_i].cpu()
            eye = torch.tensor(self.cfg.viewer.eye) + robot_pos
            lookat = torch.tensor(self.cfg.viewer.lookat) + robot_pos
            self.sim.set_camera_view(eye, lookat)

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

    def state_dict(self):
        sd = super().state_dict()
        sd["observation_spec"] = self.observation_spec
        sd["action_spec"] = self.action_spec
        sd["reward_spec"] = self.reward_spec
        return sd

    def get_extra_state(self) -> dict:
        return dict(self.extra)


def generate_mask(size: int, split: torch.Tensor, device: str):
    if isinstance(size, int):
        size = (size,)
    repeats = torch.as_tensor(split, device=device)
    masks = torch.zeros(*size, len(split), dtype=torch.bool, device=device)
    masks = masks.scatter(-1, torch.randint(len(split), (*size, 1), device=device), 1)
    masks = torch.repeat_interleave(masks, repeats, -1)
    return masks


class RewardGroup:
    def __init__(self, env: Env, name: str, funcs: OrderedDict[str, mdp.Reward]):
        self.env = env
        self.name = name
        self.funcs = funcs
        self.enabled_rewards = sum([func.enabled for func in funcs.values()])
        self.rew_buf = torch.zeros(env.num_envs, self.enabled_rewards, device=env.device)
    
    def compute(self) -> torch.Tensor:
        rewards = []
        for key, func in self.funcs.items():
            reward = func()
            self.env.stats[self.name, key].add_(reward)
            if func.enabled:
                rewards.append(reward)
        self.rew_buf[:] = torch.cat(rewards, 1)
        return self.rew_buf.sum(1, True)

