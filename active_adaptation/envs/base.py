import torch
import numpy as np
import hydra
import inspect

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite, 
    Binary,
    UnboundedContinuous,
)
from collections import OrderedDict

from abc import abstractmethod
from typing import NamedTuple, Dict
import time

import active_adaptation
import active_adaptation.envs.mdp as mdp

if active_adaptation.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    import active_adaptation.utils.symmetry as symmetry_utils

class ObsGroup:
    
    def __init__(
        self,
        name: str,
        funcs: Dict[str, mdp.Observation],
        max_delay: int = 0,
        use_flip: bool = False,
    ):
        self.name = name
        self.funcs = funcs
        self.max_delay = max_delay
        self.use_flip = use_flip
        self.raw_obs_t = OrderedDict()
        self.raw_obs_tm1: OrderedDict = None
        self.buf_obs = OrderedDict()
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
                if self.max_delay > 0:
                    shape = tensor.shape[0:1] + (1,) * (tensor.ndim - 1)
                    delay = torch.rand(shape, device=tensor.device) * self.max_delay
                    self.buf_obs[obs_key] = func.lerp(self.raw_obs_tm1[obs_key], self.raw_obs_t[obs_key], 1 - delay)
                else:
                    self.buf_obs[obs_key] = self.raw_obs_t[obs_key]
        self.timestamp = timestamp
        
        tensors = torch.cat([self.buf_obs[key] for key in self.funcs.keys()], dim=-1)
        tensordict[self.name] = tensors

        return tensordict
    
    def symmetry_transforms(self):
        transforms = []
        for obs_key, func in self.funcs.items():
            transform = func.symmetry_transforms()
            transforms.append(transform)
        transform = symmetry_utils.SymmetryTransform.cat(transforms)
        return transform


class _Env(EnvBase):
    """
    
    2024.10.10
    - disable delay
    - refactor flipping
    - no longer recompute observation upon reset

    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = active_adaptation.get_backend()

        self.setup_scene()
        
        self.max_episode_length = self.cfg.max_episode_length
        self.step_dt = self.cfg.sim.step_dt
        self.physics_dt = self.sim.get_physics_dt()
        self.decimation = int(self.step_dt / self.physics_dt)
        
        print(f"Step dt: {self.step_dt}, physics dt: {self.physics_dt}, decimation: {self.decimation}")

        super().__init__(
            device=self.sim.device,
            batch_size=[self.num_envs],
            run_type_checks=False,
        )
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=int, device=self.device)

        # parse obs and reward functions
        self.done_spec = Composite(
            done=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            terminated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            truncated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            shape=[self.num_envs],
            device=self.device
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

        members = dict(inspect.getmembers(self.__class__, inspect.isclass))
        self.command_manager: mdp.Command = hydra.utils.instantiate(self.cfg.command, env=self)

        RAND_FUNCS = mdp.RAND_FUNCS
        RAND_FUNCS.update(mdp.get_obj_by_class(members, mdp.Randomization))
        OBS_FUNCS = mdp.OBS_FUNCS
        OBS_FUNCS.update(mdp.get_obj_by_class(members, mdp.Observation))
        REW_FUNCS = mdp.REW_FUNCS
        REW_FUNCS.update(mdp.get_obj_by_class(members, mdp.Reward))
        TERM_FUNCS = mdp.TERM_FUNCS

        for k, v in inspect.getmembers(self.command_manager):
            if getattr(v, "is_reward", False):
                REW_FUNCS[k] = mdp.reward_wrapper(v)
            elif getattr(v, "is_observation", False):
                OBS_FUNCS[k] = mdp.observation_wrapper(v)
            elif getattr(v, "is_termination", False):
                TERM_FUNCS[k] = mdp.termination_wrapper(v)

        self.randomizations = OrderedDict()
        self.observation_funcs: Dict[str, ObsGroup] = OrderedDict()
        self.reward_funcs = OrderedDict()
        self._startup_callbacks = []
        self._update_callbacks = []
        self._reset_callbacks = []
        self._debug_draw_callbacks = []
        self._pre_step_callbacks = []
        self._post_step_callbacks = []

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
            if max_delay > 1e-6:
                raise NotImplementedError
            funcs = OrderedDict()            
            for key, kwargs in params.items():
                obs = OBS_FUNCS[key](self, **(kwargs if kwargs is not None else {}))
                funcs[key] = obs

                self._startup_callbacks.append(obs.startup)
                self._update_callbacks.append(obs.update)
                self._reset_callbacks.append(obs.reset)
                self._debug_draw_callbacks.append(obs.debug_draw)
                self._post_step_callbacks.append(obs.post_step)
            
            self.observation_funcs[group_key] = ObsGroup(group_key, funcs)
        
        for callback in self._startup_callbacks:
            callback()        
       
        reward_spec = Composite({})

        # parse rewards
        self.mult_dt = self.cfg.reward.pop("_mult_dt_", True)

        self._stats_ema = {}
        self._stats_ema_decay = 0.99

        self.reward_groups = OrderedDict()
        for group_name, func_specs in self.cfg.reward.items():
            print(f"Reward group: {group_name}")
            funcs = OrderedDict()
            self._stats_ema[group_name] = {}

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
                self._stats_ema[group_name][key] = (torch.tensor(0., device=self.device), torch.tensor(0., device=self.device))

            self.reward_groups[group_name] = RewardGroup(self, group_name, funcs)
            reward_spec["stats", group_name, "return"] = UnboundedContinuous(1, device=self.device)

        reward_spec["reward"] = UnboundedContinuous(max(1, len(self.reward_groups)), device=self.device)
        reward_spec["discount"] = UnboundedContinuous(1, device=self.device)
        self.reward_spec.update(reward_spec.expand(self.num_envs).to(self.device))
        self.discount = torch.ones((self.num_envs, 1), device=self.device)

        observation_spec = {}
        for group_key, group in self.observation_funcs.items():
            try:
                observation_spec.update(group.spec)
            except Exception as e:
                print(f"Error in computing observation spec for {group_key}: {e}")
                raise e

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
    
        self.input_tensordict = None
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
    def stats_ema(self):
        result = {}
        for group_key, group in self._stats_ema.items():
            result[group_key] = {}
            for rew_key, (sum, cnt) in group.items():
                result[group_key][rew_key] = (sum / cnt).item()
        return result
    
    def setup_scene(self):
        raise NotImplementedError
    
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
        return tensordict

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        raise NotImplementedError
    
    def apply_action(self, tensordict: TensorDictBase, substep: int):
        self.input_tensordict = tensordict
        self.action_manager(tensordict, substep)

    def _compute_observation(self, tensordict: TensorDictBase):
        observation_this = TensorDict({}, [self.num_envs])
    
        for group_key, obs_group in self.observation_funcs.items():
            obs_group.compute(tensordict, self.timestamp)
        
        self.observation_prev = observation_this.clone()
    
    def _compute_reward(self) -> TensorDictBase:
        if not self.reward_groups:
            return {"reward": torch.ones((self.num_envs, 1), device=self.device)}
        
        rewards = []
        for group, reward_group in self.reward_groups.items():
            reward = reward_group.compute()
            if self.mult_dt:
                reward *= self.step_dt
            rewards.append(reward)
            self.stats[group, "return"].add_(reward)

        rewards = torch.cat(rewards, 1)

        self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
        self.stats["success"][:] = (self.episode_length_buf >= self.max_episode_length * 0.9).unsqueeze(1).float()
        return {"reward": rewards}
    
    def _compute_termination(self) -> TensorDictBase:
        if not self.termination_funcs:
            return torch.zeros((self.num_envs, 1), dtype=bool, device=self.device)
        
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
        if self.sim.has_gui():
            self.sim.render()
        self.episode_length_buf.add_(1)
        self.timestamp += 1

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # start = time.perf_counter()
        for substep in range(self.decimation):
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
        self.discount.fill_(1.0)
        self._update()
        
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self._compute_reward())
        self.command_manager.update()
        self._compute_observation(tensordict)
        terminated = self._compute_termination()
        truncated = (self.episode_length_buf >= self.max_episode_length).unsqueeze(1)
        tensordict.set("terminated", terminated)
        tensordict.set("truncated", truncated)
        tensordict.set("done", terminated | truncated)
        tensordict.set("discount", self.discount.clone())
        tensordict["stats"] = self.stats.clone()

        if self.sim.has_gui():
            if hasattr(self, "debug_draw"): # isaac only
                self.debug_draw.clear()
            for callback in self._debug_draw_callbacks:
                callback()
            self.debug_vis()
            
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
    

    def state_dict(self):
        sd = super().state_dict()
        sd["observation_spec"] = self.observation_spec
        sd["action_spec"] = self.action_spec
        sd["reward_spec"] = self.reward_spec
        return sd

    def get_extra_state(self) -> dict:
        return dict(self.extra)

    def close(self):
        if not self.is_closed:
            if self.backend == "isaac":
                # destructor is order-sensitive
                del self.scene
                # clear callbacks and instance
                self.sim.clear_all_callbacks()
                self.sim.clear_instance()
                # update closing status
            super().close()

    def dump(self):
        if self.backend == "mujoco":
            self.scene.close()


class RewardGroup:
    def __init__(self, env: _Env, name: str, funcs: OrderedDict[str, mdp.Reward]):
        self.env = env
        self.name = name
        self.funcs = funcs
        self.enabled_rewards = sum([func.enabled for func in funcs.values()])
        self.rew_buf = torch.zeros(env.num_envs, self.enabled_rewards, device=env.device)
    
    def compute(self) -> torch.Tensor:
        rewards = []
        # try:
        for key, func in self.funcs.items():
            reward, count = func()
            self.env.stats[self.name, key].add_(reward)
            sum, cnt = self.env._stats_ema[self.name][key]
            sum.mul_(self.env._stats_ema_decay).add_(reward.sum())
            cnt.mul_(self.env._stats_ema_decay).add_(count)
            if func.enabled:
                rewards.append(reward)
        # except Exception as e:
        #     raise RuntimeError(f"Error in computing reward for {key}: {e}")
        if len(rewards):
            self.rew_buf[:] = torch.cat(rewards, 1)
        return self.rew_buf.sum(1, True)

