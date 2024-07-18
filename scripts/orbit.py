from typing import Dict, Optional
import gymnasium as gym
from tensordict import TensorDictBase

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs import GymWrapper
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec, 
    UnboundedDiscreteTensorSpec,
    DiscreteTensorSpec
)

from copy import deepcopy

import torch
from omni.isaac.lab.managers import RewardManager
from omni.isaac.lab.envs import RLTaskEnv


def reset(self: RewardManager, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Returns the episodic sum of individual reward terms.

    Args:
        env_ids: The environment ids for which the episodic sum of
            individual reward terms is to be returned. Defaults to all the environment ids.

    Returns:
        Dictionary of episodic sum of individual reward terms.
    """
    # resolve environment ids
    if env_ids is None:
        env_ids = slice(None)
    # store information
    extras = {}
    for key in self._episode_sums.keys():
        extras["episode_reward/" + key] = self._episode_sums[key].clone() 
        # reset episodic sum
        self._episode_sums[key][env_ids] = 0.0
    # reset all the reward terms
    for term_cfg in self._class_term_cfgs:
        term_cfg.func.reset(env_ids=env_ids)
    # return logged information
    return extras


RewardManager.reset = reset


def get_nested(info: dict, key):
    if isinstance(key, str):
        return info[key]
    elif len(key) == 1:
        return info[key[0]]
    else:
        return get_nested(info[key[0]], key[1:])


class OrbitWrapper(_EnvWrapper):
    
    _env: RLTaskEnv

    def __init__(self, env: RLTaskEnv):
        self._categorical_action_encoding=False

        super().__init__(
            device=env.unwrapped.device,
            batch_size=[env.num_envs],
            allow_done_after_reset=False,
            env=env
        )
    
    def _build_env(self, **kwargs) -> RLTaskEnv:
        env = kwargs["env"]
        env.reset()
        return env
    
    def _init_env(self) -> int | None:
        self.reset()
    
    def _check_kwargs(self, kwargs: Dict):
        pass
    
    def _set_seed(self, seed: int | None):
        self._env.seed(seed)

    @property
    def num_envs(self):
        return self._env.unwrapped.num_envs
    
    @property
    def max_peisode_length(self):
        return self._env.unwrapped.max_episode_length
    
    @property
    def extras(self):
        return deepcopy(self._env.unwrapped.extras)

    @property
    def _is_batched(self):
        return True
    
    @property
    def info_spec(self):
        return self._info_spec

    read_obs = GymWrapper.read_obs
    
    def _make_done_spec(self):  # noqa: F811
        return CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "terminated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "truncated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
            },
            shape=self.batch_size,
        )

    def _make_specs(self, env: RLTaskEnv) -> None:
        GymWrapper._make_specs(self, env)

        info_specs = {}
        for key, val in TensorDict(self.extras, []).items(True, True):
            if val.ndim > 0 and val.shape[0] == self.num_envs:
                if val.dtype.is_floating_point:
                    spec = UnboundedContinuousTensorSpec(val.shape, val.device)
                else:
                    spec = UnboundedDiscreteTensorSpec(val.shape, val.device)
                self.observation_spec[key] = spec
                info_specs[key] = spec
                print(f"[INFO] OrbitWrapper: add {key} to info_spec.")
        self._info_spec = CompositeSpec(info_specs, shape=[self.num_envs], device=self.device)

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is None:
            obs, info = self._env.reset(**kwargs)

            source = self.read_obs(obs)

            tensordict_out = TensorDict(
                source=source,
                batch_size=self.batch_size,
            )

            for key in self.info_spec.keys(True, True):
                tensordict_out[key] = get_nested(info, key)
            
            tensordict_out = tensordict_out.to(self.device, non_blocking=True)
            return tensordict_out
        else:
            _reset = tensordict.get("_reset", None)
            # TODO: manual reset
            return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key).to(self.device)

        reward = 0
        for _ in range(self.wrapper_frame_skip):
            obs, _reward, terminated, truncated, info = self._env.step(action)

            reward = reward + _reward

            terminated = terminated.reshape(self.num_envs, -1).clone()
            truncated = truncated.reshape(self.num_envs, -1).clone()
            done = terminated | truncated

        reward = reward.reshape(self.num_envs, -1) # self.read_reward(reward)
        obs_dict = self.read_obs(obs)

        obs_dict[self.reward_key] = reward

        obs_dict["truncated"] = truncated
        obs_dict["done"] = done
        obs_dict["terminated"] = terminated

        tensordict_out = TensorDict(
            obs_dict, batch_size=tensordict.batch_size, device=self.device
        )

        for key in self.info_spec.keys(True, True):
            tensordict_out[key] = get_nested(info, key)
        # tensordict_out = tensordict_out.to(self.device, non_blocking=True)
        return tensordict_out


class OrbitEnv(OrbitWrapper):

    def __init__(self, env_name, cfg, **kwargs):
        env = gym.make(env_name, cfg=cfg, render_mode="rgb_array")
        super().__init__(env)