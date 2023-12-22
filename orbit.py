import gymnasium as gym

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs.libs import GymWrapper, GymEnv
from copy import deepcopy

class OrbitWrapper(GymWrapper):
    
    def __init__(self, env=None, categorical_action_encoding=False, **kwargs):
        kwargs["device"] = env.device
        super().__init__(env, categorical_action_encoding, **kwargs)

        from omni.isaac.orbit.envs import RLTaskEnv
        self._env: RLTaskEnv
    
    @property
    def num_envs(self):
        return self._env.num_envs
    
    @property
    def max_peisode_length(self):
        return self._env.max_episode_length
    
    @property
    def extras(self):
        return deepcopy(self._env.extras)

    @property
    def _is_batched(self):
        return True
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key).to(self.device)

        reward = 0
        for _ in range(self.wrapper_frame_skip):
            obs, _reward, terminated, truncated, info = self._env.step(action)

            reward = reward + _reward

            terminated = terminated.reshape(self.num_envs, -1)
            truncated = truncated.reshape(self.num_envs, -1)
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

        if self.info_dict_reader and info is not None:
            for info_dict_reader in self.info_dict_reader:
                out = info_dict_reader(info, tensordict_out)
                if out is not None:
                    tensordict_out = out
        # tensordict_out = tensordict_out.to(self.device, non_blocking=True)
        return tensordict_out


class OrbitEnv(OrbitWrapper):

    def __init__(self, env_name, cfg, **kwargs):
        env = gym.make(env_name, cfg=cfg, render_mode="rgb_array")
        super().__init__(env)