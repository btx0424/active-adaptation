import hydra
import gymnasium as gym

from omni.isaac.orbit.app import AppLauncher

import torch
import warnings

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs.libs import GymWrapper

from torchrl.collectors import SyncDataCollector
from tqdm import tqdm

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
    def _is_batched(self):
        return True
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key).to(self.device)

        reward = 0
        for _ in range(self.wrapper_frame_skip):
            obs, _reward, terminated, truncated, info = self._env.step(action)

            reward = reward + _reward

            done = terminated | truncated

        reward = self.read_reward(reward)
        obs_dict = self.read_obs(obs)

        obs_dict[self.reward_key] = reward

        obs_dict["truncated"] = truncated
        obs_dict["done"] = done
        obs_dict["terminated"] = terminated

        tensordict_out = TensorDict(
            obs_dict, batch_size=tensordict.batch_size, device=self.device
        )

        if self.info_dict_reader and info is not None:
            if not isinstance(info, dict):
                warnings.warn(
                    f"Expected info to be a dictionary but got a {type(info)} with values {str(info)[:100]}."
                )
            else:
                for info_dict_reader in self.info_dict_reader:
                    out = info_dict_reader(info, tensordict_out)
                    if out is not None:
                        tensordict_out = out
        # tensordict_out = tensordict_out.to(self.device, non_blocking=True)
        return tensordict_out


@hydra.main(config_path="cfg")
def main(cfg):
    app = AppLauncher({"headless": True})

    import omni.isaac.orbit_tasks  # noqa: F401
    from omni.isaac.orbit_tasks.utils import parse_env_cfg

    # task_name = "Isaac-Velocity-Rough-Unitree-Go2-v0"
    task_name = "Isaac-Velocity-Flat-Unitree-Go2-v0"
    env_cfg = parse_env_cfg(task_name, use_gpu=True, num_envs=2048)
    
    env = gym.make(task_name, cfg=env_cfg)
    env = OrbitWrapper(env)

    collector = SyncDataCollector(
        env,
        policy=None,
        frames_per_batch=env.num_envs * 32,
        total_frames=-1,
        device=env.device,
        return_same_td=True
    )
    for i, data in tqdm(enumerate(collector)):
        print(data)

    env.close()


if __name__ == "__main__":
    main()
