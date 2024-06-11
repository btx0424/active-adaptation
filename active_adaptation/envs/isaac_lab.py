import torch

from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

from tensordict import TensorDict, TensorDictBase

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import RewardManager as _RewardManager


from dataclasses import dataclass
@dataclass
class IsaacLabEnvCfg:
    num_envs: int = 64
    device: str = "cuda"
    task: str = ""


class RewardManager(_RewardManager):
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super(_RewardManager, self).__init__(cfg, env)
        self._episode_sums = TensorDict({}, [self.num_envs])
        with torch.device(self.device):
            for term_name in self._term_names:
                self._episode_sums[term_name] = torch.zeros(self.num_envs, dtype=torch.float)
            self._episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float)
            self._reward_buf = torch.zeros(self.num_envs, len(self._term_names))
            self.clip_count = torch.zeros(self.num_envs)

    def compute(self, dt: float) -> torch.Tensor:
        rewards = []
        # iterate over all the reward terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            rewards.append(value)
            # update episodic sum
            self._episode_sums[name] += value
        self._reward_buf = torch.stack(rewards, dim=-1)
        reward = self._reward_buf.sum(-1)
        if False:
            self.clip_count[reward < 0.] += 1
            reward_clipped = reward.clamp_min(0.)
            self._episode_sums["total"] += reward_clipped
            return reward_clipped
        else:
            self._episode_sums["total"] += reward
            return reward
    
    def reset(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key, value in self._episode_sums.items():
            extras[f"Episode Reward/{key}"] = value.clone()
        self._episode_sums[env_ids] = 0.
        self.clip_count[env_ids] = 0.
        # reset all the reward terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras


def step_without_reset(self: ManagerBasedRLEnv, action: torch.Tensor):
    """Execute one time-step of the environment's dynamics and reset terminated environments.

    Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

    1. Process the actions.
    2. Perform physics stepping.
    3. Perform rendering if gui is enabled.
    4. Update the environment counters and compute the rewards and terminations.
    5. Reset the environments that terminated.
    6. Compute the observations.
    7. Return the observations, rewards, resets and extras.

    Args:
        action: The actions to apply on the environment. Shape is (num_envs, action_dim).

    Returns:
        A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
    """
    # process actions
    self.action_manager.process_action(action)
    # perform physics stepping
    for _ in range(self.cfg.decimation):
        # set actions into buffers
        self.action_manager.apply_action()
        # set actions into simulator
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # update buffers at sim dt
        self.scene.update(dt=self.physics_dt)
    # perform rendering if gui is enabled
    if self.sim.has_gui() or self.sim.has_rtx_sensors():
        self.sim.render()

    # post-step:
    # -- update env counters (used for curriculum generation)
    self.episode_length_buf += 1  # step in current episode (per env)
    self.common_step_counter += 1  # total step (common for all envs)
    # -- check terminations
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    # -- reward computation
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    # -- update command
    self.command_manager.compute(dt=self.step_dt)
    # -- step interval events
    if "interval" in self.event_manager.available_modes:
        self.event_manager.apply(mode="interval", dt=self.step_dt)
    # -- compute observations
    # note: done after reset to get the correct observations for reset envs
    self.obs_buf = self.observation_manager.compute()

    # return observations, rewards, resets and extras
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


class IsaacLabEnv(_EnvWrapper):
    
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: IsaacLabEnvCfg):
        self.cfg = cfg
        super().__init__(
            batch_size=[cfg.num_envs],
            device=cfg.device
        )
        self.extras = {}

    def _build_env(self, **kwargs) -> ManagerBasedRLEnv:
        import gymnasium as gym
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        
        task = self.cfg.task
        env_cfg = parse_env_cfg(task, use_gpu=True, num_envs=self.cfg.num_envs, use_fabric=True)
        env = gym.make(task, cfg=env_cfg).unwrapped
        env.reward_manager = RewardManager(env.cfg.rewards, env)
        return env

    def _init_env(self) -> int | None:
        self._env
        return
    
    def _make_specs(self, env: ManagerBasedRLEnv) -> None:
        observation_spec = _gym_to_torchrl_spec_transform(
            env.observation_space,
            device=self.device,
        )
        action_spec = _gym_to_torchrl_spec_transform(
            env.action_space,
            device=self.device,
        )
        reward_spec = CompositeSpec({})
        reward_spec["reward"] = UnboundedContinuousTensorSpec([1])
        for term in env.reward_manager.active_terms:
            reward_spec[f"Episode Reward/{term}"] = UnboundedContinuousTensorSpec([1])
        reward_spec["Episode Length"] = UnboundedContinuousTensorSpec([1])
        # reward_spec["Episode Reward/clip_ratio"] = UnboundedContinuousTensorSpec([1])
        reward_spec["Episode Reward/total"] = UnboundedContinuousTensorSpec([1])
        observation_spec.shape = self.batch_size

        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec.expand(*self.batch_size).to(self.device)
        
    def _set_seed(self, seed: int | None):
        self._env.seed(seed)
    
    def _check_kwargs(self, kwargs: torch.Dict):
        pass

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is None:
            obs, info = self._env.reset(**kwargs)
            tensordict_out = TensorDict(obs, self.batch_size, self.device)
        else:
            tensordict_out = tensordict.clone()
        
        return tensordict_out
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
            env_ids = env_mask.nonzero().squeeze(-1)
            self._env._reset_idx(env_ids)
            self._env.command_manager.compute(self._env.step_dt)
            obs = self._env.observation_manager.compute()
        else:
            obs, info = self._env.reset()
        self.extras.update(self._env.extras)
        tensordict_out = TensorDict(obs, self.batch_size, self.device)
        return tensordict_out
        
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key)
        
        # obs, reward, terminated, truncated, info = step_without_reset(self._env, action)
        obs, reward, terminated, truncated, info = self._env.step(action)

        terminated = terminated.reshape(self.num_envs, -1)
        truncated = truncated.reshape(self.num_envs, -1)
        reward = reward.reshape(self.num_envs, -1)

        tensordict_out = TensorDict(obs, self.batch_size, self.device)
        tensordict_out["truncated"] = truncated
        tensordict_out["terminated"] = terminated
        tensordict_out["done"] = terminated | truncated
        tensordict_out["reward"] = reward.clone()
        tensordict_out.update(self._read_extras())

        return tensordict_out.clone()
    
    def _read_extras(self):
        extras = {}
        for key, value in self._env.extras["log"].items():
            if isinstance(value, torch.Tensor) and value.shape[0] == self.batch_size[0]:
                extras[key] = value
            elif isinstance(value, float):
                self.extras[key] = value
        extras["Episode Length"] = self._env.episode_length_buf.clone()
        return extras
    
    @property
    def num_envs(self):
        return self._env.num_envs
    
    @property
    def max_episode_length(self):
        return self._env.max_episode_length

