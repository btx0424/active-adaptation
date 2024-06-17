import torch
import wandb
import logging
import os

from typing import Sequence
from tensordict import TensorDictBase
from termcolor import colored
from collections import OrderedDict

from active_adaptation.learning import ALGOS


class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1

class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    def add(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict = tensordict["next"]
        done = next_tensordict["done"]
        if done.any():
            done = done.squeeze(-1)
            self._episodes += done.sum().item()
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats.extend(
                next_tensordict[done].clone().unbind(0)
            )
        return len(self)
    
    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats).to_tensordict()
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)


def make_env_policy(cfg):

    from active_adaptation.envs import TASKS
    from active_adaptation.utils.torchrl import StackFrames
    from configs.rough import LocomotionEnvCfg
    from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, CatFrames, VecNorm

    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
    else:
        state_dict = {}

    env_cfg = LocomotionEnvCfg(cfg.task)

    base_env = TASKS[cfg.task.task](env_cfg)
    obs_keys = [
        key for key, spec in base_env.observation_spec.items(True, True) 
        if not (spec.dtype == bool or key.endswith("_"))
    ]
    transform = Compose(InitTracker())

    assert cfg.vecnorm in ("train", "eval", None)
    print(colored(f"[Info]: create VecNorm for keys: {obs_keys}", "green"))
    vecnorm = VecNorm(obs_keys, decay=0.9999)

    if "vecnorm" in state_dict.keys():
        print(colored("[Info]: Load VecNorm from checkpoint.", "green"))
        vecnorm.load_state_dict(state_dict["vecnorm"])
    if cfg.vecnorm == "train":
        print(colored("[Info]: Updating obervation normalizer.", "green"))
        transform.append(vecnorm)
    elif cfg.vecnorm == "eval":
        print(colored("[Info]: Not updating obervation normalizer.", "green"))
        transform.append(vecnorm.to_observation_norm())
    elif cfg.vecnorm is not None:
        raise ValueError

    long_history = cfg.algo.get("long_history", 0)
    if long_history > 0:
        print(colored(f"[Info]: Long history length {long_history}.", "green"))
        transform.append(StackFrames(long_history, ["policy"], ["policy_h"]))
    short_history = cfg.algo.get("short_history", 0)
    if short_history > 0:
        print(colored(f"[Info]: Short history length {short_history}.", "green"))
        transform.append(CatFrames(short_history, -1, ["policy"], ["policy_h"]))

    env = TransformedEnv(base_env, transform)
    env.set_seed(cfg.seed)
    
    # setup policy
    policy = ALGOS[cfg.algo.name](
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec,
        device=base_env.device
    )
    
    if "policy" in state_dict.keys():
        print(colored("[Info]: Load policy from checkpoint.", "green"))
        policy.load_state_dict(state_dict["policy"])
    
    if hasattr(policy, "make_tensordict_primer"):
        primer = policy.make_tensordict_primer()
        print(colored(f"[Info]: Add TensorDictPrimer {primer}.", "green"))
        transform.append(primer)
        env = TransformedEnv(env.base_env, transform)

    return env, policy, vecnorm