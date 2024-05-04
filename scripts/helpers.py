import torch
from typing import Sequence
from tensordict import TensorDictBase

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


def make_env(cfg):

    from active_adaptation.envs import TASKS
    from configs.rough import LocomotionEnvCfg
    from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, CatFrames, VecNorm

    env_cfg = LocomotionEnvCfg(cfg.task)

    base_env = TASKS[cfg.task.task](env_cfg)
    if cfg.get("vecnorm", True):
        vecnorm = VecNorm(list(base_env.observation_spec.keys(True, True)))
    else:
        vecnorm = None
    transform = Compose(InitTracker(), vecnorm)
    if cfg.task.short_history:
        transform.append(CatFrames(4, -1, ["policy"], ["policy"]))
    
    env = TransformedEnv(base_env, transform)
    env.set_seed(cfg.seed)
    return env, vecnorm