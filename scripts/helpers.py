import torch
import torch.nn as nn
import hydra
import numpy as np
import time
import wandb
import logging
import os
import datetime

from typing import Sequence
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModuleBase as ModBase
from torchrl.envs.transforms import VecNorm
import torch.distributed as dist

from termcolor import colored
from collections import OrderedDict
from torchvision.io import write_video
from omegaconf import OmegaConf, DictConfig
import active_adaptation.learning
from active_adaptation.utils.wandb import parse_checkpoint_path

import active_adaptation
import active_adaptation_projects

active_adaptation_projects.import_projects()


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
    def __init__(self, in_keys: Sequence[str], device: torch.device):
        self.in_keys = in_keys
        self.device = device
        self._stats = TensorDict({key: torch.tensor([0.], device=device) for key in in_keys}, [1])
        self._episodes = torch.tensor(0, device=device)

    def add(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict = tensordict["next"]
        done = next_tensordict["done"]
        if done.any():
            done = done.squeeze(-1)
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats = self._stats + next_tensordict[done].sum(dim=0)
            self._episodes += done.sum()
        return len(self)
    
    def pop(self):
        stats = self._stats / self._episodes
        self._stats.zero_()
        self._episodes.zero_()
        return stats.cpu()

    def __len__(self):
        return self._episodes.item()


def make_env_policy(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg.seed = cfg.seed + active_adaptation.get_local_rank()
    
    from active_adaptation.envs import SimpleEnv
    from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, VecNorm, StepCounter
    
    base_env = SimpleEnv(cfg.task)

    checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, weights_only=False)
    else:
        state_dict = {}
    
    policy_in_keys = cfg.algo.get("in_keys", ["policy", "priv"])

    for obs_group_key in list(cfg.task.observation.keys()):
        if (
            obs_group_key not in policy_in_keys
            and not obs_group_key.endswith("_")
        ):
            cfg.task.observation.pop(obs_group_key)
            print(colored(f"Discard obs group {obs_group_key} as it is not used.", "yellow"))
    
    obs_keys = [
        key for key, spec in base_env.observation_spec.items(True, True) 
        if not (spec.dtype == bool or key.endswith("_"))
    ]
    transform = Compose(InitTracker(), StepCounter())

    assert cfg.vecnorm in ("train", "eval", None)
    print(colored(f"[Info]: create VecNorm for keys: {obs_keys}", "green"))
    vecnorm = VecNorm(obs_keys, decay=0.9999)
    vecnorm(base_env.fake_tensordict())

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

    env = TransformedEnv(base_env, transform)
    env.set_seed(cfg.seed)
    
    # setup policy
    policy_cls = hydra.utils.get_class(cfg.algo._target_)
    print(f"Creating policy {policy_cls} on device {base_env.device}")
    policy = policy_cls(
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec,
        device=base_env.device,
        env=env
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


def check_vecnorm_state(vecnorm: VecNorm):
    WORLD_SIZE = active_adaptation.get_world_size()
    LOCAL_RANK = active_adaptation.get_local_rank()
    device = torch.device(f"cuda:{LOCAL_RANK}")

    keys = vecnorm.in_keys
    loc, scale = vecnorm._get_loc_scale()
    loc_tensors = []
    scale_tensors = []
    for key in keys:
        loc_tensors.append(loc[key].flatten().to(device))
        scale_tensors.append(scale[key].flatten().to(device))

    loc_state = torch.cat(loc_tensors)
    scale_state = torch.cat(scale_tensors)

    gathered_loc = [torch.empty_like(loc_state) for _ in range(WORLD_SIZE)]
    gathered_scale = [torch.empty_like(scale_state) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_loc, loc_state)
    dist.all_gather(gathered_scale, scale_state)
    
    diffs_loc = []
    diffs_scale = []
    for r, (loc_vec, scale_vec) in enumerate(zip(gathered_loc, gathered_scale)):
        diffs_loc.append((loc_vec - loc_state).abs().sum().item())
        diffs_scale.append((scale_vec - scale_state).abs().sum().item())
    
    return diffs_loc, diffs_scale


from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type
from tqdm import tqdm

@torch.inference_mode()
def evaluate(
    env: TransformedEnv,
    policy: torch.nn.Module,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MODE,
    render=False,
    keys=[("next", "stats")],
):
    """
    Evaluate the policy on the environment, selecting `keys` from the trajectory.
    If `render` is True, record and save the video.
    """
    keys = set(keys)
    keys.add(("next", "done"))

    env.eval()
    env.set_seed(seed)

    tensordict_ = env.reset()
    trajs = []
    frames = []

    inference_time = []
    torch.compiler.cudagraph_mark_step_begin()
    with set_exploration_type(exploration_type):
        for i in tqdm(range(env.max_episode_length), miniters=10):
            s = time.perf_counter()
            tensordict_ = policy(tensordict_)
            e = time.perf_counter()
            inference_time.append(e - s)
            tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
            trajs.append(tensordict.select(*keys, strict=False).cpu())
            if render:
                frames.append(env.render("rgb_array"))
    inference_time = np.mean(inference_time[5:])
    print(f"Average inference time: {inference_time:.4f} s")

    trajs: TensorDictBase = torch.stack(trajs, dim=1)
    done = trajs.get(("next", "done"))
    episode_cnt = len(done.nonzero())
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    info = {}
    stats = {}
    compute_std_for = ["return", "survival"]
    for k, v in trajs["next", "stats"].items(True, True):
        v = take_first_episode(v)
        key = "eval/" + ("/".join(k) if isinstance(k, tuple) else k)
        stats[key] = v
        info[key] = torch.mean(v.float()).item()
        if k in compute_std_for:
            info[key + "_std"] = torch.std(v.float()).item()

    # log video
    if len(frames):
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        video_array = np.stack(frames)
        frames.clear()
        video_path = os.path.join(os.path.dirname(__file__), f"recording-{time_str}.mp4")
        write_video(
            video_path, video_array=video_array, fps=int(1 / env.step_dt), video_codec="h264"
        )

    info["episode_cnt"] = episode_cnt
    return dict(sorted(info.items())), trajs, stats
