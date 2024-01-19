import torch
import hydra
import numpy as np
import einops
from omegaconf import OmegaConf

from omni.isaac.orbit.app import AppLauncher
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import SyncDataCollector

from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs.transforms import (
    TransformedEnv, Compose, InitTracker, History
)
from active_adaptation.learning import ALGOS

import wandb
import logging
from tqdm import tqdm
from helpers import EpisodeStats, Every

import os
import time

@hydra.main(config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)

    # load cheaper kit config in headless
    if cfg.headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    
    app_launcher = AppLauncher(
        {"headless": cfg.headless, "offscreen_render": True},
        experience=app_experience
    )
    simulation_app = app_launcher.app

    from active_adaptation.envs import TASKS
    from configs.rough import LocomotionEnvCfg

    # setup environment
    env_cfg = LocomotionEnvCfg(cfg.task)
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**21
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**21
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**20
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**22
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**19
    env_cfg.sim.physx.gpu_collision_stack_size = 2**24
    env_cfg.sim.physx.gpu_heap_capacity = 2**24

    env_cfg.history_length = cfg.task.history_length

    base_env = TASKS[cfg.task.task](env_cfg)
    transform = Compose(
        InitTracker(),
        History(["policy"], steps=16)
    )
    env = TransformedEnv(base_env, transform)
    env.set_seed(0)

    # setup policy
    policy = ALGOS[cfg.algo.name](
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec, 
        device=base_env.device
    )

    path = os.path.join(os.path.dirname(__file__), "policy.pt")
    torch.save(policy.cpu(), path)
    logging.info(F"Export policy to {path}")

    if hasattr(policy, "make_tensordict_primer"):
        transform.append(policy.make_tensordict_primer())

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch

    stats_keys = [
        k for k in env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
        exploration_type=ExplorationType.MODE
    )
    
    pbar = tqdm(collector, total=total_frames//frames_per_batch)

    env.eval()
    if hasattr(collector.policy, "mode"):
        collector.policy.mode = "adapt"
    
    for i, data in enumerate(pbar):
        info = {}
        episode_stats.add(data)

        if len(episode_stats) >= base_env.num_envs:
            info = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }

            print()
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
    
    base_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

