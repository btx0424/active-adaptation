import torch
import hydra
import numpy as np
import einops
from omegaconf import OmegaConf

from omni.isaac.lab.app import AppLauncher
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import SyncDataCollector

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictSequential
from active_adaptation.learning import ALGOS
from collections import OrderedDict

import wandb
import logging
from tqdm import tqdm

import os
import datetime

@hydra.main(config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app

    from scripts.helpers import EpisodeStats, make_env_policy
    env, policy, vecnorm = make_env_policy(cfg)
    
    if cfg.export_policy:
        import time
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        fake_input = env.observation_spec[0].zero().cpu()
        fake_input["is_init"] = torch.tensor(1, dtype=bool)
        fake_input["context_adapt_hx"] = torch.zeros(128)
        fake_input = fake_input.unsqueeze(0)

        def test(m, x):
            start = time.perf_counter()
            for _ in range(1000):
                m(x)
            return (time.perf_counter() - start) / 1000
        
        FILE_PATH = os.path.dirname(__file__)
        _policy = TensorDictSequential(
            vecnorm.to_observation_norm(),
            policy.get_rollout_policy("deploy")
        ).cpu()
        
        print(f"Inference time of policy: {test(_policy, fake_input)}")

        torch.save(_policy, os.path.join(FILE_PATH, f"policy-{time_str}.pt"))

    frames_per_batch = env.num_envs * 32
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    policy = policy.get_rollout_policy("eval")

    with (
        torch.inference_mode(), 
        set_exploration_type(ExplorationType.MODE)
    ):
        td_ = env.reset()
        
        while True:
            policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)

    collector = SyncDataCollector(
        env,
        policy=policy.get_rollout_policy("eval"),
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
        exploration_type=ExplorationType.MODE
    )
    
    pbar = tqdm(collector, total=total_frames//frames_per_batch)

    env.eval()
    
    for i, data in enumerate(pbar):
        info = {}
        episode_stats.add(data)

        if len(episode_stats) >= env.num_envs:
            info = {}
            for k, v in sorted(episode_stats.pop().items(True, True)):
                if isinstance(v, torch.Tensor):
                    info["train/" + (".".join(k) if isinstance(k, tuple) else k)] = torch.mean(v.float()).item()

            print()
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

