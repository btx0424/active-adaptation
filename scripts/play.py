import torch
import hydra
import numpy as np
import einops
import itertools
from omegaconf import OmegaConf

from omni.isaac.lab.app import AppLauncher
# from omni_drones.utils.wandb import init_wandb
# from omni_drones.utils.torchrl import SyncDataCollector

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
@torch.inference_mode()
@set_exploration_type(ExplorationType.MODE)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app

    from scripts.helpers import EpisodeStats, make_env_policy, ObsNorm, export_onnx
    env, policy, vecnorm = make_env_policy(cfg)
    
    if cfg.export_policy:
        import time
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        fake_input = env.observation_spec[0].rand().cpu()
        fake_input["is_init"] = torch.tensor(1, dtype=bool)
        fake_input["context_adapt_hx"] = torch.zeros(128)
        fake_input = fake_input.unsqueeze(0)

        def test(m, x):
            start = time.perf_counter()
            for _ in range(1000):
                m(x)
            return (time.perf_counter() - start) / 1000
        
        FILE_PATH = os.path.dirname(__file__)
        
        deploy_policy = policy.get_rollout_policy("deploy")
        obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy_policy.in_keys)
        _policy = TensorDictSequential(obs_norm, deploy_policy).cpu()
        
        print(f"Inference time of policy: {test(_policy, fake_input)}")

        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        path = os.path.join(FILE_PATH, f"policy-{time_str}.pt")
        torch.save(_policy, path)

        meta = {}
        meta["action_scaling"] = dict(cfg.task.action.get("action_scaling"))
        export_onnx(_policy, fake_input, path.replace(".pt", ".onnx"), meta)
        

    frames_per_batch = env.num_envs * 32
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    policy = policy.get_rollout_policy("eval")


    td_ = env.reset()
    
    for i in itertools.count():
        td_ = policy(td_)
        td, td_ = env.step_and_maybe_reset(td_)
        # td_.update(td["next"])
        episode_stats.add(td)

        if len(episode_stats) >= env.num_envs:
            print("Step", i)
            for k, v in sorted(episode_stats.pop().items(True, True)):
                print(k, torch.mean(v).item())
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

