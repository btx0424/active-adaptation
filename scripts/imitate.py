import torch
import hydra
import numpy as np
import einops
from omegaconf import OmegaConf

from omni.isaac.lab.app import AppLauncher
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import SyncDataCollector

from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs.transforms import (
    TransformedEnv, 
    Compose, 
    InitTracker,
    RewardSum,
    CatFrames
)
from active_adaptation.learning import BCPolicy, ALGOS

# local import
from scripts.helpers import make_env_policy, EpisodeStats, Every

import wandb
import logging
from tqdm import tqdm
from collections import OrderedDict

import os
import time

@hydra.main(config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.vecnorm = "eval"
    
    # load cheaper kit config in headless
    if cfg.headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    
    app_launcher = AppLauncher(
        {"headless": cfg.headless, "offscreen_render": cfg.offscreen_render},
        experience=app_experience,
        # experience=f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    )
    simulation_app = app_launcher.app

    run = init_wandb(cfg)

    # setup environment
    env, teacher, vecnorm = make_env_policy(cfg)

    policy = BCPolicy(
        {},
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec,
        teacher=teacher.get_rollout_policy("eval"),
        device=env.device
    )

    if hasattr(policy, "make_tensordict_primer"):
        env.transform.append(policy.make_tensordict_primer())

    frames_per_batch = env.num_envs * cfg.algo.train_every
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )
    
    def save(policy, checkpoint_name: str, artifact: bool=False):
        ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["policy"] = policy.state_dict()
        if "vecnorm" in locals():
            state_dict["vecnorm"] = vecnorm.state_dict()
        torch.save(state_dict, ckpt_path)
        if artifact:
            artifact = wandb.Artifact(
                f"{type(env).__name__}-{type(policy).__name__}", 
                type="model"
            )
            artifact.add_file(ckpt_path)
            run.log_artifact(artifact)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    pbar = tqdm(collector, total=total_frames//frames_per_batch)
    
    for i, data in enumerate(pbar):
        start = time.perf_counter()
        
        info = {}

        episode_stats.add(data)

        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + (".".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        
        info.update(policy.train_op(data))

        info["env_frames"] = collector._frames
        info["rollout_fps"] = collector._fps
        info["training_time"] = time.perf_counter() - start
        
        if save_interval > 0  and i % save_interval == 0:
            save(policy, f"checkpoint_{i}")

        run.log(info)

        print()
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
    
    info["env_frames"] = collector._frames
    run.log(info)

    save(policy, "checkpoint_final")

    wandb.finish()
    exit(0)
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

