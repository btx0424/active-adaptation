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
    TransformedEnv, 
    Compose, 
    InitTracker,
    History,
    RewardSum,
    CatFrames
)
from active_adaptation.learning import BCPolicy, ALGOS

from helpers import EpisodeStats, Every

import wandb
import logging
from tqdm import tqdm

import os
import time

@hydra.main(config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    
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

    from active_adaptation.envs import TASKS
    from configs.rough import LocomotionEnvCfg

    run = init_wandb(cfg)

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
        # CatFrames(4, -1, ["policy"], ["priv"]),
        History(["policy"], steps=16)
    )
    env = TransformedEnv(base_env, transform)
    env.set_seed(0)

    # setup policy
    teacher = ALGOS[cfg.algo.name](
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec, 
        device=base_env.device
    )
    policy = BCPolicy(
        env.observation_spec, 
        env.action_spec, 
        teacher=teacher,
        device=base_env.device
    )

    if hasattr(policy, "make_tensordict_primer"):
        transform.append(policy.make_tensordict_primer())

    frames_per_batch = env.num_envs * cfg.algo.train_every
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)
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
    )
    
    def save(policy, checkpoint_name: str, artifact: bool=False):
        try:
            ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            if artifact:
                artifact = wandb.Artifact(
                    f"{type(base_env).__name__}-{type(policy).__name__}", 
                    type="model"
                )
                artifact.add_file(ckpt_path)
                run.log_artifact(artifact)
            logging.info(f"Saved checkpoint to {str(ckpt_path)}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

    pbar = tqdm(collector, total=total_frames//frames_per_batch)
    
    for i, data in enumerate(pbar):
        start = time.perf_counter()
        
        info = {}

        episode_stats.add(data)

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
        
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
    
    base_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

