import torch
# import warp
import hydra
import numpy as np
import einops
import wandb
import logging
import os
import time
import datetime

from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

from omni.isaac.lab.app import AppLauncher
# from omni_drones.utils.wandb import init_wandb
from active_adaptation.utils.torchrl import SyncDataCollector

# local import
from scripts.helpers import make_env_policy, EpisodeStats, evaluate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra.main(config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    run = wandb.init(**cfg.wandb)
    run.config.update(OmegaConf.to_container(cfg))
    setproctitle(run.name)

    env, policy, vecnorm = make_env_policy(cfg)

    import inspect
    import shutil
    source_path = inspect.getfile(policy.__class__)
    target_path = os.path.join(run.dir, source_path.split("/")[-1])
    shutil.copy(source_path, target_path)
    wandb.save(target_path, policy="now")

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy.get_rollout_policy("train"),
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
        wandb.save(ckpt_path, policy="now")
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    pbar = tqdm(collector, total=total_iters)
    
    for i, data in enumerate(pbar):
        start = time.perf_counter()
        
        info = {}

        episode_stats.add(data)

        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + (".".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        
        info.update(policy.train_op(data))
        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters)

        info["env_frames"] = collector._frames
        info["rollout_fps"] = collector._fps
        info["training_time"] = time.perf_counter() - start
        
        if save_interval > 0  and i % save_interval == 0:
            save(policy, f"checkpoint_{i}")

        run.log(info)

        print()
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, (float, int))}))
    
    save(policy, "checkpoint_final")

    policy_eval = policy.get_rollout_policy("eval")
    info, trajs = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
    info["env_frames"] = collector._frames
    run.log(info)

    wandb.finish()
    exit(0)
    
    base_env.close()
    simulation_app.close()
    exit(0)


if __name__ == "__main__":
    main()

