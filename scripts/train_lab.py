import torch
import os.path as osp
import hydra
import time
import wandb
import termcolor

from tqdm import tqdm
from pprint import pprint
from collections import OrderedDict
from omegaconf import OmegaConf, DictConfig

from active_adaptation.utils.torchrl import SyncDataCollector
from active_adaptation.learning import ALGOS

from omni.isaac.lab.app import AppLauncher

from torchrl.envs.transforms import (
    TransformedEnv, 
    Compose, 
    InitTracker,
    VecNorm
)

from scripts.helpers import EpisodeStats

def make_env_agent(cfg):
    from active_adaptation.envs.isaac_lab import IsaacLabEnv, IsaacLabEnvCfg
    
    env_cfg = IsaacLabEnvCfg(**cfg.env)
    env = IsaacLabEnv(env_cfg)
    transform = [
        InitTracker(), 
        VecNorm(["policy"])
    ] 
    transform = Compose(*transform)

    env = TransformedEnv(env, transform)

    agent = ALGOS[cfg.algo.name](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=env.device
    )

    if cfg.get("checkpoint") is not None:
        checkpoint = torch.load(checkpoint)
        agent.load_state_dict(checkpoint["agent"])
        env.transform.load_state_dict(checkpoint.get("transform", {}))

    return env, agent


@hydra.main(config_path="../cfg", config_name="train_lab")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    
    run = wandb.init(**cfg.wandb)
    run.config.update(OmegaConf.to_container(cfg))

    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    env, agent = make_env_agent(cfg)
    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if k.startswith("Episode")
    ]
    episode_stats = EpisodeStats(stats_keys)

    frames_per_batch = cfg.training.update_every * env.num_envs
    total_frames = ((cfg.training.total_frames // frames_per_batch) + 1) * frames_per_batch
    total_iters = total_frames // frames_per_batch
    collector = SyncDataCollector(
        env,
        policy=agent.get_rollout_policy("train"),
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=env.device,
        return_same_td=True,
    )
    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    
    print(f"[INFO] Log interval: {log_interval}")

    start_time = time.time()
    p = tqdm(collector, total=total_iters)
    env.train()
    agent.train()
    try:
        for i, tensordict in enumerate(p):
            episode_stats.add(tensordict)

            info = {}
            info["env_steps"] = collector._frames
            info["rollout_fps"] = collector._fps
            info["overall_fps"] = collector._frames / (time.time() - start_time)

            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + (".".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()
            
            for k, v in sorted(env.extras["log"].items()):
                if not isinstance(v, torch.Tensor):
                    info[f"train/{k}"] = v
            
            t = time.perf_counter()
            info.update(agent.train_op(tensordict))
            info["training_time"] = time.perf_counter() - t

            pprint(info)
            run.log(info)

            if i % cfg.training.save_interval == 0:
                state_dict = OrderedDict()
                state_dict["agent"] = agent.state_dict()
                state_dict["transform"] = env.transform.state_dict()
                
                ckpt_path = osp.join(run.dir, f"checkpoint_{i}.pt")
                torch.save(state_dict, ckpt_path)
                print(f"Save checkpoint to {ckpt_path}")
    
    except KeyboardInterrupt:
        pass

    state_dict = OrderedDict()
    state_dict["agent"] = agent.state_dict()
    
    ckpt_path = osp.join(run.dir, f"checkpoint_{i}.pt")
    torch.save(state_dict, ckpt_path)
    print(f"Save checkpoint to {ckpt_path}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()