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

@hydra.main(config_path="../cfg", config_name="eval")
def main(cfg):
    OmegaConf.resolve(cfg)

    # load cheaper kit config in headless
    if cfg.headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    
    app_launcher = AppLauncher(
        {"headless": cfg.headless, "offscreen_render": True},
        # experience=app_experience
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

    # path = os.path.join(os.path.dirname(__file__), "policy.pt")
    # torch.save(policy.cpu(), path)
    # logging.info(F"Export policy to {path}")

    if hasattr(policy, "make_tensordict_primer"):
        transform.append(policy.make_tensordict_primer())

    stats_keys = [
        k for k in env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    
    @torch.no_grad()
    def evaluate(
        seed: int=0, 
        exploration_type: ExplorationType=ExplorationType.MODE,
        render=False,
        mode=None,
    ):
        frames = []

        base_env.eval()
        env.eval()
        env.set_seed(seed)
        policy.eval()

        if mode is not None and hasattr(policy, "mode"):
            policy.mode = mode

        from tqdm import tqdm
        t = tqdm(total=base_env.max_episode_length)
        def record_frame(*args, **kwargs):
            if render:
                frame = base_env.render(mode="rgb_array")
                frames.append(frame)
            t.update(2)
        
        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=Every(record_frame, 2),
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item() 
            for k, v in traj_stats.items()
        }

        # log video
        if len(frames):
            video_array = einops.rearrange(np.stack(frames), "t h w c -> t c h w")
            frames.clear()
            info["recording"] = wandb.Video(
                video_array, 
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), 
                format="mp4"
            )
        
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8, 8))
        axes[0].hist(traj_stats["return"])
        axes[1].hist(traj_stats["episode_len"])
        fig.tight_layout()
        path = os.path.join(os.path.dirname(__file__), "plot.png")
        fig.savefig(path)

        return info
    
    info = evaluate(render=False)
    print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
    
    base_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

