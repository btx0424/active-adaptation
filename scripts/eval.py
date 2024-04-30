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
    TransformedEnv, Compose, InitTracker, VecNorm
)
from active_adaptation.learning import ALGOS

import wandb
import logging
from tqdm import tqdm
from helpers import EpisodeStats, Every

import os
import datetime

@hydra.main(config_path="../cfg", config_name="eval", version_base=None)
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

    base_env = TASKS[cfg.task.task](env_cfg)
    if cfg.get("vecnorm", True):
        vecnorm = VecNorm(list(base_env.observation_spec.keys(True, True)))
    else:
        vecnorm = None
    transform = Compose(InitTracker(), vecnorm)
    env = TransformedEnv(base_env, transform)
    env.set_seed(cfg.seed)

    # setup policy
    policy = ALGOS[cfg.algo.name](
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec,
        vecnorm,
        device=base_env.device
    )

    try:
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        path = os.path.join(os.path.dirname(__file__), f"policy-{time_str}.pt")
        _policy = policy.get_rollout_policy("eval").cpu()
        torch.save(_policy, path)
        logging.info(F"Export policy to {path}")
    except Exception as e:
        print(e)
    
    if hasattr(policy, "make_tensordict_primer"):
        env = TransformedEnv(env, policy.make_tensordict_primer())

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    
    @torch.no_grad()
    def evaluate(
        seed: int=0, 
        exploration_type: ExplorationType=ExplorationType.RANDOM,
        render=False,
        mode=None,
    ):
        frames = []

        env.eval()
        env.set_seed(seed)

        if mode is not None and hasattr(policy, "mode"):
            policy.mode = mode

        from tqdm import tqdm
        t = tqdm(total=base_env.max_episode_length)
        def record_frame(*args, **kwargs):
            if render:
                frame = base_env.render(mode="rgb_array")
                frames.append(frame)
            t.update(1)
        
        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy.get_rollout_policy(mode="eval").to(base_env.device),
                callback=record_frame,
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
            from torchvision.io import write_video
            time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
            video_array = np.stack(frames)
            frames.clear()
            write_video(
                os.path.join(os.path.dirname(__file__), f"recording-{time_str}.mp4"),
                video_array,
                fps=1 / base_env.step_dt
            )

        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        path = os.path.join(os.path.dirname(__file__), f"trajs-{time_str}.pt")
        trajs = trajs.select(
            ("next", "done"), 
            ("next", "stats", "return"), 
            "value_priv",
            "value_adapt",
            "context_expert",
            "context_adapt",
            "context_adapt_std",
            strict=False
        )
        torch.save(trajs, path)
        return info
    
    info = evaluate(render=cfg.eval_render, seed=cfg.seed)
    info = {k: v for k, v in info.items() if isinstance(v, float)}
    info["task"] = cfg.task.name
    info["algo"] = cfg.algo.name
    info["checkpoint_path"] = cfg.algo.checkpoint_path
    print(OmegaConf.to_yaml(info))
    
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    path = os.path.join(os.path.dirname(__file__), f"eval/{time_str}.yaml")
    with open(path, "w") as f:
        OmegaConf.save(info, f)
    
    base_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

