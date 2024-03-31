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
    RewardSum,
    CatFrames,
    VecNorm
)
from active_adaptation.learning import ALGOS, PPODualPolicy
from helpers import EpisodeStats, Every

import wandb
import logging
from tqdm import tqdm

import os
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

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
    
    base_env = TASKS[cfg.task.task](env_cfg)
    transform = Compose(
        InitTracker(),
        # VecNorm(in_keys=["policy"], out_keys=["policy"]),
        # CatFrames(4, -1, ["policy"], ["priv"]),
        # History(["policy"], steps=16)
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

    if hasattr(policy, "make_tensordict_primer"):
        transform.append(policy.make_tensordict_primer())

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    log_interval = (base_env.max_episode_length // cfg.algo.train_every) + 1
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
        
        # log distributions
        # df = pd.DataFrame(traj_stats)
        # table = wandb.Table(dataframe=df)
        # info["eval/return"] = wandb.plot.histogram(table, "return")
        # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

        return info
    
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

    pbar = tqdm(collector, total=total_iters)
    
    
    for i, data in enumerate(pbar):
        start = time.perf_counter()
        
        info = {}

        episode_stats.add(data)

        if i % log_interval == 0:
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + (".".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        
        info.update(policy.train_op(data))
        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters)

        info["env_frames"] = collector._frames
        info["rollout_fps"] = collector._fps
        info["training_time"] = time.perf_counter() - start
        
        if eval_interval > 0 and (i + 1) % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate(render=cfg.eval_render))
            env.train()
            policy.train()
        
        if save_interval > 0  and i % save_interval == 0:
            save(policy, f"checkpoint_{i}")

        run.log(info)

        print()
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
    
    save(policy, "checkpoint_final")

    info = evaluate(render=cfg.eval_render, mode="expert")
    info["env_frames"] = collector._frames
    run.log(info)

    wandb.finish()
    
    base_env.close()
    simulation_app.close()
    exit(0)


if __name__ == "__main__":
    main()

