import torch
import hydra
import numpy as np
import einops
from omegaconf import OmegaConf

from omni.isaac.orbit.app import AppLauncher
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import SyncDataCollector

from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker
from active_adaptation.learning import PPOPolicy, PPORNNPolicy

import wandb
import logging
from tqdm import tqdm
from helpers import EpisodeStats, Every


@hydra.main(config_path="cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)

    app_launcher = AppLauncher({"headless": True, "offscreen_render": True})
    simulation_app = app_launcher.app

    from active_adaptation.envs import LocomotionEnv
    from configs import UNITREE_A1_ENV

    run = init_wandb(cfg)

    # setup environment
    UNITREE_A1_ENV.scene.num_envs = cfg.task.env.num_envs
    base_env = LocomotionEnv(UNITREE_A1_ENV)
    transform = InitTracker()
    env = TransformedEnv(base_env, transform)
    env.set_seed(0)

    # setup policy
    policy = PPORNNPolicy(
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec, 
        device=base_env.device
    )

    frames_per_batch = env.num_envs * cfg.algo.train_every
    eval_interval = cfg.get("eval_interval", -1)
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

    @torch.no_grad()
    def evaluate(
        seed: int=0, 
        exploration_type: ExplorationType=ExplorationType.MODE
    ):
        frames = []

        base_env.eval()
        env.eval()
        env.set_seed(seed)

        from tqdm import tqdm
        t = tqdm(total=base_env.max_episode_length)
        def record_frame(*args, **kwargs):
            frame = base_env.render(mode="rgb_array")
            frames.append(frame)
            t.update(2)
        
        base_env.enable_render = True
        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=Every(record_frame, 2),
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render = False
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
    
    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        info = {
            "env_frames": collector._frames, 
            "rollout_fps": collector._fps
        }
        pbar.set_postfix(info)
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
        
        info.update(policy.train_op(data))

        if eval_interval > 0 and (i + 1) % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())

        run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

    
    info = evaluate()
    run.log(info)

    simulation_app.close()


if __name__ == "__main__":
    main()

