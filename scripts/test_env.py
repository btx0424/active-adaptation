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

from torchrl.envs.utils import set_exploration_type, ExplorationType

from omni.isaac.lab.app import AppLauncher
from omni_drones.utils.wandb import init_wandb
from active_adaptation.utils.torchrl import SyncDataCollector

# local import
from scripts.helpers import make_env_policy, EpisodeStats, Every

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

    run = init_wandb(cfg)

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
    
    @torch.inference_mode()
    def evaluate(
        seed: int=0, 
        exploration_type: ExplorationType=ExplorationType.MODE,
        render=False,
    ):
        frames = []

        env.eval()
        env.set_seed(seed)

        keys = [
            ("next", "done"), 
            ("next", "stats", "return"),
            "value_obs",
            "value_priv",
            "value_adapt",
            "context_expert",
            "context_scale",
            "context_adapt",
            "context_adapt_scale",
            "action_kl",
        ]

        td_ = env.reset()
        trajs = []
        frames = []
        _policy = policy.get_rollout_policy("eval")

        t = tqdm(range(env.max_episode_length), miniters=50)
        with set_exploration_type(exploration_type):
            for i in enumerate(t):
                _policy(td_)
                td, td_ = env.step_and_maybe_reset(td_)
                trajs.append(td.select(*keys, strict=False).cpu())
                if render:
                    frames.append(env.render("rgb_array").cpu())

        trajs = torch.stack(trajs)
        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }
        
        info = {}
        compute_std_for = ["return", "survival"]
        for k, v in sorted(traj_stats.items()):
            info["eval/stats." + k] = torch.mean(v.float()).item()
            if k in compute_std_for:
                info["eval/stats." + k + "_std"] = torch.std(v.float()).item()

        # log video
        if len(frames):
            from torchvision.io import write_video
            time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
            video_array = np.stack(frames)
            frames.clear()
            video_path = os.path.join(os.path.dirname(__file__), f"recording-{time_str}.mp4")
            write_video(video_path, video_array, fps=1 / env.step_dt)

        # time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        # path = os.path.join(os.path.dirname(__file__), f"trajs-{time_str}.pt")
        # print(termcolor.colored(trajs, "light_yellow"))
        # torch.save(trajs, path)
        return info
    
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
        
        if eval_interval > 0 and (i + 1) % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate(render=cfg.eval_render))
            env.train()
            policy.train()
        
        if save_interval > 0  and i % save_interval == 0:
            save(policy, f"checkpoint_{i}")

        run.log(info)

        print()
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, (float, int))}))
    
    save(policy, "checkpoint_final")

    info = evaluate(render=cfg.eval_render)
    info["env_frames"] = collector._frames
    run.log(info)

    try:
        path = os.path.join(run.dir, f"policy.pt")
        _policy = policy.get_rollout_policy("eval").cpu()
        torch.save(_policy, path)
        logging.info(F"Export policy to {path}")
    except Exception as e:
        print(f"Cannot save policy due to {e}")

    wandb.finish()
    exit(0)
    
    base_env.close()
    simulation_app.close()
    exit(0)


if __name__ == "__main__":
    main()

