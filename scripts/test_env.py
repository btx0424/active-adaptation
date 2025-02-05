import torch
import torchvision
# import warp
import hydra
import numpy as np
import einops
import wandb
import logging
import os
import time
import datetime

from omegaconf import OmegaConf, DictConfig
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


def log_video(env, it, render_interval, render_decimation):
    if it == 0 or it - env.last_recording_it >= render_interval:
        env.start_recording(render_decimation)
        env.last_recording_it = it

    frames = env.get_complete_frames()
    if len(frames) > 0:
        env.pause_recording()
        video_array = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
        video_tensor = torch.from_numpy(video_array)

        run_dir = wandb.run.dir
        video_path = os.path.join(run_dir, f"video_{it}.mp4")
        torchvision.io.write_video(
            video_path,
            video_tensor.permute(0, 2, 3, 1),  # Change to (T, H, W, C) format
            fps=1 / env.step_dt / env.render_decimation
        )
        
        wandb.log({"video": wandb.Video(video_path)}, step=it)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")

@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    run = wandb.init(
        job_type=cfg.wandb.job_type,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
    )
    run.config.update(OmegaConf.to_container(cfg))
    
    default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    run_idx = run.name.split("-")[-1]
    run.name = f"{run_idx}-{default_run_name}"
    setproctitle(run.name)

    cfg_save_path = os.path.join(run.dir, "cfg.yaml")
    OmegaConf.save(cfg, cfg_save_path)
    run.save(cfg_save_path, policy="now")
    run.save(os.path.join(run.dir, "config.yaml"), policy="now")

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
    render_interval = cfg.get("render_interval", -1)
    render_decimation = cfg.get("render_decimation", 1)
    save_interval = cfg.get("save_interval", -1)

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    rollout_policy = policy.get_rollout_policy("train")
    compile_policy = cfg.get("compile", False)
    assert compile_policy in (True, False, "auto")
    if compile_policy or compile_policy == "auto":
        fake_td = env.fake_tensordict()
        rollout_policy_compiled = torch.compile(rollout_policy)
        for _ in range(16): 
            rollout_policy_compiled(fake_td)
    if compile_policy == "auto":
        @torch.inference_mode()
        def _timeit(policy):
            start = time.perf_counter()
            for _ in range(128): 
                policy(fake_td)
            return (time.perf_counter() - start) / 128
        inference_time = _timeit(rollout_policy)
        inference_time_compiled = _timeit(rollout_policy_compiled)
        print(f"Inference time: {inference_time:.4f} -> {inference_time_compiled:.4f}")
        if inference_time_compiled < inference_time:
            rollout_policy = rollout_policy_compiled
            print("Using compiled policy")

    collector = SyncDataCollector(
        env,
        policy=rollout_policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )
    
    def save(policy, checkpoint_name: str, artifact: bool=False):
        ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
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
        run.save(ckpt_path, policy="now", base_path=run.dir)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    pbar = tqdm(collector, total=total_iters)
    
    for i, data in enumerate(pbar):
        start = time.perf_counter()
        
        info = {}

        episode_stats.add(data)

        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        
        info.update(policy.train_op(data))
        info.update(env.extra)
        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters)

        info["env_frames"] = collector._frames
        info["rollout_fps"] = collector._fps
        info["training_time"] = time.perf_counter() - start
        
        if save_interval > 0  and i % save_interval == 0:
            save(policy, f"checkpoint_{i}")

        if render_interval > 0:
            log_video(env, i, render_interval, render_decimation)

        run.log(info)

        print()
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, (float, int))}))
    
    save(policy, "checkpoint_final")

    policy_eval = policy.get_rollout_policy("eval")
    info, trajs, stats = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
    info["env_frames"] = collector._frames
    run.log(info)

    wandb.finish()
    exit(0)
    
    base_env.close()
    simulation_app.close()
    exit(0)


if __name__ == "__main__":
    main()

