import torch
import torchvision
import hydra
import numpy as np
import wandb
import logging
import os
import time
import datetime

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

import active_adaptation as aa
from isaaclab.app import AppLauncher
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModuleBase

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")


@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    print(f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}")
    app_launcher = AppLauncher(
        OmegaConf.to_container(cfg.app),
        distributed=aa.is_distributed(),
        device=f"cuda:{aa.get_local_rank()}"
    )
    simulation_app = app_launcher.app

    if aa.is_main_process():
        run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        run.config.update(OmegaConf.to_container(cfg))
        run.config["world_size"] = aa.get_world_size()
        
        default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        run_idx = run.name.split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"
        setproctitle(run.name)

        cfg_save_path = os.path.join(run.dir, "cfg.yaml")
        OmegaConf.save(cfg, cfg_save_path)
        run.save(cfg_save_path, policy="now")
        run.save(os.path.join(run.dir, "config.yaml"), policy="now")

    from helpers import make_env_policy, EpisodeStats, evaluate, check_vecnorm_state
    env, policy, vecnorm = make_env_policy(cfg)

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    rollout_policy = policy.get_rollout_policy("train")

    def save(checkpoint_name: str, best_ckpt_path: str, best_ckpt_metric: float):
        """
        Save the checkpoint and update the best checkpoint if the current metric is better.
        If the current metric is worse than the best checkpoint by 20%, recover to the best checkpoint.
        
        Args:
            checkpoint_name: The name of the checkpoint.
            best_ckpt_path: The path of the best checkpoint.
            best_ckpt_metric: The metric of the best checkpoint.

        Returns:
            The path of the best checkpoint and the metric of the best checkpoint.
        """
        ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
        metric = getattr(policy, "metric", best_ckpt_metric) # maximize

        if metric < best_ckpt_metric * 0.8 and best_ckpt_path is not None:
            # recover to the best checkpoint
            ckpt = torch.load(best_ckpt_path, weights_only=False)
            policy.load_state_dict(ckpt["policy"])
            
            if "vecnorm" in ckpt.keys():
                vecnorm.load_state_dict(ckpt["vecnorm"])
            
            logging.info(f"Recovered to the best checkpoint {best_ckpt_path}")
            return best_ckpt_path, best_ckpt_metric

        if metric > best_ckpt_metric:
            best_ckpt_metric = metric
            best_ckpt_path = ckpt_path
        
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
        if "vecnorm" in locals():
            state_dict["vecnorm"] = vecnorm.state_dict()
        torch.save(state_dict, ckpt_path)
        run.save(ckpt_path, policy="now", base_path=run.dir)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
        return best_ckpt_path, max(metric, best_ckpt_metric)

    assert env.training
    if aa.is_main_process():
        progress = tqdm(range(total_iters))
    else:
        progress = range(total_iters)
    
    def should_save(i):
        if not aa.is_main_process():
            return False
        return i > 0 and i % save_interval == 0
    
    carry = env.reset()
    rollout_policy: TensorDictModuleBase = policy.get_rollout_policy("train")
    amp_enabled = cfg.get("amp", False)

    @torch.inference_mode()
    @set_exploration_type(ExplorationType.RANDOM)
    def collect(carry):
        torch.cuda.empty_cache()
        data = []
        with torch.autocast("cuda", enabled=amp_enabled):
            for _ in range(cfg.algo.train_every):
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                td["next"] = td["next"].exclude(*rollout_policy.in_keys)

                private_keys = [key for key in td.keys(True, True) if isinstance(key, str) and key.startswith('_')]
                td = td.exclude(*private_keys)
                
                data.append(td.to(policy.device))
            data = torch.stack(data, dim=1)
            if (values := data.get("state_value")) is None:
                policy.critic(data)
                values = data["state_value"]
            data["next", "state_value"] = torch.where(
                data["next", "done"],
                values, # a walkaround to avoid storing the next states
                torch.cat([values[:, 1:], policy.critic(carry.copy())["state_value"].unsqueeze(1)], dim=1)
            )
        return data, carry
    
    best_ckpt_path = None
    best_ckpt_metric = 0 # maximize

    env_frames = 0
    for i in progress:
        rollout_start = time.perf_counter()
        data, carry = collect(carry)
        loc_diff, scale_diff = check_vecnorm_state(vecnorm)
        rollout_time = time.perf_counter() - rollout_start

        episode_stats.add(data)
        env_frames += data.numel()

        info = {}
        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        training_start = time.perf_counter()
        info.update(policy.train_op(data))
        training_time = time.perf_counter() - training_start
        info.update(env.extra)
        info.update(env.stats_ema) # step-wise exponential moving average of stats
        
        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters)
        
        info["env_frames"] = env_frames * aa.get_world_size()
        info["performance/rollout_fps"] = data.numel() / rollout_time * aa.get_world_size()
        info["performance/training_time"] = training_time
        info["performance/iter_time"] = (time.perf_counter() - rollout_start)
        info["debug/max_loc_diff"] = max(loc_diff)
        info["debug/max_scale_diff"] = max(scale_diff)
        info["debug/mean_loc_diffs"] = sum(loc_diff) / len(loc_diff)
        info["debug/mean_scale_diffs"] = sum(scale_diff) / len(scale_diff)
        if should_save(i):
            best_ckpt_path, best_ckpt_metric = save(f"checkpoint_{i}", best_ckpt_path, best_ckpt_metric)

        if aa.is_main_process():
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, (float, int))}))
            run.log(info)
    
    if aa.is_main_process():
        save(policy, "checkpoint_final")

        policy_eval = policy.get_rollout_policy("eval")
        info, trajs, stats = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
        run.log(info)

    wandb.finish()
    exit(0)
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

