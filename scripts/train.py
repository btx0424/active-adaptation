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
from setproctitle import setproctitle

from isaaclab.app import AppLauncher
import active_adaptation.learning
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, StepCounter

# local import

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

    from active_adaptation.envs import TASKS
    from configs.rough import LocomotionEnvCfg

    env_cfg = LocomotionEnvCfg(cfg.task)
    base_env = TASKS[cfg.task.task](env_cfg)
    transform = Compose(InitTracker(), StepCounter())
    env = TransformedEnv(base_env, transform)
    env.set_seed(cfg.seed)

    policy_cls = hydra.utils.get_class(cfg.algo._target_)
    policy = policy_cls(
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec,
        device=base_env.device
    )
    if cfg.checkpoint_path is not None:
        state_dict = torch.load(cfg.checkpoint_path)
        policy.load_state_dict(state_dict)

    policy.learn(env, cfg)
    
    wandb.finish()
    base_env.close()
    simulation_app.close()
    return

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")


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
        
        info = {}

        episode_stats.add(data)

        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
            info.update(env.extra)
        
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
    info, trajs, stats = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
    info["env_frames"] = collector._frames
    run.log(info)

    exit(0)
    
    exit(0)


if __name__ == "__main__":
    main()

