import torch
import hydra
import numpy as np
import einops
import time
import sys
from tqdm import tqdm
from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

from active_adaptation.learning import ALGOS

import wandb
import logging
from tqdm import tqdm
from scripts.helpers import make_env_policy, evaluate

import os
import datetime
import termcolor

@hydra.main(config_path="../cfg", config_name="eval", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    env, agent, vecnorm = make_env_policy(cfg)
    
    keys = [
        ("next", "stats"),
        ("next", "done"), 
        ("next", "reward"),
        "value_obs",
        "value_priv",
        "value_adapt",
        "context_expert",
        "context_scale",
        "context_adapt",
        "context_adapt_scale",
        "action_kl",
    ]
    
    policy_eval = agent.get_rollout_policy("eval")
    info, trajs, stats = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed, keys=keys)
    
    print(termcolor.colored(trajs, "light_yellow"))
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    path = os.path.join(os.path.dirname(__file__), f"trajs-{time_str}.pt")
    torch.save(trajs, path)

    path = os.path.join(os.path.dirname(__file__), f"stats-{time_str}.pt")
    torch.save(stats, path)

    info["task"] = cfg.task.name
    info["algo"] = cfg.algo.name
    info["checkpoint_path"] = cfg.checkpoint_path
    info["argv"] = sys.argv
    print(OmegaConf.to_yaml(info))
    
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    dir_path = os.path.join(os.path.dirname(__file__), f"eval", cfg.task.name)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"{cfg.task.name}-{time_str}.yaml")
    with open(path, "w") as f:
        OmegaConf.save(info, f)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

