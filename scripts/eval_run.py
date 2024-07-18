import torch
import wandb
import os
import sys
import hydra
import argparse

from omegaconf import OmegaConf
from omni.isaac.lab.app import AppLauncher
from scripts.helpers import make_env_policy, evaluate

FILE_PATH = os.path.dirname(__file__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--play", action="store_true", default=False)
    args = parser.parse_args()

    api = wandb.Api()
    
    run = api.run(args.run_path)
    print(f"Loading run {run.name}")

    root = os.path.dirname(__file__)
    checkpoints = []
    for file in run.files():
        print(file.name)
        if "checkpoint" in file.name:
            checkpoints.append(file)
        elif file.name == "files/cfg.yaml":
            file.download(root, replace=True)
        elif file.name == "config.yaml":
            file.download(root, replace=True)
    checkpoint = checkpoints[-1]
    print(f"Downloading {checkpoint.name}")
    checkpoint.download(replace=True)
    
    # `run.config` does not preserve order of the keys
    # so we need to manually load the config file :(
    if os.path.exists(os.path.join(root, "config.yaml")):
        cfg = OmegaConf.load(os.path.join(root, "config.yaml"))
        for k, v in run.config.items():
            cfg[k] = cfg[k]["value"]
    else:
        cfg = OmegaConf.load(os.path.join(root, "files", "cfg.yaml"))
    OmegaConf.set_struct(cfg, False)

    cfg["checkpoint_path"] = os.path.join(os.path.dirname(__file__), checkpoint.name)
    cfg["vecnorm"] = "eval"
    # cfg["algo"]["phase"] = "adapt"

    if args.play:
        cfg["app"]["headless"] = False
        cfg["task"]["num_envs"] = 16
    else:
        cfg["task"]["num_envs"] = 2048

    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="eval", version_base=None):
            _cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["reward"] = _cfg.task.reward
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    env, agent, vecnorm = make_env_policy(cfg)

    policy_eval = agent.get_rollout_policy("eval")
    
    keys = [
        ("next", "stats"),
        ("next", "done"), 
        ("next", "reward"),
    ]
    info, trajs, stats = evaluate(env, policy_eval, seed=cfg.seed, keys=keys)
    
    path = os.path.join(os.path.dirname(__file__), f"{run.name}.pt")
    torch.save(stats, path)

    print(OmegaConf.to_yaml(info))
    with open(os.path.join(os.path.dirname(__file__), f"{run.name}.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(info))

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()