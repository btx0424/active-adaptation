import torch
import wandb
import os
import sys
import hydra
import argparse

from omegaconf import OmegaConf
from omni.isaac.lab.app import AppLauncher
from scripts.play import main as play_main
from scripts.eval import main as eval_main

play = play_main.__wrapped__
eval = eval_main.__wrapped__

FILE_PATH = os.path.dirname(__file__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_path", type=str)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("-p", "--play", action="store_true", default=False)
    # whether to override terrain and command
    parser.add_argument("-t", "--terrain", action="store_true", default=False)
    parser.add_argument("-c", "--command", action="store_true", default=False)
    
    parser.add_argument("-e", "--export", action="store_true", default=False)
    parser.add_argument("-v", "--video", action="store_true", default=False)
    parser.add_argument("-i", "--interations", type=int, default=None)
    args = parser.parse_args()

    api = wandb.Api()
    
    run = api.run(args.run_path)
    print(f"Loading run {run.name}")

    root = os.path.join(os.path.dirname(__file__), "wandb", run.name)
    os.makedirs(root, exist_ok=True)

    checkpoints = []
    for file in run.files():
        print(file.name)
        if "checkpoint" in file.name:
            checkpoints.append(file)
        elif file.name == "files/cfg.yaml":
            file.download(root, replace=True)
        elif file.name == "config.yaml":
            file.download(root, replace=True)
    
    if args.interations is None:
        def sort_by_time(file):
            number_str = file.name[:-3].split("_")[-1]
            if number_str == "final":
                return 100000
            else:
                return int(number_str)

        checkpoints.sort(key=sort_by_time)
        checkpoint = checkpoints[-1]
    else:
        for file in checkpoints:
            if file.name == f"checkpoint_{args.interations}.pt":
                checkpoint = file
                break
    print(f"Downloading {checkpoint.name}")
    checkpoint.download(root, replace=True)

    # `run.config` does not preserve order of the keys
    # so we need to manually load the config file :(
    # if os.path.exists(os.path.join(root, "config.yaml")):
    #     cfg = OmegaConf.load(os.path.join(root, "config.yaml"))
    #     for k, v in run.config.items():
    #         cfg[k] = cfg[k]["value"]
    # else:
    cfg = OmegaConf.load(os.path.join(root, "files", "cfg.yaml"))
    OmegaConf.set_struct(cfg, False)

    cfg["checkpoint_path"] = os.path.join(root, checkpoint.name)
    cfg["vecnorm"] = "eval"
    cfg["algo"]["phase"] = "adapt"
    # cfg['algo']["phase"] = "finetune"

    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="eval", version_base=None):
            _cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        # cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["reward"] = _cfg.task.reward
        if args.terrain:
            cfg["task"]["terrain"] = _cfg.task.terrain
        if args.command:
            cfg["task"]["command"] = _cfg.task.command
    
    if args.play:
        cfg["app"]["headless"] = False
        cfg["task"]["num_envs"] = 16
        cfg["export_policy"] = args.export
        play(cfg)
    else:
        if args.video:
            cfg["task"]["num_envs"] = 16
            cfg["eval_render"] = True
            cfg["app"]["enable_cameras"] = True
            cfg["app"]["headless"] = False
        eval(cfg)


if __name__ == "__main__":
    main()