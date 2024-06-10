import torch
import os.path as osp
import hydra
import time
from tqdm import tqdm
from pprint import pprint
from collections import OrderedDict

from omegaconf import OmegaConf, DictConfig

from active_adaptation.utils.torchrl import SyncDataCollector
from active_adaptation.learning import ALGOS

from omni.isaac.lab.app import AppLauncher

from torchrl.envs.transforms import (
    TransformedEnv, 
    Compose, 
    InitTracker,
    VecNorm
)

from scripts.helpers import EpisodeStats

def make_env_agent(cfg):
    from active_adaptation.envs.isaac_lab import IsaacLabEnv, IsaacLabEnvCfg
    
    env_cfg = IsaacLabEnvCfg(**cfg.env)
    env = IsaacLabEnv(env_cfg)
    transform = [
        InitTracker(),
        VecNorm(["policy"])
    ] 
    transform = Compose(*transform)

    env = TransformedEnv(env, transform)

    agent = ALGOS[cfg.algo.name](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=env.device
    )

    if cfg.get("checkpoint") is not None:
        checkpoint = torch.load(cfg.checkpoint)
        agent.load_state_dict(checkpoint["agent"])
        env.transform.load_state_dict(checkpoint.get("transform"))

    return env, agent


@hydra.main(config_path="../cfg", config_name="play_lab")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    env, agent = make_env_agent(cfg)

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k.startswith("Episode")
    ]
    episode_stats = EpisodeStats(stats_keys)

    policy = agent.get_rollout_policy("eval")
    td_ = env.reset()
    
    try:
        while True:
            policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            episode_stats.add(td)

            if len(episode_stats) > env.num_envs:
                info = {}
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    info[k] = torch.mean(v.float()).item()
                
                print(info)
   
    except KeyboardInterrupt:    
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()