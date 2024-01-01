import hydra
import os
import wandb
import torch
import logging
import numpy as np
import einops

from tqdm import tqdm
from omegaconf import OmegaConf
from omni.isaac.orbit.app import AppLauncher

from helpers import EpisodeStats, Every

from torchrl.envs.gym_like import default_info_dict_reader
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    RewardSum, 
    Compose
)
from torchrl.data import UnboundedDiscreteTensorSpec

from active_adaptation.utils.torchrl import SyncDataCollector
from active_adaptation.utils.wandb import init_wandb
from active_adaptation.learning import (
    PPOPolicy, 
    PPORNNPolicy, 
    PPODualPolicy, 
    PPOTConvPolicy, 
    PPORMAPolicy
)

policies = {
    "ppo": PPOPolicy,
    "ppo_dual": PPODualPolicy,
    "ppo_rnn": PPORNNPolicy,
    "ppo_tconv": PPOTConvPolicy,
    "ppo_rma": PPORMAPolicy
}


@hydra.main(config_path="cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)

    # load cheaper kit config in headless
    if cfg.headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    app = AppLauncher(
        {"headless": cfg.headless, "offscreen_render": cfg.offscreen_render}, 
        experience=app_experience
    )

    import omni.isaac.orbit_tasks  # noqa: F401
    from configs.orbit import UnitreeGo2RecoveryEnvCfg
    from configs.rough import ROUGH_EASY
    from omni.isaac.orbit_tasks.utils import parse_env_cfg

    # task_name = "Isaac-Velocity-Rough-Unitree-Go2-v0"
    task_name = "Isaac-Velocity-Flat-Unitree-Go2-v0"
    # task_name = "Isaac-Velocity-Flat-Unitree-A1-v0"
    # task_name = "Go2-Recovery"
    env_cfg = parse_env_cfg(task_name, use_gpu=True, num_envs=cfg.task.env.num_envs)
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = ROUGH_EASY
    env_cfg.scene.terrain.curriculum = False

    if cfg.get("camera_follow", False):
        def follow(env):
            asset = env.scene["robot"]
            pos = asset.data.root_pos_w[0].cpu()
            return torch.as_tensor(env.cfg.viewer.eye) + pos, pos
        env_cfg.viewer.func = follow

    if hasattr(env_cfg.scene, "height_scanner"):
        env_cfg.scene.height_scanner = None
        env_cfg.observations.policy.height_scan = None
    
    # create TorchRL environment
    from orbit import OrbitEnv
    env = OrbitEnv(task_name, cfg=env_cfg)
    episode_stats = EpisodeStats(["episode_reward", *env.info_spec.keys(True, True)])

    transform = Compose(InitTracker(), RewardSum())
    env = TransformedEnv(env, transform)

    policy = policies[cfg.algo.name](
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec, 
        device=env.device
    )
    
    total_frames = cfg.get("total_frames", -1)
    frames_per_batch = env.num_envs * 32

    # create collector
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=env.device,
        return_same_td=True
    )
    
    def find_numerics(source: dict):
        result = {}
        for k, v in source.items():
            if isinstance(v, dict):
                children = find_numerics(v)
                if children:
                    result[k] = children
            elif isinstance(v, float):
                result[k] = v
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                result[k] = v.item()
        return result

    t = tqdm(collector)
    for i, data in enumerate(t):
        episode_stats.add(data)

        info = policy.train_op(data)
        if len(episode_stats) >= env.num_envs:
            info_log = {}
            for k, v in episode_stats.pop().items(True, True):
                if isinstance(k, tuple):
                    k = "/".join(k)
                info_log[k] = v.mean().item()
            info["train"] = info_log
        
        info["env_frames"] = collector._frames
        info["rollout_fps"] = collector._fps
        info["extras"] = find_numerics(env.base_env.extras)

        print()
        print(OmegaConf.to_yaml(find_numerics(info)))
    
    env.close()


if __name__ == "__main__":
    main()
