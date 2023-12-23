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

from orbit import OrbitEnv
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
        {"headless": cfg.headless, "offscreen_render": True}, 
        experience=app_experience
    )

    import omni.isaac.orbit_tasks  # noqa: F401
    from configs import UnitreeGo2RecoveryEnvCfg
    from omni.isaac.orbit_tasks.utils import parse_env_cfg

    # task_name = "Isaac-Velocity-Rough-Unitree-Go2-v0"
    # task_name = "Isaac-Velocity-Flat-Unitree-Go2-v0"
    # task_name = "Isaac-Velocity-Flat-Unitree-A1-v0"
    task_name = "Go2-Recovery"
    env_cfg = parse_env_cfg(task_name, use_gpu=True, num_envs=cfg.task.env.num_envs)
    
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
    env = OrbitEnv(task_name, cfg=env_cfg)

    transform = Compose(InitTracker(), RewardSum())
    reader = default_info_dict_reader(
        ["episode_len"], 
        [UnboundedDiscreteTensorSpec([env.num_envs, 1], device=env.device)]
    )

    env.set_info_dict_reader(reader)
    env = TransformedEnv(env, transform)

    policy = PPOPolicy(
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec, 
        device=env.device
    )

    run = init_wandb(cfg)

    episode_stats = EpisodeStats(["episode_reward", "episode_len"])
    
    eval_interval = cfg.get("eval_interval", -1)
    eval_render = cfg.get("eval_render", False)
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

    @torch.no_grad()
    def evaluate(
        seed: int=0, 
        exploration_type: ExplorationType=ExplorationType.MODE,
        render=False,
        expert=False,
    ):
        frames = []

        env.eval()
        env.set_seed(seed)
        policy.eval()

        if expert and isinstance(policy, PPODualPolicy):
            policy.mode = "expert"

        from tqdm import tqdm
        t = tqdm(total=env.max_episode_length, desc="Evaluating")
        def record_frame(*args, **kwargs):
            frame = env.render()
            frames.append(frame)
            t.update(2)
        
        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=env.max_episode_length,
                policy=policy,
                callback=Every(record_frame, 2) if render else None,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        info = {}
        for key in episode_stats.in_keys:
            info[key] = take_first_episode(trajs["next"][key]).float().mean().item()

        # log video
        if len(frames):
            video_array = einops.rearrange(np.stack(frames), "t h w c -> t c h w")
            frames.clear()
            info["recording"] = wandb.Video(
                video_array, 
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), 
                format="mp4"
            )

        return info
    
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

    t = tqdm(collector, total=total_frames//frames_per_batch)
    for i, data in enumerate(t):
        episode_stats.add(data)

        info = policy.train_op(data)
        if len(episode_stats) >= env.num_envs:
            info_log = {}
            for k, v in episode_stats.pop().items(True, True):
                info_log[k] = v.mean().item()
            info["train"] = info_log
        
        if eval_interval > 0 and (i + 1) % eval_interval == 0:
            info["eval"] = evaluate(render=eval_render)
        
        info["env_frames"] = collector._frames
        info["rollout_fps"] = collector._fps
        info["extras"] = find_numerics(env.base_env.extras)

        run.log(info)

        print()
        print(OmegaConf.to_yaml(find_numerics(info)))

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)
        artifact = wandb.Artifact(
            f"{type(env).__name__}-{type(policy).__name__}", 
            type="model"
        )
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
    
    env.close()


if __name__ == "__main__":
    main()
