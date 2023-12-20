import hydra
import os

from tqdm import tqdm
from pprint import pprint

from omni.isaac.orbit.app import AppLauncher
from torchrl.collectors import SyncDataCollector

from orbit import OrbitEnv
from helpers import EpisodeStats, Every
from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    RewardSum, 
    Compose
)

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
    # load cheaper kit config in headless
    if cfg.headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    app = AppLauncher(
        {"headless": cfg.headless}, 
        experience=app_experience
    )

    import omni.isaac.orbit_tasks  # noqa: F401
    from omni.isaac.orbit_tasks.utils import parse_env_cfg

    # task_name = "Isaac-Velocity-Rough-Unitree-Go2-v0"
    task_name = "Isaac-Velocity-Flat-Unitree-Go2-v0"
    # task_name = "Isaac-Velocity-Flat-Unitree-A1-v0"
    env_cfg = parse_env_cfg(task_name, use_gpu=True, num_envs=2048)
    
    transform = Compose(InitTracker(), RewardSum())
    env = OrbitEnv(task_name, cfg=env_cfg)
    env = TransformedEnv(env, transform)

    policy = PPOPolicy(
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec, 
        device=env.device
    )

    episode_stats = EpisodeStats(["episode_reward"])
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=env.num_envs * 32,
        total_frames=-1,
        device=env.device,
        return_same_td=True
    )

    for i, data in tqdm(enumerate(collector)):
        episode_stats.add(data)

        info = policy.train_op(data)
        if len(episode_stats) >= env.num_envs:
            for k, v in episode_stats.pop().items(True, True):
                info[k] = v.mean().item()

        pprint(info)

    env.close()


if __name__ == "__main__":
    main()
