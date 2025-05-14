import torch
import hydra
import numpy as np
import einops
import itertools
import os
import datetime
from omegaconf import OmegaConf

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictSequential

import active_adaptation
from active_adaptation.utils.export import export_onnx


@hydra.main(config_path="../cfg", config_name="play", version_base=None)
def main(cfg):
    active_adaptation.set_backend("mujoco")
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    from scripts.helpers import EpisodeStats, make_env_policy, ObsNorm
    # TODO: maybe implement these
    cfg.task.randomization = {}
    cfg.task.termination = {}
    cfg.task.reward = {}
    env, policy, vecnorm = make_env_policy(cfg)
    
    if cfg.export_policy:
        import time
        import copy
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        fake_input = env.observation_spec[0].rand().cpu()
        fake_input["is_init"] = torch.tensor(1, dtype=bool)
        fake_input["context_adapt_hx"] = torch.zeros(128)
        fake_input = fake_input.unsqueeze(0)

        def test(m, x):
            start = time.perf_counter()
            for _ in range(1000):
                m(x)
            return (time.perf_counter() - start) / 1000
        
        FILE_PATH = os.path.dirname(__file__)
        
        deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy"))
        obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy_policy.in_keys)
        _policy = TensorDictSequential(obs_norm, deploy_policy).cpu()
        
        print(f"Inference time of policy: {test(_policy, fake_input)}")

        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        os.makedirs(os.path.join(FILE_PATH, "exports", cfg.task.name), exist_ok=True)
        path = os.path.join(FILE_PATH, "exports", cfg.task.name, f"policy-{time_str}.pt")
        torch.save(_policy, path)

        meta = {}
        meta["action_scaling"] = dict(cfg.task.action.get("action_scaling"))
        meta["stiffness"] = dict(cfg.task.robot.stiffness)
        meta["damping"] = dict(cfg.task.robot.damping)
        meta["effort_limit"] = dict(cfg.task.robot.effort_limit)
        export_onnx(_policy, fake_input, path.replace(".pt", ".onnx"), meta)
    
    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    policy = policy.get_rollout_policy("eval")

    env.base_env.eval()
    td_ = env.reset()
    
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            try:
                td_ = policy(td_)
                td, td_ = env.step_and_maybe_reset(td_)
                episode_stats.add(td)

                if len(episode_stats) >= env.num_envs:
                    print("Step", i)
                    for k, v in sorted(episode_stats.pop().items(True, True)):
                        print(k, torch.mean(v).item())
            except KeyboardInterrupt:
                break
    
    env.dump()    
    env.close()


if __name__ == "__main__":
    main()

