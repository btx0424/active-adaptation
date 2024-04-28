import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict

api = wandb.Api()

plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3)

run_paths = {
    "Go2": [
        "btx0424/adaptation/sfu1wfts",
        "btx0424/adaptation/y6ew7ims"
    ],
    "H1": [
        "btx0424/adaptation/sfu1wfts",
        "btx0424/adaptation/y6ew7ims"
    ],
}

runs = defaultdict(list)

for robot, paths in run_paths.items():
    for run_path in paths:
        run = api.run(run_path)
        runs[robot].append(run.history())

axes[0].set_ylabel("Episode Return (Normalized)")
for (robot, runs), axis in zip(runs.items(), axes):
    env_frames = runs[0]["env_frames"]
    print(env_frames.shape)
    axis.plot(env_frames, runs[0]["train/stats.return"][:len(env_frames)], label="")
    axis.plot(env_frames, runs[1]["train/stats.return"][:len(env_frames)], label="")
    axis.set_xlabel("Environment Frames")
axes[-1].legend()

fig.suptitle("Ablation of Choices for Privileged Observations")
fig.tight_layout()
fig.savefig("choice.pdf")