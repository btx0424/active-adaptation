import torch
import math
from omni.isaac.orbit.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg

COMMAND_CFG = UniformVelocityCommandCfg(
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.02,
    rel_heading_envs=1.0,
    heading_command=True,
    debug_vis=False,
    ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    ),
)
