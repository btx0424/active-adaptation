from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm

import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.envs import RLTaskEnv

import torch


def joint_torques(env: RLTaskEnv):
    asset: Articulation = env.scene["robot"]
    base_legs = asset.actuators["base_legs"]
    return base_legs.applied_effort / base_legs.effort_limit


def joint_params(env: RLTaskEnv):
    asset: Articulation = env.scene["robot"]
    base_legs = asset.actuators["base_legs"]
    return torch.cat([base_legs.stiffness, base_legs.damping], dim=-1) 


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_quat = ObsTerm(func=mdp.base_quat)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

    @configclass
    class PrivCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        feet_pos = ObsTerm(func=mdp.body_pos_rel)
        joint_acc = ObsTerm(func=mdp.joint_acc, scale=0.01)
        joint_torques = ObsTerm(func=joint_torques)


    # observation groups
    policy: PolicyCfg = PolicyCfg(concatenate_terms=True)
    priv: PrivCfg = PrivCfg(concatenate_terms=True)

