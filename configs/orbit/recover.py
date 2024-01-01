from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

from omni.isaac.orbit_tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from omni.isaac.orbit.assets.config.unitree import UNITREE_GO2_CFG  # isort: skip
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup

import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
import gymnasium as gym
import torch
import einops
from omni.isaac.orbit.envs import RLTaskEnv


def upright_l2(env: RLTaskEnv):
    asset = env.scene.articulations["robot"]
    return - (-1. - asset.data.projected_gravity_b[:, 2]).square()


def upright_l1(env: RLTaskEnv):
    asset = env.scene.articulations["robot"]
    return 1 - asset.data.projected_gravity_b[:, 2]


def base_height_l1(env: RLTaskEnv):
    asset = env.scene.articulations["robot"]
    return asset.data.root_pos_w[:, 2] / 0.3
    return (asset.data.root_pos_w[:, 2] - 0.30).clamp_max(0.) / 0.3


def com(env: RLTaskEnv):
    asset = env.scene.articulations["robot"]
    com = asset.data.root_pos_w
    feet_center = asset.data.body_pos_w[:, [4, 8, 14, 18]].mean(dim=1)
    error = ((com[:, :2] - feet_center[:, :2]) / 0.15).square().sum(dim=1)
    return torch.exp(- error) * (asset.data.projected_gravity_b[:, 2] < 0.).float()


def joint_deviation_l2(env: RLTaskEnv):
    asset = env.scene.articulations["robot"]
    err = (asset.data.joint_pos - asset.data.default_joint_pos).square().sum(dim=-1)
    return err


def joint_vel_l1(env: RLTaskEnv):
    asset = env.scene.articulations["robot"]
    return asset.data.joint_vel.abs().sum(dim=-1)


def undesired_contact(env: RLTaskEnv):
    sensor: ContactSensor = env.scene["contact_forces"]
    contact_forces = sensor.data.net_forces_w_history.clone()
    contact_forces[:, :, [4, 8, 14, 18]] = 0.
    undesired_contact_forces = (contact_forces.norm(dim=-1) > 1.0).float()
    return - undesired_contact_forces.sum(dim=1).any(1).float()


def feet_contact(env: RLTaskEnv):
    sensor: ContactSensor = env.scene["contact_forces"]
    contact_forces = sensor.data.net_forces_w_history.clone()
    contact_forces_feet = contact_forces[:, :, [4, 8, 14, 18]].norm(dim=-1)
    return (contact_forces_feet > 1.0).float().sum(dim=(1, 2))


def reset_joints_uniform(env: RLTaskEnv, env_ids: torch.Tensor, eps=.05):
    asset = env.scene.articulations["robot"]
    lower, higher = asset.data.soft_joint_pos_limits[env_ids].unbind(-1)

    joint_pos = torch.rand(lower.shape, device=lower.device) * (higher - lower)
    joint_pos = joint_pos.clamp(lower+eps, higher-eps)
    joint_vel = torch.zeros_like(joint_pos)
    
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def time(env: RLTaskEnv): # obs
    if hasattr(env, "episode_length_buf"):
        t = (env.episode_length_buf / env.max_episode_length) * 2. - 1.
        return einops.repeat(t, "n -> n 4")
    else:
        return torch.zeros(env.num_envs, 4, device=env.device)


def root_quat(env: RLTaskEnv): # obs
    asset = env.scene.articulations["robot"]
    return asset.data.root_quat_w


@configclass
class RewardsCfg:
    
    upright = RewTerm(func=upright_l1, weight=2.0)
    base_height = RewTerm(func=base_height_l1, weight=1.0)
    com = RewTerm(func=com, weight=2.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # joint_pos = RewTerm(func=joint_deviation_l2, weight=-0.1)
    # joint_vel = RewTerm(func=joint_vel_l1, weight=-0.001)
    feet_contact = RewTerm(func=feet_contact, weight=0.2)
    undesired_contact = RewTerm(func=undesired_contact, weight=0.2)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        feet_pos = ObsTerm(func=mdp.body_pos_rel)
        time = ObsTerm(func=time)
        root_quat = ObsTerm(func=root_quat)

    # observation groups
    policy: PolicyCfg = PolicyCfg(concatenate_terms=True)


@configclass
class UnitreeGo2RecoveryEnvCfg(LocomotionVelocityRoughEnvCfg):
    
    rewards = RewardsCfg()
    observations = ObservationsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 4.

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # reduce action scale
        self.actions.joint_pos.scale = 0.5

        self.randomization.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), 
                "roll": (-1., 1.),
                "pitch": (-0.5, 0.5),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-1., 1.),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.randomization.reset_robot_joints = RandTerm(
            func=reset_joints_uniform,
            mode="reset",
        )
        self.terminations.base_contact = None


gym.register(
    id="Go2-Recovery",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnitreeGo2RecoveryEnvCfg,
    },
)