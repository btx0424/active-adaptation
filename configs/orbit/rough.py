from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm

import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.envs import RLTaskEnv, BaseEnv

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg

import torch
import gymnasium as gym

from omni.isaac.orbit_tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from omni.isaac.orbit.assets.config.unitree import UNITREE_GO2_CFG  # isort: skip


def joint_torques(env: RLTaskEnv):
    asset: Articulation = env.scene["robot"]
    base_legs = asset.actuators["base_legs"]
    return base_legs.applied_effort / base_legs.effort_limit


def joint_params(env: RLTaskEnv):
    asset: Articulation = env.scene["robot"]
    base_legs = asset.actuators["base_legs"]
    return torch.cat([base_legs.stiffness, base_legs.damping], dim=-1) 


def base_quat(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root orientation in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_quat = ObsTerm(func=base_quat)
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
        joint_params = ObsTerm(func=joint_params)


    # observation groups
    policy: PolicyCfg = PolicyCfg(concatenate_terms=True)
    priv: PrivCfg = PrivCfg(concatenate_terms=True)


def motor_params(env: BaseEnv, env_ids: torch.Tensor):
    asset: Articulation = env.scene["robot"]
    base_legs = asset.actuators["base_legs"]
    if not hasattr(base_legs, "default_stiffness"):
        base_legs.default_stiffness = base_legs.stiffness.clone()
        base_legs.default_damping = base_legs.damping.clone()
    base_legs.stiffness[env_ids] = sample_around(base_legs.default_stiffness[env_ids])
    base_legs.damping[env_ids] = sample_around(base_legs.default_damping[env_ids])


def sample_around(x: torch.Tensor, low: float=0.8, high: float=1.2):
    scale = torch.rand(x.shape, device=x.device) * (high - low) + low
    return x * scale


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        # self.sim.dt = 0.01
        self.decimation = 2

        # randomization
        self.randomization.push_robot = None
        self.randomization.add_base_mass.params["mass_range"] = (-1.0, 3.0)
        self.randomization.add_base_mass.params["asset_cfg"].body_names = "base"
        self.randomization.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.randomization.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.randomization.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.randomization.motor_params = RandTerm(motor_params, mode="reset")

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


gym.register(
    id="Go2-Rough",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnitreeGo2RoughEnvCfg,
    },
)