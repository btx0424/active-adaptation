# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



from omni.isaac.orbit.robots.legged_robot.legged_robot_cfg import LeggedRobotCfg

from omni.isaac.core.utils.viewports import set_camera_view
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG
from omni.isaac.orbit.robots.config.unitree import UNITREE_A1_CFG
from omni.isaac.orbit.actuators.model import IdealActuator
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit_envs.locomotion.velocity.velocity_cfg import ISAAC_NUCLEUS_DIR

from omni_drones.envs import IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import normalize, quat_rotate_inverse

import torch
import torch.distributions as D
import numpy as np

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec

from .robot import LeggedRobot
from .utils import attach_payload

class Pan:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.scale = high - low
    
    def sample(self, size: torch.Size):
        theta = torch.rand(size, device=self.low.device) * 2. * torch.pi
        sample = torch.rand(size + (2,), device=self.low.device) * self.scale + self.low
        sample = torch.stack([
            sample[..., 0] * theta.cos(),
            sample[..., 0] * theta.sin(),
            sample[..., 1]
        ], dim=-1)
        return sample


class Velocity(IsaacEnv):

    OBS_SPECS = {
        # bodies
        "base_height": UnboundedContinuousTensorSpec((1, 1)),
        "base_mass": UnboundedContinuousTensorSpec((1, 1)),
        "payload_mass": UnboundedContinuousTensorSpec((1, 1)),
        "legs_masses": UnboundedContinuousTensorSpec((1, 8)),
        # joints
        "p_gains": UnboundedContinuousTensorSpec((1, 12)),
        "d_gains": UnboundedContinuousTensorSpec((1, 12)),
        "applied_torques": UnboundedContinuousTensorSpec((1, 13)),
        # sensory
        "feet_pos": UnboundedContinuousTensorSpec((1, 4 * 3)),
        "feet_vel": UnboundedContinuousTensorSpec((1, 4 * 3)),
        "feet_height": UnboundedContinuousTensorSpec((1, 4)),
        "normalized_forces": UnboundedContinuousTensorSpec((1, 3 * 9)),
        # "normalized_torques": UnboundedContinuousTensorSpec((1, 3)),
        "base_linvel": UnboundedContinuousTensorSpec((1, 6)),
    }


    def __init__(self, cfg, headless):
        self.action_scaling = cfg.task.get("action_scaling", 1.0)
        self.randomization = cfg.task.get("randomization", {})
        self.command_interval = cfg.task.get("command_interval")

        super().__init__(cfg, headless)
        # -- history
        self.actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        self.init_vel_dist = D.Uniform(
            torch.tensor([-.8, -.8, 0.], device=self.device),
            torch.tensor([.8, .8, 0.8], device=self.device)
        )
        # -- command: target velocity
        low, high = cfg.task.get("command_speed")
        self.commands_dist = Pan(
            torch.tensor([low, -0.], device=self.device),
            torch.tensor([high, 0.], device=self.device)
        )

        import math
        self.n_commands = math.ceil(self.max_episode_length / self.command_interval)
        self.commands_queue = torch.zeros(self.num_envs, self.n_commands, 3, device=self.device)
        self.commands_i = torch.zeros(self.num_envs, 1, 1, dtype=torch.long, device=self.device)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.push_interval = 300
        self.push_force_dist = Pan(
            torch.tensor([1., -.25], device=self.device) * 5./ (self.dt * self.substeps),
            torch.tensor([1.6, .25], device=self.device) * 5. / (self.dt * self.substeps)
        )
        self.push_force = torch.zeros(self.num_envs, 3, device=self.device)

        self.actuator = self.robot.actuator_groups["base_legs"]
        self.actuator_model = self.actuator.model

        self.base_mass = self.robot.base.get_masses(clone=True)
        self.base_inertia = self.robot.base.get_inertias(clone=True)[..., [0, 4, 8]]

        import pprint
        randomization_cfg = self.randomization.get("train", {})
        pprint.pprint(randomization_cfg)
        if randomization_cfg.get("motor", None) is not None:
            # randomization of motor parameters
            cfg = randomization_cfg["motor"]
            if isinstance(self.actuator_model, IdealActuator):
                self.init_p_gains = self.actuator_model._p_gains.clone()
                self.init_d_gains = self.actuator_model._d_gains.clone()
                self.motor_p_gains_dist = D.Uniform(
                    torch.ones(12, device=self.device) * cfg["p_gains"][0],
                    torch.ones(12, device=self.device) * cfg["p_gains"][1],
                )
                self.motor_d_gains_dist = D.Uniform(
                    torch.ones(12, device=self.device) * cfg["d_gains"][0],
                    torch.ones(12, device=self.device) * cfg["d_gains"][1],
                )
        if randomization_cfg.get("base_mass", None) is not None:
            low, high = randomization_cfg["base_mass"]
            self.base_mass_dist = D.Uniform(
                torch.tensor([low], device=self.device),
                torch.tensor([high], device=self.device)
            )
            self.legs_masses = self.robot.legs.get_masses(clone=True)
        
        if randomization_cfg.get("legs_mass", None) is not None:
            low, high = randomization_cfg["legs_mass"]
            self.legs_mass_dist = D.Uniform(
                torch.tensor([low], device=self.device),
                torch.tensor([high], device=self.device)
            )

        if randomization_cfg.get("payload_mass", None) is not None:
            self.payload = RigidPrimView(
                "/World/envs/env_*/Robot/payload",
                reset_xform_properties=False,
            )
            self.payload.initialize()
            low, high = randomization_cfg["payload_mass"]
            self.payload_mass_dist = D.Uniform(
                torch.tensor([low], device=self.device),
                torch.tensor([high], device=self.device)
            )

        # self.pos_buffer = torch.zeros(self.num_envs, 3, 3, device=self.device)

        self.base_target_height = 0.3
        self.base_height_error = torch.zeros(self.num_envs, 1, device=self.device)

        # visulization
        self.feet_pos_traj = []

    def _design_scene(self):
        # self.robot = LeggedRobot(ANYMAL_C_CFG)
        # UNITREE_A1_CFG.actuator_groups["base_legs"].control_cfg.command_types=["p_rel"]
        self.robot = LeggedRobot(UNITREE_A1_CFG)
        self.robot.spawn("/World/envs/env_0/Robot")
        attach_payload("/World/envs/env_0/Robot")

        if True:
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                improve_patch_friction=True,
                combine_mode="max",
            )
        else:
            prim_utils.create_prim(
                "/World/defaultGroundPlane", 
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd"
            )
        return ["/World/defaultGroundPlane"]
    
    def _set_specs(self):
        self.robot.initialize("/World/envs/env_*/Robot")
        observation_dim = (
            10
            + 3 
            + 12 + 12 # dof_pos, dof_vel
            + 12 + 12 # actions
            + 2
        )
        
        intrinsics_spec = CompositeSpec(self.OBS_SPECS)
        self.intrinsics = (
            intrinsics_spec
            .expand(self.num_envs)
            .to(self.device)
            .zero()
        )
        obs_priv_dim = sum(
            self.OBS_SPECS[key].shape[-1]
            for key in self.cfg.task.priv_obs
        )
        self.priv_obs_manager = _Observation(self.cfg.task.priv_obs)
        self.reward_manager = _Reward(self.cfg.task.reward)
        self.observation_spec = CompositeSpec(
            {
                "agents": {
                    "observation": UnboundedContinuousTensorSpec((1, observation_dim)),
                    "observation_h": UnboundedContinuousTensorSpec((1, observation_dim, 32)),
                    "observation_priv": UnboundedContinuousTensorSpec((1, obs_priv_dim)),
                },
            }
        ).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((1, self.robot.num_actions)),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": BinaryDiscreteTensorSpec(1, dtype=bool, device=self.device),
            "terminated": BinaryDiscreteTensorSpec(1, dtype=bool, device=self.device),
            "truncated": BinaryDiscreteTensorSpec(1, dtype=bool, device=self.device),
        }).expand(self.num_envs).to(self.device)

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            **{
                key: UnboundedContinuousTensorSpec(1) 
                for key in self.reward_manager.reward_funcs.keys()
            }
        })
        self.observation_spec["stats"] = stats_spec.expand(self.num_envs).to(self.device)
        self.stats = stats_spec.zero()
        self.observation_h = self.observation_spec[("agents", "observation_h")].zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        # -- dof state (handled by the robot)
        dof_pos, dof_vel = self.robot.get_random_dof_state(env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # -- root state (custom)
        root_state = self.robot.get_default_root_state(env_ids)
        root_state[:, :3] += self.envs_positions[env_ids]
        root_state[:, 7:10] = self.init_vel_dist.sample(env_ids.shape)
        root_state[:, 10:13] = 0.
        # set into robot
        self.robot.set_root_state(root_state, env_ids=env_ids)
        self.robot.reset_buffers(env_ids)
        
        # randomize motor parameters
        if (
            isinstance(self.actuator_model, IdealActuator)
            and hasattr(self, "motor_p_gains_dist")
        ):
            p_gains = self.motor_p_gains_dist.sample(env_ids.shape)
            d_gains = self.motor_d_gains_dist.sample(env_ids.shape)
            self.actuator_model._p_gains[env_ids] = p_gains * self.init_p_gains[env_ids]
            self.actuator_model._d_gains[env_ids] = d_gains * self.init_d_gains[env_ids]
            self.intrinsics["p_gains"][env_ids] = p_gains.unsqueeze(1)
            self.intrinsics["d_gains"][env_ids] = d_gains.unsqueeze(1)

        # sample commands
        commands_queue = self.commands_dist.sample(env_ids.shape+(self.n_commands,))
        # commands_queue *= (commands_queue.norm(dim=-1, keepdim=True) > 0.6).float()
        self.commands_queue[env_ids] = commands_queue
        self.commands_i[env_ids] = 0.
        self.commands[env_ids] = 0.

        if (env_ids == self.central_env_idx).any():
            self.feet_pos_traj.clear()

        if hasattr(self, "base_mass_dist"):
            ## randomize base mass
            base_mass = self.base_mass_dist.sample(env_ids.shape)
            self.intrinsics["base_mass"][env_ids] = base_mass.unsqueeze(1)
            self.robot.base.set_masses(
                base_mass * self.base_mass[env_ids], 
                env_indices=env_ids
            )

        if hasattr(self, "legs_mass_dist"):
            ## randomize leg mass
            legs_mass = self.legs_mass_dist.sample(env_ids.shape + (8,))
            self.intrinsics["legs_masses"][env_ids] = legs_mass.reshape(-1, 1, 8)
            self.robot.legs.set_masses(
                legs_mass * self.legs_masses[env_ids], 
                env_indices=env_ids
            )

        if hasattr(self, "payload_mass_dist"):
            # randomize payload mass
            payload_mass = self.payload_mass_dist.sample(env_ids.shape)
            self.intrinsics["payload_mass"][env_ids] = payload_mass.unsqueeze(-1)
            self.payload.set_masses(
                payload_mass * self.base_mass[env_ids], 
                env_indices=env_ids
            )
    
        # -- reset history
        self.previous_actions[env_ids] = 0.
        self.observation_h[env_ids] = 0.
        self.stats[env_ids] = 0.

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.actions[:] = tensordict[("agents", "action")].squeeze(1)
        push = ((self.progress_buf + 1) % self.push_interval == 0).nonzero().squeeze(1)
        self.push_force[push] = self.push_force_dist.sample(push.shape)

        actions = self.actions.clip(-10., 10.) * self.action_scaling
        for substep in range(self.substeps):
            self.robot.apply_action(actions)
            self.robot.base.apply_forces(self.push_force)
            self.sim.step(self._should_render(substep) and substep == self.substeps-1)
        
        self.push_force.mul_(0.7)
        self._post_sim_step(tensordict)
        self.progress_buf += 1
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(self._compute_state_and_obs())
        tensordict.update(self._compute_reward_and_done())
        return tensordict
    
    def _compute_state_and_obs(self):
        self.robot.update_buffers(dt=self.dt * self.substeps)

        commands_target = (
            self.commands_queue
            .take_along_dim(self.commands_i, dim=1)
            .squeeze(1)
        )
        self.commands.add_(clip_norm(0.2 * (commands_target - self.commands), 0.1))
        self.cmd_linvel_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.commands)
        
        dof_pos = self.robot.data.dof_pos - self.robot.data.actuator_pos_offset
        dof_vel = self.robot.data.dof_vel - self.robot.data.actuator_vel_offset
        obs = [
            # basic
            self.robot.data.root_quat_w,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            self.cmd_linvel_b,
            noise(dof_pos, 0.05),
            noise(dof_vel, 0.05),
            self.actions, # a_{t-1}
            self.previous_actions, # a_{t-2}
        ]
        
        obs = torch.cat(obs, dim=-1).unsqueeze(1)
        self.observation_h[..., :-1] = self.observation_h[..., 1:]
        self.observation_h[..., -1] = obs

        if self._should_render(0):
            self.debug_draw.clear()
            robot_pos = self.robot.data.root_pos_w[self.central_env_idx].cpu()
            feet_pos = self.robot.feet_pos_w[self.central_env_idx].cpu()
            self.feet_pos_traj.append(feet_pos)
            # feet_pos = self.robot.feet_pos_w[self.central_env_idx].cpu()
            # feet_contact_forces = self.robot.feet_contact_forces[self.central_env_idx].cpu()
            force = self.robot.force_sensor_forces[self.central_env_idx, 0].cpu() / 5.
            robot_top = robot_pos + torch.tensor([0., 0., 0.5])
            linvel = self.robot.data.root_lin_vel_w[self.central_env_idx].cpu()
            # diff_linvel = self.diff_linvel_w[self.central_env_idx].cpu()
            push_force = self.push_force[self.central_env_idx].cpu()

            command = self.commands[self.central_env_idx].cpu()
            self.debug_draw.vector(robot_top, command, color=(1., 1., 1., 1.))
            self.debug_draw.vector(robot_top, linvel, color=(1., 0.5, 0.5, 1.))
            # self.debug_draw.vector(robot_top, diff_linvel, color=(0.5, 0.5, 1., 1.))

            self.debug_draw.vector(robot_pos, force, color=(0.5, 0.5, 1., 1.))
            self.debug_draw.vector(robot_pos, push_force, color=(0.5, 1., 0.5, 1.))

            if len(self.feet_pos_traj[1:]) > 1:
                feet_pos_traj = torch.stack(self.feet_pos_traj[1:], dim=1)
                self.debug_draw.plot(feet_pos_traj[0], 1., color=(1., 0., 0., .8))
                self.debug_draw.plot(feet_pos_traj[1], 1., color=(0., 1., 0., .8))
                self.debug_draw.plot(feet_pos_traj[2], 1., color=(0., 0., 1., .8))
                self.debug_draw.plot(feet_pos_traj[3], 1., color=(1., 1., 0., .8))
            
            contact_forces = self.robot.contact_forces[self.central_env_idx].cpu()
            self.debug_draw.vector(feet_pos, contact_forces)
            
            set_camera_view(
                eye=robot_pos.numpy() + np.asarray(self.cfg.viewer.eye),
                target=robot_pos.numpy() + np.asarray(self.cfg.viewer.lookat)                        
            )

        priv_obs = self.priv_obs_manager.compute(self, self.robot)
        return TensorDict({
            "agents": {
                "observation": obs,
                "observation_h": self.observation_h,
                "observation_priv": priv_obs
            },
            "stats": self.stats.clone(),
        }, self.num_envs)

    def _compute_reward_and_done(self):
        
        rewards = self.reward_manager.compute(self, self.robot)
        reward = sum(rewards.values())

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated = (
            (self.robot.data.root_pos_w[:, 2] <= self.base_target_height * 0.5)
            | (self.robot.data.projected_gravity_b[:, 2] >= -0.3)
        ).unsqueeze(1)
        
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        for key, value in rewards.items():
            self.stats[key].add_(value)

        # resample commands
        change_commands = ((self.progress_buf % self.command_interval) == 0).nonzero().squeeze(1)
        self.commands_i[change_commands] += 1

        return TensorDict({
            "agents": {
                "reward": reward.reshape(-1, 1, 1)
            },
            "done": terminated | truncated,
            "terminated": terminated,
            "truncated": truncated,
        }, self.num_envs)


def square_norm(x: torch.Tensor):
    return x.square().sum(dim=-1, keepdim=True)


def noise(x: torch.Tensor, scale: float):
    return x + torch.randn_like(x).clip(-3., 3.) * scale


def clip_norm(x: torch.Tensor, max_norm: float):
    norm = x.norm(dim=-1, keepdim=True)
    return x * (norm.clamp(max=max_norm) / norm.clamp(min=1e-6))


class _Observation:

    def __init__(self, obs_keys) -> None:
        self.obs_funcs = [
            getattr(self, key)
            for key in obs_keys
        ]

    def compute(self, env: Velocity, robot: LeggedRobot):
        obs = torch.cat([
            func(env, robot)
            for func in self.obs_funcs
        ], dim=-1)
        return obs
    
    @staticmethod
    def base_mass(env: Velocity, robot: LeggedRobot):
        return env.intrinsics["base_mass"]

    @staticmethod
    def payload_mass(env: Velocity, robot: LeggedRobot):
        return env.intrinsics["payload_mass"]

    @staticmethod
    def p_gains(env: Velocity, robot: LeggedRobot):
        return env.intrinsics["p_gains"]

    @staticmethod
    def d_gains(env: Velocity, robot: LeggedRobot):
        return env.intrinsics["d_gains"]
    
    @staticmethod
    def base_height(env: Velocity, robot: LeggedRobot):
        return (
            robot.data.root_pos_w[..., [2]] - env.base_target_height
        ).unsqueeze(1)
    
    @staticmethod
    def base_linvel(env: Velocity, robot: LeggedRobot):
        return torch.cat([
            robot.data.root_lin_vel_b,
            robot.data.root_lin_vel_b - env.cmd_linvel_b
        ], dim=-1).unsqueeze(1)

    @staticmethod
    def feet_pos(env: Velocity, robot: LeggedRobot):
        return robot.feet_pos_b.reshape(-1, 1, 12)

    @staticmethod
    def feet_vel(env: Velocity, robot: LeggedRobot):
        return robot.feet_vel_b.reshape(-1, 1, 12)

    @staticmethod
    def feet_height(env: Velocity, robot: LeggedRobot):
        return robot.feet_pos_w[..., [2]].reshape(-1, 1, 4)

    @staticmethod
    def applied_torques(env: Velocity, robot: LeggedRobot):
        return robot.data.applied_torques.unsqueeze(1) / 30.

    @staticmethod
    def contact_forces(env: Velocity, robot: LeggedRobot):
        return robot.contact_forces.reshape(-1, 1, 12)

from collections import OrderedDict
class _Reward:
    def __init__(self, cfg):
        self.reward_funcs = OrderedDict()
        for key, weight in cfg.items():
            func = getattr(self, key)
            self.reward_funcs[key] = (func, weight)
    
    def compute(self, env: Velocity, robot: LeggedRobot):
        reward = {
            key: func(env, robot) * weight
            for key, (func, weight) in self.reward_funcs.items()
        }
        return reward

    def linvel(self, env: Velocity, robot: LeggedRobot):
        lin_vel_w = robot.data.root_lin_vel_w
        lin_vel_error = square_norm(env.commands[:, :2] - lin_vel_w[:, :2])
        reward_linvel = 1. / (1. + lin_vel_error / 0.25)
        return reward_linvel

    def heading(self, env: Velocity, robot: LeggedRobot):
        heading_projection = (
            (normalize(robot.heading[:, :2]) * env.commands[:, :2])
            .sum(-1, keepdim=True)
        )
        return heading_projection

    def base_height(self, env: Velocity, robot: LeggedRobot):
        base_height_error = (
            (robot.data.root_pos_w[:, [2]] - env.base_target_height)
            .abs()
        )
        return 1. / (1 + base_height_error / 0.25)
    
    def energy(self, env: Velocity, robot: LeggedRobot):
        energy = (
            (robot.data.dof_vel * robot.data.applied_torques)
            .abs()
            .sum(dim=-1, keepdim=True)
        )
        return -energy

    def flat_orientation_l2(self, env: Velocity, robot: LeggedRobot):
        flat_orientation_l2 = square_norm(robot.data.projected_gravity_b[:, :2])
        return -flat_orientation_l2