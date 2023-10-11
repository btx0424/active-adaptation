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
        # -- command: x vel, y vel, yaw vel, heading
        self.commands_dist = Pan(
            torch.tensor([0., -0.], device=self.device),
            torch.tensor([2.8, 0.], device=self.device)
        )

        import math
        self.n_commands = math.ceil(self.max_episode_length / self.command_interval)
        self.commands_queue = torch.zeros(self.num_envs, self.n_commands, 3, device=self.device)
        self.commands_i = torch.zeros(self.num_envs, 1, 1, dtype=torch.long, device=self.device)

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

        self.pos_buffer = torch.zeros(self.num_envs, 3, 3, device=self.device)

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
        
        intrinsics_spec = CompositeSpec({
            key: self.OBS_SPECS[key]
            for key in self.cfg.task.priv_obs
        })

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, observation_dim)),
                "observation_h": UnboundedContinuousTensorSpec((1, observation_dim, 32)),
                "intrinsics": intrinsics_spec,
            },
        }).expand(self.num_envs).to(self.device)

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
            "lin_vel_error": UnboundedContinuousTensorSpec(1),
            "base_height_error": UnboundedContinuousTensorSpec(1),
            "energy": UnboundedContinuousTensorSpec(1),
            "dof_torques": UnboundedContinuousTensorSpec(1),
            "dof_acc": UnboundedContinuousTensorSpec(1),
            "action_rate": UnboundedContinuousTensorSpec(1),
            "feet_symmetry": UnboundedContinuousTensorSpec(1),
            "feet_clearance": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()
        self.intrinsics = self.observation_spec[("agents", "intrinsics")].zero()
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
            # self.intrinsics["p_gains"][env_ids] = (
            #     (p_gains - self.motor_p_gains_dist.low)
            #     / (self.motor_p_gains_dist.high - self.motor_p_gains_dist.low)
            # ).unsqueeze(1)
            # self.intrinsics["d_gains"][env_ids] = (
            #     (d_gains - self.motor_d_gains_dist.low)
            #     / (self.motor_d_gains_dist.high - self.motor_d_gains_dist.low)
            # ).unsqueeze(1)

        # sample commands
        commands_queue = self.commands_dist.sample(env_ids.shape+(self.n_commands,))
        commands_queue *= (commands_queue.norm(dim=-1, keepdim=True) > 0.6).float()
        self.commands_queue[env_ids] = commands_queue
        self.commands_i[env_ids] = 0

        if (env_ids == self.central_env_idx).any():
            self.feet_pos_traj.clear()

        if hasattr(self, "base_mass_dist"):
            # randomize base mass
            base_mass = self.base_mass_dist.sample(env_ids.shape)
            self.intrinsics["base_mass"][env_ids] = (
                (base_mass - self.base_mass_dist.low) 
                / (self.base_mass_dist.high - self.base_mass_dist.low)
            ).reshape(-1, 1, 1)
            self.robot.base.set_masses(base_mass * self.base_mass[env_ids], env_indices=env_ids)

            # randomize leg mass
            # legs_mass = self.legs_mass_dist.sample(env_ids.shape + (8,))
            # self.intrinsics["legs_masses"][env_ids] = (
            #     (legs_mass - self.legs_mass_dist.low)
            #     / (self.legs_mass_dist.high - self.legs_mass_dist.low)
            # ).reshape(-1, 1, 8)
            # self.robot.legs.set_masses(legs_mass * self.legs_masses[env_ids], env_indices=env_ids)

        if hasattr(self, "payload_mass_dist"):
            # randomize payload mass
            payload_mass = self.payload_mass_dist.sample(env_ids.shape)
            self.intrinsics["payload_mass"][env_ids] = (
                (payload_mass.unsqueeze(-1) - self.payload_mass_dist.low)
                / (self.payload_mass_dist.high - self.payload_mass_dist.low)
            )
            self.payload.set_masses(payload_mass * self.base_mass[env_ids], env_indices=env_ids)
    
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

        self.commands = (
            self.commands_queue
            .take_along_dim(self.commands_i, dim=1)
            .squeeze(1)
        )
        cmd_lin_vel_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.commands)
       
        normalized_forces = (
            self.robot.force_sensor_forces / self.base_mass.unsqueeze(1)
        )

        if self.intrinsics.get("base_height", None) is not None:
            self.intrinsics["base_height"][:] = (
                self.robot.data.root_pos_w[..., [2]]- self.base_target_height
            ).unsqueeze(1)
        if self.intrinsics.get("base_linvel", None) is not None:
            self.intrinsics["base_linvel"][:] = torch.cat([
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_lin_vel_b - cmd_lin_vel_b
            ], dim=-1).unsqueeze(1)
        if self.intrinsics.get("feet_pos", None) is not None:
            self.intrinsics["feet_pos"][:] = self.robot.feet_pos_b.reshape(-1, 1, 12)
        if self.intrinsics.get("feet_vel", None) is not None:
            self.intrinsics["feet_vel"][:] = self.robot.feet_vel_b.reshape(-1, 1, 12)
        if self.intrinsics.get("feet_height", None) is not None:
            self.intrinsics["feet_height"][:] = self.robot.feet_pos_w[..., [2]].reshape(-1, 1, 4)
        if self.intrinsics.get("normalized_forces", None) is not None:
            self.intrinsics["normalized_forces"][:] = normalized_forces.reshape(-1, 1, 27)
        if self.intrinsics.get("applied_torques", None) is not None:
            self.intrinsics["applied_torques"][:] = self.robot.data.applied_torques.unsqueeze(1) / 30.
        
        dof_pos = self.robot.data.dof_pos - self.robot.data.actuator_pos_offset
        dof_vel = self.robot.data.dof_vel - self.robot.data.actuator_vel_offset
        obs = [
            # basic
            self.robot.data.root_quat_w,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            cmd_lin_vel_b,
            noise(dof_pos, 0.05),
            noise(dof_vel, 0.05),
            self.actions, # a_{t-1}
            self.previous_actions, # a_{t-2}
        ]
        
        obs = torch.cat(obs, dim=-1).unsqueeze(1)
        self.observation_h[..., :-1] = self.observation_h[..., 1:]
        self.observation_h[..., -1] = obs

        self.diff_linvel_w = (
            (self.robot.data.root_pos_w - self.pos_buffer[:, 0]) 
            / (self.pos_buffer.shape[1] * (self.dt * self.substeps))
        )
        self.pos_buffer[:, :-1] = self.pos_buffer[:, 1:]
        self.pos_buffer[:, -1] = self.robot.data.root_pos_w

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
            diff_linvel = self.diff_linvel_w[self.central_env_idx].cpu()
            push_force = self.push_force[self.central_env_idx].cpu()

            command = self.commands[self.central_env_idx].cpu()
            self.debug_draw.vector(robot_top, command, color=(1., 1., 1., 1.))
            self.debug_draw.vector(robot_top, linvel, color=(1., 0.5, 0.5, 1.))
            self.debug_draw.vector(robot_top, diff_linvel, color=(0.5, 0.5, 1., 1.))

            self.debug_draw.vector(robot_pos, force, color=(0.5, 0.5, 1., 1.))
            self.debug_draw.vector(robot_pos, push_force, color=(0.5, 1., 0.5, 1.))

            if len(self.feet_pos_traj[1:]) > 1:
                feet_pos_traj = torch.stack(self.feet_pos_traj[1:], dim=1)
                self.debug_draw.plot(feet_pos_traj[0], 1., color=(1., 0., 0., .8))
                self.debug_draw.plot(feet_pos_traj[1], 1., color=(0., 1., 0., .8))
                self.debug_draw.plot(feet_pos_traj[2], 1., color=(0., 0., 1., .8))
                self.debug_draw.plot(feet_pos_traj[3], 1., color=(1., 1., 0., .8))
            
            set_camera_view(
                eye=robot_pos.numpy() + np.asarray(self.cfg.viewer.eye),
                target=robot_pos.numpy() + np.asarray(self.cfg.viewer.lookat)                        
            )


        return TensorDict({
            "agents": {
                "observation": obs,
                "intrinsics": self.intrinsics.clone(),
            },
            "stats": self.stats.clone(),
        }, self.num_envs)

    def _compute_reward_and_done(self):
        # -- compute reward
        lin_vel_w = self.robot.data.root_lin_vel_w
        lin_vel_error = square_norm(self.commands[:, :2] - lin_vel_w[:, :2])
        # ang_vel_error = torch.square(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        # lin_vel_proj = (lin_vel_w * self.commands).sum(-1, keepdim=True)
        # lin_vel_error_projected = (
        #     (self.commands.norm(dim=-1, keepdim=True) - lin_vel_proj).clip(0.)
        # )
        
        base_height_error = (
            (self.robot.data.root_pos_w[:, [2]] - self.base_target_height)
            .abs()
        )
        heading_projection = (
            (normalize(self.robot.heading[:, :2]) * self.commands[:, :2])
            .sum(-1, keepdim=True)
        )
        feet_symmetry_error = (
            self.robot.feet_pos_b[:, [0, 1], 1].sum(dim=-1, keepdim=True).abs()
            + self.robot.feet_pos_b[:, [2, 3], 1].sum(dim=-1, keepdim=True).abs()
        )
        feet_clearance = self.robot.feet_pos_w[:, :, 2].sum(dim=-1, keepdim=True)

        # lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)
        # lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.5)
        # ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)
        lin_vel_z_l2 = torch.square(self.robot.data.root_lin_vel_w[:, [2]])
        ang_vel_xy_l2 = square_norm(self.robot.data.root_ang_vel_w[:, :2])
        flat_orientation_l2 = square_norm(self.robot.data.projected_gravity_b[:, :2])
        dof_torques_l2 = square_norm(self.robot.data.applied_torques)
        dof_acc_l2 = square_norm(self.robot.data.dof_acc)
        action_rate_l2 = square_norm(self.previous_actions - self.actions)
        self.previous_actions[:] = self.actions
        energy = (self.robot.data.dof_vel * self.robot.data.applied_torques).abs().sum(dim=-1, keepdim=True)
        
        reward_linvel = 1. / (1. + lin_vel_error / 0.25)
        reward = (
            2.0  * reward_linvel    
            # 1.2 / (1. + lin_vel_error_projected / 0.5)
            # lin_vel_proj.clamp_max(self.commands[:, :2].norm(dim=-1, keepdim=True))
            + 0.25 * heading_projection
            # + 0.5 * ang_vel_z_exp.unsqueeze(1)
            + 0.5 / (1 + base_height_error / 0.25)
            - 2.0 * lin_vel_z_l2
            - 0.05 * ang_vel_xy_l2
            - 2.0 * flat_orientation_l2
            - 0.000025 * dof_torques_l2
            - 2.5e-7 * dof_acc_l2
            - 0.01 * action_rate_l2
            - 0.0005 * energy
            - 2. * reward_linvel * feet_symmetry_error
            + 0.5 * reward_linvel * feet_clearance
        ).clip(min=0.)

        self.base_height_error[:] = base_height_error

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated = (
            (self.robot.data.root_pos_w[:, 2] <= self.base_target_height * 0.5)
            | (self.robot.data.projected_gravity_b[:, 2] >= -0.3)
        ).unsqueeze(1)
        
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        self.stats["energy"].add_(energy)
        self.stats["lin_vel_error"].add_(lin_vel_error)
        self.stats["base_height_error"].add_(base_height_error)
        self.stats["dof_torques"].add_(dof_torques_l2)
        self.stats["dof_acc"].add_(dof_acc_l2)
        self.stats["action_rate"].add_(action_rate_l2)
        self.stats["feet_symmetry"].add_(feet_symmetry_error)
        self.stats["feet_clearance"].add_(feet_clearance)

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

