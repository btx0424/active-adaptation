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


from typing import List, Optional

from omni.isaac.core.utils.viewports import set_camera_view
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.utils.kit as kit_utils

from omni.isaac.orbit.robots.config.unitree import UNITREE_A1_CFG
from omni.isaac.orbit.actuators.model import IdealActuator
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit_envs.locomotion.velocity.velocity_cfg import ObservationsCfg, RewardsCfg

from omni_drones.envs import IsaacEnv
from omni_drones.views import RigidPrimView, disable_warnings
from omni_drones.utils.torch import quat_axis, normalize, quat_rotate_inverse

import torch
import torch.distributions as D
import numpy as np
import einops

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec

from omni.isaac.debug_draw import _debug_draw
from .robot import LeggedRobot


def attach_payload(parent_path):
    import omni.physx.scripts.utils as script_utils
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core import objects
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.18, 0.16, 0.12]),
        mass=0.0001,
        translation=torch.tensor([0.0, 0.0, 0.1]),
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.0)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.0)


class Tracking(IsaacEnv):
    def __init__(self, cfg, headless):
        self.action_scaling = cfg.task.get("action_scaling", 1.0)
        self.randomization = cfg.task.get("randomization", {})

        super().__init__(cfg, headless)
        # -- history
        self.actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        self.init_vel_dist = D.Uniform(
            torch.tensor([-.5, -.5, 0.], device=self.device),
            torch.tensor([.5, .5, 0.], device=self.device)
        )

        self.push_interval = 600
        self.push_acc_dist = D.Uniform(
            torch.tensor([-3, -3, -0.25], device=self.device) / (self.dt * self.substeps), 
            torch.tensor([3., 3., 0.25], device=self.device) / (self.dt * self.substeps),
        )

        self.actuator_model = self.robot.actuator_groups["base_legs"].model
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
            self.base_mass_dist = D.Uniform(
                torch.tensor([0.8], device=self.device),
                torch.tensor([1.2], device=self.device)
            )
            self.legs_masses = self.robot.legs.get_masses(clone=True)
            self.legs_mass_dist = D.Uniform(
                torch.tensor([0.8], device=self.device),
                torch.tensor([1.2], device=self.device)
            )

        if randomization_cfg.get("payload_mass", None) is not None:
            self.payload = RigidPrimView(
                "/World/envs/env_*/Robot/payload",
                reset_xform_properties=False,
            )
            self.payload.initialize()
            self.payload_mass_dist = D.Uniform(
                torch.tensor([0.05], device=self.device),
                torch.tensor([0.6], device=self.device)
            )

        self.heading_target = torch.zeros(self.num_envs, device=self.device)

        self.base_target_height = 0.3
        self.base_height_error = torch.zeros(self.num_envs, 1, device=self.device)

        self.trajectory_scaling = torch.zeros(self.num_envs, 2, device=self.device)
        self.trajectory_scaling_dist = D.Uniform(
            torch.tensor([2.0, 2.0], device=self.device),
            torch.tensor([2.5, 2.5], device=self.device),
        )

    def _design_scene(self):
        # self.robot = LeggedRobot(ANYMAL_C_CFG)
        # UNITREE_A1_CFG.actuator_groups["base_legs"].control_cfg.command_types=["p_rel"]
        self.robot = LeggedRobot(UNITREE_A1_CFG)
        self.robot.spawn("/World/envs/env_0/Robot")
        attach_payload("/World/envs/env_0/Robot")

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            improve_patch_friction=True,
            combine_mode="max",
        )
        return ["/World/defaultGroundPlane"]
    
    def _set_specs(self):
        self.robot.initialize("/World/envs/env_*/Robot")
        observation_dim = (
            28
            + 4 * 2 # ref_pos
            + 3 + 3 # linvel_b, angvel_b
            + 12 + 12 # dof_pos, dof_vel
            + 2
            # + 2 # base and payload mass
            # + 12 # + 12 # feet_pos, feet_vel
            # + 9 * 3 # forces
        )

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, observation_dim)),
                "intrinsics": CompositeSpec({
                    "base_mass": UnboundedContinuousTensorSpec((1, 1)),
                    "payload_mass": UnboundedContinuousTensorSpec((1, 1)),
                    "legs_masses": UnboundedContinuousTensorSpec((1, 8)),
                    "p_gains": UnboundedContinuousTensorSpec((1, 12)),
                    "d_gains": UnboundedContinuousTensorSpec((1, 12)),
                    "feet_pos": UnboundedContinuousTensorSpec((1, 4 * 3)),
                    "feet_vel": UnboundedContinuousTensorSpec((1, 4 * 3)),
                    # "normalized_forces": UnboundedContinuousTensorSpec((1, 3)),
                    # "normalized_torques": UnboundedContinuousTensorSpec((1, 3)),
                })
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
            "truncated": BinaryDiscreteTensorSpec(1, dtype=bool, device=self.device)
        }).expand(self.num_envs).to(self.device)

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "base_height_error": UnboundedContinuousTensorSpec(1),
            "energy": UnboundedContinuousTensorSpec(1),
            "dof_torques": UnboundedContinuousTensorSpec(1),
            "dof_acc": UnboundedContinuousTensorSpec(1),
            "action_rate": UnboundedContinuousTensorSpec(1),
            "feet_symmetry": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()
        self.intrinsics = self.observation_spec[("agents", "intrinsics")].zero()

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
            self.intrinsics["p_gains"][env_ids] = (
                (p_gains - self.motor_p_gains_dist.low)
                / (self.motor_p_gains_dist.high - self.motor_p_gains_dist.low)
            ).unsqueeze(1)
            self.intrinsics["d_gains"][env_ids] = (
                (d_gains - self.motor_d_gains_dist.low)
                / (self.motor_d_gains_dist.high - self.motor_d_gains_dist.low)
            ).unsqueeze(1)

        if hasattr(self, "base_mass_dist"):
            # randomize base mass
            base_mass = self.base_mass_dist.sample(env_ids.shape)
            self.intrinsics["base_mass"][env_ids] = (
                (base_mass - self.base_mass_dist.low) 
                / (self.base_mass_dist.high - self.base_mass_dist.low)
            ).reshape(-1, 1, 1)
            self.robot.base.set_masses(base_mass * self.base_mass[env_ids], env_indices=env_ids)

            # randomize leg mass
            legs_mass = self.legs_mass_dist.sample(env_ids.shape + (8,))
            self.intrinsics["legs_masses"][env_ids] = (
                (legs_mass - self.legs_mass_dist.low)
                / (self.legs_mass_dist.high - self.legs_mass_dist.low)
            ).reshape(-1, 1, 8)
            self.robot.legs.set_masses(legs_mass * self.legs_masses[env_ids], env_indices=env_ids)

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
        self.stats[env_ids] = 0.

        # -- sample trajectory params
        self.trajectory_scaling[env_ids] = (
            self.trajectory_scaling_dist.sample(env_ids.shape)
             * torch.randn(env_ids.shape+(2,), device=self.device).sign()
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.actions[:] = tensordict[("agents", "action")].squeeze(1)
        # push = ((self.progress_buf % self.push_interval) == 0).nonzero().squeeze(1)
        # if len(push) > 0:
        #     push_force = self.base_mass[push] * self.push_acc_dist.sample(push.shape)
        actions = self.actions.clip(-100., 100.) * self.action_scaling
        for substep in range(self.substeps):
            self.robot.apply_action(actions)
            # if len(push) > 0:
            #     self.robot.base.apply_forces(push_force, indices=push)
            self.sim.step(self._should_render(substep) and substep == self.substeps-1)
        self._post_sim_step(tensordict)
        self.progress_buf += 1
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(self._compute_state_and_obs())
        tensordict.update(self._compute_reward_and_done())
        return tensordict
    
    def _compute_state_and_obs(self):
        # compute current command
        t = (
            + (self.progress_buf.unsqueeze(1) + torch.arange(4, device=self.device) * 5) 
            * self.dt * self.substeps
        )
        self.ref_pos = (
            lemniscate(torch.pi/2 + scale_time(-0.5 * t)) 
            * self.trajectory_scaling.unsqueeze(1)
            - self.robot.data.root_pos_w[:, :2].unsqueeze(1)
            + self.envs_positions[:, :2].unsqueeze(1)
        )
        ref_pos_b = quat_rotate_inverse(
            einops.repeat(self.robot.data.root_quat_w, "b q -> b q t", t=4), 
            torch.cat([
                self.ref_pos, 
                torch.zeros(*self.ref_pos.shape[:-1], 1, device=self.device)
                ], dim=-1)
        )[:, :, :2]

        self.robot.update_buffers(dt=self.dt * self.substeps)

        normalized_forces = (
            self.robot.force_sensor_forces
            / self.base_mass.unsqueeze(-2)
        )
        self.intrinsics["feet_pos"][:] = self.robot.feet_pos_b.reshape(-1, 1, 12)
        self.intrinsics["feet_vel"][:] = self.robot.feet_vel_b.reshape(-1, 1, 12)
        # self.intrinsics["normalized_forces"][:] = normalized_forces.reshape(-1, 1, 3)
        # self.intrinsics["normalized_torques"][:] = normalized_torques.reshape(-1, 1, 3)

        obs = [
            # base orientation
            self.robot.data.root_pos_w[:, [2]] - self.base_target_height,
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            ref_pos_b.reshape(-1, 8),
            self.robot.data.dof_pos - self.robot.data.actuator_pos_offset,
            self.robot.data.dof_vel - self.robot.data.actuator_vel_offset,
            self.actions, # a_{t-1}
            self.previous_actions, # a_{t-2}

            # self.intrinsics["base_mass"].squeeze(1),
            # self.intrinsics["payload_mass"].squeeze(1),
            # self.robot.feet_pos_b.flatten(1) * 0.1,
            # self.robot.feet_vel_b.flatten(1) * 0.1,
            # normalized_forces.flatten(1) * 0.1,
        ]
        
        obs = torch.cat(obs, dim=-1)

        if self._should_render(0):
            self.debug_draw.clear()
            robot_pos = self.robot.data.root_pos_w[self.central_env_idx].cpu()
            
            ref_pos = torch.zeros(4, 3)
            ref_pos[:, :2] = (
                robot_pos[:2].unsqueeze(0)
                + self.ref_pos[self.central_env_idx].cpu()
                # + self.envs_positions[self.central_env_idx, :2]
            )
            ref_pos[:, 2] = self.base_target_height
            self.debug_draw.plot(ref_pos)

            set_camera_view(
                eye=robot_pos.numpy() + np.asarray(self.cfg.viewer.eye),
                target=robot_pos.numpy() + np.asarray(self.cfg.viewer.lookat)                        
            )

        return TensorDict({
            "agents": {
                "observation": obs.unsqueeze(1),
                "intrinsics": self.intrinsics,
            },
            "stats": self.stats.clone(),
        }, self.num_envs)

    def _compute_reward_and_done(self):
        # -- compute reward
        pos_error = self.ref_pos[:, 0].norm(dim=-1, keepdim=True)
        
        base_height_error = (
            (self.robot.data.root_pos_w[:, [2]] - self.base_target_height)
            .abs()
        )
        heading_projection = normalize(self.robot.data.root_lin_vel_b)[:, [0]]
            
        feet_symmetry_error = (
            self.robot.feet_pos_b[:, [0, 1], 1].sum(dim=-1, keepdim=True).abs()
            + self.robot.feet_pos_b[:, [2, 3], 1].sum(dim=-1, keepdim=True).abs()
        )

        # lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)
        # lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.5)
        # ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)
        lin_vel_z_l2 = torch.square(self.robot.data.root_lin_vel_w[:, 2])
        ang_vel_xy_l2 = square_norm(self.robot.data.root_ang_vel_w[:, :2])
        flat_orientation_l2 = square_norm(self.robot.data.projected_gravity_b[:, :2])
        dof_torques_l2 = square_norm(self.robot.data.applied_torques)
        dof_acc_l2 = square_norm(self.robot.data.dof_acc)
        action_rate_l2 = square_norm(self.previous_actions - self.actions)
        self.previous_actions[:] = self.actions
        energy = (self.robot.data.dof_vel * self.robot.data.applied_torques).abs().sum(dim=-1, keepdim=True)
        
        reward = (
            # 2.0 * lin_vel_xy_exp
            2.0 / (1. + pos_error / 0.5)
            # lin_vel_proj.clamp_max(self.commands[:, :2].norm(dim=-1, keepdim=True))
            + 0.5 * heading_projection
            # + 0.5 * ang_vel_z_exp.unsqueeze(1)
            + 0.5 / (1 + base_height_error)
            - 2.0 * lin_vel_z_l2.unsqueeze(1)
            - 0.05 * ang_vel_xy_l2
            - 2.0 * flat_orientation_l2
            - 0.000025 * dof_torques_l2
            - 2.5e-7 * dof_acc_l2
            - 0.01 * action_rate_l2
            - 0.0005 * energy
            - 2 * feet_symmetry_error
        ).clip(min=0.)

        self.base_height_error[:] = base_height_error

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated = (
            (self.robot.data.root_pos_w[:, 2] <= self.base_target_height * 0.5)
            | (self.robot.data.projected_gravity_b[:, 2] >= -0.5)
            | (pos_error > 0.6).squeeze(-1)
        ).unsqueeze(1)
        
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        self.stats["energy"].add_(energy)
        self.stats["pos_error"].add_(pos_error)
        self.stats["base_height_error"].add_(base_height_error)
        self.stats["dof_torques"].add_(dof_torques_l2)
        self.stats["dof_acc"].add_(dof_acc_l2)
        self.stats["action_rate"].add_(action_rate_l2)
        self.stats["feet_symmetry"].add_(feet_symmetry_error)

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


def lemniscate(t: torch.Tensor):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    return torch.stack([cos_t, sin_t * cos_t], dim=-1)


def scale_time(t, a: float=1.0):
    return t / (1 + 1/(a*torch.abs(t)))

