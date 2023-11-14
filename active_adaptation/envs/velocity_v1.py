import torch

from omni_drones.envs.isaac_env import DebugDraw
from omni.isaac.orbit.sensors import ContactSensor, RayCaster

from active_adaptation.envs.base import Env, observation_func, reward_func, termination_func
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

from tensordict.tensordict import TensorDictBase
from torchrl.data import (
    CompositeSpec, 
    UnboundedContinuousTensorSpec
)

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)


class LocomotionEnv(Env):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.robot = self.scene.articulations["robot"]
        print(self.robot.body_names)
        self.calf_indices = [i for i, name in enumerate(self.robot.body_names) if "calf" in name]
        self.thigh_indices = [i for i, name in enumerate(self.robot.body_names) if "thigh" in name]

        self.contact_sensor: ContactSensor = self.scene.sensors.get("contact_forces", None)
        self.FEET_OFFSET = (
            torch.tensor([0., 0., -0.2], device=self.device)
            .reshape(1, 1, 3)
            .expand(self.num_envs, 4, 3)
        )
        self.height_scanner: RayCaster = self.scene.sensors.get("height_scanner", None)

        self.init_root_state = self.robot.data.default_root_state_w.clone()
        self.init_root_state[..., :3] += self.scene._default_env_origins
        self.init_joint_pos = self.robot.data.default_joint_pos.clone()
        self.init_joint_vel = self.robot.data.default_joint_vel.clone()
        self.debug_draw = DebugDraw()
        self.lookat_env_i = (
            self.scene._default_env_origins.cpu() 
            - torch.tensor(self.cfg.viewer.lookat)
        ).norm(dim=-1).argmin()

        self.target_base_height = self.cfg.target_base_height

        with torch.device(self.device):
            self._command = torch.zeros(self.num_envs, 3)
            self.target_pos = torch.zeros(self.num_envs, 4, 3)
            self.target_speed = torch.zeros(self.num_envs, 4, 1)
            self.offset = torch.tensor([
                [-1., -1.],
                [-1., 0.],
                [0., -1.],
                [0., 0.]
            ])
            self._actions = torch.zeros(self.num_envs, 12)
            self._prev_actions = torch.zeros(self.num_envs, 12)
            self._feet_pos = torch.zeros(self.num_envs, 4, 3)

        obs = super()._compute_observation()
        reward = self._compute_reward()

        self.action_spec = CompositeSpec(
            {
                ("agents", "action"): UnboundedContinuousTensorSpec((self.num_envs, 12))
            }, 
            shape=[self.num_envs]
        ).to(self.device)

        self._observation_h = (
            obs[("agents", "observation")]
            .unsqueeze(-1)
            .expand(self.num_envs, -1, 32)
            .clone()
            .zero_()
        )
        obs[("agents", "observation_h")] = self._observation_h

        for key, value in obs.items(True, True):
            if key not in self.observation_spec.keys(True, True):
                self.observation_spec[key] = UnboundedContinuousTensorSpec(value.shape, device=self.device)
        
        self.reward_spec = CompositeSpec(
            {
                key: UnboundedContinuousTensorSpec(value.shape)
                for key, value in reward.items()
            },
            shape=[self.num_envs]
        ).to(self.device)

    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.robot.write_root_state_to_sim(self.init_root_state[env_ids], env_ids)
        self.robot.write_joint_state_to_sim(
            random_scale(self.init_joint_pos[env_ids], 0.8, 1.2),
            self.init_joint_vel[env_ids],
            env_ids=env_ids
        )
        self.stats[env_ids] = 0.

        self.target_pos[env_ids, :, :2] = torch.gather(
            torch.rand(len(env_ids), 4, 2, device=self.device) + self.offset,
            dim=1,
            index=torch.rand(len(env_ids), 4, 1, device=self.device).argsort(dim=1).expand(-1, -1, 2)
        ) * 2.5
        self.target_speed[env_ids] = sample_uniform((len(env_ids), 4, 1), 0.8, 1.2, device=self.device)

        self._prev_actions[env_ids] = 0.
        self._actions[env_ids] = 0.
        self._observation_h[env_ids] = 0.

        self.scene.reset(env_ids)
        self.scene.update(dt=self.physics_dt)

    def apply_action(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            self._prev_actions[:] = self._actions
            actions = tensordict[("agents", "action")]
            self._actions[:] = actions + self.init_joint_pos
            self.robot.set_joint_position_target(self._actions)
        self.robot.write_data_to_sim()
    
    def _compute_observation(self) -> TensorDictBase:
        index = (self.episode_length_buf) // 300
        current_target_pos = self.target_pos.take_along_dim(
            index.reshape(-1, 1, 1),
            dim=1
        ).squeeze(1)
        current_target_speed = self.target_speed.take_along_dim(
            index.reshape(-1, 1, 1),
            dim=1
        ).squeeze(1)
        pos_diff_xy = (
            current_target_pos[:, :2] 
            - self.robot.data.root_pos_w[:, :2]
            + self.scene._default_env_origins[:, :2]
        )
        pos_error_xy = pos_diff_xy.norm(dim=-1, keepdim=True)
        self._command[:, :2] = (
            (pos_diff_xy / pos_error_xy)
            * current_target_speed
            * (pos_error_xy > 0.1).float()
        )
        self._feet_pos[:] = (
            self.robot.data.body_pos_w[:, self.calf_indices]
            + quat_rotate(self.robot.data.body_quat_w[:, self.calf_indices], self.FEET_OFFSET)
        )
        obs = super()._compute_observation()
        self._observation_h[:, :, :-1] = self._observation_h[:, :, 1:]
        self._observation_h[:, :, -1] = obs[("agents", "observation")]
        obs[("agents", "observation_h")] = self._observation_h.clone()
        return obs

    def render(self, mode: str="human"):
        if self.enable_render:
            self.debug_draw.clear()
            robot_pos = (
                self.robot.data.root_pos_w.cpu()
                + torch.tensor([0., 0., 0.2])
            )
            self.debug_draw.clear()
            self.debug_draw.vector(
                robot_pos, 
                self._command,
                color=(1., 1., 1., 1.)
            )
            self.debug_draw.vector(
                robot_pos, 
                self.robot.data.root_lin_vel_w,
                color=(1., .5, .5, 1.)
            )
        return super().render(mode)

    @observation_func
    def command(self):
        command_b = quat_rotate_inverse(
            self.robot.data.root_quat_w,
            self._command
        )
        return command_b
    
    @observation_func
    def root_quat_w(self):
        return self.robot.data.root_quat_w
    
    @observation_func
    def root_angvel_b(self):
        return self.robot.data.root_ang_vel_b
    
    @observation_func
    def joint_pos(self):
        return random_noise(self.robot.data.joint_pos, 0.05)
    
    @observation_func
    def joint_vel(self):
        return self.robot.data.joint_vel
    
    @observation_func
    def projected_gravity_b(self):
        return self.robot.data.projected_gravity_b
    
    @observation_func
    def root_linvel_b(self):
        return self.robot.data.root_lin_vel_b
    
    @observation_func
    def prev_actions(self):
        return self._actions
    
    @observation_func
    def applied_torques(self):
        return self.robot.data.applied_torque / 30.
    
    @observation_func
    def contact_forces(self):
        forces = self.contact_sensor.data.net_forces_w[:, self.calf_indices]
        forces_norm = forces.norm(dim=-1, keepdim=True)
        return (forces / forces_norm.clamp_min(1e-6) * symlog(forces_norm)).reshape(self.num_envs, -1)
    
    @observation_func
    def contact_indicator(self):
        forces = self.contact_sensor.data.net_forces_w_history.mean(dim=1)
        return (forces.norm(dim=-1) > 1.).float()

    @observation_func
    def feet_pos_b(self):
        feet_pos_b = quat_rotate_inverse(
            self.robot.data.root_quat_w.unsqueeze(1),
            self._feet_pos - self.robot.data.root_pos_w.unsqueeze(1)
        )
        return feet_pos_b.reshape(self.num_envs, -1)
    
    @reward_func
    def linvel(self):
        linvel_w = self.robot.data.root_lin_vel_w
        linvel_error = square_norm(linvel_w - self._command)
        return 1. / (1. + linvel_error / 0.25)
    
    @reward_func
    def heading(self):
        return noarmalize(self.robot.data.root_lin_vel_b)[:, [0]]
    
    @reward_func
    def base_height(self):
        return self.robot.data.root_pos_w[:, [2]] - self.target_base_height

    @reward_func
    def energy(self):
        energy = (
            (self.robot.data.joint_vel * self.robot.data.applied_torque)
            .abs()
            .sum(dim=-1, keepdim=True)
        )
        return - energy
    
    @reward_func
    def joint_acc_l2(self):
        return - self.robot.data.joint_acc.square().sum(dim=-1, keepdim=True)
    
    @reward_func
    def joint_torques_l2(self):
        return - self.robot.data.applied_torque.square().sum(dim=-1, keepdim=True)

    @reward_func
    def action_rate_l2(self):
        return - (self._actions - self._prev_actions).square().sum(dim=-1, keepdim=True)

    @reward_func
    def survive(self):
        return torch.ones(self.num_envs, 1, device=self.device)
    
    @reward_func
    def orientation(self):
        return self.robot.data.projected_gravity_b[:, [2]].square()
    
    @termination_func
    def crash(self):
        terminated = (
            (self.robot.data.root_pos_w[:, 2] <= self.target_base_height * 0.5)
            | (self.robot.data.projected_gravity_b[:, 2] >= -0.5)
            | (self.contact_sensor.data.net_forces_w[:, 0].norm(dim=-1) > 1.)
        ).unsqueeze(1)
        return terminated

    def motor_params(self, env_ids: torch.Tensor):
        if not hasattr(self, "base_legs"):
            self.base_legs = self.robot.actuators["base_legs"]
            self.base_legs.default_stiffness = self.base_legs.stiffness.clone()
            self.base_legs.default_damping = self.base_legs.damping.clone()
        self.base_legs.stiffness[env_ids] = random_shift(self.base_legs.default_stiffness[env_ids], -.2, .2)
        self.base_legs.damping[env_ids] = random_shift(self.base_legs.default_damping[env_ids], -.2, .2)

    def body_masses(self):
        self.robot.body_physx_view.get_masses()
        pass


def random_scale(x: torch.Tensor, low: float, high: float):
    return x * (torch.rand_like(x) * (high - low) + low)

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def square_norm(x: torch.Tensor):
    return x.square().sum(dim=-1, keepdim=True)

def noarmalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(dim=-1, keepdim=True)

def symlog(x: torch.Tensor):
    return x.sign() * torch.log(x.abs() + 1.)
