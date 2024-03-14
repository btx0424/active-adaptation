import torch

from omni.isaac.orbit.sensors import ContactSensor, RayCaster
from omni.isaac.orbit.actuators import DCMotor
from active_adaptation.envs.base import Env
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
import active_adaptation.envs.mdp as mdp

from tensordict.tensordict import TensorDictBase
from torchrl.data import (
    CompositeSpec, 
    UnboundedContinuousTensorSpec
)

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

from .mdp import CommandManager1
from collections import OrderedDict

class Filter:
    def __init__(self, shape, device):
        self.data_history = torch.zeros(shape, 4, device=device)
        self.data = torch.zeros(shape, device=device)

    def reset(self, env_ids: torch.Tensor, value=0.):
        self.data[env_ids] = value
    
    def update(self, value: torch.Tensor):
        self.data_history[..., :-1] = self.data_history[..., 1:]
        self.data_history[..., -1] = value
        self.data[:] = self.data_history.mean(dim=-1)


class LocomotionEnv(Env):

    feet_name_expr: str

    def __init__(self, cfg):
        super().__init__(cfg)
        self.action_scaling = 0.5

        self.robot = self.scene.articulations["robot"]
        self.feet_indices, self.feet_names = self.robot.find_bodies(self.feet_name_expr)
        self.num_feet = len(self.feet_indices)

        self.contact_sensor: ContactSensor = self.scene.sensors.get("contact_forces", None)
        self.height_scanner: RayCaster = self.scene.sensors.get("height_scanner", None)

        self.init_root_state = self.robot.data.default_root_state.clone()
        self.init_joint_pos = self.robot.data.default_joint_pos.clone()
        self.init_joint_vel = self.robot.data.default_joint_vel.clone()
        
        self.default_joint_pos = self.init_joint_pos.clone()
        
        try:
            from active_adaptation.utils.debug import DebugDraw
            self.debug_draw = DebugDraw()
            print("[INFO] Debug Draw API enabled.")
        except ModuleNotFoundError:
            pass
        
        self.lookat_env_i = (
            self.scene._default_env_origins.cpu() 
            - torch.tensor(self.cfg.viewer.lookat)
        ).norm(dim=-1).argmin()

        self.command_manager = CommandManager1(self)

        with torch.device(self.device):
            # self.action_scale = torch.ones(self.num_envs, 1)
            self.action_alpha = torch.ones(self.num_envs, 1)
            self.action_buf = torch.zeros(self.num_envs, self.action_dim, 4)
            self.last_action = torch.zeros(self.num_envs, self.action_dim)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)
            self.root_pos_history = torch.zeros(self.num_envs, 5, 3)
            self.last_contact = torch.zeros(self.num_envs, self.num_feet)
            
        # set by subclass
        self.resample_interval = 300
        self.resample_prob = 0.6

    @property
    def action_dim(self):
        return sum(actuator.num_joints for actuator in self.robot.actuators.values())

    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.init_root_state[env_ids]
        init_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        self.robot.write_root_state_to_sim(
            init_root_state, 
            env_ids=env_ids
        )
        # self.robot.write_joint_state_to_sim(
        #     random_scale(self.init_joint_pos[env_ids], 0.8, 1.2),
        #     self.init_joint_vel[env_ids],
        #     env_ids=env_ids
        # )
        self.stats[env_ids] = 0.
        self.action_buf[env_ids] = 0.
        self.last_action[env_ids] = 0.
        self.delay[env_ids] = torch.randint(0, 4, (len(env_ids), 1), device=self.device)

        self.scene.reset(env_ids)
        self.scene.update(dt=self.physics_dt)
        for _, randomization in self.randomizations.items():
            randomization.reset(env_ids)
        
        self.command_manager.reset(env_ids=env_ids)
    
    def _update(self):
        super()._update()
        should_resample = (
            (self.episode_length_buf % self.resample_interval == 0)
            & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        )
        self.command_manager.update(resample=should_resample.nonzero().squeeze(-1))

        if self.sim.has_gui() and hasattr(self, "debug_draw"):
            self.debug_draw.clear()
            robot_pos = (
                self.robot.data.root_pos_w.cpu()
                + torch.tensor([0., 0., 0.2])
            )
            command_linvel_w = quat_rotate(
                self.robot.data.root_quat_w,
                self.command_manager._command_linvel
            )
            self.debug_draw.vector(
                robot_pos, 
                command_linvel_w,
                color=(1., 1., 1., 1.)
            )
            self.debug_draw.vector(
                robot_pos,
                self.command_manager._command_heading,
                color=(.2, .2, 1., 1.)
            )
            self.debug_draw.vector(
                robot_pos, 
                self.robot.data.root_lin_vel_w,
                color=(1., .5, .5, 1.)
            )
            for group in self.observation_funcs.values():
                for obs in group.values():
                    obs.debug_draw()
            for rand in self.randomizations.values():
                rand.debug_draw()

    def render(self, mode: str = "human"):
        robot_pos = self.robot.data.root_pos_w[self.lookat_env_i].cpu()
        if mode == "rgb_array":
            eye = torch.tensor(self.cfg.viewer.eye) + robot_pos
            lookat = torch.tensor(self.cfg.viewer.lookat) + robot_pos
            self.sim.set_camera_view(eye, lookat)
        return super().render(mode)

    def apply_action(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            # random packet loss: repeat previous actions
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = tensordict["action"]
            action = self.action_buf.take_along_dim(self.delay.unsqueeze(1), dim=-1)
            self.last_action.lerp_(action.squeeze(-1), self.action_alpha)

            pos_target = self.last_action * self.action_scaling + self.default_joint_pos
            self.robot.set_joint_position_target(pos_target)
        self.robot.write_data_to_sim()

    @mdp.observation_func
    def height_scan(self, prim_path):
        asset = self.scene["robot"]
        height_scanner = self.scene.sensors["height_scanner"]
        if not hasattr(height_scanner, "_view"):
            return torch.zeros(self.num_envs, 1, 11, 17, deivce=self.device)
        else:
            root_pos_w = asset.data.root_pos_w
            ray_hits_w = height_scanner.data.ray_hits_w
            height_scan = root_pos_w[:, [2]].unsqueeze(1) - ray_hits_w[:, :, [2]]
            return height_scan.reshape(self.num_envs, 1, 11, 17).clamp(-2., 2.)
    
    @mdp.observation_func
    def command(self):
        if not hasattr(self, "command_manager"):
            return torch.zeros(self.num_envs, 3, device=self.device)
        return self.command_manager.command
    
    @mdp.observation_func
    def prev_command(self):
        return self.command_manager.command_prev

    @mdp.observation_func
    def projected_gravity_b(self):
        return self.scene["robot"].data.projected_gravity_b
    
    @mdp.observation_func
    def root_linvel_b(self):
        return self.scene["robot"].data.root_lin_vel_b
    
    @mdp.observation_func
    def prev_actions(self):
        if not hasattr(self, "action_buf"):
            return torch.zeros(self.num_envs, self.action_dim * 4, device=self.device)
        return self.action_buf.reshape(self.num_envs, -1)
    
    @mdp.observation_func
    def applied_action(self):
        if not hasattr(self, "last_action"):
            return torch.zeros(self.num_envs, self.action_dim, device=self.device)
        return self.last_action
    
    @mdp.reward_func
    def linvel_projection(self):
        linvel_b = self.scene["robot"].data.root_lin_vel_b[:, :2]
        command_linvel_b = self.command_manager._command_linvel[:, :2]
        projection = (linvel_b * command_linvel_b).sum(dim=-1, keepdim=True) 
        return projection.clamp_max(self.command_manager._command_speed)
    
    @mdp.reward_func
    def linvel_exp(self):
        linvel_b = self.scene["robot"].data.root_lin_vel_b[:, :2]
        linvel_error = square_norm(linvel_b - self.command_manager._command_linvel[:, :2])
        return torch.exp( - linvel_error / 0.25)
    
    @mdp.reward_func
    def angvel_z_exp(self):
        angvel_error = (self.command_manager.command[:, [2]] - self.scene["robot"].data.root_ang_vel_b[:, [2]]).square()
        return torch.exp( - angvel_error / 0.25)
    
    @mdp.reward_func
    def heading(self):
        root_quat = self.scene["robot"].data.root_quat_w
        heading_b_x = quat_rotate_inverse(root_quat, self.command_manager._command_heading)[:, [0]]
        return 0.5 * (heading_b_x + heading_b_x.sign() * heading_b_x.square())
    
    @mdp.reward_func
    def base_height(self):
        height = self.scene["robot"].data.feet_pos_b[:, :, 2].mean(1, keepdim=True).abs()
        return (height / self.target_base_height).square().clamp_max(1.)
    
    @mdp.reward_func
    def joint_torques_l2(self):
        return - self.scene["robot"].data.applied_torque.square().sum(dim=-1, keepdim=True)

    @mdp.reward_func
    def action_rate_l2(self):
        action_diff = self.action_buf[:, :, 0] - self.action_buf[:, :, 1]
        return - action_diff.square().sum(dim=-1, keepdim=True)
    
    @mdp.reward_func
    def action_rate2_l2(self):
        action_diff = (
            self.action_buf[:, :, 0] - 2 * self.action_buf[:, :, 1] + self.action_buf[:, :, 2]
        )
        return - action_diff.square().sum(dim=-1, keepdim=True)

    @mdp.reward_func
    def orientation(self):
        return -self.scene["robot"].data.projected_gravity_b[:, :2].square().sum(-1, True)
        return self.scene["robot"].data.projected_gravity_b[:, [2]].square()
    
    @mdp.reward_func
    def feet_slip(self):
        i = self.contact_indicator()
        feet_vel = self.scene["robot"].data.body_lin_vel_w[:, self.foot_indices]
        return - (i * feet_vel.norm(dim=-1)).sum(dim=1, keepdim=True)
    
    @mdp.reward_func
    def stand(self):
        jpos_error = square_norm(self.scene["robot"].data.joint_pos - self.scene["robot"].data.default_joint_pos)
        cost = - (jpos_error) * self.command_manager._command_stand
        return cost

    class feet_pos_b(mdp.body_pos):
        def __init__(self, env: "LocomotionEnv"):
            super().__init__(env, env.feet_name_expr)
            self.asset.data.feet_pos_b = self.body_pos_b
    
    class feet_air_time(mdp.Reward):
        def __init__(self, env: "LocomotionEnv"):
            super().__init__(env)
            self.asset = self.env.scene["robot"]
            self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
            self.feet_ids, _ = self.asset.find_bodies(self.env.feet_name_expr)

        def __call__(self):
            first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.feet_ids]
            last_air_time = self.contact_sensor.data.last_air_time[:, self.feet_ids]
            reward = torch.sum(last_air_time.clamp(max=0.2) * first_contact, dim=1, keepdim=True)
            return reward
    

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

def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def flip_lr(joints: torch.Tensor):
    return joints.reshape(-1, 3, 2, 2).flip(-1).reshape(-1, 12)

def flip_fb(joints: torch.Tensor):
    return joints.reshape(-1, 3, 2, 2).flip(-2).reshape(-1, 12)

