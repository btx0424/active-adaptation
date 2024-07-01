from math import inf
import torch

from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.actuators import DCMotor
from omni.isaac.lab.assets import Articulation, RigidObject
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from omni.isaac.lab.utils.math import yaw_quat


quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

from collections import OrderedDict
from .locomotion import LocomotionEnv, sample_quat
from . import mdp


class Dribble(LocomotionEnv):

    feet_name_expr: str = ".*foot"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ball: RigidObject = self.scene["ball"]
        self.init_ball_state = self.ball.data.default_root_state.clone()

    @property
    def action_dim(self):
        return 12
    
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.init_root_state[env_ids]
        origins = self.scene.env_origins[torch.randint(0, self.scene.num_envs, (len(env_ids),), device=self.device)]
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = sample_quat(len(env_ids), device=self.device)
        
        self.robot.write_root_state_to_sim(init_root_state, env_ids=env_ids)
        
        init_root_state = self.init_ball_state[env_ids]
        init_root_state[:, :3] += origins
        init_root_state[:, 0] += 0.2
        init_root_state[:, 1] += 0.2
        self.ball.write_root_state_to_sim(init_root_state, env_ids=env_ids)
        self.stats[env_ids] = 0.
        self.action_buf[env_ids] = 0.
        self.last_action[env_ids] = 0.
        self.delay[env_ids] = torch.randint(0, 4, (len(env_ids), 1), device=self.device)

        self.scene.reset(env_ids)
        self.scene.update(dt=self.physics_dt)
        self.command_manager.reset(env_ids=env_ids)
    
    class ball_state_b(mdp.Observation):
        def __init__(self, env):
            super().__init__(env)
            self.robot: Articulation =  self.env.scene["robot"]
            self.ball: RigidObject = self.env.scene["ball"]
        
        def __call__(self) -> torch.Tensor:
            robot_quat_yaw = yaw_quat(self.robot.data.root_quat_w)
            ball_pos = quat_rotate_inverse(
                robot_quat_yaw,
                self.ball.data.root_pos_w - self.robot.data.root_pos_w
            )
            ball_linvel = quat_rotate_inverse(
                robot_quat_yaw,
                self.ball.data.root_lin_vel_w
            )
            return torch.cat([ball_pos, ball_linvel], 1)
    
    class ball_command(mdp.Observation):
        def __init__(self, env):
            super().__init__(env)
            self.robot: Articulation =  self.env.scene["robot"]
            self.ball: RigidObject = self.env.scene["ball"]
            self.command_vel = torch.zeros(self.num_envs, 3, device=self.device)
            self.ball.data.command_vel = self.command_vel
        
        def reset(self, env_ids: torch.Tensor):
            command_vel = torch.zeros(len(env_ids), 3, device=self.device)
            command_vel[:, 0].uniform_(0.3, 0.6)
            self.command_vel[env_ids] = command_vel

        def __call__(self) -> torch.Tensor:
            robot_quat_yaw = yaw_quat(self.robot.data.root_quat_w)
            command_vel_b = quat_rotate_inverse(robot_quat_yaw, self.command_vel)
            return command_vel_b
        
        def debug_draw(self):
            self.env.debug_draw.vector(
                self.ball.data.root_pos_w,
                self.command_vel,
            )

    class ball_pos_exp(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation =  self.env.scene["robot"]
            self.ball: RigidObject = self.env.scene["ball"]
        
        def compute(self) -> torch.Tensor:
            d = - self.ball.data.root_pos_w[:, :2] + self.robot.data.root_pos_w[:, :2]
            target_vel = clip_norm(1. * d, 0.6)
            error = (self.robot.data.root_lin_vel_w - target_vel).square().sum(1, True)
            return 1 / (1. + 2 * error)
        
        def debug_draw(self):
            self.env.debug_draw.vector(
                self.robot.data.root_pos_w,
                self.ball.data.root_pos_w - self.robot.data.root_pos_w,
                color=(.8, .6, .2, 1.)
            )

    class ball_vel_exp(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation =  self.env.scene["robot"]
            self.ball: RigidObject = self.env.scene["ball"]

        def compute(self) -> torch.Tensor:
            linvel_error = (
                self.ball.data.command_vel[:, :2]
                - self.ball.data.root_lin_vel_w[:, :2]
            ).square().sum(1, True)
            linvel_exp = torch.exp(- 2. * linvel_error)
            self.robot.data.linvel_exp = linvel_exp
            return linvel_exp
    
    class ball_too_far(mdp.Termination):
        def __init__(self, env, thres: float):
            super().__init__(env)
            self.thres = thres
            self.robot: Articulation =  self.env.scene["robot"]
            self.ball: RigidObject = self.env.scene["ball"]
        
        def __call__(self) -> torch.Tensor:
            distance = (self.ball.data.root_pos_w - self.robot.data.root_pos_w).norm(dim=-1, keepdim=True)
            return distance > self.thres


def clip_norm(x: torch.Tensor, max_norm):
    x_norm = x.norm(dim=-1, keepdim=True)
    return torch.where(x_norm > max_norm, x / x_norm * max_norm, x)