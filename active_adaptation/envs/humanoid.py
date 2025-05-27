from math import inf
import torch

from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.actuators import DCMotor
from isaaclab.assets import Articulation
from isaaclab.utils.math import yaw_quat, quat_mul
from isaaclab.utils.warp import raycast_mesh
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

from active_adaptation.envs.locomotion import Env, LocomotionEnv

import active_adaptation.envs.mdp as mdp

class Humanoid(LocomotionEnv):
    
    feet_name_expr = ".*ankle_link"

    class feet_too_close(mdp.Termination):
        def __init__(self, env, feet_names: str, thres: float=0.1):
            super().__init__(env)
            self.threshold = thres
            self.asset: Articulation = self.env.scene["robot"]
            self.body_ids = self.asset.find_bodies(feet_names)[0]
            assert len(self.body_ids) == 2, "Only support two feet"

        def __call__(self):
            feet_pos = self.asset.data.body_pos_w[:, self.body_ids]
            distance_xy = (feet_pos[:, 0, :2] - feet_pos[:, 1, :2]).norm(dim=-1)
            return (distance_xy < self.threshold).reshape(-1, 1)
    
    class arm_swing(mdp.Reward):
        
        l: float = 0.2

        def __init__(self, env, arm_names: str,weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.arm_ids = self.asset.find_bodies(arm_names)[0]
            self.phase: torch.Tensor = self.asset.data.phase
            self.fwd_vec = torch.tensor([1., 0., 0.], device=self.device)
            self.command_manager = self.env.command_manager

        def compute(self) -> torch.Tensor:
            quat_root = yaw_quat(self.asset.data.root_quat_w)
            arm_displacement = (
                + self.asset.data.body_pos_w[:, self.arm_ids[0]]
                - self.asset.data.body_pos_w[:, self.arm_ids[1]]
            )
            arm_displacement = (quat_rotate(quat_root, self.fwd_vec) * arm_displacement).sum(-1, True)
            reward = (self.phase.cos().sign().unsqueeze(1) * arm_displacement).clamp(max=self.l)
            return reward.reshape(self.num_envs, 1) * (~self.command_manager.is_standing_env)

        
    class step_up(mdp.Reward):
        
        env: "Humanoid"

        def __init__(self, env, feet_names: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.body_ids = self.asset.find_bodies(feet_names)[0]
            assert len(self.body_ids) == 2, "Only support two feet"

            self.height_scan: torch.Tensor = self.asset.data.height_scan
            self.phase: torch.Tensor = self.asset.data.phase
            self.scan_size = self.height_scan.shape[-2:]
        
        def update(self):
            self.feet_pos = self.asset.data.body_pos_w[:, self.body_ids]
            self.feet_height = self.feet_pos[:, :, 2]
            height_scan = (
                self.height_scan 
                - self.asset.data.root_pos_w[:, 2].reshape(-1, 1, 1)
                + self.feet_height.min(dim=1).values.reshape(-1, 1, 1)
            )
            height_front = height_scan[:, :, self.scan_size[1]//2:].mean(dim=(1, 2))
            self.stairs_front = (height_front < -0.0).unsqueeze(1)
            
        def compute(self) -> torch.Tensor:
            phase_sin = self.phase.sin().unsqueeze(1)
            feet_height_diff = (self.feet_height[:, 0] - self.feet_height[:, 1]).unsqueeze(1)
            feet_height_diff = torch.where(phase_sin > 0, feet_height_diff, -feet_height_diff)
            r = (feet_height_diff.clamp(0, 0.15) / 0.15).sqrt()
            r = (self.stairs_front & (phase_sin.abs() > 0.1)) * r
            return r.reshape(self.num_envs, 1)

        def debug_draw(self):
            phase_sin = self.phase.sin().unsqueeze(1)
            with torch.device(self.device):
                feet_pos = self.asset.data.body_pos_w[:, self.body_ids]
                lift_foot = torch.where(phase_sin > 0, feet_pos[:, 0], feet_pos[:, 1])
                lift = torch.tensor([0, 0, 1.5]).expand_as(lift_foot)
            self.env.debug_draw.vector(
                lift_foot, 
                lift * self.stairs_front,
                size=5,
                color=(1, 0, 0, 1)
            )

    class root_orientation(mdp.Reward):
            
        env: "Humanoid"

        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]

        def compute(self) -> torch.Tensor:
            z = - self.asset.data.projected_gravity_b[:, 2]
            # y = self.asset.data.projected_gravity_b[:, 1].abs()
            x = self.asset.data.projected_gravity_b[:, 0].clamp_max(0.1)
            return (z + 2 * x).unsqueeze(1)
    

    class arm_velocity_exp(mdp.Reward):

        def __init__(self, env, arm_names: str, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.arm_ids = self.asset.find_bodies(arm_names)[0]

            self.action_manager: mdp.action.HumanoidWithArm = self.env.action_manager
            if not isinstance(self.action_manager, mdp.action.HumanoidWithArm):
                raise ValueError("`HumanoidWithArm` action manager required")
            
            with torch.device(self.device):
                self.arm_linvel_w = torch.zeros(self.num_envs, len(self.arm_ids), 3)
                self.arm_linvel_b = torch.zeros(self.num_envs, len(self.arm_ids), 3)
                self.error = torch.zeros(self.num_envs, len(self.arm_ids))
                self.cum_error = torch.zeros(self.num_envs, len(self.arm_ids))
                self.action_manager.cum_error = self.cum_error

        def reset(self, env_ids):
            self.cum_error[env_ids] = 0

        def update(self):
            arm_linvel_w = self.asset.data.body_lin_vel_w[:, self.arm_ids]
            arm_linvel_b = quat_rotate_inverse(self.asset.data.root_quat_w.unsqueeze(1), arm_linvel_w)
            self.error = (arm_linvel_b - self.action_manager.command_arm_linvel).square().sum(dim=-1)
            self.cum_error.add_(self.error * self.env.step_dt).mul_(0.99)
            self.arm_linvel_w[:] = arm_linvel_w
            self.arm_linvel_b[:] = arm_linvel_b
            
        def compute(self) -> torch.Tensor:
            r = torch.exp(- self.error / 0.25 ).mean(1, True)
            return r

        def debug_draw(self):
            arm_pos_w = self.asset.data.body_pos_w[:, self.arm_ids]
            command_arm_linvel = quat_rotate(self.asset.data.root_quat_w.unsqueeze(1), self.action_manager.command_arm_linvel)
            self.env.debug_draw.vector(
                arm_pos_w.reshape(-1, 3),
                command_arm_linvel.reshape(-1, 3),
                color=(0.5, 0.6, 0.5, 1),
            )
            self.env.debug_draw.vector(
                arm_pos_w.reshape(-1, 3),
                self.arm_linvel_w.reshape(-1, 3),
                color=(0.6, 0.5, 0.5, 1),
            )
    
    class arm_velocity_cum_error(mdp.Termination):
        def __init__(self, env, thres: float=0.8):
            super().__init__(env)
            self.threshold = thres
            self.asset: Articulation = self.env.scene["robot"]
            self.action_manager: mdp.action.HumanoidWithArm = self.env.action_manager
            if not isinstance(self.action_manager, mdp.action.HumanoidWithArm):
                raise ValueError("`HumanoidWithArm` action manager required")
            self.cum_error = self.action_manager.cum_error

        def __call__(self):
            return (self.cum_error > self.threshold).any(1, True)
    
    class command_arm_linvel(mdp.Observation):
        def __init__(self, env):
            super().__init__(env)
            self.asset: Articulation = self.env.scene["robot"]
            self.action_manager: mdp.action.HumanoidWithArm = self.env.action_manager
            if not isinstance(self.action_manager, mdp.action.HumanoidWithArm):
                raise ValueError("`HumanoidWithArm` action manager required")

        def compute(self) -> torch.Tensor:
            return self.action_manager.command_arm_linvel.reshape(self.num_envs, -1)

    class symmetry(mdp.Observation):
        def __init__(self, env, body_names: str):
            super().__init__(env)
            self.asset: Articulation = self.env.scene["robot"]
            self.body_ids, self.body_names = self.asset.find_bodies(body_names)
            
            self.flipy = torch.tensor([1., -1., 1.], device=self.device)

        def compute(self) -> torch.Tensor:
            root_quat = self.asset.data.root_quat_w
            root_pos  = self.asset.data.root_pos_w
            
            body_pos = quat_rotate(
                root_quat.unsqueeze(1),
                self.asset.data.body_pos_w[:, self.body_ids] - root_pos.unsqueeze(1)
            ).reshape(self.num_envs, -1, 2, 3)
            body_vel = quat_rotate(
                root_quat.unsqueeze(1),
                self.asset.data.body_lin_vel_w[:, self.body_ids]
            ).reshape(self.num_envs, -1, 2, 3)
            
            gravity = self.asset.data.projected_gravity_b
            lin_vel_b = self.asset.data.root_lin_vel_b
            ang_vel_b = self.asset.data.root_ang_vel_b
            left = torch.cat([
                gravity,
                lin_vel_b,
                ang_vel_b,
                body_pos.reshape(self.num_envs, -1),
                body_vel.reshape(self.num_envs, -1)
            ], dim=-1)
            right = torch.cat([
                gravity * self.flipy, 
                lin_vel_b * self.flipy,
                ang_vel_b * torch.tensor([-1., 1., -1.], device=self.device),
                (body_pos.flip(dims=(2,)) * self.flipy).reshape(self.num_envs, -1), 
                (body_vel.flip(dims=(2,)) * self.flipy).reshape(self.num_envs, -1)
            ], dim=-1)
            return torch.stack([left, right], dim=1)


    class hand_pose(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.hand_ids = self.asset.find_bodies(".*arm_link6")[0]
            self.hand_pos_target = torch.tensor([0.3, 0.0, 0.1], device=self.device)
        
        def compute(self) -> torch.Tensor:
            hand_pos = self.asset.data.body_pos_w[:, self.hand_ids]
            hand_pos_target = (
                self.asset.data.root_pos_w.unsqueeze(1) 
                + quat_rotate(self.asset.data.root_quat_w.unsqueeze(1), self.hand_pos_target)
            )
            diff = hand_pos - hand_pos_target
            return - diff.square().sum(dim=-1).sum(1, True)


    class attach_z(mdp.Reward):
        def __init__(self, env, weight, enabled: bool, target_height: float):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.target_height = target_height
            
            from .mdp.observations import _initialize_warp_meshes
            self.body_ids = self.asset.find_bodies(".*attach_point.*")[0]
            self.num_attach_points = len(self.body_ids)
            self.mesh = _initialize_warp_meshes("/World/ground", "cuda")

            with torch.device(self.device):
                self.attach_point_height = torch.full((self.num_envs, self.num_attach_points), self.target_height)
                self.ray_direction = torch.tensor([0., 0., -1.]).expand(self.num_envs, self.num_attach_points, 3)
                self.kp = torch.zeros(self.num_envs, 1)
                self.kd = 20

        def reset(self, env_ids):
            kp = 1200
            self.kp[env_ids] = kp

        def update(self):
            self.ray_hit_w = raycast_mesh(
                self.asset.data.root_pos_w,
                self.ray_direction,
                self.mesh,
                max_dist=100.0
            )[0]
            self.attach_point_height = self.asset.data.body_pos_w[:, self.body_ids, 2] - self.ray_hit_w[:, 2].unsqueeze(1)

        def step(self, substep):
            attach_point_linvel = self.asset.data.body_lin_vel_w[:, self.body_ids]
            self.force = torch.zeros(self.num_envs, 4, 3, device=self.device)
            self.force[:, :, 2] = (
                self.kp * (self.target_height - self.attach_point_height) + 
                self.kd * (0. - attach_point_linvel[:, :, 2])
            ) * (self.target_height > self.attach_point_height)
            self.force[:, :, :2] = - 1.0 * attach_point_linvel[:, :, :2] * self.force[:, :, 2].unsqueeze(2)
            force = quat_rotate_inverse(self.asset.data.body_quat_w[:, self.body_ids], self.force)
            self.asset._external_force_b[:, self.body_ids] += force
            self.asset.has_external_wrench = True

        def compute(self) -> torch.Tensor:
            return -(self.force / 50).square().sum(dim=-1).mean(1, True)

        def debug_draw(self):
            self.env.debug_draw.vector(
                self.asset.data.body_pos_w[:, self.body_ids],
                self.force / 100,
                color=(1., 0., 0., 1.),
                size=5.,
            )
