import torch

from typing import TYPE_CHECKING
from isaaclab.utils.math import yaw_quat
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor
    from active_adaptation.envs.mdp.commands.locomotion import Command2

from .locomotion import Reward, normalize

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(-1, True)

class feet_clearance(Reward):
    """
    Avoid self-tripping by penalize a distance too small between the feet.
    """
    def __init__(self, env, feet_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id, feet_names = self.asset.find_bodies(feet_names)
        self.feet_id = torch.tensor(self.feet_id, device=self.device)
        self.thres = 0.16
    
    def compute(self) -> torch.Tensor:
        feet_pos_w = self.asset.data.body_pos_w[:, self.feet_id]
        distance_xy = (feet_pos_w[:, 0, :2] - feet_pos_w[:, 1, :2]).norm(dim=-1, keepdim=True)
        return (- self.thres + distance_xy).clamp_max(0.) / self.thres


class knee_distance(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=...):
        super().__init__(env, weight, enabled, clip_range)


class feet_swing(Reward):
    def __init__(self, env, feet_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id = self.asset.find_bodies(feet_names)[0]
        self.phase: torch.Tensor = self.asset.data.phase
        self.command_manager = self.env.command_manager

        self.feet_vel_buf = torch.zeros(self.num_envs, 2, 3, 4, device=self.device)
    
    def update(self):
        self.feet_vel_buf[..., 1:] = self.feet_vel_buf[..., :-1]
        self.feet_vel_buf[..., 0] = self.asset.data.body_lin_vel_w[:, self.feet_id]

    def reset(self, env_ids):
        self.feet_vel_buf[env_ids] = 0.

    def compute(self) -> torch.Tensor:
        feet_linvel = self.feet_vel_buf.mean(-1)
        swing_vel = torch.zeros_like(feet_linvel)
        swing_vel[:] = self.command_manager.command_linvel.unsqueeze(1)
        phase_sin = self.phase.sin()
        swing_vel[:, 0] *= (phase_sin > +0.15).float().unsqueeze(1)
        swing_vel[:, 1] *= (phase_sin < -0.15).float().unsqueeze(1)
        swing_vel = quat_rotate(yaw_quat(self.asset.data.root_quat_w).unsqueeze(1), swing_vel)
        # reward = torch.exp(- 2 * (feet_linvel - swing_vel).abs().sum(-1)).sum(1, True)

        max_speed = self.command_manager.command_speed.unsqueeze(1)
        reward = dot(normalize(swing_vel), feet_linvel).clamp_max(max_speed)
        reward = torch.where(reward>0, reward.sqrt(), reward).sum(1, True)
        return reward.reshape(self.num_envs, 1) * (~self.command_manager.is_standing_env)
    
    def debug_draw(self):
        feet_pos = self.asset.data.body_pos_w[:, self.feet_id]
        feet_linvel = self.feet_vel_buf.mean(-1)
        swing_vel = torch.zeros_like(feet_pos)
        swing_vel[:] = self.command_manager.command_linvel.unsqueeze(1)
        swing_vel[:, 0] *= (self.phase.sin() > +0.15).float().unsqueeze(1)
        swing_vel[:, 1] *= (self.phase.sin() < -0.15).float().unsqueeze(1)
        swing_vel = quat_rotate(yaw_quat(self.asset.data.root_quat_w).unsqueeze(1), swing_vel)
        self.env.debug_draw.vector(feet_pos.reshape(-1, 3), swing_vel.reshape(-1, 3))
        self.env.debug_draw.vector(feet_pos.reshape(-1, 3), feet_linvel.reshape(-1, 3), color=(1., 0., 0.2, 1.))


class leg_swing(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.phase: torch.Tensor = self.asset.data.phase
        self.command_manager = self.env.command_manager
        self.hip_ids = self.asset.find_joints(".*leg_joint1")[0]
        self.knee_ids = self.asset.find_joints(".*leg_joint4")[0]
    
    def compute(self) -> torch.Tensor:
        phase_sin = self.phase.sin()
        self.hip_pos = self.asset.data.joint_pos[:, self.hip_ids]
        self.knee_pos = self.asset.data.joint_pos[:, self.knee_ids]
        r1 = torch.where(self.knee_pos < -0.25, 0., -(-0.25 - self.knee_pos).abs())
        r2 = torch.where(self.hip_pos > -0.25, 0., -(-0.25 - self.hip_pos).abs())
        r = r1 + r2
        r[:, 0] *= (phase_sin > +0.15).float()
        r[:, 1] *= (phase_sin < -0.15).float()
        r = r.sum(1, True) * (~self.command_manager.is_standing_env)
        return r


class feet_orientation(Reward):

    def __init__(self, env, feet_names: str, weight: float, enabled: bool = True, body_name: str=None):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id = self.asset.find_bodies(feet_names)[0]
        self.heading_feet = torch.tensor([[[1., 0., 0.]]], device=self.device)
        self.heading_root = torch.tensor([[[1., 0., 0.]]], device=self.device)
        if body_name is not None:
            self.body_id = self.asset.find_bodies(body_name)[0][0]
        else:
            self.body_id = None
        self.command_manager: Command2 = self.env.command_manager
    
    def update(self):
        quat_feet = yaw_quat(self.asset.data.body_quat_w[:, self.feet_id])
        self.feet_fwd = quat_rotate(quat_feet, self.heading_feet)
        
        if self.body_id is not None:
            quat_body = yaw_quat(self.asset.data.body_quat_w[:, self.body_id]).unsqueeze(1)
        else:
            quat_body = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        self.body_fwd = quat_rotate(quat_body, self.heading_root)

    def compute(self) -> torch.Tensor:
        reward_alignment = - (self.feet_fwd - self.body_fwd).square().sum(-1)
        # reward_alignment = dot(self.feet_fwd, self.body_fwd).square().squeeze(-1)
        return (reward_alignment).sum(1, True)

    # def debug_draw(self):
    #     feet_pos = self.asset.data.body_pos_w[:, self.feet_id]
    #     self.env.debug_draw.vector(
    #         feet_pos.reshape(-1, 3),
    #         self.feet_fwd.reshape(-1, 3),
    #         color=(1., 1., 0., 1.)
    #     )
    #     self.env.debug_draw.vector(
    #         self.asset.data.root_pos_w, 
    #         self.body_fwd,
    #         color=(1., 1., 0., 1.)
    #     )


class joint_pos_default(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
    
    def compute(self) -> torch.Tensor:
        dev = self.asset.data.joint_pos[:, self.joint_ids] - self.default_joint_pos
        return - dev.square().mean(1, True)

class feet_step(Reward):
    def __init__(self, env, feet_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id, feet_names = self.asset.find_bodies(feet_names)
        self.phase: torch.Tensor = self.asset.data.phase
        self.heading_root = torch.tensor([[1., 0., 0.]], device=self.device)
        self.command_manager = self.env.command_manager
        print(f"Feet names: {feet_names}, be aware of the order!")
    
    def compute(self) -> torch.Tensor:
        quat_root = yaw_quat(self.asset.data.root_quat_w)
        feet_displacement = (
            - self.asset.data.body_pos_w[:, self.feet_id[0]]
            + self.asset.data.body_pos_w[:, self.feet_id[1]]
        )
        feet_displacement = dot(quat_rotate(quat_root, self.heading_root), feet_displacement)
        phase_cos = self.phase.cos()
        reward = (phase_cos.sign().unsqueeze(1) * feet_displacement).clamp(max=0.4)
        return reward.reshape(self.num_envs, 1) * (~self.command_manager.is_standing_env)

    def debug_draw(self):
        quat_root = yaw_quat(self.asset.data.root_quat_w)
        feet_displacement = (
            - self.asset.data.body_pos_w[:, self.feet_id[0]]
            + self.asset.data.body_pos_w[:, self.feet_id[1]]
        )
        feet_displacement_projected = dot(quat_rotate(quat_root, self.heading_root), feet_displacement)
        phase_cos = self.phase.cos()
        reward = phase_cos.sign() * feet_displacement_projected.squeeze(1)
        positive = reward > 0
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[positive][:, self.feet_id[0]],
            feet_displacement[positive],
            color=(0., 1., 0., 1.)
        )
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[~positive][:, self.feet_id[0]],
            feet_displacement[~positive],
            color=(1., 0., 0., 1.)
        )


class body_orientation(Reward):
    def __init__(self, env, body_name, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id, body_name = self.asset.find_bodies(body_name)
        self.body_id = self.body_id[0]
        self.x_vec = torch.tensor([1., 0., 0.], device=self.device)
        self.y_vec = torch.tensor([0., -1., 0.], device=self.device)
        self.body_heading_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.root_heading_vec = torch.zeros(self.num_envs, 3, device=self.device)
    
    def update(self):
        body_yaw_quat = yaw_quat(self.asset.data.body_quat_w[:, self.body_id])
        root_yaw_quat = yaw_quat(self.asset.data.root_quat_w)
        self.body_heading_vec[:] = quat_rotate(body_yaw_quat, self.y_vec)
        self.root_heading_vec[:] = quat_rotate(root_yaw_quat, self.x_vec)
    
    def compute(self) -> torch.Tensor:
        reward = dot(self.body_heading_vec, self.root_heading_vec)
        return (reward.square() * reward.sign()).reshape(self.num_envs, 1)

    # def debug_draw(self):
    #     self.env.debug_draw.vector(
    #         self.asset.data.body_pos_w[:, self.body_id],
    #         self.body_heading_vec
    #     )


class body_upright(Reward):
    """
    Reward for keeping the specified body upright.
    """
    def __init__(self, env, body_name: str, weight, enabled = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id, body_name = self.asset.find_bodies(body_name)
    
    def compute(self) -> torch.Tensor:
        down = torch.tensor([[0., 0., -1.]], device=self.device)
        g = quat_rotate_inverse(
            self.asset.data.body_quat_w[:, self.body_id],
            down.expand(self.num_envs, len(self.body_id), 3)
        )
        rew = 1. - g[:, :, :2].square().sum(-1)
        return rew.mean(1, True)


class arm_swing(Reward):
    def __init__(self, env, arm_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, body_names = self.asset.find_bodies(arm_names)
        self.phase: torch.Tensor = self.asset.data.phase
        self.command_manager = self.env.command_manager

        self.arm_vel_buf = torch.zeros(self.num_envs, 2, 3, 4, device=self.device)
    
    def update(self):
        self.arm_vel_buf[..., 1:] = self.arm_vel_buf[..., :-1]
        self.arm_vel_buf[..., 0] = self.asset.data.body_lin_vel_w[:, self.body_ids]

    def reset(self, env_ids):
        self.arm_vel_buf[env_ids] = 0.
    
    def compute(self) -> torch.Tensor:
        arm_linvel = self.arm_vel_buf.mean(-1)
        phase_sin = self.phase.sin()
        swing_vel = torch.zeros_like(arm_linvel)
        swing_vel[:] = self.command_manager.command_linvel.unsqueeze(1)
        phase_sin = self.phase.sin()
        swing_vel[:, 0] *= (phase_sin < +0.15).float().unsqueeze(1)
        swing_vel[:, 1] *= (phase_sin > -0.15).float().unsqueeze(1)
        swing_vel = quat_rotate(yaw_quat(self.asset.data.root_quat_w).unsqueeze(1), swing_vel)
        
        reward = (normalize(swing_vel) * arm_linvel).sum(-1)
        reward = reward.clamp(max=self.command_manager.command_speed).sum(1, True)
        # reward = torch.where(reward>0, reward.log1p().clamp(max=1.0), reward).sum(1, True)
        return reward.reshape(self.num_envs, 1) * (~self.command_manager.is_standing_env)


class arm_velocity(Reward):
    def __init__(self, env, arm_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, body_names = self.asset.find_bodies(arm_names)
    
    def compute(self) -> torch.Tensor:
        arm_linvel = self.asset.data.body_lin_vel_w[:, self.body_ids]
        root_linvel = self.asset.data.root_lin_vel_w
        d = (arm_linvel - root_linvel.unsqueeze(1)).square().sum(-1)
        return - d.mean(1, True)


class arm_well_being(Reward):
    def __init__(self, env, arm_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, body_names = self.asset.find_bodies(arm_names)
        # self.target_x = torch.tensor([0.0, 0.0], device=self.device)
        # self.target_y = torch.tensor([0.21, -0.21], device=self.device)
    
    def update(self):
        self.arm_pos = self.asset.data.body_pos_w[:, self.body_ids]
        self.root_pos = self.asset.data.root_pos_w
        self.root_quat = self.asset.data.root_quat_w
    
    def compute(self) -> torch.Tensor:
        arm_pos_b = quat_rotate_inverse(
            self.root_quat.unsqueeze(1), 
            self.arm_pos - self.root_pos.unsqueeze(1)
        )
        rx = - (arm_pos_b[:, :, 0].mean(1)).square()
        ry = - (arm_pos_b[:, :, 1].mean(1)).square()
        return (rx + ry).reshape(self.num_envs, 1)


class feet_impact(Reward):
    def __init__(self, env, feet_names: str, weight, enabled = True):
        super().__init__(env, weight, enabled)

        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(feet_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self.asset_body_ids, self.asset_body_names = self.asset.find_bodies(feet_names)

        self.in_contact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.impact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.detach = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.has_impact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.impact_point = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)
        self.detach_point = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)

    def reset(self, env_ids):
        self.has_impact[env_ids] = False
    
    def update(self):
        self.contact_force = self.contact_sensor.data.net_forces_w_history[:, :, self.body_ids]
        feet_pos_w = self.asset.data.body_pos_w[:, self.asset_body_ids]
        in_contact = (self.contact_force.norm(dim=-1) > 0.01).any(dim=1)
        self.impact = (~self.in_contact) & in_contact
        self.detach = self.in_contact & (~in_contact)
        self.in_contact = in_contact
        self.has_impact.logical_or_(self.impact)
        self.impact_point[self.impact] = feet_pos_w[self.impact]
        self.detach_point[self.detach] = feet_pos_w[self.detach]
    
    def compute(self):
        r = self.contact_force.mean(1).square().sum(-1)
        return - (self.impact * r).sum(1, True)


class angular_momentum(Reward):
    """
    https://arxiv.org/pdf/2409.16611
    """
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_mass = self.asset.root_physx_view.get_masses().to(self.device).reshape(self.num_envs, -1, 1)
        self.body_inertia = self.asset.root_physx_view.get_inertias().to(self.device).reshape(self.num_envs, -1, 3, 3)
        self.com_offset = self.asset.root_physx_view.get_coms()[:, :, :3].to(self.device)
        self.body_mass_weight = self.body_mass / self.body_mass.sum(1, keepdim=True)

    def update(self):
        self.body_pos = self.asset.data.body_pos_w
        self.body_lin_vel = self.asset.data.body_lin_vel_w
        self.body_ang_vel = self.asset.data.body_ang_vel_w
        self.root_pos = self.asset.data.root_pos_w
        self.root_lin_vel = self.asset.data.root_lin_vel_w
    
    def compute(self):
        body_com_pos = self.body_pos + quat_rotate(
            self.asset.data.body_quat_w,
            self.com_offset
        )
        com_pos = (body_com_pos * self.body_mass_weight).sum(1)
        body_pos = body_com_pos - com_pos.unsqueeze(1)
        
        m1 = body_pos.cross(self.body_lin_vel * self.body_mass, dim=-1)
        body_ang_vel_b = quat_rotate_inverse(
            self.asset.data.body_quat_w,
            self.body_ang_vel
        )
        m2 = (self.body_inertia @ body_ang_vel_b.unsqueeze(-1)).squeeze(-1)
        m2 = quat_rotate(self.asset.data.body_quat_w, m2)
        r = (m1 + m2).sum(1).square().sum(-1)
        return r.reshape(self.num_envs, 1)


class com_pos(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids = slice(None)
        self.body_mass = self.asset.root_physx_view.get_masses()[:, self.body_ids].to(self.device).reshape(self.num_envs, -1, 1)
        self.mass_weight = self.body_mass / self.body_mass.sum(1, keepdim=True)
        self.com_offset = self.asset.root_physx_view.get_coms()[:, self.body_ids, :3].to(self.device)

    def update(self):
        self.body_pos_w = self.asset.data.body_pos_w[:, self.body_ids]
        self.body_quat_w = self.asset.data.body_quat_w[:, self.body_ids]
        self.root_pos_w = self.asset.data.root_pos_w
    
    def compute(self):
        self.body_com_pos = self.body_pos_w + quat_rotate(
            self.body_quat_w,
            self.com_offset
        )
        self.com_pos_w = (self.body_com_pos * self.mass_weight).sum(1)
        self.com_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.com_pos_w - self.root_pos_w
        )
        return self.com_pos_b[:, 0].clamp_max(0.).unsqueeze(1)
    
    def debug_draw(self):
        self.env.debug_draw.point(
            self.com_pos_w.reshape(-1, 3),
            size=40,
            color=(1.0, 0.0, 0.0, 1.0),
        )


class oscillator_humanoid(Reward):
    def __init__(
        self,
        env,
        feet_names: str="[l,r]leg_link6",
        margin: float=0.,
        weight=1.0,
        enabled = True,
    ):
        super().__init__(env, weight, enabled)
        self.margin = margin
        self.target_swing_height = 0.20

        self.asset: Humanoid = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.command_manager: Command2 = self.env.command_manager
        
        self.feet_ids, feet_names = self.contact_sensor.find_bodies(feet_names)
        self.mass = self.asset.data.default_mass[0].sum().to(self.device)
        self.gravity = self.mass * 9.81
        
        self.phi: torch.Tensor = self.asset.phi
        self.phi[:, 0] = torch.pi
        self.phi[:, 1] = 0.
        self.phi_dot: torch.Tensor = self.asset.phi_dot
        self.grf_substep = torch.zeros(self.num_envs, self.env.decimation, len(self.feet_ids), device=self.device)
        
        self.omega = torch.zeros(self.num_envs, 1, device=self.device)
        self.omega.uniform_(2., 3.).mul_(torch.pi)
        
        self.rest_target = torch.pi * 3 / 2
        self.keep_steping = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)

    def reset(self, env_ids):
        self.keep_steping[env_ids] = torch.rand(len(env_ids), 1, device=self.device) < 0.5
    
    def post_step(self, substep):
        grf = self.contact_sensor.data.net_forces_w[:, self.feet_ids].norm(dim=-1)
        self.grf_substep[:, substep] = grf
    
    def update(self):
        self.grf = self.grf_substep.mean(1) / self.gravity
        inp = (self.command_manager.command_speed + self.command_manager.command_angvel.reshape(-1, 1).abs()) > 0.1
        phi_dot = self.omega + torch.randn_like(self.omega).clamp(-3., 3.) * 0.1
        self.phi[:] = (self.phi + self.phi_dot * self.env.step_dt) % (2 * torch.pi)
        self.phi_dot[:] = phi_dot

    def compute(self):
        phi_sin = self.phi.sin()
        feet_height = self.asset.data.feet_height.clamp_max(self.target_swing_height)
        r = (feet_height - self.grf.clamp_max(0.4)) * phi_sin * (phi_sin.abs() > self.margin)
        return r.sum(1, True)

