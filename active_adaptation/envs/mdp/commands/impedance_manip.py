from typing import Optional, Tuple
import torch
import einops
from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.envs.mdp import observation, reward
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat,
    clamp_norm,
    wrap_to_pi,
    euler_from_quat,
    quat_from_euler_xyz,
    yaw_rotate,
    EMA
)

from tensordict import TensorClass
from pathlib import Path
from active_adaptation.envs.mdp.utils.forces import ConstantForce, SpringForce


import active_adaptation
if active_adaptation.get_backend() == "isaac":
    from isaaclab.markers import (
        BLUE_ARROW_X_MARKER_CFG,
        VisualizationMarkers,
        VisualizationMarkersCfg,
        sim_utils
    )


class ImpedanceCommand(TensorClass):
    setpoint: torch.Tensor
    setpoint_eef: torch.Tensor
    kp_base: torch.Tensor
    kd_base: torch.Tensor
    kp_eef: torch.Tensor
    kd_eef: torch.Tensor
    virtual_mass_base: torch.Tensor
    virtual_mass_eef: torch.Tensor
    mode: torch.Tensor # 0: world; 1: set lin vel
    set_lin_vel: torch.Tensor
    # set_ang_vel: torch.Tensor
    transmission: torch.Tensor # whether `eef_spring_force` is transmitted to the base

    @classmethod
    def zero(cls, num_envs: int, device: torch.device):
        return cls(
            setpoint=torch.zeros(num_envs, 6, device=device),
            setpoint_eef=torch.zeros(num_envs, 3, device=device),
            kp_base=torch.zeros(num_envs, 1, device=device),
            kd_base=torch.zeros(num_envs, 1, device=device),
            kp_eef=torch.zeros(num_envs, 1, device=device),
            kd_eef=torch.zeros(num_envs, 1, device=device),
            virtual_mass_base=torch.ones(num_envs, 1, device=device),
            virtual_mass_eef=torch.ones(num_envs, 1, device=device),
            mode=torch.zeros(num_envs, dtype=torch.long, device=device),
            set_lin_vel=torch.zeros(num_envs, 3, device=device),
            transmission=torch.zeros(num_envs, 1, device=device),
            batch_size=[num_envs],
        )


def round_robbin(buf: torch.Tensor, val: torch.Tensor):
    buf = buf.roll(1, dims=1)
    buf[:, 0] = val
    return buf


def make_point_marker(name: str, color: Tuple[float, float, float]):
    marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=f"/Visuals/Command/{name}",
            markers={
                name: sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            }
        )
    )
    marker.set_visibility(True)
    return marker


class ImpedanceCommandManager(Command):
    
    # training
    MODE0_PROB = 0.5
    FORCE_PROB = 0.5

    # evaluation
    # MODE0_PROB = 0.0
    # FORCE_PROB = 1.0

    def __init__(
        self,
        env,
        eef_body_name: str,
        arm_body_name: str,
        virtual_mass_range,
        ref_steps=[8, 16, 32],
        temporal_smoothing: int=32,
    ) -> None:
        super().__init__(env)
        self.virtual_mass_range = torch.tensor(virtual_mass_range, device=self.device).float()
        self.eef_body_id = self.asset.find_bodies(eef_body_name)[0][0]
        self.arm_body_id = self.asset.find_bodies(arm_body_name)[0][0]
        self.ref_steps = ref_steps
        self.temporal_smoothing = temporal_smoothing
        self.base_kp_range = (4., 40.)
        self.eef_kp_range = (4., 40.)
        
        self.cmd = ImpedanceCommand.zero(self.num_envs, self.device)
        
        with torch.device(self.device):
            bshape = (self.num_envs, self.temporal_smoothing + 1)
            self.ref_pos_w = torch.zeros(*bshape, 3)
            self.ref_lin_vel_w = torch.zeros(*bshape, 3)
            self.ref_lin_acc_w = torch.zeros(*bshape, 3)
            self.ref_rpy_w = torch.zeros(*bshape, 3)
            self.ref_ang_vel_w = torch.zeros(*bshape, 3)
            self.ref_ang_acc_w = torch.zeros(*bshape, 3)
            
            # assume ee is always forward
            self.eef_ref_pos_w = torch.zeros(*bshape, 3)
            self.eef_ref_lin_vel_w = torch.zeros(*bshape, 3)
            self.eef_ref_lin_acc_w = torch.zeros(*bshape, 3)

            self.setpoint_w = torch.zeros(self.num_envs, 3 + 3)
            self.setpoint_b = torch.zeros(self.num_envs, 3 + 3)
            self._cum_error = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        
        path = Path(__file__).parent / "target_archive.pt"
        self.ee_target_archive = torch.load(path)["target_pos"].to(self.device)
        self.ee_target_archive += self.asset.data.body_pos_w[0, self.arm_body_id] - self.asset.data.root_pos_w[0]

        self.eef_const_force = ConstantForce.sample(self.num_envs, (30., 30., 10.), device=self.device)
        self.eef_const_force.force.zero_()
        self.eef_spring_force = SpringForce.sample(self.num_envs, device=self.device)
        self.eef_spring_force.duration.zero_()

        self.base_const_force = ConstantForce.sample(self.num_envs, (30., 30., 10.), device=self.device)
        self.base_const_force.force.zero_()

        self.lin_vel_ema = EMA(self.asset.data.root_lin_vel_w, [0.0, 0.5, 0.8])
        self.ang_vel_ema = EMA(self.asset.data.root_ang_vel_w, [0.0, 0.5, 0.8])
        self.eef_lin_vel_ema = EMA(self.asset.data.body_lin_vel_w[:, self.eef_body_id], [0.75])
        self.eef_lin_vel_ema.update(self.asset.data.body_lin_vel_w[:, self.eef_body_id])

        self.dim_weights = torch.tensor([1.0, 1.0, 0.5], device=self.device)
        self.update()

        if True: # self.env.sim.has_gui() and self.env.backend == "isaac":
            self.marker = VisualizationMarkers(BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/target_yaw"))
            self.marker.set_visibility(True)
            self.setpoint_marker = make_point_marker("setpoint", (0.0, 0.8, 0.0))
            self.spring_marker = make_point_marker("spring", (0.0, 0.0, 0.8))
            

    def _integrate(self, dt: float):
        base_pos_error = torch.where(
            (self.cmd.mode == 0).reshape(self.num_envs, 1, 1),
            self.cmd.setpoint[:, None, :3] - self.ref_pos_w,
            (self.cmd.kd_base / self.cmd.kp_base * self.cmd.set_lin_vel).unsqueeze(1)
        )
        ee_setpoint = self.ref_pos_w + yaw_rotate(self.ref_rpy_w[:, :, 2], self.cmd.setpoint_eef.unsqueeze(1))
        eef_pos_error = ee_setpoint - self.eef_ref_pos_w
        
        coriolis_vel = self.ref_ang_vel_w.cross(self.eef_ref_pos_w - self.ref_pos_w, dim=-1)
        eef_vel_error = self.ref_lin_vel_w + coriolis_vel - self.eef_ref_lin_vel_w

        eef_spring_force = (
            self.cmd.kp_eef.unsqueeze(1) * eef_pos_error
            + self.cmd.kd_eef.unsqueeze(1) * eef_vel_error
        )
        base_spring_force = (
            self.cmd.kp_base.unsqueeze(1) * base_pos_error
            + self.cmd.kd_base.unsqueeze(1) * (0 - self.ref_lin_vel_w)
        )
        des_lin_acc = (
            base_spring_force
            - eef_spring_force * self.cmd.transmission.unsqueeze(1)
        ) / self.cmd.virtual_mass_base.unsqueeze(1)

        eef_des_lin_acc = (
            eef_spring_force 
            + self.get_eef_force(self.eef_ref_pos_w, self.eef_ref_lin_vel_w)
        ) / self.cmd.virtual_mass_eef.unsqueeze(1)

        rpy_error = wrap_to_pi(self.cmd.setpoint[:, None, 3:] - self.ref_rpy_w)
        des_ang_acc = (
            self.cmd.kp_base.unsqueeze(1) * rpy_error
            + self.cmd.kd_base.unsqueeze(1) * (0 - self.ref_ang_vel_w)
        ) / 1.5

        self.ref_lin_acc_w = clamp_norm(des_lin_acc, max=10.0)
        self.ref_lin_acc_w[:, :, 2] = 0.
        self.ref_ang_acc_w = des_ang_acc
        self.ref_ang_acc_w[:, :, 2].clamp_(-20.0, 20.0)
        self.eef_ref_lin_acc_w = eef_des_lin_acc

        # integrate base
        self.ref_lin_vel_w.add_(self.ref_lin_acc_w * dt)
        self.ref_lin_vel_w[:, :, 2] = 0.
        self.ref_lin_vel_w = clamp_norm(self.ref_lin_vel_w, max=3.0)
        self.ref_pos_w.add_(self.ref_lin_vel_w * dt)

        self.ref_ang_vel_w.add_(self.ref_ang_acc_w * dt)
        self.ref_ang_vel_w[:, :, 2].clamp_(-2.2, 2.2)
        self.ref_rpy_w.add_(self.ref_ang_vel_w * dt)
        
        # integrate eef
        self.eef_ref_lin_vel_w.add_(self.eef_ref_lin_acc_w * dt)
        self.eef_ref_pos_w.add_(self.eef_ref_lin_vel_w * dt)
        

    @property
    def command(self):
        eef_pos_error = self.cmd.setpoint_eef - self.eef_pos_b
        return torch.cat([
            self.setpoint_b, 
            self.setpoint_b * self.cmd.kp_base,
            eef_pos_error,
            eef_pos_error * self.cmd.kp_eef,
        ], dim=1)

    @observation
    def command_hidden(self):
        return torch.cat([
            self.w2b(self.ref_pos_w[:, self.ref_steps] - self.asset.data.root_pos_w.unsqueeze(1)).reshape(self.num_envs, -1),
            self.w2b(self.ref_lin_vel_w[:, self.ref_steps]).reshape(self.num_envs, -1),
            wrap_to_pi(self.ref_rpy_w[:, self.ref_steps] - self.base_rpy_w.unsqueeze(1)).reshape(self.num_envs, -1),
            self.w2b(self.ref_ang_vel_w[:, self.ref_steps]).reshape(self.num_envs, -1),
            self.w2b(self.eef_ref_lin_vel_w[:, self.ref_steps]).reshape(self.num_envs, -1),
        ], dim=1)

    @reward
    def impedance_ee_pos(self):
        diff = self.eef_pos_w.unsqueeze(1) - self.eef_ref_pos_w[:, self.ref_steps]
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = torch.exp(- error_l2 / 0.1).mean(1)
        return r
    
    @reward
    def impedance_ee_vel(self):
        diff = self.eef_lin_vel_w.unsqueeze(1) - self.eef_ref_lin_vel_w[:, self.ref_steps]
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = torch.exp(- error_l2 / 0.25).mean(1)
        return r
    
    @reward
    def impedance_pos(self):
        diff = self.surrogate_pos_target - self.asset.data.root_pos_w.unsqueeze(1)
        error_l2 = diff[:, :, :2].square().sum(dim=-1, keepdim=True)
        r = (- error_l2 / 0.25).exp().mean(1)
        return r
    
    @reward
    def impedance_vel(self):
        diff = einops.rearrange(self.surrogate_lin_vel_target, "n t1 d -> n t1 1 d") \
            - einops.rearrange(self.lin_vel_ema.ema, "n t2 d-> n 1 t2 d")
        error_l2 = (diff * self.dim_weights).square().sum(dim=-1, keepdim=True)
        r = ((- error_l2 / 0.25).exp() - 0.25 * error_l2).mean(1)
        return r.max(dim=1).values
    
    @reward
    def impedance_yaw_vel(self):
        diff = einops.rearrange(self.surrogate_yaw_vel_target, "n t1 d -> n t1 1 d") \
            - einops.rearrange(self.ang_vel_ema.ema[:, :, 2:3], "n t2 d-> n 1 t2 d")
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = ((- error_l2 / 0.25).exp() - 0.25 * error_l2).mean(1)
        return r.max(dim=1).values
    
    @reward
    def ee_angvel_penalty(self):
        return - self.asset.data.body_ang_vel_w[:, self.eef_body_id].square().sum(1, True)
    
    # evaluation metrics
    @reward
    def impedance_pos_error(self):
        diff = self.ref_pos_w[:, -1] - self.asset.data.root_pos_w
        error_l2 = diff[:, :2].square().sum(dim=-1, keepdim=True)
        return error_l2
    
    @reward
    def impedance_vel_error(self):
        diff = self.ref_lin_vel_w[:, -1] - self.asset.data.root_lin_vel_w
        error_l2 = diff[:, :2].square().sum(dim=-1, keepdim=True)
        return error_l2
    
    @reward
    def impedance_acc_error(self):
        diff = self.ref_lin_acc_w[:, 0] - self.asset.data.body_acc_w[:, 0, :3]
        error_l2 = diff[:, :2].square().sum(dim=-1, keepdim=True)
        return error_l2
    
    def w2b(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            quat = self.asset.data.root_quat_w
        elif x.ndim == 3:
            quat = self.asset.data.root_quat_w.unsqueeze(1)
        else:
            raise ValueError(f"Invalid input dimension: {x.ndim}")
        return quat_rotate_inverse(quat, x)
    
    def b2w(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            quat = self.asset.data.root_quat_w
        elif x.ndim == 3:
            quat = self.asset.data.root_quat_w.unsqueeze(1)
        else:
            raise ValueError(f"Invalid input dimension: {x.ndim}")
        return quat_rotate(quat, x)

    def step(self, substep: int):
        self.asset._external_force_b[:, self.eef_body_id] += quat_rotate_inverse(
            self.asset.data.body_quat_w[:, self.eef_body_id],
            self.get_eef_force(self.asset.data.body_pos_w[:, self.eef_body_id], self.asset.data.body_lin_vel_w[:, self.eef_body_id])
        )
        self.asset._external_force_b[:, 0] += quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.base_const_force.get_force()
        )
        self.asset.has_external_wrench = True
        self.lin_vel_ema.update(self.asset.data.root_lin_vel_w)
        self.ang_vel_ema.update(self.asset.data.root_ang_vel_w)
        self.eef_lin_vel_ema.update(self.asset.data.body_lin_vel_w[:, self.eef_body_id])

    def reset(self, env_ids: torch.Tensor):
        self.ref_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids].unsqueeze(1)
        self.ref_lin_vel_w[env_ids] = self.asset.data.root_lin_vel_w[env_ids].unsqueeze(1)
        self.ref_rpy_w[env_ids] = euler_from_quat(self.asset.data.root_quat_w[env_ids].unsqueeze(1))
        self.ref_ang_vel_w[env_ids] = self.asset.data.root_ang_vel_w[env_ids].unsqueeze(1)

        self.eef_ref_pos_w[env_ids] = self.asset.data.body_pos_w[env_ids, self.eef_body_id].unsqueeze(1)
        self.eef_ref_lin_vel_w[env_ids] = self.asset.data.body_lin_vel_w[env_ids, self.eef_body_id].unsqueeze(1)
        
        self.sample_base_mode0(env_ids)
        self.sample_eef(env_ids, torch.tensor([0.60, 0.0, 0.25], device=self.device))
        
        self.setpoint_w[:, :3] = torch.where(
            (self.cmd.mode == 0).reshape(self.num_envs, 1),
            self.cmd.setpoint[:, :3],
            self.cmd.kd_base / self.cmd.kp_base * self.cmd.set_lin_vel + self.asset.data.root_pos_w
        )
        self.setpoint_b[:, :3] = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.setpoint_w[:, :3] - self.asset.data.root_pos_w
        )
    
    def update(self):
        self.eef_pos_w = self.asset.data.body_pos_w[:, self.eef_body_id]
        self.eef_pos_b = self.w2b(self.eef_pos_w - self.asset.data.root_pos_w)
        self.eef_lin_vel_w = self.eef_lin_vel_ema.ema[:, 0]
        self.base_rpy_w = euler_from_quat(self.asset.data.root_quat_w)
        self.setpoint_eef_w = self.asset.data.root_pos_w + yaw_rotate(self.asset.data.heading_w, self.cmd.setpoint_eef)

        # closed-loop update
        take_xy = torch.tensor([1., 1., 0.], device=self.device)
        take_z = torch.tensor([0., 0., 1.], device=self.device)
        self.ref_pos_w = round_robbin(self.ref_pos_w, self.asset.data.root_pos_w)
        self.ref_pos_w[..., 2] = 0.6

        self.ref_lin_vel_w = round_robbin(self.ref_lin_vel_w, self.asset.data.root_lin_vel_w * take_xy)
        self.ref_rpy_w = round_robbin(self.ref_rpy_w, self.base_rpy_w * take_z)
        self.ref_ang_vel_w = round_robbin(self.ref_ang_vel_w, self.asset.data.root_ang_vel_w * take_z)
        self.eef_ref_pos_w = round_robbin(self.eef_ref_pos_w, self.eef_pos_w)
        self.eef_ref_lin_vel_w = round_robbin(self.eef_ref_lin_vel_w, self.eef_lin_vel_w)
        
        # self._integrate(self.env.step_dt)
        for _ in range(self.env.decimation):
            self._integrate(self.env.physics_dt)

        self.surrogate_pos_target = self.ref_pos_w[:, self.ref_steps]
        self.surrogate_lin_vel_target = self.ref_lin_vel_w[:, self.ref_steps]
        self.surrogate_yaw_target = self.ref_rpy_w[:, self.ref_steps, 2, None]
        self.surrogate_yaw_vel_target = self.ref_ang_vel_w[:, self.ref_steps, 2, None]

        self.update_command_and_force()
        self.setpoint_w[:, :3] = torch.where(
            (self.cmd.mode == 0).reshape(self.num_envs, 1),
            self.cmd.setpoint[:, :3],
            self.cmd.kd_base / self.cmd.kp_base * self.cmd.set_lin_vel + self.asset.data.root_pos_w
        )
        self.setpoint_b[:, :3] = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.setpoint_w[:, :3] - self.asset.data.root_pos_w
        )
        self.setpoint_b[:, 3:] = wrap_to_pi(self.setpoint_w[:, 3:] - self.base_rpy_w)

    def update_command_and_force(self):
        resample = ((self.env.episode_length_buf-100) % 150 == 0)
        if resample.any():
            r = torch.rand(self.num_envs, device=self.device) < self.MODE0_PROB
            self.sample_base_mode0(self.mask2id(resample & r))
            self.sample_base_mode1(self.mask2id(resample & ~r))
            self.sample_eef(self.mask2id(resample))

            r = torch.randint(0, 3, (self.num_envs,), device=self.device)
            resample_ids = self.mask2id(resample & (r == 0))
            force = ConstantForce.sample(len(resample_ids), (25., 25., 5.), device=self.device)
            self.eef_const_force[resample_ids] = force

            resample_ids = self.mask2id(resample & (r == 1))
            force = SpringForce.sample(len(resample_ids), device=self.device)
            force.setpoint += self.eef_pos_w[resample_ids]
            self.eef_spring_force[resample_ids] = force

        self.eef_const_force.time.add_(self.env.step_dt)
        self.eef_spring_force.step(self.eef_pos_w, self.eef_lin_vel_w, self.env.step_dt)
        self.base_const_force.time.add_(self.env.step_dt)

    def get_eef_force(self, eef_pos_w: torch.Tensor, eef_vel_w: torch.Tensor):
        if eef_pos_w.ndim == 2:
            const_force = self.eef_const_force.get_force()
            spring_force = self.eef_spring_force.get_force(eef_pos_w, eef_vel_w)
        elif eef_pos_w.ndim == 3:
            const_force = self.eef_const_force.get_force().unsqueeze(1)
            spring_force = self.eef_spring_force.unsqueeze(1).get_force(eef_pos_w, eef_vel_w)
        return const_force + spring_force

    def mask2id(self, mask: torch.Tensor) -> torch.Tensor:
        return mask.nonzero().squeeze(-1)
    
    def sample_base_mode0(self, env_ids: torch.Tensor):
        cmd = ImpedanceCommand.zero(len(env_ids), self.device)
        cmd.mode[:] = 0
        cmd.setpoint[:, :2].uniform_(0.8, 1.4)
        cmd.setpoint[:, :2].mul_(torch.randn(len(env_ids), 2, device=self.device).sign())
        cmd.setpoint[:, :3].add_(self.asset.data.root_pos_w[env_ids])
        cmd.setpoint[:, 5].uniform_(-torch.pi, torch.pi)
        cmd.kp_base.uniform_(*self.base_kp_range)
        cmd.kd_base = 1.8 * cmd.kp_base.sqrt() # near-critical damping
        self.cmd[env_ids] = cmd
    
    def sample_base_mode1(self, env_ids: torch.Tensor, vel: Optional[torch.Tensor]=None):
        """
        Walk at a constant velocity.
        """
        cmd = ImpedanceCommand.zero(len(env_ids), self.device)
        cmd.mode[:] = 1
        if vel is not None:
            cmd.set_lin_vel[:] = vel
        else:
            cmd.set_lin_vel[:, 0].uniform_(0.4, 1.4)
            cmd.set_lin_vel[:, 0].mul_(torch.randn(len(env_ids), device=self.device).sign())
        cmd.setpoint[:, 5].uniform_(-0.25 * torch.pi, 0.25 * torch.pi)
        cmd.kp_base.uniform_(*self.base_kp_range)
        cmd.kd_base = 1.8 * cmd.kp_base.sqrt() # near-critical damping
        self.cmd[env_ids] = cmd
    
    def sample_eef(self, env_ids: torch.Tensor, setpoint: Optional[torch.Tensor]=None):
        cmd: ImpedanceCommand = self.cmd[env_ids]
        if setpoint is None:
            idx = torch.randint(0, self.ee_target_archive.shape[0], (len(env_ids),), device=self.device)
            cmd.setpoint_eef = self.ee_target_archive[idx]
        else:
            cmd.setpoint_eef[:] = setpoint
        cmd.kp_eef.uniform_(*self.eef_kp_range)
        cmd.kd_eef = 1.8 * cmd.kp_eef.sqrt() # near-critical damping
        cmd.transmission.random_(0, 2)
        self.cmd[env_ids] = cmd
    
    def eef_draw_circle(self):
        t = self.env.step_dt * self.env.episode_length_buf * torch.pi
        self.cmd.setpoint_eef[:, 0] = 0.60
        self.cmd.setpoint_eef[:, 1] = 0.25 * torch.cos(t)
        self.cmd.setpoint_eef[:, 2] = 0.25 + 0.25 * torch.sin(t)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.setpoint_w[:, :3] - self.asset.data.root_pos_w,
            color=(1.0, 0.0, 0.0, 1.0),
            size=5.0,
        )
        self.env.debug_draw.vector( # set lin vel (if applicable), purple
            self.asset.data.root_pos_w,
            self.cmd.set_lin_vel * (self.cmd.mode == 1).reshape(self.num_envs, 1),
            color=(1.0, 0.0, 1.0, 1.0),
            size=5.0,
        )
        self.env.debug_draw.vector( # reference lin vel, green
            self.asset.data.root_pos_w,
            self.ref_lin_vel_w[:, -2],
            color=(0.0, 1.0, 0.0, 1.0),
            size=5.0,
        )
        self.marker.visualize(
            self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            quat_from_euler_xyz(*self.cmd.setpoint[:, 3:].unbind(-1)),
            scales=torch.tensor([[4., 1., 0.1]]).expand(self.num_envs, 3),
        )
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.setpoint_eef_w - self.eef_pos_w,
            color=(1.0, 0.0, 0.0, 1.0),
            size=5.0,
        )
        # self.env.debug_draw.vector(
        #     self.eef_pos_w,
        #     # self.eef_ref_lin_vel_w[:, -2],
        #     self.eef_lin_vel_w,
        #     color=(0.0, 1.0, 0.0, 1.0),
        #     size=5.0,
        # )
        self.env.debug_draw.vector(
            self.eef_pos_w,
            quat_rotate(self.asset.data.body_quat_w[:, self.eef_body_id], self.asset._external_force_b[:, self.eef_body_id]) / 9.81,
            color=(1.0, 1.0, 0.0, 1.0),
            size=5.0,
        )
        eef_ref_pos = self.eef_ref_pos_w[:, self.ref_steps].reshape(-1, 3)
        self.setpoint_marker.visualize(eef_ref_pos)
        
        has_spring = self.eef_spring_force.is_valid().squeeze(1)
        if has_spring.any():
            spring_setpoint = self.eef_spring_force.setpoint[has_spring]
            self.spring_marker.visualize(spring_setpoint)

        # self.setpoint_vis.visualize(ref_pos, marker_indices=[1] * len(ref_pos))
        # self.env.debug_draw.vector(
        #     self.eef_pos_w,
        #     self.eef_ref_pos_w[:, -2] - self.eef_pos_w,
        #     color=(0.0, 0.0, 1.0, 1.0),
        #     size=5.0,
        # )


class ImpedanceCommandDemo(ImpedanceCommandManager):
    def reset(self, env_ids: torch.Tensor):
        self.ref_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids].unsqueeze(1)
        self.ref_lin_vel_w[env_ids] = self.asset.data.root_lin_vel_w[env_ids].unsqueeze(1)
        self.ref_rpy_w[env_ids] = euler_from_quat(self.asset.data.root_quat_w[env_ids].unsqueeze(1))
        self.ref_ang_vel_w[env_ids] = self.asset.data.root_ang_vel_w[env_ids].unsqueeze(1)

        self.eef_ref_pos_w[env_ids] = self.asset.data.body_pos_w[env_ids, self.eef_body_id].unsqueeze(1)
        self.eef_ref_lin_vel_w[env_ids] = self.asset.data.body_lin_vel_w[env_ids, self.eef_body_id].unsqueeze(1)
        
        self.sample_base_mode0(env_ids)
        self.cmd.setpoint[:, 5].data.zero_()
        self.sample_eef(env_ids, torch.tensor([0.60, 0.0, 0.25], device=self.device))
        
        self.setpoint_w[:, :3] = torch.where(
            (self.cmd.mode == 0).reshape(self.num_envs, 1),
            self.cmd.setpoint[:, :3],
            self.cmd.kd_base / self.cmd.kp_base * self.cmd.set_lin_vel + self.asset.data.root_pos_w
        )
        self.setpoint_b[:, :3] = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.setpoint_w[:, :3] - self.asset.data.root_pos_w
        )

    def update_command_and_force(self):
        if self.env.episode_length_buf[0] == 100:
            cmd = ImpedanceCommand.zero(self.num_envs, self.device)
            cmd.mode[:] = 1
            cmd.kp_base.uniform_(4., 24.)
            cmd.kd_base = 1.8 * cmd.kp_base.sqrt()
            cmd.setpoint_eef[:] = torch.tensor([0.5, 0.0, 0.3], device=self.device)
            cmd.kp_eef.uniform_(24., 24.)
            cmd.kd_eef = 1.8 * cmd.kp_eef.sqrt()
            cmd.transmission.fill_(1.)
            self.cmd = cmd

            eef_const_force = ConstantForce.sample(self.num_envs, device=self.device)
            eef_const_force.force[:, 0] = 20.
            eef_const_force.duration[:] = 3.0
            self.eef_const_force = eef_const_force
        
        if self.env.episode_length_buf[0] == 250:
            cmd = ImpedanceCommand.zero(self.num_envs, self.device)
            cmd.mode[:] = 0
            cmd.kp_base.uniform_(4., 24.)
            cmd.kd_base = 1.8 * cmd.kp_base.sqrt()
            cmd.setpoint[:, :3] = self.asset.data.root_pos_w
            cmd.setpoint[:, 5] = - torch.pi * 0.2
            cmd.setpoint_eef[:] = torch.tensor([0.5, 0.0, 0.3], device=self.device)
            cmd.kp_eef.uniform_(24., 24.)
            cmd.kd_eef = 1.8 * cmd.kp_eef.sqrt()
            cmd.transmission.fill_(0.)
            self.cmd = cmd

        if self.env.episode_length_buf[0] == 600:
            cmd = ImpedanceCommand.zero(self.num_envs, self.device)
            cmd.mode[:] = 0
            cmd.kp_base.uniform_(40., 40.)
            cmd.kd_base = 1.8 * cmd.kp_base.sqrt()
            cmd.setpoint[:, :3] = self.asset.data.root_pos_w[:, :3] - torch.tensor([1.5, 0.0, 0.0], device=self.device)
            cmd.setpoint[:, 5] = - torch.pi * 0.2
            cmd.setpoint_eef[:] = torch.tensor([0.4, 0.1, 0.3], device=self.device)
            cmd.kp_eef.uniform_(40., 40.)
            cmd.kd_eef = 1.8 * cmd.kp_eef.sqrt()
            cmd.transmission.fill_(1.)
            self.cmd = cmd

            eef_spring_force = SpringForce.sample(self.num_envs, device=self.device)
            eef_spring_force.setpoint += self.eef_pos_w
            self.eef_spring_force = eef_spring_force
            self.eef_spring_force.duration[:] = 3.0

        self.eef_const_force.time.add_(self.env.step_dt)
        self.eef_spring_force.step(self.eef_pos_w, self.eef_lin_vel_w, self.env.step_dt)
        
        if self.env.episode_length_buf[0] == 80:
            eye = self.asset.data.root_pos_w[0].cpu()
            eye[0] = 4.0
            eye[2] = 0.5
            self.env.sim.set_camera_view(
                eye + torch.tensor([0., 6.0, 0.5]),
                eye + torch.tensor([0., 0.0, 0.0])
            )
        elif self.env.episode_length_buf[0] == 600:
            eye = self.asset.data.root_pos_w[0].cpu()
            eye[2] = 0.5
            self.env.sim.set_camera_view(
                eye + torch.tensor([0., 6.0, 0.5]),
                eye + torch.tensor([0., 0.0, 0.0])
            )
        has_spring = self.eef_spring_force.is_valid().squeeze(1)
        if has_spring.any():
            spring_setpoint = self.eef_spring_force.setpoint[has_spring]
            self.spring_marker.visualize(spring_setpoint)

    def get_eef_force(self, eef_pos_w: torch.Tensor, eef_vel_w: torch.Tensor):
        if self.env.episode_length_buf[0] < 250:
            if eef_pos_w.ndim == 2:
                return self.eef_const_force.get_force()
            elif eef_pos_w.ndim == 3:
                return self.eef_const_force.get_force().unsqueeze(1)
        elif self.env.episode_length_buf[0] < 600:
            radius = 0.3
            t = (self.env.episode_length_buf-300) * self.env.step_dt
            offset = torch.zeros(self.num_envs, 3, device=self.device)
            offset[:, 0] = 0.5
            offset[:, 1] = radius * torch.cos(t * torch.pi)
            offset[:, 2] = 0.3 + radius * torch.sin(t * torch.pi)
            target_pos = self.asset.data.root_pos_w + offset
            target_vel = torch.zeros_like(target_pos)
            target_vel[:, 1] = -radius * torch.pi * torch.sin(t * torch.pi)
            target_vel[:, 2] = radius * torch.pi * torch.cos(t * torch.pi)
            if eef_pos_w.ndim == 2:
                pos_error = (target_pos - eef_pos_w)
                vel_error = (target_vel - eef_vel_w)
            elif eef_pos_w.ndim == 3:
                pos_error = (target_pos.unsqueeze(1) - eef_pos_w)
                vel_error = (target_vel.unsqueeze(1) - eef_vel_w)
            return 80. * pos_error + 10. * vel_error
        else:
            if eef_pos_w.ndim == 2:
                return self.eef_spring_force.get_force(eef_pos_w, eef_vel_w)
            elif eef_pos_w.ndim == 3:
                return self.eef_spring_force.unsqueeze(1).get_force(eef_pos_w, eef_vel_w)


def expand_time_as(input: torch.Tensor, other: torch.Tensor):
    shape = input.shape[:1] + (1,) * other.ndim-input.ndim + input.shape[-1:]
    return input.reshape(shape)