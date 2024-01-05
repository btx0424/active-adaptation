import torch

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


class Recover(Env):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.action_scaling = 0.5

        self.robot = self.scene.articulations["robot"]
        body_masses = self.robot.root_physx_view.get_masses()[0]
        for name, mass in zip(self.robot.body_names, body_masses):
            print(name, mass)
        
        self.foot_indices = [i for i, name in enumerate(self.robot.body_names) if "foot" in name]
        self.calf_indices = [i for i, name in enumerate(self.robot.body_names) if "calf" in name]
        self.thigh_indices = [i for i, name in enumerate(self.robot.body_names) if "thigh" in name]
        self.main_body_indices = list(set(range(self.robot.num_bodies)) - set(self.calf_indices) - set(self.foot_indices))

        self.contact_sensor: ContactSensor = self.scene.sensors.get("contact_forces", None)
        self.height_scanner: RayCaster = self.scene.sensors.get("height_scanner", None)

        import os.path as osp
        init_state = torch.load(osp.join(osp.dirname(__file__), "init_states.pt")).to(self.device)
        self.init_root_state = init_state[:, :7]
        self.init_joint_pos = init_state[:, 7:]
        self.default_root_state = self.robot.data.default_root_state.clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.default_joint_friction = self.robot.root_physx_view.get_dof_friction_coefficients().clone()
        
        try:
            from omni_drones.envs.isaac_env import DebugDraw
            self.debug_draw = DebugDraw()
            print("[INFO] Debug Draw API enabled.")
        except ModuleNotFoundError:
            pass
        
        self.lookat_env_i = (
            self.scene._default_env_origins.cpu() 
            - torch.tensor(self.cfg.viewer.lookat)
        ).norm(dim=-1).argmin()

        self.target_base_height = self.cfg.target_base_height

        with torch.device(self.device):
            self._command = torch.zeros(self.num_envs, 3 + 3)
            self._command_linvel = self._command[:, :3]
            self._command_heading = self._command[:, 3:6]
            self._command_speed = torch.zeros(self.num_envs, 1)
            # self.action_scale = torch.ones(self.num_envs, 1)
            self.action_alpha = torch.ones(self.num_envs, 1)
            self._actions_t = torch.zeros(self.num_envs, 12)
            self._actions_tm1 = torch.zeros_like(self._actions_t)
            self._actions_tm2 = torch.zeros_like(self._actions_t)

        obs = super()._compute_observation()
        reward = self._compute_reward()

        self.action_spec = CompositeSpec(
            {
                "action": UnboundedContinuousTensorSpec((self.num_envs, 12))
            }, 
            shape=[self.num_envs]
        ).to(self.device)

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

        self.rigig_body_material()

    def _reset_idx(self, env_ids: torch.Tensor):
        sample = torch.randint(0, len(self.init_joint_pos), env_ids.shape, device=self.device)
        
        init_root_state = self.default_root_state[env_ids]
        init_root_state[:, :7] = self.init_root_state[sample]
        init_root_state[:, :3] += self.scene.env_origins[env_ids]
        init_root_state[:, 2] += 0.04
        
        self.robot.write_root_state_to_sim(
            init_root_state, 
            env_ids=env_ids
        )
        
        self.robot.write_joint_state_to_sim(
            # random_scale(self.init_joint_pos[env_ids], 0.8, 1.2),
            self.init_joint_pos[sample],
            self.default_joint_pos[env_ids],
            env_ids=env_ids
        )
        self.stats[env_ids] = 0.
        self._actions_t[env_ids] = 0.
        self._actions_tm1[env_ids] = 0.
        self._actions_tm2[env_ids] = 0.
        
        self.motor_params(env_ids)
        # self.body_masses(env_ids)

        self.scene.reset(env_ids)
        self.scene.update(dt=self.physics_dt)

    def apply_action(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            self._actions_tm2[:] = self._actions_tm1
            self._actions_tm1[:] = self._actions_t
            self._actions_t.lerp_(tensordict["action"], self.action_alpha)
            pos_target = self._actions_t * self.action_scaling + self.default_joint_pos
            self.robot.set_joint_position_target(pos_target)
        self.robot.write_data_to_sim()

    @observation_func
    def command(self):
        quat_w = self.robot.data.root_quat_w
        command_linvel = quat_rotate_inverse(quat_w, self._command_linvel)
        command_heading = quat_rotate_inverse(quat_w, self._command_heading)
        return torch.cat([command_linvel, command_heading], dim=-1)
    
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
    def joint_acc(self):
        return self.robot.data.joint_acc / 30.

    @observation_func
    def projected_gravity_b(self):
        return self.robot.data.projected_gravity_b
    
    @observation_func
    def root_linvel_b(self):
        return self.robot.data.root_lin_vel_b
    
    @observation_func
    def prev_actions(self):
        return self._actions_t
    
    @observation_func
    def applied_torques(self):
        return self.robot.data.applied_torque / 30.
    
    @observation_func
    def contact_forces(self):
        forces = self.contact_sensor.data.net_forces_w_history[:, :, self.foot_indices].mean(dim=1)
        return forces * self.physics_dt
        forces_norm = forces.norm(dim=-1, keepdim=True)
        return (forces / forces_norm.clamp_min(1e-6) * symlog(forces_norm)).reshape(self.num_envs, -1)
    
    @observation_func
    def contact_indicator(self):
        forces = self.contact_sensor.data.net_forces_w_history[:, :, self.foot_indices].mean(dim=1)
        return (forces.norm(dim=-1) > 1.).float()

    @observation_func
    def feet_pos_b(self):
        feet_pos_w = self.robot.data.body_pos_w[:, self.foot_indices]
        feet_pos_b = quat_rotate_inverse(
            self.robot.data.root_quat_w.unsqueeze(1),
            feet_pos_w - self.robot.data.root_pos_w.unsqueeze(1)
        )
        return feet_pos_b.reshape(self.num_envs, -1)
    
    @reward_func
    def linvel(self):
        linvel_w = self.robot.data.root_lin_vel_w
        return (linvel_w * self._command_linvel).sum(dim=1, keepdim=True).clamp_max(self._command_speed)
        linvel_error = square_norm(linvel_w - self._command_linvel)
        return 1. / (1. + linvel_error / 0.25)
    
    @reward_func
    def heading(self):
        heading_b = quat_rotate_inverse(self.robot.data.root_quat_w, self._command_heading)
        return heading_b[:, [0]]
    
    @reward_func
    def base_height(self):
        height = self.robot.data.root_pos_w[:, [2]]
        height = height - self.robot.data.body_pos_w[:, self.foot_indices, 2].mean(1, keepdim=True)
        return (height / self.target_base_height).clamp_max(1.1)

    @reward_func
    def energy(self):
        energy = (
            (self.robot.data.joint_vel * self.robot.data.applied_torque)
            .square()
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
        return - (self._actions_t - self._actions_tm1).square().sum(dim=-1, keepdim=True)

    @reward_func
    def action_rate2_l2(self):
        return (
            (self._actions_t - self._actions_tm1 - self._actions_tm1 + self._actions_tm2)
            .square()
            .sum(dim=-1, keepdim=True)
        )

    @reward_func
    def survive(self):
        return torch.ones(self.num_envs, 1, device=self.device)
    
    @reward_func
    def orientation(self):
        up = - self.robot.data.projected_gravity_b[:, [2]]
        return up + up.square() * up.sign()
        return gravity_z.square() * -gravity_z.sign()
    
    @reward_func
    def feet_slip(self):
        i = self.contact_indicator()
        feet_vel = self.robot.data.body_lin_vel_w[:, self.foot_indices]
        return - (i * feet_vel.norm(dim=-1)).sum(dim=1, keepdim=True)
    
    @reward_func
    def stand_on_feet(self):
        contact_forces = self.contact_sensor.data.net_forces_w.norm(dim=-1)
        contact = (contact_forces > 1.).float()
        contact_others = contact[:, self.main_body_indices].any(1, keepdim=True).float()
        contact_feet = contact[:, self.foot_indices].mean(dim=1, keepdim=True)
        return contact_feet - contact_others

    @termination_func
    def crash(self):
        return torch.zeros((self.num_envs, 1), dtype=bool, device=self.device)

    def motor_params(self, env_ids: torch.Tensor):
        if not hasattr(self, "base_legs"):
            self.base_legs = self.robot.actuators["base_legs"]
            self.base_legs.default_stiffness = self.base_legs.stiffness.clone()
            self.base_legs.default_damping = self.base_legs.damping.clone()
        self.base_legs.stiffness[env_ids] = random_shift(self.base_legs.default_stiffness[env_ids], -.3, .3)
        self.base_legs.damping[env_ids] = random_shift(self.base_legs.default_damping[env_ids], -.3, .3)

    def body_masses(self, env_ids: torch.Tensor):
        if not hasattr(self, "default_masses"):
            self.default_masses = self.robot.root_view.get_body_masses().clone()
            self.default_inertias = self.robot.root_view.get_body_inertias().clone()
        body_masses = random_shift(self.default_masses[env_ids], -0.2, 0.2)
        self.robot.root_view.set_body_masses(body_masses, indices=env_ids)

    def rigig_body_material(
        self, 
        num_buckets=64, 
        static_friction_range=(0.8, 1.0),
        dynamic_friction_range=(0.6, 0.8),
        restitution_range=(0.0, 0.0)
    ):
        asset = self.robot
        material_buckets = torch.zeros(num_buckets, 3)
        material_buckets[:, 0].uniform_(*static_friction_range)
        material_buckets[:, 1].uniform_(*dynamic_friction_range)
        material_buckets[:, 2].uniform_(*restitution_range)
        material_ids = torch.randint(0, num_buckets, (asset.body_physx_view.count, asset.body_physx_view.max_shapes))
        materials = material_buckets[material_ids]
        # resolve the global body indices from the env_ids and the env_body_ids
        bodies_per_env = asset.body_physx_view.count // self.num_envs  # - number of bodies per spawned asset
        indices = torch.tensor(self.foot_indices, dtype=torch.int).repeat(self.num_envs, 1)
        indices += torch.arange(self.num_envs).unsqueeze(1) * bodies_per_env

        # set the material properties into the physics simulation
        # TODO: Need to use CPU tensors for now. Check if this changes in the new release
        asset.body_physx_view.set_material_properties(materials, indices)
        

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
