import torch

from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.actuators import DCMotor
from active_adaptation.envs.base import Env
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
import active_adaptation.envs.mdp as mdp

from tensordict.tensordict import TensorDictBase

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

class LocomotionEnv(Env):

    feet_name_expr: str

    def __init__(self, cfg):
        super().__init__(cfg)

        self.robot = self.scene.articulations["robot"]
        self.action_split = [act.num_joints for act in self.robot.actuators.values()]

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

        self.action_buf: torch.Tensor = self.action_manager.action_buf
        self.last_action: torch.Tensor = self.action_manager.applied_action

    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        
        self.robot.write_root_state_to_sim(
            init_root_state, 
            env_ids=env_ids
        )
        self.stats[env_ids] = 0.

        self.scene.reset(env_ids)
        # in `self._reset_callbacks`
        # self.command_manager.reset(env_ids=env_ids)
        # self.action_manager.reset(env_ids=env_ids)

    def render(self, mode: str = "human"):
        robot_pos = self.robot.data.root_pos_w[self.lookat_env_i].cpu()
        if mode == "rgb_array":
            eye = torch.tensor(self.cfg.viewer.eye) + robot_pos
            lookat = torch.tensor(self.cfg.viewer.lookat) + robot_pos
            self.sim.set_camera_view(eye, lookat)
        return super().render(mode)
    
    @mdp.observation_func
    def command(self):
        return self.command_manager.command
    
    @mdp.observation_func
    def prev_command(self):
        return self.command_manager.command_prev
    
    # @mdp.reward_func
    # def heading(self):
    #     root_quat = self.scene["robot"].data.root_quat_w
    #     heading_b_x = quat_rotate_inverse(root_quat, self.command_manager._command_heading)[:, [0]]
    #     return 0.5 * (heading_b_x + heading_b_x.sign() * heading_b_x.square())

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
    def stand(self):
        jpos_error = square_norm(self.scene["robot"].data.joint_pos - self.scene["robot"].data.default_joint_pos)
        cost = - (jpos_error) * self.command_manager.is_standing_env
        return cost

    class feet_pos_b(mdp.body_pos):
        def __init__(self, env: "LocomotionEnv", feet_names=None, yaw_only: bool=False):
            if feet_names is None:
                feet_names = env.feet_name_expr
            super().__init__(env, feet_names, yaw_only)
            self.asset.data.feet_pos_b = self.body_pos_b
        
        def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
            obs = obs.reshape(self.num_envs, 4, 3)[:, [1, 0, 3, 2]] * torch.tensor([1., -1., 1.])
            return obs.reshape(self.num_envs, -1)
    
    class feet_vel_b(mdp.body_vel):
        def __init__(self, env: "LocomotionEnv", feet_names=None, yaw_only: bool=False):
            if feet_names is None:
                feet_names = env.feet_name_expr
            super().__init__(env, feet_names, yaw_only)
            self.asset.data.feet_vel_b = self.body_vel_b
        
        def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
            obs = obs.reshape(self.num_envs, 4, 3)[:, [1, 0, 3, 2]] * torch.tensor([1., -1., 1.])
            return obs.reshape(self.num_envs, -1)
    
    class base_height_l2(mdp.Reward):
        def __init__(self, env, target_height: float, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset = self.env.scene["robot"]
            if isinstance(target_height, str) and target_height == "command":
                self.target_height = self.env.command_manager._target_base_height
            else:
                self.target_height = float(target_height)
        
        def compute(self) -> torch.Tensor:
            height = self.asset.data.feet_pos_b[:, :, 2].mean(1, keepdim=True).abs()
            height_errot = (height - self.target_height) / self.target_height
            return - height_errot.square()

    class feet_force_distribution(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.asset = self.env.scene["robot"]
            self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
            self.default_mass_total = (
                self.asset.root_physx_view.get_masses()[0]
                .sum().to(self.env.device)
                * 9.81
            )

        def compute(self) -> torch.Tensor:
            force = self.contact_sensor.data.net_forces_w_history.mean(dim=1)
            force_norm = force.norm(dim=-1) # / self.default_mass_total
            return force_norm.std(dim=1, keepdim=True)


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

def sample_quat(size, yaw_range = (0, torch.pi * 2), device: torch.device = "cpu"):
    yaw = torch.rand(size, device=device).uniform_(*yaw_range)
    # in (w x y z)
    quat = torch.cat([
        torch.cos(yaw / 2).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.sin(yaw / 2).unsqueeze(-1),
    ], dim=-1)
    return quat