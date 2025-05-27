import torch

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class AddOn:
    def __init__(self, env):
        self.env: _Env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device
    
    def reset(self):
        pass
    
    def update(self):
        pass
    
    def debug_draw(self):
        pass


class oscillator_quadruped(AddOn):
    def __init__(self, env, omega_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.omega_range = omega_range
        self.omega = torch.zeros(self.num_envs, 1, device=self.device)
        self.omega.uniform_(*self.omega_range).mul_(torch.pi)

        self.asset.phi = torch.zeros(self.num_envs, 4, device=self.device)
        self.asset.phi_dot = torch.zeros(self.num_envs, 4, device=self.device)
        self.asset.phi[:, 0] = torch.pi
        self.asset.phi[:, 3] = torch.pi

        self.rest_target = torch.pi * 3 / 2
        self.keep_steping = torch.ones(self.num_envs, 1, dtype=bool, device=self.device)
    
    def reset(self, env_ids):
        omega = torch.zeros(len(env_ids), 1, device=self.device)
        omega.uniform_(*self.omega_range).mul_(torch.pi)
        self.asset.phi_dot[env_ids] = omega

    def update(self):
        self.asset.phi_dot[:] = self.omega
        self.asset.phi += self.asset.phi_dot * self.env.step_dt
        self.asset.phi = torch.where((self.asset.phi > torch.pi * 2).all(1, True), self.asset.phi - torch.pi * 2, self.asset.phi)

    def stand(self, phi: torch.Tensor, phi_dot: torch.Tensor,):
        two_pi = torch.pi * 2
        target = self.rest_target
        dt = self.env.step_dt
        a = ((phi % two_pi) < target - 1e-4) & (((phi + phi_dot * dt) % two_pi) > target + 1e-4)
        b = ((phi % two_pi) - target).abs() < 1e-4
        phi_dot = torch.where(a, (((target - phi) % two_pi) / dt), phi_dot)
        return phi_dot * (~b)

    def trot(self, phi: torch.Tensor, phi_dot: torch.Tensor):
        phi_dot = torch.zeros_like(phi)
        phi_dot[:, 0] = (phi[:, 3] - phi[:, 0]) + (phi[:, 1] + torch.pi - phi[:, 0]) 
        phi_dot[:, 1] = (phi[:, 2] - phi[:, 1]) + (phi[:, 0] - torch.pi - phi[:, 1]) 
        phi_dot[:, 2] = (phi[:, 1] - phi[:, 2]) + (phi[:, 0] - torch.pi - phi[:, 2])
        phi_dot[:, 3] = (phi[:, 0] - phi[:, 3]) + (phi[:, 1] + torch.pi - phi[:, 3])
        return phi_dot
    