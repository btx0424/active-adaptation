import torch
from tensordict import TensorClass
from active_adaptation.utils.math import clamp_norm


class SpringForce(TensorClass):
    duration: torch.Tensor
    time: torch.Tensor # the time elapsed since the start of the force
    setpoint : torch.Tensor
    setpoint_mass: torch.Tensor
    setpoint_vel: torch.Tensor
    
    kp: torch.Tensor
    kd: torch.Tensor
    
    @classmethod
    def sample(cls, size: int, device: str):
        scalar = torch.empty(size, 1, device=device)
        offset = torch.zeros(size, 3, device=device)
        offset[:, 0] = -0.5
        offset[:, 2].uniform_(-0.15, 0.15)
        return cls(
            duration=scalar.uniform_(2., 4.).clone(),
            time=torch.zeros(size, 1, device=device),
            setpoint=offset,
            setpoint_mass=400.*torch.ones(size, 1, device=device),
            setpoint_vel=torch.zeros(size, 3, device=device),
            kp=scalar.uniform_(80., 120.).clone(),
            kd=scalar.uniform_(10., 20.).clone(),
        )
    
    def get_force(self, pos: torch.Tensor, vel: torch.Tensor):
        """Return the world-frame force."""
        force = self.kp * (self.setpoint - pos) - self.kd * vel
        force = clamp_norm(force, 100.)
        force *= (self.time < self.duration)
        return force


class ConstantForce(TensorClass):
    duration: torch.Tensor
    time: torch.Tensor # the time elapsed since the start of the force
    offset: torch.Tensor
    force: torch.Tensor
    
    @classmethod
    def sample(cls, size: int, device: str):
        duration = torch.zeros(size, 1, device=device)
        duration.uniform_(1.0, 4.0)
        offset = torch.rand(size, 3, device=device) * 2. - 1.
        offset *= torch.tensor([0.2, 0.1, 0.1], device=device)
        force = torch.rand(size, 3, device=device) * 2. - 1.
        force *= torch.tensor([40., 40., 15.], device=device)
        return cls(
            duration=duration,
            time=torch.zeros(size, 1, device=device),
            offset=offset,
            force=force,
        )

    def get_force(self):
        """Return the world-frame force."""
        return self.force * (self.time < self.duration)


class ImpulseForce(TensorClass):
    duration: torch.Tensor
    time: torch.Tensor # the time elapsed since the start of the force
    peak: torch.Tensor

    @classmethod
    def sample(cls, size: int, device: str):
        duration = torch.zeros(size, 1, device=device)
        duration.uniform_(0.40, 0.60)
        peak = torch.zeros(size, 3, device=device)
        peak[:, 0].uniform_(80., 200.)
        peak[:, 1].uniform_(80., 200.)
        peak[:, 2].uniform_(0., 20.)
        peak *= (torch.rand(size, 3, device=device) - 0.5).sign()
        return cls(
            duration=duration,
            time=torch.zeros(size, 1, device=device),
            peak=peak,
        )

    def get_force(self):
        """Return the world-frame force."""
        t = (self.time / self.duration).clamp(0., 1.)
        force = torch.where(t < 0.5, t * 2 * self.peak, (1 - t) * 2 * self.peak)
        return force