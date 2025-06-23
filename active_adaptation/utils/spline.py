import torch


def cubic_bezier(t: torch.Tensor, ps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate a cubic Bezier curve with four control points. Return the position and velocity.
    
    Args:
        t: (N, T), time
        ps: (N, 4, 2), control points
    
    Returns:
        x: (N, T, 2), position
        v: (N, T, 2), velocity
    """
    t = t.unsqueeze(2) # (..., 1)
    p0, p1, p2, p3 = ps.unsqueeze(2).unbind(1) # (N, 4, 2) -> [(N, 1, 2)] * 4
    x = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    v = 3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t ** 2 * (p3 - p2)
    # a = 6 * (1 - t) * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)
    return x, v


def create_from(x0: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """
    Create control points from initial position and velocity.
    
    Args:
        x0: (N, 2), initial position
        v0: (N, 2), initial velocity
    
    Returns:
        ps: (N, 4, 2), control points
    """
    x0 = torch.atleast_2d(x0)
    v0 = torch.atleast_2d(v0)
    device = x0.device
    N = x0.shape[0]
    p0 = x0
    p1 = p0 + v0 / 3
    # p1 and p2 are randomly sampled
    ls = torch.zeros(N, 2, 1, device=device).uniform_(0.7, 1.2)
    theta0 = torch.where(
        v0.norm(dim=-1) < 1e-6,
        torch.zeros(N, device=device),
        torch.atan2(v0[:, 1], v0[:, 0]),
    )
    thetas = torch.zeros(N, 2, device=device).uniform_(-1., 1.)
    thetas = thetas.cumsum(1) + theta0.unsqueeze(1)
    offsets = ls * torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1)
    p2 = p1 + offsets[:, 0]
    p3 = p2 + offsets[:, 1]
    return torch.stack([p0, p1, p2, p3], dim=1) # (N, 4, 2)