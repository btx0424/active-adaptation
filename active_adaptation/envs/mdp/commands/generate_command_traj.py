import pickle
import torch
import numpy as np
from tqdm import tqdm

def pitch_yaw_to_vec(pitch, yaw):
    """Convert pitch and yaw to 3D unit vectors."""
    return torch.concatenate([
        torch.cos(yaw) * torch.cos(pitch),
        torch.sin(yaw) * torch.cos(pitch),
        -torch.sin(pitch)
    ], dim=-1)

def vec_to_pitch_yaw(vec):
    """Convert 3D unit vectors to pitch and yaw."""
    pitch = -torch.arcsin(vec[..., 2])
    yaw = torch.atan2(vec[..., 1], vec[..., 0])
    return pitch.unsqueeze(-1), yaw.unsqueeze(-1)

def slerp_vec(v1, v2, t):
    """Spherical linear interpolation for vectors."""
    dot = torch.sum(v1 * v2, axis=-1, keepdims=True)
    dot = torch.clip(dot, -1.0, 1.0)
    theta = torch.arccos(dot)
    sin_theta = torch.sin(theta)

    mask = sin_theta > 1e-6
    t1 = torch.where(mask, torch.sin((1 - t) * theta) / sin_theta, 1 - t)
    t2 = torch.where(mask, torch.sin(t * theta) / sin_theta, t)

    result = v1 * t1 + v2 * t2
    return result / torch.linalg.norm(result, axis=-1, keepdims=True)

def slerp_angle(a1, a2, t):
    """Spherical linear interpolation for angles."""
    return a1 + t * (a2 - a1)

def slerp_angle_with_limit(a1, a2, t, angle_range):
    """Spherical linear interpolation for angles with range limit."""
    a1 = torch.clip(a1, *angle_range)
    a2 = torch.clip(a2, *angle_range)
    return a1 + t * (a2 - a1)

def interpolate_position(last_pos, next_pos, t, yaw_range):
    last_r, last_pitch, last_yaw = torch.split(last_pos, 1, -1)
    next_r, next_pitch, next_yaw = torch.split(next_pos, 1, -1)

    current_r = last_r + t * (next_r - last_r)
    current_pitch = slerp_angle(last_pitch, next_pitch, t)
    current_yaw = slerp_angle_with_limit(last_yaw, next_yaw, t, yaw_range)
    
    return current_r, current_pitch, current_yaw

def interpolate_orientation(last_ori, next_ori, t):
    last_roll, last_ori_pitch, last_ori_yaw = torch.split(last_ori, 1, -1)
    next_roll, next_ori_pitch, next_ori_yaw = torch.split(next_ori, 1, -1)

    current_roll = last_roll + t * (next_roll - last_roll)
    last_ori_dir = pitch_yaw_to_vec(last_ori_pitch, last_ori_yaw)
    next_ori_dir = pitch_yaw_to_vec(next_ori_pitch, next_ori_yaw)
    current_ori_dir = slerp_vec(last_ori_dir, next_ori_dir, t)
    current_ori_pitch, current_ori_yaw = vec_to_pitch_yaw(current_ori_dir)
    
    return current_roll, current_ori_pitch, current_ori_yaw

    
def generate_random_trajectory(yaw_range, pitch_range, radius_range, ee_lin_vel_bounds, episode_len, step_dt, device="cpu"):
    last_pos_waypoint = torch.tensor([
        np.random.uniform(*radius_range),
        np.random.uniform(*pitch_range),
        np.random.uniform(*yaw_range)
    ]).unsqueeze(0).to(device)
    last_ori_waypoint = torch.tensor([
        0.0,
        last_pos_waypoint[0, 1] + np.random.uniform(-torch.pi/4, torch.pi/4),
        last_pos_waypoint[0, 2] + np.random.uniform(-torch.pi/4, torch.pi/4)
    ]).unsqueeze(0).to(device)

    ee_pos = []
    ee_forward = []
    
    while sum([pos.shape[0] for pos in ee_pos]) < episode_len:
        # generate a new waypoint
        new_pos_waypoint = torch.tensor([
            np.random.uniform(*radius_range),
            np.random.uniform(*pitch_range),
            np.random.uniform(*yaw_range)
        ]).unsqueeze(0).to(device)
        new_ori_waypoint = torch.tensor([
            0.0,
            new_pos_waypoint[0, 1] + np.random.uniform(-torch.pi/4, torch.pi/4),
            new_pos_waypoint[0, 2] + np.random.uniform(-torch.pi/4, torch.pi/4)
        ]).unsqueeze(0).to(device)
        
        distance = torch.linalg.norm(new_pos_waypoint - last_pos_waypoint)
            
        time = distance / np.random.uniform(*ee_lin_vel_bounds)
        n_steps = int(time / step_dt)

        if n_steps < 50:
            continue

        t = torch.linspace(1 / n_steps, 1, n_steps, device=device)[:, None]
        
        r, pitch, yaw = interpolate_position(last_pos_waypoint, new_pos_waypoint, t, yaw_range)
        pos = torch.concatenate((
            r * torch.cos(yaw) * torch.cos(pitch),
            r * torch.sin(yaw) * torch.cos(pitch),
            -r * torch.sin(pitch)
        ), axis=-1)
        ee_pos.append(pos)
        
        _, ori_pitch, ori_yaw = interpolate_orientation(last_ori_waypoint, new_ori_waypoint, t)
        forward = pitch_yaw_to_vec(ori_pitch, ori_yaw)
        ee_forward.append(forward)
        
        last_pos_waypoint = new_pos_waypoint
        last_ori_waypoint = new_ori_waypoint

    ee_pos = torch.concatenate(ee_pos)[:episode_len]
    ee_forward = torch.concatenate(ee_forward)[:episode_len]
    
    return ee_pos, ee_forward

if __name__ == "__main__":
    sampling_rate = 50
    step_dt = 1 / sampling_rate
    yaw_range = [-2.5, 2.5]
    pitch_range = [-1.5, 0.5]
    radius_range = [0.4, 0.8]
    ee_lin_vel_bounds = torch.tensor([0.05, 2.0])
    episode_len = 300
    num_trajectories = 40

    parsed_plan = []
    for _ in tqdm(range(num_trajectories)):
        ee_pos, ee_forward = generate_random_trajectory(yaw_range, pitch_range, radius_range, ee_lin_vel_bounds, episode_len, step_dt)
        t = torch.linspace(0, episode_len * step_dt, episode_len + 1)
        parsed_plan.append({
            "t": t,
            "pos": ee_pos,
            "forward": ee_forward,
        })

    pickle.dump(parsed_plan, open("command_traj.pkl", "wb"))