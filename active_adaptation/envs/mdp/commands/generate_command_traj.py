import pickle
import torch
import numpy as np
from tqdm import tqdm

def pitch_yaw_to_vec(pitch, yaw):
    return torch.concatenate([
        torch.cos(yaw) * torch.cos(pitch),
        torch.sin(yaw) * torch.cos(pitch),
        -torch.sin(pitch)
    ], dim=-1)

def vec_to_pitch_yaw(vec):
    pitch = -torch.arcsin(vec[..., 2])
    yaw = torch.atan2(vec[..., 1], vec[..., 0])
    return pitch.unsqueeze(-1), yaw.unsqueeze(-1)

def slerp_vec(v1, v2, t):
    dot = torch.sum(v1 * v2, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.arccos(dot)
    sin_theta = torch.sin(theta)

    mask = sin_theta > 1e-6
    t1 = torch.where(mask, torch.sin((1 - t) * theta) / sin_theta, 1 - t)
    t2 = torch.where(mask, torch.sin(t * theta) / sin_theta, t)

    result = v1 * t1 + v2 * t2
    return result / torch.linalg.norm(result, dim=-1, keepdim=True)

def slerp_angle(a1, a2, t):
    return a1 + t * (a2 - a1)

def slerp_angle_with_limit(a1, a2, t, angle_range):
    a1 = torch.clamp(a1, *angle_range)
    a2 = torch.clamp(a2, *angle_range)
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

def generate_random_trajectories(num_trajectories, yaw_range, pitch_range, radius_range, ee_lin_vel_bounds, episode_len, step_dt, device="cuda", init_pos_waypoint=None, init_ori_waypoint=None):
    # Pre-allocate tensors for all trajectories
    ee_pos = torch.zeros((num_trajectories, episode_len, 3), device=device)
    ee_forward = torch.zeros((num_trajectories, episode_len, 3), device=device)

    # Generate initial waypoints for all trajectories
    last_pos_waypoint = torch.stack([
        torch.empty(num_trajectories, device=device).uniform_(*radius_range),
        torch.empty(num_trajectories, device=device).uniform_(*pitch_range),
        torch.empty(num_trajectories, device=device).uniform_(*yaw_range)
    ], dim=-1) if init_pos_waypoint is None else init_pos_waypoint

    last_ori_waypoint = torch.stack([
        torch.zeros(num_trajectories, device=device),
        last_pos_waypoint[:, 1] + torch.empty(num_trajectories, device=device).uniform_(-torch.pi/4, torch.pi/4),
        last_pos_waypoint[:, 2] + torch.empty(num_trajectories, device=device).uniform_(-torch.pi/4, torch.pi/4)
    ], dim=-1) if init_ori_waypoint is None else init_ori_waypoint

    current_step = torch.zeros(num_trajectories, dtype=torch.long, device=device)
    active_trajectories = torch.ones(num_trajectories, dtype=torch.bool, device=device)

    while active_trajectories.any():
        # Generate new waypoints for active trajectories
        new_pos_waypoint = torch.stack([
            torch.empty_like(last_pos_waypoint[:, 0]).uniform_(*radius_range),
            torch.empty_like(last_pos_waypoint[:, 1]).uniform_(*pitch_range),
            torch.empty_like(last_pos_waypoint[:, 2]).uniform_(*yaw_range)
        ], dim=-1)

        new_ori_waypoint = torch.stack([
            torch.zeros_like(last_ori_waypoint[:, 0]),
            new_pos_waypoint[:, 1] + torch.empty_like(new_pos_waypoint[:, 1]).uniform_(-torch.pi/4, torch.pi/4),
            new_pos_waypoint[:, 2] + torch.empty_like(new_pos_waypoint[:, 2]).uniform_(-torch.pi/4, torch.pi/4)
        ], dim=-1)

        distance = torch.linalg.norm(new_pos_waypoint - last_pos_waypoint, dim=-1)
        time = distance / torch.empty_like(distance).uniform_(*ee_lin_vel_bounds)
        n_steps = (time / step_dt).long()

        # Filter out short segments
        valid_segments = n_steps >= 50
        if not valid_segments.any():
            continue
        n_steps = torch.where(valid_segments, n_steps, torch.zeros_like(n_steps))

        min_steps = n_steps[valid_segments].min().item()
        if min_steps == 0:
            continue
        # Create t for each valid trajectory
        t = torch.stack([torch.linspace(1/steps, 1, steps, device=device)[:min_steps] for steps in n_steps[valid_segments]])
        t = t.unsqueeze(-1)  # Shape: [num_valid, min_steps, 1]
        
        r, pitch, yaw = interpolate_position(
            last_pos_waypoint[valid_segments].unsqueeze(1),
            new_pos_waypoint[valid_segments].unsqueeze(1),
            t, yaw_range
        )
        pos = torch.concatenate([
            r * torch.cos(yaw) * torch.cos(pitch),
            r * torch.sin(yaw) * torch.cos(pitch),
            -r * torch.sin(pitch)
        ], dim=-1)

        _, ori_pitch, ori_yaw = interpolate_orientation(
            last_ori_waypoint[valid_segments].unsqueeze(1),
            new_ori_waypoint[valid_segments].unsqueeze(1),
            t
        )
        forward = pitch_yaw_to_vec(ori_pitch, ori_yaw)

        # Update positions and orientations for valid segments
        valid_indices = torch.where(valid_segments)[0]
        for i, valid_idx in enumerate(valid_indices):
            start = current_step[valid_idx]
            end = min(start + min_steps, episode_len)
            try:
                ee_pos[valid_idx, start:end] = pos[i, :end-start]
            except RuntimeError:
                breakpoint()
            ee_forward[valid_idx, start:end] = forward[i, :end-start]
            current_step[valid_idx] = end

        # Update waypoints for next iteration
        last_pos_waypoint[valid_segments] = torch.concatenate([
            r[:, -1, :],
            pitch[:, -1, :],
            yaw[:, -1, :]
        ], dim=-1)
        last_ori_waypoint[valid_segments] = torch.concatenate([
            torch.zeros_like(ori_pitch[:, -1, :]),
            ori_pitch[:, -1, :],
            ori_yaw[:, -1, :]
        ], dim=-1)

        # Check which trajectories are complete
        active_trajectories = current_step < episode_len

    return ee_pos, ee_forward

if __name__ == "__main__":
    sampling_rate = 50
    step_dt = 1 / sampling_rate
    yaw_range = [-2.5, 2.5]
    pitch_range = [-1.5, 0.5]
    radius_range = [0.4, 0.8]
    ee_lin_vel_bounds = [0.05, 2.0]
    episode_len = 300
    num_trajectories = 4096

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ee_pos, ee_forward = generate_random_trajectories(
        num_trajectories, yaw_range, pitch_range, radius_range,
        ee_lin_vel_bounds, episode_len, step_dt, device
    )

    parsed_plan = []
    for pos, forward in zip(ee_pos, ee_forward):
        t = torch.linspace(0, pos.shape[0] * step_dt, pos.shape[0] + 1, device=device)
        parsed_plan.append({
            "t": t.cpu().numpy(),
            "pos": pos.cpu().numpy(),
            "forward": forward.cpu().numpy(),
        })

    pickle.dump(parsed_plan, open("command_traj.pkl", "wb"))