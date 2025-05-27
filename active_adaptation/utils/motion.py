import torch
import numpy as np
import json
import functools
from tqdm import tqdm
from pathlib import Path
from tensordict import TensorClass, MemoryMappedTensor
from typing import List
from scipy.spatial.transform import Rotation as sRot, Slerp
from concurrent.futures import ThreadPoolExecutor
from isaaclab.utils.string import resolve_matching_names


def lerp(x, xp, fp):
    return np.stack([np.interp(x, xp, fp[:, i]) for i in range(fp.shape[1])], axis=-1)


def slerp(x, xp, fp):
    s = Slerp(xp, sRot.from_quat(fp, scalar_first=True))
    return s(x).as_quat(scalar_first=True)


def interpolate(motion, target_fps: int = 50):
    if motion["fps"] != target_fps:
        T = motion["joint_pos"].shape[0]
        end_t = T / motion["fps"]
        xp = np.arange(0, end_t, 1 / motion["fps"])
        x = np.arange(0, end_t, 1 / target_fps)
        if x[-1] > xp[-1]:
            x = x[:-1]
        motion["body_pos_w"] = lerp(x, xp, motion["body_pos_w"].reshape(T, -1)).reshape(len(x), -1, 3)
        motion["root_pos_w"] = lerp(x, xp, motion["root_pos_w"])
        motion["joint_pos"] = lerp(x, xp, motion["joint_pos"])
        motion["root_quat_w"] = slerp(x, xp, motion["root_quat_w"])
        motion["fps"] = 50
    return motion


class MotionData(TensorClass):
    motion_id: torch.Tensor
    step: torch.Tensor
    root_pos_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_pos: torch.Tensor
    body_pos_w: torch.Tensor
    body_pos_b: torch.Tensor
    body_lin_vel_w: torch.Tensor


class MotionDataset:
    def __init__(
        self,
        body_names: List[str],
        joint_names: List[str],
        starts: List[int],
        ends: List[int],
        data: MotionData,
    ):
        self.body_names = body_names
        self.joint_names = joint_names
        self.starts = torch.as_tensor(starts)
        self.ends = torch.as_tensor(ends)
        self.lengths = self.ends - self.starts
        self.data = data

    @classmethod
    def create_from_path(cls, root_path: str, target_fps: int = 50):
        meta_path = Path(root_path) / "meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        motion_paths = list(Path(root_path).rglob("*.npz"))
        if not motion_paths:
            raise RuntimeError(f"No motions found in {root_path}")
        print(f"Found {len(motion_paths)} motion files under {root_path}")
        
        motions = []
        total_length = 0
        for motion_path in tqdm(motion_paths):
            motion = dict(np.load(motion_path))
            motion = interpolate(motion, target_fps=target_fps)
            total_length += motion["root_pos_w"].shape[0]
            motions.append(motion)
        
        step = MemoryMappedTensor.empty(total_length, dtype=int)
        motion_id = MemoryMappedTensor.empty(total_length, dtype=int)
        root_pos_w = MemoryMappedTensor.empty(total_length, 3)
        root_lin_vel_w = MemoryMappedTensor.empty(total_length, 3)
        root_quat_w = MemoryMappedTensor.empty(total_length, 4)
        joint_pos = MemoryMappedTensor.empty(total_length, len(meta["joint_names"]))
        body_pos_w = MemoryMappedTensor.empty(total_length, len(meta["body_names"]), 3)
        body_pos_b = MemoryMappedTensor.empty(total_length, len(meta["body_names"]), 3)
        body_lin_vel_w = MemoryMappedTensor.empty(total_length, len(meta["body_names"]), 3)
        
        cursor = 0
        
        starts = []
        ends = []

        for i, motion in enumerate(motions):
            motion_length = motion["root_pos_w"].shape[0]
            step[cursor: cursor + motion_length] = torch.arange(motion_length)
            motion_id[cursor:cursor + motion_length] = i
            root_pos_w[cursor:cursor + motion_length] = torch.as_tensor(motion["root_pos_w"])
            
            root_lin_vel_w[cursor:cursor + motion_length-1] = torch.as_tensor(motion["root_pos_w"]).diff(dim=0) * target_fps
            root_lin_vel_w[cursor + motion_length-1] = root_lin_vel_w[cursor + motion_length-2]
            
            root_quat_w[cursor:cursor + motion_length] = torch.as_tensor(motion["root_quat_w"])
            joint_pos[cursor:cursor + motion_length] = torch.as_tensor(motion["joint_pos"])
            body_pos_w[cursor:cursor + motion_length] = torch.as_tensor(motion["body_pos_w"])
            
            body_lin_vel_w[cursor:cursor + motion_length-1] = torch.as_tensor(motion["body_pos_w"]).diff(dim=0) * target_fps
            body_lin_vel_w[cursor + motion_length-1] = body_lin_vel_w[cursor + motion_length-2]

            body_pos_b[cursor:cursor + motion_length] = torch.as_tensor(motion["body_pos_b"])
            starts.append(cursor)
            cursor += motion_length
            ends.append(cursor)
        
        data = MotionData(
            motion_id=motion_id, 
            step=step,
            root_pos_w=root_pos_w,
            root_lin_vel_w=root_lin_vel_w,
            root_quat_w=root_quat_w,
            joint_pos=joint_pos,
            body_pos_w=body_pos_w,
            body_pos_b=body_pos_b,
            body_lin_vel_w=body_lin_vel_w,
            batch_size=[total_length]
        )
        
        return cls(
            body_names=meta["body_names"],
            joint_names=meta["joint_names"],
            starts=starts,
            ends=ends,
            data=data
        )

    @property
    def num_motions(self):
        return len(self.starts)
    
    @property
    def num_steps(self):
        return len(self.data)

    @functools.cache
    def valid_starts(self, seq_len: int=1):
        idx = torch.arange(self.num_steps)
        same = self.data.motion_id[idx] == self.data.motion_id[(idx+seq_len-1) % self.num_steps]
        idx = idx[same]
        return idx
    
    def sample_transitions(self, size, seq_len: int=1) -> MotionData:
        if isinstance(size, int):
            size = (size,)
        valid_starts = self.valid_starts(seq_len)
        idx = torch.randint(0, len(valid_starts), size)
        return self.data[idx.unsqueeze(-1) + torch.arange(seq_len)]

    def sample_subset(self, num_motions: int) -> "MotionDataset":
        if num_motions > self.num_motions:
            raise ValueError()
        episodes = torch.randperm(self.num_motions)[:num_motions].sort().values
        starts = self.starts[episodes]
        ends = self.ends[episodes]
        idx = []
        for start, end in zip(starts, ends):
            idx.append(torch.arange(start, end))
        idx = torch.cat(idx)
        data = self.data[idx]
        return MotionDataset(self.body_names, self.joint_names, starts, ends, data)
    
    def get_slice(self, motion_ids: torch.Tensor, starts: torch.Tensor, steps: int=1) -> MotionData:
        if isinstance(steps, int):
            idx = (self.starts[motion_ids] + starts).unsqueeze(1) + torch.arange(steps)
        elif isinstance(steps, torch.Tensor):
            idx = (self.starts[motion_ids] + starts).unsqueeze(1) + steps
        else:
            raise TypeError()
        if not (idx[:, -1] <= self.ends[motion_ids]).all():
            raise ValueError()
        return self.data[idx]

    def find_joints(self, joint_names, preserve_order: bool=False):
        return resolve_matching_names(joint_names, self.joint_names, preserve_order)

    def find_bodies(self, body_names, preserve_order: bool=False):
        return resolve_matching_names(body_names, self.body_names, preserve_order)


# class MotionIterator:
#     def __init__(
#         self, dataset: MotionDataset,
#         num_motions: int,
#         steps: int,
#         prefetch: bool=False,
#     ):
#         self.dataset = dataset
#         self.num_motions = num_motions
#         self.steps = steps
#         self.prefetch = prefetch
    
#     def _get(self, t: torch.Tensor):
#         return self.dataset.get_slice(self.motion_ids, t, self.steps)
    
#     def reset(self):
#         self.motion_ids = torch.randint(self.dataset.num_motions, (self.num_motions,))
#         self.t = torch.zeros((self.num_motions,), dtype=torch.int32)
#         self._slice = self.dataset.get_slice(self.motion_ids, self.t, self.steps)
#         if self.prefetch:
#             self.executor = ThreadPoolExecutor(max_workers=1)
#             self.future = self.executor.submit(self._get, self.t+1)
#         return self._slice

#     def step(self, flag: torch.Tensor):
#         if not flag.dtype == torch.BoolTensor:
#             raise ValueError
        
#         self.t += flag.bool()
#         ended = self.t + self.steps > self.dataset.lengths[self.motion_ids]

#         if self.prefetch:
#             _slice = self.future.result()
#             if flag.all():
#                 self._slice = _slice
#             else:
#                 self._slice = torch.where(flag, _slice, self._slice)
#             self.future = self.executor.submit(self._get, self.t+1)
#         else:
#             self._slice = self._get(self.t)
        
#         self.t = torch.where(ended, 0, self.t)
#         self.motion_ids = torch.where(ended, torch.randint(self.dataset.num_motions, (self.num_motions,)), self.motion_ids)
#         return self._slice, ended

