import torch
import numpy as np
import mujoco, mujoco_viewer
import time
import os
import itertools
import imageio
import threading
import argparse

from tensordict import TensorDict
from torchrl.envs import set_exploration_type, ExplorationType
from scipy.spatial.transform import Rotation as R
from go2deploy import ONNXModule

np.set_printoptions(precision=2, suppress=True)

# joint order transformation
ISAAC_JOINTS = [
    'lleg_joint1', 'rleg_joint1', 
    'waist_yaw_joint', 
    'lleg_joint2', 'rleg_joint2', 
    'larm_joint1', 'rarm_joint1', 
    'lleg_joint3', 'rleg_joint3', 
    'larm_joint2', 'rarm_joint2', 
    'lleg_joint4', 'rleg_joint4', 
    'larm_joint3', 'rarm_joint3', 
    'lleg_joint5', 'rleg_joint5', 
    'larm_joint4', 'rarm_joint4', 
    'lleg_joint6', 'rleg_joint6', 
    'larm_joint5', 'rarm_joint5', 
    'larm_joint6', 'rarm_joint6'
]

MJC_JOINTS = [
    'lleg_joint1', 'lleg_joint2', 'lleg_joint3', 'lleg_joint4', 'lleg_joint5', 'lleg_joint6', 
    'rleg_joint1', 'rleg_joint2', 'rleg_joint3', 'rleg_joint4', 'rleg_joint5', 'rleg_joint6', 
    'waist_yaw_joint', 
    'larm_joint1', 'larm_joint2', 'larm_joint3', 'larm_joint4', 'larm_joint5', 'larm_joint6', 
    'rarm_joint1', 'rarm_joint2', 'rarm_joint3', 'rarm_joint4', 'rarm_joint5', 'rarm_joint6'
]

CTRL_JOINTS = [
    'lleg_joint1', 'rleg_joint1', 
    'waist_yaw_joint', 
    'lleg_joint2', 'rleg_joint2', 
    'larm_joint1', 'rarm_joint1', 
    'lleg_joint3', 'rleg_joint3', 
    'larm_joint2', 'rarm_joint2', 
    'lleg_joint4', 'rleg_joint4', 
    'larm_joint3', 'rarm_joint3', 
    'lleg_joint5', 'rleg_joint5', 
    'larm_joint4', 'rarm_joint4', 
    'lleg_joint6', 'rleg_joint6'
]

ISAAC_ACTION_SCALE = [0.8, 0.8, 0.8, 0.8, 0.8, 0.25, 0.25, 0.8, 0.8, 0.25, 0.25, 0.8, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

isaac2mjc = [ISAAC_JOINTS.index(joint) for joint in MJC_JOINTS]
mjc2isaac = [MJC_JOINTS.index(joint) for joint in ISAAC_JOINTS]

MJC_KP = [75.0, 50.0, 50.0, 75.0, 30.0, 15.0, 75.0, 50.0, 50.0, 75.0, 30.0, 15.0, 75.0, 75.0, 50.0, 30.0, 30.0, 15.0, 15.0, 75.0, 50.0, 30.0, 30.0, 15.0, 15.0]
MJC_KD = [6.0, 3.0, 3.0, 6.0, 2.0, 1.0, 6.0, 3.0, 3.0, 6.0, 2.0, 1.0, 3.0, 6.0, 3.0, 0.5, 1.0, 1.0, 1.0, 6.0, 3.0, 0.5, 1.0, 1.0, 1.0]
MJC_QPOS = [0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.1, 0.0, 0.3, 0.0, 0.0]

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(FILE_PATH, "orca_stable/mjcf/orca_description_stable.xml")

from sim2sim.go2async import Robot
from torch.utils._pytree import tree_map

@set_exploration_type(ExplorationType.MODE)
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-s", "--sync", default=False, action="store_true")
    args = parser.parse_args()

    ISAAC_QPOS = np.array(MJC_QPOS)[mjc2isaac]
    robot = Robot(XML_PATH, ISAAC_JOINTS, ISAAC_QPOS, CTRL_JOINTS)
    robot.kp = np.array(MJC_KP)
    robot.kd = np.array(MJC_KD)
    robot.action_scaling = np.array(ISAAC_ACTION_SCALE)

    if args.sync:
        robot.run_sync()
    else:
        robot.run_async()
    print(robot.num_joints, robot.action_dim)

    path = args.path
    if path.endswith(".onnx"):
        backend = "onnx"
        policy_module = ONNXModule(path)
        def policy(inp):
            out = policy_module(inp)
            action = out["action"].reshape(-1)
            carry = {k[1]: v for k, v in out.items() if k[0] == "next"}
            return action, carry
    else:
        backend = "torch"
        policy_module = torch.load(path)
        def policy(inp):
            inp = TensorDict(tree_map(torch.as_tensor, inp), []).unsqueeze(0)
            out = policy_module(inp)
            action = out["action"].numpy().reshape(-1)
            carry = dict(out["next"].squeeze(0))
            return action, carry
    
    inp = {
        "is_init": np.array([True]),
        "context_adapt_hx": np.zeros((1, 128), dtype=np.float32),
    }

    command = np.zeros(4, dtype=np.float32)
    command[0] = 1.0
    command[3] = 0.3

    cycle = 1.2
    omega = np.pi * 2 / cycle
    offset = np.pi / 2
    
    def get_obs(self: Robot):
        t = self.data.time

        phase = omega * t + offset
        sin_t = np.sin(phase)
        cos_t = np.cos(phase)

        obs = [
            command, # 4
            self.gyro, # 3
            self.gravity, # 3
            self.jpos[self.mjc2isaac], # 25
            self.jvel[self.mjc2isaac], # 25
            self.action_buf[:, :3].flatten(), # 21 * 3
            np.array([sin_t, omega * cos_t, cos_t, -omega * sin_t]), # 4
        ]
        return np.concatenate(obs, dtype=np.float32)
    

    t0 = time.perf_counter()
    for i in itertools.count():
        iter_start = time.perf_counter()
        yaw_diff = 0. - robot.rot.as_euler("xyz")[2]
        command[2:3] = yaw_diff

        inp["command"] = command[None, ...]
        inp["policy"] = get_obs(robot)[None, ...]
        inp["is_init"] = np.array([False])
        action, carry = policy(inp)
        if i > 50:
            robot.apply_action(action)
        inp = carry

        time.sleep(max(0, 0.02 - (time.perf_counter() - iter_start)))
        control_freq = i / (time.perf_counter() - t0)
        if i % 50 == 0:
            print("Control freq:", control_freq)


if __name__ == "__main__":
    main()