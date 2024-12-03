import numpy as np
import os
import itertools
import time
import argparse
import math

from typing import Sequence, Dict
from scipy.spatial.transform import Rotation as R
from go2deploy import ONNXModule, SecondOrderLowPassFilter
from sim2sim.robot import MJCRobot

np.set_printoptions(precision=3, suppress=True)

# joint order transformation
ISAAC_JOINTS = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

ISAAC_QPOS = [
    0.1, -0.1, 0.1, -0.1,
    0.7, 0.7, 0.8, 0.8,
    -1.5, -1.5, -1.5, -1.5,

    # 0.1, -0.1, 0.1, -0.1,
    # 1.0, 1.0, 1.0, 1.0,
    # -1.8, -1.8, -1.8, -1.8,
]


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(FILE_PATH, "go2/go2_plane.xml")


class Go2Manager:
    
    oscillator_history: bool = True
    command_dim: int = 11 + (4 * 2 if not oscillator_history else 4 * 4 * 2) 

    def __init__(
        self,
        robot: MJCRobot,
        action_steps: int=3,
    ):
        self.robot = robot
        self.action_steps = action_steps

        self.jpos_multistep = np.zeros((4, 12))
        self.jvel_multistep = np.zeros((4, 12))
        self.gyro_multistep = np.zeros((4, 3))

        obs = self.update_obs()
        self.obs_dim = obs.shape[-1]
        self.obs_buf = np.zeros((4, self.obs_dim), dtype=np.float32)
        self.command = np.zeros(self.command_dim, dtype=np.float32)
        self.last_command_update = time.perf_counter()
        
        # oscillators
        self.phi = np.zeros(4)
        self.phi[0] = np.pi
        self.phi[3] = np.pi
        self.phi_history = np.zeros((4, 4)) # [t, legs]
        self.phi_dot = np.zeros(4)
        
        self.update_command()
    
    def update(self):
        # common
        self.jpos_multistep = np.roll(self.jpos_multistep, shift=1, axis=0)
        self.jpos_multistep[0] = self.robot.jpos_isaac
        self.jvel_multistep = np.roll(self.jvel_multistep, shift=1, axis=0)
        self.jvel_multistep[0] = self.robot.jvel_isaac
        self.gyro_multistep = np.roll(self.gyro_multistep, shift=1, axis=0)
        # noise = 0.2 * math.sin(time.perf_counter() * math.pi * 210)
        # noise += 0.1 * math.sin(time.perf_counter() * math.pi * 390)
        self.gyro_multistep[0] = 0. * self.robot.gyro # + noise

        obs = self.update_obs()
        command = self.update_command()
        self.obs_buf = np.roll(self.obs_buf, shift=1, axis=0)
        self.obs_buf[0] = obs
        return obs, command
    
    def update_obs(self):
        jpos_multistep = self.jpos_multistep.copy()
        jpos_multistep[1:] = self.jpos_multistep[1:] - self.jpos_multistep[:-1]
        jvel_multistep = self.jvel_multistep.copy()
        jvel_multistep[1:] = self.jvel_multistep[1:] - self.jvel_multistep[:-1]
        obs = [
            # self.gyro_multistep.reshape(-1),
            self.robot.gravity,
            jpos_multistep.reshape(-1),
            jvel_multistep.reshape(-1),
            self.robot.action_buf[:, :self.action_steps].reshape(-1),
        ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs
    
    def update_command(self):
        raise NotImplementedError


class Go2DICManager(Go2Manager):
    
    oscillator_history: bool = True
    command_dim: int = 11 + (4 * 2 if not oscillator_history else 4 * 4 * 2) 
    
    def update_command(self):
        mass = 8.0
        kp = 20
        kd = 2 * math.sqrt(kp)

        target_vel = np.array([1.0, 0.0])
        setpoint = target_vel * kd / kp

        rpy = self.robot.rot.as_euler("xyz")
        self.command[0:2] = setpoint
        self.command[2:4] = 0. - rpy[1:3]
        self.command[4:6] = self.command[:2] * kp
        self.command[6:7] = kd
        self.command[7:9] = self.command[2:4] * kp
        self.command[9:10] = kd
        self.command[10:11] = mass

        t = time.perf_counter()
        dt = t - self.last_command_update
        self.last_command_update = t

        self.phi = (self.phi + np.pi * 4 * dt) % (2 * np.pi)
        self.phi_history = np.roll(self.phi_history, 1, axis=1)
        self.phi_history[0] = self.phi
        if self.oscillator_history:
            phi_sin = np.sin(self.phi_history)
            phi_cos = np.cos(self.phi_history)
        else:
            phi_sin = np.sin(self.phi)
            phi_cos = np.cos(self.phi)
        
        osc = np.concatenate([phi_sin, phi_cos], axis=-1)
        self.command[11:] = osc.reshape(-1)
        return self.command


class Go2LocoManager(Go2Manager):

    oscillator_history: bool = False
    command_dim: int = 4 + (4 * 2 + 4 if not oscillator_history else 4 * 4 * 2) 

    def update_command(self):
        rpy = self.robot.rot.as_euler("xyz")
        self.command[0] = 0 # min(self.command[0] + 0.01, 0.8)
        self.command[1] = 0.
        self.command[2] = 0.5 * (0. - rpy[2])
        self.command[3] = 0.75

        t = time.perf_counter()
        dt = t - self.last_command_update
        self.last_command_update = t

        self.phi_dot[:] = np.pi * 4
        self.phi = (self.phi + self.phi_dot * dt) % (2 * np.pi)
        self.phi_history = np.roll(self.phi_history, 1, axis=0)
        self.phi_history[0] = self.phi

        if self.oscillator_history:
            phi_sin = np.sin(self.phi_history)
            phi_cos = np.cos(self.phi_history)
        else:
            phi_sin = np.sin(self.phi)
            phi_cos = np.cos(self.phi)
        
        osc = np.concatenate([phi_sin, phi_cos, self.phi_dot], axis=-1)
        self.command[4:] = osc.reshape(-1)
        return self.command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-s", "--sync", default=False, action="store_true")
    args = parser.parse_args()
    
    robot = MJCRobot(XML_PATH, ISAAC_JOINTS, ISAAC_QPOS)
    robot.action_scaling = 0.5
    robot.jnt_kp = 25.
    robot.jnt_kd = 0.5
    robot.effort_limit = 40.

    if args.sync:
        robot.run_sync()
    else:
        robot.run_async()
    
    path = args.path
    if path.endswith(".onnx"):
        backend = "onnx"
        policy_module = ONNXModule(path)
        def policy(inp):
            out = policy_module(inp)
            action = out["action"].reshape(-1)
            carry = {k[1]: v for k, v in out.items() if k[0] == "next"}
            # carry["retro_pred"] = out[("info", "retro_pred")]
            return action, carry
    else:
        import torch
        from torch.utils._pytree import tree_map
        from tensordict import TensorDict
        backend = "torch"
        policy_module = torch.load(path)
        @torch.inference_mode()
        def policy(inp):
            inp = TensorDict(tree_map(torch.as_tensor, inp), [])
            out = policy_module(inp)
            action = out["action"].numpy().reshape(-1)
            carry = dict(out["next"])
            return action, carry
    
    inp = {
        "is_init": np.array([True]),
        "adapt_hx": np.zeros((1, 128), dtype=np.float32),
    }

    manager = Go2LocoManager(robot)

    t0 = time.perf_counter()
    for i in itertools.count():
        iter_start = time.perf_counter()

        obs, command = manager.update()
        inp["command_"] = command[None, ...]
        inp["policy"] = obs[None, ...]
        inp["is_init"] = np.array([False])
        action, carry = policy(inp)
        if i > 20:
            robot.enable_control = True
            # robot.data.xfrc_applied[1, 0] = 30.
            # robot.data.xfrc_applied[1, 1] = 20.
        
        robot.apply_action(action, 0.8)
        inp = carry

        time.sleep(max(0, 0.02 - (time.perf_counter() - iter_start)))
        control_freq = i / (time.perf_counter() - t0)
        if i % 50 == 0:
            # print(obs)
            # print(command)
            # print(action)
            print(f"Control freq: {control_freq:.2f}")
            print(robot.lin_vel_b)
            # print(command[:4])
            print(carry["retro_pred"].squeeze(0))
            # print(robot.action_rate_l2)
            # print(robot.action_rate2_l2)
            # print(np.abs(carry["_dyn_pred"] - robot.jpos_isaac).mean())

if __name__ == "__main__":
    main()