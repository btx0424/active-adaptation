from go2deploy.build import go2py

import time
import datetime
import numpy as np
import math
import torch
import itertools
import h5py
import argparse

from scipy.spatial.transform import Rotation as R
from tensordict import TensorDict

from setproctitle import setproctitle

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

ORBIT_JOINT_ORDER = [
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
]

SDK_JOINT_ORDER = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
]


def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)


class Robot:

    smoothing_length: int = 5
    smoothing_ratio: float = 0.4

    def __init__(self, cfg, log_file: h5py.File=None):
        self.cfg = cfg
        self.log_file = log_file

        self._robot = go2py.RobotIface()
        self._robot.start_control()
        self.default_joint_pos = np.array(
            [
                0.1, -0.1,  0.1, -0.1,  
                0.8,  0.8,  1.0,  1.0, 
                -1.5, -1.5, -1.5, -1.5
            ], 
        )
        
        self.dt = 0.02
        self.latency = 0.0
        self.rpy = np.zeros(3)
        self.angvel_history = np.zeros((3, self.smoothing_length))
        self.projected_gravity_history = np.zeros((3, self.smoothing_length))
        self.angvel = np.zeros(3)
        self.action_buf = np.zeros((12, 4))
        self.command = np.zeros(4) # linvel_xy, angvel_z, base_height
        self.lxy = 0.
        self.rxy = 0.
        self.action_buf_steps = 2
        self.last_action = np.zeros(12)
        self.start_t = time.perf_counter()
        self.timestamp = time.perf_counter()

        self.update()
        _obs = self._compute_obs()
        self.obs_dim = _obs.shape[0]
        self.obs_buf = np.zeros((self.obs_dim * 6))
        
        if self.log_file is not None:
            default_len = 50 * 60
            self.log_file.attrs["cursor"] = 0
            log_file.create_dataset("observation", (default_len, self.obs_dim), maxshape=(None, self.obs_dim))
            log_file.create_dataset("action", (default_len, 12), maxshape=(None, 12))

            log_file.create_dataset("jpos", (default_len, 12), maxshape=(None, 12))
            log_file.create_dataset("jvel", (default_len, 12), maxshape=(None, 12))
            log_file.create_dataset("quat", (default_len, 4), maxshape=(None, 4))
            log_file.create_dataset("linvel", (default_len, 3), maxshape=(None, 3))
            log_file.create_dataset("angvel", (default_len, 3), maxshape=(None, 3))
    
    def reset(self):
        self.start_t = time.perf_counter()
        self.update()
        return self._compute_obs()
    
    def update(self):
        # self.linvel = self._robot.get_velocity()
        # self.feet_pos_b = self._robot.get_feet_pos().reshape(4, 3)
        # self.quat = self._robot.get_quat()
        # self.rot = R.from_quat(self.quat[[1, 2, 3, 0]])
        # self.jpos_sdk = self._robot.get_joint_pos()
        # self.jvel_sdk = self._robot.get_joint_vel()
        # self.yaw_speed = self._robot.get_yaw_speed()
        
        self.prev_rpy = self.rpy
        # self.rpy = self._robot.get_rpy()
        (
            self.jpos_sdk, self.jvel_sdk, self.tau_sdk,
            self.rpy, angvel
        ) = np.split(self._robot.get_full_state(), [12, 24, 36, 39])
        
        self.jpos_sim = self.sdk_to_orbit(self.jpos_sdk)
        self.jvel_sim = self.sdk_to_orbit(self.jvel_sdk)

        dt = time.perf_counter() - self.timestamp
        
        # angvel = ((self.rpy - self.prev_rpy) / dt).clip(-3, 3)
        self.angvel_history = np.roll(self.angvel_history, -1, axis=1)
        self.angvel_history[:, -1] = angvel
        self.angvel = mix(self.angvel, self.angvel_history.mean(axis=1), self.smoothing_ratio)
        
        self.projected_gravity_history[:, 1:] = self.projected_gravity_history[:, :-1]
        self.projected_gravity_history[:, 0] = self._robot.get_projected_gravity()
        self.projected_gravity = normalize(self.projected_gravity_history.mean(1))

        self.lxy = mix(self.lxy, self._robot.lxy(), 0.5)
        self.rxy = mix(self.rxy, self._robot.rxy(), 0.5)
        self.latency = (datetime.datetime.now() - self._robot.timestamp).total_seconds()

        t = time.perf_counter() - self.start_t
        vx = np.sin(t * 0.75)
        self.command[0] = mix(self.command[0] * 0.95, self.lxy[1] * 1.0, 0.2)
        self.command[1] = mix(self.command[1], -self.lxy[0], 0.2)

        self.command[2] = -self.rxy[0] * 1.2
        self.command[3] = 0.0 #
        # heading = np.array([1., -self.rxy[0] * 1.3])
        # heading = heading / np.linalg.norm(heading)
        # heading = np.array([1., 0.])
        # self.command[2:4] = heading
        self.timestamp = time.perf_counter()
        self.step_count = 0
    
    def step(self, action=None):
        if action is not None:
            self.action_buf[:, 1:] = self.action_buf[:, :-1]
            self.action_buf[:, 0] = action.clip(-6, 6)

            # _action = 0.2 * self.action_buf[:, 1] + 0.8 * self.action_buf[:, 0]
            self.last_action = self.last_action * 0.2 + self.action_buf[:, 0] * 0.8
            jpos_target = self.last_action * 0.5 + self.default_joint_pos
            jpos_target = jpos_target.clip(-np.pi, np.pi)
            self._robot.set_command(self.orbit_to_sdk(jpos_target))
        self.update()
        self._maybe_log()
        self.step_count += 1
        obs = self._compute_obs()

        self.obs_buf = np.roll(self.obs_buf, -self.obs_dim)
        self.obs_buf[-self.obs_dim:] = obs
        
        return obs

    def _compute_obs(self):
        # self.rot = R.from_euler("xyz", self.rpy)
        # angvel = self.rot.inv().apply(self.angvel)
        angvel = self.angvel
        
        obs = [
            self.command,
            # angvel,
            self.projected_gravity,
            self.jpos_sim,
            self.jvel_sim,
            self.action_buf[:, :self.action_buf_steps].reshape(-1),
        ]
        # obs = [
        #     self.command,
        #     # np.array([self.yaw_speed]),
        #     angvel[2:],
        #     self.projected_gravity,
        #     self.sdk_to_orbit(self.jpos_sdk),
        #     self.sdk_to_orbit(self.jvel_sdk),
        #     self.action_buf[:, :self.action_buf_steps].reshape(-1),
        #     self.linvel
        # ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs
    
    def _maybe_log(self):
        if self.log_file is None:
            return
        self.log_file["action"][self.step_count] = self.action_buf[:, 0]
        self.log_file["angvel"][self.step_count] = self.angvel
        self.log_file["linvel"][self.step_count] = self._robot.get_velocity()
        self.log_file["jpos"][self.step_count] = self.jpos_sim
        self.log_file["jvel"][self.step_count] = self.jvel_sim
        self.log_file.attrs["cursor"] = self.step_count

        if self.step_count == self.log_file["jpos"].len() - 1:
            new_len = self.step_count + 1 + 3000
            print(f"Extend log size to {new_len}.")
            for key, value in self.log_file.items():
                value.resize((new_len, value.shape[1]))

    @staticmethod
    def orbit_to_sdk(joints: np.ndarray):
        return np.flip(joints.reshape(3, 2, 2), axis=2).transpose(1, 2, 0).reshape(-1)
    
    @staticmethod
    def sdk_to_orbit(joints: np.ndarray):
        return np.flip(joints.reshape(2, 2, 3), axis=1).transpose(2, 0, 1).reshape(-1)

    def process_action(self, action: np.ndarray):
        return self.orbit_to_sdk(action * 0.5 + self.default_joint_pos)
    
    def process_action_inv(self, jpos_sdk: np.ndarray):
        return (self.sdk_to_orbit(jpos_sdk) - self.default_joint_pos) / 0.5


def mix(a, b, alpha):
    return a * (1 - alpha) + alpha * b

from torchrl.envs.utils import set_exploration_type, ExplorationType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", action="store_true", default=False)
    args = parser.parse_args()

    timestr = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    setproctitle("play_go2")

    go2py.init_channel("enp3s0")

    init_pos = np.array([
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8
    ])

    if args.log:
        log_file = h5py.File(f"log_{timestr}.h5py", "a")
    else:
        log_file = None

    robot = Robot({}, log_file)
    
    robot._robot.set_kp(20.)
    robot._robot.set_kd(0.5)

    path = "policy_noangvel.pt"
    policy = torch.load(path)
    policy.module[0].set_missing_tolerance(True)
    # policy = lambda td: torch.zeros(12)

    robot._robot.set_command(init_pos)
    obs = robot.reset()
    obs = robot._compute_obs()
    print(obs.shape)
    print(policy)
    # policy.module.pop(0)

    try:
        td = TensorDict({
            "policy": torch.as_tensor(obs),
            "is_init": torch.tensor(1, dtype=bool),
            "context_adapt_hx": torch.zeros(128),
            "estimator_hx": torch.zeros(128),
        }, []).unsqueeze(0)
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for i in itertools.count():
                start = time.perf_counter()
                policy(td)
                action = td["action"].cpu().numpy()
                # print(action)
                # print(td["state_value"].item())
                # print(processed_actions)
                # print(robot._robot.get_joint_pos_target())
                # obs = torch.as_tensor(robot._compute_obs())

                obs = torch.as_tensor(robot.step(action))
                td["next", "policy"] = obs.unsqueeze(0)
                td["next", "is_init"] = torch.tensor([0], dtype=bool)

                if i % 25 == 0:
                    # print(robot.projected_gravity)
                    print(robot.command)
                    # print(robot.jpos_sdk.reshape(4, 3))
                    # print(robot.sdk_to_orbit(robot.jpos_sdk).reshape(3, 4))

                td = td["next"]
                time.sleep(max(0, 0.02 - (time.perf_counter() - start)))

    except KeyboardInterrupt:
        print("End")
        
if __name__ == "__main__":
    main()

    
