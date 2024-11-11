import numpy as np
import mujoco
import mujoco.viewer
import mujoco_viewer
import threading
import os
import itertools
import time
import argparse
import zmq

from typing import Sequence
from scipy.spatial.transform import Rotation as R
from go2deploy import ONNXModule, SecondOrderLowPassFilter

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
    0.78, 0.78, 0.78, 0.78,
    -1.5, -1.5, -1.5, -1.5,

    # 0.1, -0.1, 0.1, -0.1,
    # 1.0, 1.0, 1.0, 1.0,
    # -1.8, -1.8, -1.8, -1.8,
]

def normalize(x: np.ndarray):
    return x / np.linalg.norm(x)


class Robot:
    def __init__(
        self,
        xml_path: str,
        joint_names_isaac: Sequence[str],
        jpos_default_isaac: Sequence[float],
        action_joint_names: Sequence[str]=None,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = []
        self.joint_ranges = []
        for i in range(self.model.njnt):
            jnt = self.model.joint(i)
            if jnt.type.item() == 3:
                self.joint_names.append(jnt.name)
                self.joint_ranges.append(jnt.range)
        self.num_joints = len(self.joint_names)

        actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        
        if not (np.array(self.joint_names) == np.array(actuator_names)).all():
            raise ValueError("Joint names and actuator names don't match")

        self.isaac2mjc = [joint_names_isaac.index(joint) for joint in self.joint_names]
        self.mjc2isaac = [self.joint_names.index(joint) for joint in joint_names_isaac]
        
        if action_joint_names is None:
            action_joint_names = joint_names_isaac
        self.action_joint_ids_mjc = [self.joint_names.index(name) for name in action_joint_names]
        self.action_joint_ids_isaac = [joint_names_isaac.index(name) for name in action_joint_names]
        self.action_dim = len(self.action_joint_ids_isaac)

        self.action_buf = np.zeros((self.action_dim, 4), dtype=np.float32)
        self.applied_action = np.zeros(self.action_dim, dtype=np.float32)
        self._action_rate_l2 = np.zeros(self.action_dim, dtype=np.float32)
        self._action_rate2_l2 = np.zeros(self.action_dim, dtype=np.float32)

        self.kp = 20.
        self.kd = 0.5
        self.action_scaling = 0.5
        
        mujoco.mj_step(self.model, self.data)
        self.jpos_default_isaac = np.array(jpos_default_isaac, dtype=np.float32)
        self.jpos_default_mjc = self.jpos_default_isaac[self.isaac2mjc]
        self.jpos_des_isaac = self.jpos_default_isaac.copy()
        self.jpos_des_mjc = self.jpos_default_mjc.copy()
        self.jpos = self.data.qpos[-self.num_joints:].astype(np.float32)
        
        smoothing = 2
        self.gyro_buf = np.zeros((3, smoothing), dtype=np.float32)
        self.gravity_buf = np.zeros((3, smoothing), dtype=np.float32)

        self.synchronized = True
        self.enable_control = False
        self.filter = SecondOrderLowPassFilter(50., 400.)
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")

    def run_async(self):
        self.thread_state = threading.Thread(target=self.state_thread_func)
        self.thread_step = threading.Thread(target=self.step_thread_func)
        self.thread_control = threading.Thread(target=self.control_thread_func)
        self.thread_viewer = threading.Thread(target=self.viewer_func)

        self.sim_freq = 0.
        self.state_update_freq = 0.
        self.synchronized = False
        self.step_cnt = 0.

        self.thread_state.start()
        self.thread_step.start()
        self.thread_control.start()
        self.thread_viewer.start()
    
    def run_sync(self):
        self.thread_viewer = threading.Thread(target=self.viewer_func)

        self.sim_freq = int(1. / self.model.opt.timestep)
        self.state_update_freq = 100
        self.synchronized = True
        self.step_cnt = 0.

        self._update_state()

        self.thread_viewer.start()

    def apply_action(self, action: np.ndarray, alpha: float=0.8):
        self.action_buf = np.roll(self.action_buf, 1, axis=1)
        self.action_buf[:, 0] = action
        self.applied_action = alpha * action + (1. - alpha) * self.applied_action

        self.jpos_des_isaac = self.jpos_default_isaac.copy()
        self.jpos_des_isaac[self.action_joint_ids_isaac] += self.action_scaling * self.applied_action

        if self.synchronized:
            self.jpos_des_mjc = self.jpos_des_isaac[self.isaac2mjc]
            for i in range(int(0.02 / self.model.opt.timestep)):
                self._step_physics()
            self._update_state()
        
        # examine action smoothness
        decay = 0.98
        self.step_cnt = self.step_cnt * decay + 1.

        action_rate_l2 = np.square(self.action_buf[:, 0] - self.action_buf[:, 1])
        action_rate2_l2 = np.square(self.action_buf[:, 0] + 2 * self.action_buf[:, 1] - self.action_buf[:, 2])
        
        self._action_rate_l2 = self._action_rate_l2 * decay + action_rate_l2
        self._action_rate2_l2 = self._action_rate2_l2 * decay + action_rate2_l2

    def viewer_func(self):
        viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        viewer.cam.type = 1
        viewer.cam.trackbodyid = 1
        expected_freq = 1 / self.model.opt.timestep
        for i in itertools.count():
            viewer.render()
            if not self.synchronized and (i + 1) % 100 == 0:
                print(
                    f"Simulation freq: {self.sim_freq:.2f}/{expected_freq:.2f}, "
                    f"State update freq: {self.state_update_freq:.2f}, "
                    f"Control update freq: {self.control_update_freq:.2f}"
                )
            time.sleep(0.02)

    def step_thread_func(self):
        dt = self.model.opt.timestep - 0.00022
        t0 = time.perf_counter()

        for i in itertools.count():
            self._step_physics()
            time.sleep(dt)
            self.sim_freq = i / (time.perf_counter() - t0)
        
    def state_thread_func(self):
        t0 = time.perf_counter()
        dt = 0.005
        
        for i in itertools.count():
            self._update_state(dt)
            time.sleep(dt)
            self.state_update_freq = i / (time.perf_counter() - t0)

    def control_thread_func(self):
        t0 = time.perf_counter()
        dt = 0.005

        for i in itertools.count():
            if self.enable_control:
                self.jpos_des_mjc = self.filter.update(self.jpos_des_isaac[self.isaac2mjc])
            time.sleep(dt)
            self.control_update_freq = i / (time.perf_counter() - t0)

    def _step_physics(self):
        self.pd_control()
        mujoco.mj_step(self.model, self.data)

    def _update_state(self, dt=0.005):
        self.jpos_prev = self.jpos.copy()
        self.jpos = self.data.qpos[-self.num_joints:].astype(np.float32)
        self.jvel = self.data.qvel[-self.num_joints:].astype(np.float32)

        self.jpos_isaac = self.jpos[self.mjc2isaac]
        self.jvel_isaac = self.jvel[self.mjc2isaac]

        self.quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]
        self.rot = R.from_quat(self.quat)
        self.gravity = self.rot.inv().apply([0., 0., -1.])
        self.gravity_buf = np.roll(self.gravity_buf, 1, 1)
        self.gravity_buf[:, 0] = self.gravity
        self.gravity_smoothed = normalize(self.gravity_buf.mean(1))

        self.gyro = self.data.sensor("imu_gyro").data.astype(np.float32)
        self.gyro_buf = np.roll(self.gyro_buf, 1, 1)
        self.gyro_buf[:, 0] = self.gyro
        self.gyro_smoothed = self.gyro_buf.mean(1)
        
        self.acc = self.data.sensor("imu_gyro").data.astype(np.float32)
        self.linvel = self.data.sensor("linvel").data.astype(np.float32)

        self.socket.send_pyobj({
            "jpos": self.jpos,
            "jpos_des": self.jpos_des_mjc,
            "rpy": self.gyro,
            "tau": self.data.ctrl,
        })

    def pd_control(self):
        pos_error = self.jpos_des_mjc - self.data.qpos[-self.num_joints:]
        vel_error = 0. - self.data.qvel[-self.num_joints:]
        tau_ctrl = self.kp * pos_error + self.kd * vel_error
        
        self.data.ctrl = tau_ctrl

    @property
    def action_rate_l2(self):
        return self._action_rate_l2 / self.step_cnt
    
    @property
    def action_rate2_l2(self):
        return self._action_rate2_l2 / self.step_cnt


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(FILE_PATH, "go2/go2_plane.xml")


class InferenceManager:
    def __init__(self, robot: Robot, action_steps: int=3):
        self.robot = robot
        self.action_steps = action_steps

        obs = self.update_obs()
        self.obs_dim = obs.shape[-1]
        self.obs_buf = np.zeros((4, self.obs_dim), dtype=np.float32)
        self.command = self.update_command()    
    
    def update(self):
        obs = self.update_obs()
        command = self.update_command()
        self.obs_buf = np.roll(self.obs_buf, shift=1, axis=0)
        self.obs_buf[0] = obs
        return obs, command
    
    def update_obs(self):
        obs = [
            self.robot.gravity,
            self.robot.jpos_isaac,
            self.robot.jvel_isaac,
            self.robot.action_buf[:, :self.action_steps].reshape(-1),
        ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs
    
    def update_command(self):
        mass = 2.0
        kd = 4 * np.sqrt(mass)
        kp = np.square(0.5 * kd)

        target_vel = np.array([1.0, 0.0])
        setpoint = target_vel * kd / kp

        rpy = self.robot.rot.as_euler("xyz")
        command = np.zeros(11, dtype=np.float32)
        command[0:2] = setpoint
        command[2:4] = 0. - rpy[1:3]
        command[4:6] = command[:2] * kp
        command[6:7] = kd
        command[7:9] = command[2:4] * kp
        command[9:10] = kd
        command[10:11] = mass
        return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-s", "--sync", default=False, action="store_true")
    args = parser.parse_args()
    
    robot = Robot(XML_PATH, ISAAC_JOINTS, ISAAC_QPOS)
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
            return action, carry
    else:
        import torch
        from torch.utils._pytree import tree_map
        from tensordict import TensorDict
        backend = "torch"
        policy_module = torch.load(path)
        @torch.inference_mode()
        def policy(inp):
            inp = TensorDict(tree_map(torch.as_tensor, inp), []).unsqueeze(0)
            out = policy_module(inp)
            action = out["action"].numpy().reshape(-1)
            carry = dict(out["next"].squeeze(0))
            return action, carry
    
    inp = {
        "is_init": np.array([True]),
        "adapt_hx": np.zeros((1, 128), dtype=np.float32),
    }

    manager = InferenceManager(robot)

    t0 = time.perf_counter()
    for i in itertools.count():
        iter_start = time.perf_counter()

        obs, command = manager.update()
        inp["command"] = command[None, ...]
        inp["policy"] = obs[None, ...]
        inp["is_init"] = np.array([False])
        action, carry = policy(inp)
        if i > 100:
            robot.enable_control = True
        
        robot.apply_action(action, 1.0)
        inp = carry

        time.sleep(max(0, 0.02 - (time.perf_counter() - iter_start)))
        control_freq = i / (time.perf_counter() - t0)
        if i % 50 == 0:
            # print(obs)
            # print(command)
            # print(action)
            print(f"Control freq: {control_freq:.2f}")
            print(robot.action_rate_l2)
            print(robot.action_rate2_l2)

if __name__ == "__main__":
    main()