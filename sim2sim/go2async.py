import numpy as np
import mujoco
import mujoco.viewer
import mujoco_viewer
import threading
import os
import itertools
import time
import argparse

from scipy.spatial.transform import Rotation as R
from go2deploy import ONNXModule

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

class Robot:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = []
        self.joint_ranges = []
        self.act_names = []
        for i in range(self.model.njnt):
            jnt = self.model.joint(i)
            if jnt.type.item() == 3:
                self.joint_names.append(jnt.name)
                self.joint_ranges.append(jnt.range)
        self.num_joints = len(self.joint_names)

        self.act_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        
        assert self.num_joints == 12
        assert (np.array(self.joint_names) == np.array(self.act_names)).all()

        self.isaac2mjc = [ISAAC_JOINTS.index(joint) for joint in self.joint_names]
        self.mjc2isaac = [self.joint_names.index(joint) for joint in ISAAC_JOINTS]
        
        self.prev_action_steps = 1
        self.action_buf = np.zeros((self.num_joints, self.prev_action_steps), dtype=np.float32)
        self.applied_action = np.zeros(self.num_joints, dtype=np.float32)

        self.kp = 20.
        self.kd = 0.5
        self.action_scaling = 0.5
        
        mujoco.mj_step(self.model, self.data)
        self.jpos_default = np.array(ISAAC_QPOS, dtype=np.float32)[self.isaac2mjc]
        self.jpos_des = self.jpos_default.copy()

    def run(self):
        self.thread_state = threading.Thread(target=self.state_func)
        self.thread_step = threading.Thread(target=self.step_func)
        self.thread_viewer = threading.Thread(target=self.viewer_func)

        self.sim_freq = 0.
        self.state_update_freq = 0.

        self.thread_state.start()
        self.thread_step.start()
        self.thread_viewer.start()

    def get_obs(self):
        obs = [
            self.gravity,
            self.jpos[self.mjc2isaac],
            self.jvel[self.mjc2isaac],
            self.action_buf.reshape(-1),
        ]
        return np.concatenate(obs, dtype=np.float32)

    def apply_action(self, action: np.ndarray):
        self.action_buf = np.roll(self.action_buf, 1, axis=1)
        self.action_buf[:, 0] = action
        alpha = 0.8
        self.applied_action = alpha * action + (1. - alpha) * self.applied_action

        self.jpos_des = (
            self.action_scaling * self.applied_action[self.isaac2mjc]
            + self.jpos_default
        )

    def viewer_func(self):
        viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        viewer.cam.type = 1
        viewer.cam.trackbodyid = 1
        for i in itertools.count():
            viewer.render()
            if i % 50 == 0:
                print(self.sim_freq, self.state_update_freq)
                # print(self.jvel)
                # print(self.jvel_diff)
            time.sleep(0.02)

    def step_func(self):
        dt = self.model.opt.timestep - 0.0003
        t0 = time.perf_counter()

        for i in itertools.count():
            self.pd_control()
            mujoco.mj_step(self.model, self.data)
            time.sleep(dt)
            self.sim_freq = i / (time.perf_counter() - t0)
        
    def state_func(self):
        t0 = time.perf_counter()
        dt = 0.005

        self.jpos = self.data.qpos[-self.num_joints:].astype(np.float32)
        for i in itertools.count():
            self.jpos_prev = self.jpos.copy()
            self.jpos = self.data.qpos[-self.num_joints:].astype(np.float32)
            self.jvel = self.data.qvel[-self.num_joints:].astype(np.float32)
            self.jvel_diff = (self.jpos - self.jpos_prev) / dt

            self.quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]
            self.rot = R.from_quat(self.quat)
            self.gravity = self.rot.inv().apply([0., 0., -1.])
            self.gyro = self.data.sensor("imu_gyro").data.astype(np.float32)
            self.acc = self.data.sensor("imu_gyro").data.astype(np.float32)
            time.sleep(dt)
            self.state_update_freq = i / (time.perf_counter() - t0)

    def pd_control(self):
        pos_error = self.jpos_des - self.data.qpos[-self.num_joints:]
        vel_error = 0. - self.data.qvel[-self.num_joints:]
        tau_ctrl = self.kp * pos_error + self.kd * vel_error
        
        self.data.ctrl = tau_ctrl


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(FILE_PATH, "go2/go2_plane.xml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    args = parser.parse_args()
    
    robot = Robot(XML_PATH)
    robot.run()
    
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
        "adapt_hx": np.zeros((1, 128), dtype=np.float32),
    }
    
    kd = 2
    kp = np.square(0.5 * kd)

    command = np.zeros(10, dtype=np.float32)
    command[0] = 1.0
    command[1] = 0.0
    command[2:3] = 0.
    command[3:5] = kp * command[0:2]
    command[5:6] = kd
    command[6:7] = kp
    command[7:8] = kd
    command[8:9] = 3.0

    t0 = time.perf_counter()
    for i in itertools.count():
        iter_start = time.perf_counter()
        yaw_diff = 0. - robot.rot.as_euler("xyz")[2]
        command[2:3] = yaw_diff

        inp["command"] = command[None, ...]
        inp["policy"] = robot.get_obs()[None, ...]
        inp["is_init"] = np.array([False])
        action, carry = policy(inp)
        if i > 100:
            robot.apply_action(action)
        inp = carry

        time.sleep(max(0, 0.02 - (time.perf_counter() - iter_start)))
        control_freq = i / (time.perf_counter() - t0)
        if i % 50 == 0:
            print("Control freq:", control_freq)

if __name__ == "__main__":
    main()