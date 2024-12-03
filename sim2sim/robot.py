import numpy as np
import mujoco
import mujoco_viewer

import itertools
import time
import datetime
import zmq
import threading
import re
import h5py
import os

from typing import Sequence, Dict
from sim2sim.utils import normalize
from go2deploy import SecondOrderLowPassFilter
from scipy.spatial.transform import Rotation as R


class MJCRobot:
    def __init__(
        self,
        xml_path: str,
        joint_names_isaac: Sequence[str],
        jpos_default_isaac: Sequence[float],
        action_joint_names: Sequence[str]=None,
        policy_freq: float=50,
        state_update_freq: float=200,
        control_update_freq: float=200,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.policy_freq = policy_freq
        self.state_update_freq = state_update_freq
        self.control_update_freq = control_update_freq
        
        self.state_update_dt = 1. / state_update_freq
        self.control_update_dt = 1. / control_update_freq

        self.joint_names_isaac = joint_names_isaac
        self.joint_names_mjc = []
        self.joint_ranges = []
        for i in range(self.model.njnt):
            jnt = self.model.joint(i)
            if jnt.type.item() == 3:
                self.joint_names_mjc.append(jnt.name)
                self.joint_ranges.append(jnt.range)
        self.num_joints = len(self.joint_names_mjc)

        actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        
        if not (np.array(self.joint_names_mjc) == np.array(actuator_names)).all():
            raise ValueError("Joint names and actuator names don't match")

        self.isaac2mjc = [joint_names_isaac.index(joint) for joint in self.joint_names_mjc]
        self.mjc2isaac = [self.joint_names_mjc.index(joint) for joint in joint_names_isaac]
        
        if action_joint_names is None:
            action_joint_names = joint_names_isaac
        self.action_joint_ids_mjc = [self.joint_names_mjc.index(name) for name in action_joint_names]
        self.action_joint_ids_isaac = [joint_names_isaac.index(name) for name in action_joint_names]
        self.action_dim = len(self.action_joint_ids_isaac)

        self.action_buf = np.zeros((self.action_dim, 4), dtype=np.float32)
        self.applied_action = np.zeros(self.action_dim, dtype=np.float32)
        self._action_rate_l2 = np.zeros(self.action_dim, dtype=np.float32)
        self._action_rate2_l2 = np.zeros(self.action_dim, dtype=np.float32)

        self.jnt_kp = np.zeros(self.num_joints) # in mjc order
        self.jnt_kd = np.zeros(self.num_joints) # in mjc order
        self.effort_limit = np.zeros(self.num_joints)
        self.action_scaling = 0.5
        
        mujoco.mj_step(self.model, self.data)
        self.jpos_mjc = self.data.qpos[-self.num_joints:].astype(np.float32)
        self.jvel_mjc = self.data.qvel[-self.num_joints:].astype(np.float32)
        self.jpos_default_isaac = np.array(jpos_default_isaac, dtype=np.float32)
        self.jpos_default_mjc = self.jpos_default_isaac[self.isaac2mjc]
        self.jpos_des_isaac = self.jpos_default_isaac.copy()
        self.jpos_des_mjc = self.jpos_default_mjc.copy()

        self.synchronized = True
        self.enable_control = False
        self.filter = SecondOrderLowPassFilter(50., 200.)
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")

        os.makedirs("logs", exist_ok=True)
        timestr = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        self.log_file = h5py.File(f"logs/{timestr}.h5py", "a")
        init_length = 200 * 10
        self.log_file.attrs["cursor"] = 0

        # joint state
        self.log_file.create_dataset("jpos", (init_length, self.num_joints), dtype=np.float32, maxshape=(None, self.num_joints))
        self.log_file.create_dataset("jpos_des", (init_length, self.num_joints), dtype=np.float32, maxshape=(None, self.num_joints))
        self.log_file.create_dataset("jvel", (init_length, self.num_joints), dtype=np.float32, maxshape=(None, self.num_joints))
        self.log_file.create_dataset("tau", (init_length, self.num_joints), dtype=np.float32, maxshape=(None, self.num_joints))
        self.log_file.create_dataset("tau_kp", (init_length, self.num_joints), dtype=np.float32, maxshape=(None, self.num_joints))
        self.log_file.create_dataset("tau_kd", (init_length, self.num_joints), dtype=np.float32, maxshape=(None, self.num_joints))

        # sensor state
        self.log_file.create_dataset("gyro", (init_length, 3), dtype=np.float32, maxshape=(None, 3))
        self.log_file.create_dataset("acc", (init_length, 3), dtype=np.float32, maxshape=(None, 3))
    
    def get_gains(self, jnt_name: str):
        i = self.joint_names_mjc.index(jnt_name)
        return self.jnt_kp[i], self.jnt_kd[i]

    def set_gains(self, jnt_name: str, kp: float, kd: float):
        i = self.joint_names_mjc.index(jnt_name)
        self.jnt_kp[i] = kp
        self.jnt_kd[i] = kd
    
    def set_jpos_target(self, jnt_name: str, value: float):
        i = self.joint_names_isaac.index(jnt_name)
        self.jpos_des_isaac[i] = value
    
    def get_jpos_offset(self, jnt_name: str):
        i = self.joint_names_isaac.index(jnt_name)
        return self.jpos_default_isaac[i]
    
    def set_jpos_offset(self, jnt_name: str, value: float):
        i = self.joint_names_isaac.index(jnt_name)
        self.jpos_default_isaac[i] = value

    def run_async(self):
        self.thread_state = threading.Thread(target=self.state_thread_func)
        self.thread_step = threading.Thread(target=self.step_thread_func)
        self.thread_control = threading.Thread(target=self.control_thread_func)
        self.thread_viewer = threading.Thread(target=self.viewer_func)

        self.sim_freq = 0.
        self.state_update_freq = 0.
        self.synchronized = False
        self.step_cnt = 0.
        self.state_update_cnt = 0
        self.running = True

        self.thread_state.start()
        self.thread_step.start()
        self.thread_control.start()
        self.thread_viewer.start()
    
    def run_sync(self):
        self.thread_viewer = threading.Thread(target=self.viewer_func)

        self.sim_freq = int(1. / self.model.opt.timestep)
        self.state_update_freq = 100
        self.synchronized = True
        self.step_cnt = 0
        self.state_update_cnt = 0

        self._update_state()

        self.thread_viewer.start()

    def apply_action(self, action: np.ndarray, alpha: float=0.8):
        self.action_buf = np.roll(self.action_buf, 1, axis=1)
        self.action_buf[:, 0] = action
        self.applied_action = action # alpha * action + (1. - alpha) * self.applied_action

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
            self._step_physics(i)
            time.sleep(dt)
            self.sim_freq = i / (time.perf_counter() - t0)
            if not self.running: break
        
    def state_thread_func(self):
        t0 = time.perf_counter()
        dt = 0.005
        
        for i in itertools.count():
            self._update_state(dt)
            time.sleep(self.state_update_dt)
            self.state_update_freq = i / (time.perf_counter() - t0)
            if not self.running: break

    def control_thread_func(self):
        t0 = time.perf_counter()

        for i in itertools.count():
            if self.enable_control:
                # self.jpos_des_mjc = self.jpos_des_isaac[self.isaac2mjc]
                self.jpos_des_mjc = self.filter.update(self.jpos_des_isaac[self.isaac2mjc])
            time.sleep(self.control_update_dt)
            self.control_update_freq = i / (time.perf_counter() - t0)
            if not self.running: break

    def _step_physics(self, i):
        self.pd_control(i)
        mujoco.mj_step(self.model, self.data)

    def _update_state(self, dt=0.005):
        self.jpos_mjc_prev = self.jpos_mjc
        self.jvel_mjc_prev = self.jvel_mjc
        self.jpos_mjc = self.data.qpos[-self.num_joints:].astype(np.float32)
        self.jvel_mjc = self.data.qvel[-self.num_joints:].astype(np.float32)
        self.jvel_mjc_ = (self.jpos_mjc - self.jpos_mjc_prev) / dt

        self.jpos_isaac = (self.jpos_mjc + self.jpos_mjc_prev)[self.mjc2isaac] / 2.0
        self.jvel_isaac = (self.jvel_mjc + self.jvel_mjc_prev)[self.mjc2isaac] / 2.0
        self.jvel_isaac_ = self.jvel_mjc_[self.mjc2isaac]

        self.quat = self.data.sensor("orientation").data
        self.rot = R.from_quat(self.quat, scalar_first=True)
        self.gravity = self.rot.inv().apply([0., 0., -1.])
        self.gyro = self.data.sensor("imu_gyro").data.astype(np.float32)
        self.acc = self.data.sensor("imu_acc").data.astype(np.float32)
        self.lin_vel_w = self.data.sensor("base_linvel").data.astype(np.float32)
        self.lin_vel_b = self.rot.inv().apply(self.lin_vel_w)

        self.log_file["jpos"][self.state_update_cnt] = self.jpos_isaac
        self.log_file["jvel"][self.state_update_cnt] = self.jvel_isaac
        self.log_file["jpos_des"][self.state_update_cnt] = self.jpos_des_mjc[self.mjc2isaac]
        self.log_file["tau"][self.state_update_cnt] = self.data.ctrl
        self.log_file["tau_kp"][self.state_update_cnt] = self.tau_kp
        self.log_file["tau_kd"][self.state_update_cnt] = self.tau_kd

        self.log_file["gyro"][self.state_update_cnt] = self.gyro
        self.log_file["acc"][self.state_update_cnt] = self.acc

        self.log_file.attrs["cursor"] = self.state_update_cnt
        if self.log_file.attrs["cursor"] % 200 == 0: 
            self.log_file.flush()
        if self.log_file.attrs["cursor"] == self.log_file["jpos"].len() - 1:
            new_len = self.log_file.attrs["cursor"] + 1 + 200 * 20
            print(f"Extend log size to {new_len}.")
            for key, value in self.log_file.items():
                value.resize((new_len, *value.shape[1:]))

        self.state_update_cnt += 1
        # self.socket.send_pyobj({
        #     "jpos": self.jpos_mjc,
        #     "jpos_des": self.jpos_des_mjc,
        #     "rpy": self.gyro,
        #     # "tau": self.data.ctrl,
        #     "acc": self.acc,
        # })

    def pd_control(self, i: int):
        pos_error = self.jpos_des_mjc - self.data.qpos[-self.num_joints:]
        vel_error = 0. - self.data.qvel[-self.num_joints:]
        self.tau_kp = self.jnt_kp * pos_error
        self.tau_kd = self.jnt_kd * vel_error
        self.tau_ctrl = np.clip(self.tau_kp + self.tau_kd, -self.effort_limit, self.effort_limit)
        self.data.ctrl[:] = self.tau_ctrl

    @property
    def action_rate_l2(self):
        return self._action_rate_l2 / self.step_cnt
    
    @property
    def action_rate2_l2(self):
        return self._action_rate2_l2 / self.step_cnt
    
    def resolve_mjc(self, data: Dict[str, float]):
        jnt_names = np.array(self.joint_names_mjc)
        matched = np.zeros(len(jnt_names), dtype=bool)
        values = np.zeros(len(jnt_names))
        for name, value in data.items():
            pattern = re.compile(name)
            for i, joint in enumerate(self.joint_names_mjc):
                if pattern.match(joint):
                    assert not matched[i]
                    matched[i] = True
                    values[i] = value
        ids = np.arange(len(jnt_names))[matched]
        names = jnt_names[matched]
        values = values[matched]
        return ids, names, values
    
    def resolve_isaac(self, data: Dict[str, float]):
        jnt_names = np.array(self.joint_names_isaac)
        matched = np.zeros(len(jnt_names), dtype=bool)
        values = np.zeros(len(jnt_names))
        for name, value in data.items():
            pattern = re.compile(name)
            for i, joint in enumerate(self.joint_names_isaac):
                if pattern.match(joint):
                    assert not matched[i]
                    matched[i] = True
                    values[i] = value
        ids = np.arange(len(jnt_names))[matched]
        names = jnt_names[matched]
        values = values[matched]
        return ids, names, values

