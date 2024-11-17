import torch
import numpy as np
import time
import os
import itertools
import argparse
import math
import threading
import lcm

from tensordict import TensorDict
from torchrl.envs import set_exploration_type, ExplorationType
from go2deploy import ONNXModule
from scipy.spatial.transform import Rotation as R

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

REAL_JOINTS = [
    'lleg_joint6', 'lleg_joint5', 'lleg_joint4', 'lleg_joint3', 'lleg_joint2', 'lleg_joint1',
    'rleg_joint6', 'rleg_joint5', 'rleg_joint4', 'rleg_joint3', 'rleg_joint2', 'rleg_joint1',
    'larm_joint6', 'larm_joint5', 'larm_joint4', 'larm_joint3', 'larm_joint2', 'larm_joint1',
    'rarm_joint6', 'rarm_joint5', 'rarm_joint4', 'rarm_joint3', 'rarm_joint2', 'rarm_joint1',
    'waist_yaw_joint'
]

MJC2REAL: list = [
    5, 4, 3, 2, 1, 0,       # lleg 
    11, 10, 9, 8, 7, 6,     # rleg
    18, 17, 16, 15, 14, 13, # larm
    24, 23, 22, 21, 20, 19, # rarm
    12                      # waist
]

REAL2ISAAC = [REAL_JOINTS.index(joint) for joint in ISAAC_JOINTS]
ISAAC2REAL = [ISAAC_JOINTS.index(joint) for joint in REAL_JOINTS]

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

from torch.utils._pytree import tree_map
from sim2sim.go2async import Robot as _Robot
from sim2sim.lcm_types.python.cyan_legged_cmd_lcmt import *
from sim2sim.lcm_types.python.cyan_armwaisthead_cmd_lcmt import *
from sim2sim.lcm_types.python.cyan_legged_data_lcmt import *
from sim2sim.lcm_types.python.cyan_armwaisthead_data_lcmt import *
from sim2sim.lcm_types.python.cyan_lower_bodyimu_data_lcmt import *

class Robot(_Robot):
    
    def _setup_lcm(self):
        ttl = 255
        self.lowlevel_legs_lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))
        self.lowlevel_ahw_lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))
        self.lowlevel_sensors_lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))

        self._pub_topic_legs = 'robot2controller_legs'
        self._pub_topic_ahw = 'robot2controller_ahw'
        self._pub_topic_sensors = 'robot2controller_sensors'
        self.__ll_msg_legs = cyan_legged_data_lcmt()
        self.__ll_msg_ahw = cyan_armwaisthead_data_lcmt()
        self.__ll_msg_sensor = cyan_lower_bodyimu_data_lcmt()

    def run_async(self):
        self._setup_lcm()
        self._update_state()
        super().run_async()
    
    def run_sync(self):
        self._setup_lcm()
        self._update_state()
        super().run_sync()
    
    def _update_state(self):
        super()._update_state()
        jpos = self.jpos[MJC2REAL]
        jvel = self.jvel[MJC2REAL]
        self.__ll_msg_legs.q = jpos[:12]
        self.__ll_msg_legs.qd = jvel[:12]
        self.__ll_msg_legs.tauIq = np.zeros(12)
        
        self.__ll_msg_ahw.q[:13] = jpos[12:]
        self.__ll_msg_ahw.qd[:13] = jvel[12:]
        self.__ll_msg_ahw.tauIq = np.zeros(18)

        self.__ll_msg_sensor.quat = self.quat
        self.__ll_msg_sensor.gyro = self.gyro

        self.lowlevel_legs_lcm.publish(self._pub_topic_legs, self.__ll_msg_legs.encode())
        self.lowlevel_ahw_lcm.publish(self._pub_topic_ahw, self.__ll_msg_ahw.encode())
        self.lowlevel_sensors_lcm.publish(self._pub_topic_sensors, self.__ll_msg_sensor.encode())


class PolicyClient:
    
    sub_topic_name_legs="robot2controller_legs"
    sub_topic_name_ahw="robot2controller_ahw"
    sub_topic_name_sensor="robot2controller_sensors"
    sub_topic_name_gamepad="gamepad2controller"

    def __init__(
        self,
        action_dim: int,
        action_buf_steps: int = 3
    ):
        self.action_dim = action_dim
        self.action_buf_steps = action_buf_steps
        
        ttl = 255
        self.lc_sub = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))
        
        self.lc_sub.subscribe(self.sub_topic_name_legs, self._handler_legs)
        self.lc_sub.subscribe(self.sub_topic_name_ahw, self._handler_ahw)
        self.lc_sub.subscribe(self.sub_topic_name_sensor, self._handler_sensor)
        # self.lc_sub.subscribe(self.sub_topic_name_gamepad, self.my_handler_gamepad)
        
        self.__p2c_msg_legs = cyan_legged_cmd_lcmt()
        self.__p2c_msg_ahw = cyan_armwaisthead_cmd_lcmt()
        
        self.command = np.zeros(4, dtype=np.float32)
        self.jpos_real = np.zeros(len(ISAAC_JOINTS), dtype=np.float32)
        self.jvel_real = np.zeros(len(ISAAC_JOINTS), dtype=np.float32)
        self.rot = R.from_quat([1., 0., 0., 0.])
        
        self.action_buf = np.zeros((action_dim, self.action_buf_steps), dtype=np.float32)
        
        thread = threading.Thread(target=self._handle)
        thread.start()
        while not hasattr(self, "gyro"):
            time.sleep(0.1)
        print("Connected to robot!")

    def set_cycle(self, cycle: float):
        self.omega = np.pi * 2 / cycle
        self.offset = np.pi / 2

    def get_obs(self, time: float):
        t = self.omega * time + self.offset
        sin_t = math.sin(t)
        cos_t = math.cos(t)
        
        self.command[0] = 1.0
        yaw_diff = 0. - self.rot.as_euler("xyz")[2]
        self.command[2:3] = yaw_diff
        self.command[3] = 0.3
        
        obs = [
            self.command, # 4
            self.gyro, # 3
            self.gravity, # 3
            self.jpos_real[REAL2ISAAC], # 25
            self.jvel_real[REAL2ISAAC], # 25
            self.action_buf.reshape(-1), # 4
            np.array([sin_t, self.omega * cos_t, cos_t, -self.omega * sin_t]), # 4
        ]
        return np.concatenate(obs, dtype=np.float32)

    def apply_action(self, action: np.ndarray):
        self.action_buf = np.roll(self.action_buf, 1, axis=1)
        self.action_buf[:, 0] = action

    def _handle(self):
        while True:
            self.lc_sub.handle()
            pass

    def _handler_legs(self, channel, data):
        msg = cyan_legged_data_lcmt.decode(data)
        self.jpos_real[:12] = msg.q
        self.jvel_real[:12] = msg.qd

    def _handler_ahw(self, channel, data):
        msg = cyan_armwaisthead_data_lcmt.decode(data)
        self.jpos_real[12:] = msg.q[:13]
        self.jvel_real[12:] = msg.qd[:13]

    def _handler_sensor(self, channel, data):
        msg = cyan_lower_bodyimu_data_lcmt.decode(data)
        self.quat = msg.quat
        self.gyro = msg.gyro
        self.rot = R.from_quat(self.quat)
        self.gravity = self.rot.inv().apply([0., 0., -1.])


@set_exploration_type(ExplorationType.MODE)
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-s", "--sync", default=False, action="store_true")
    parser.add_argument("-d", "--dry", default=False, action="store_true")
    args = parser.parse_args()

    ISAAC_QPOS = np.array(MJC_QPOS)[mjc2isaac]
    robot = Robot(XML_PATH, ISAAC_JOINTS, ISAAC_QPOS, CTRL_JOINTS)
    robot.kp = np.array(MJC_KP) * 0.8
    robot.kd = np.array(MJC_KD) * 0.8
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

    t0 = time.perf_counter()
    client = PolicyClient(action_dim=robot.action_dim)
    client.set_cycle(1.0)
    
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
        
        inp["policy"] = client.get_obs(time=robot.data.time)[None, ...]
        # inp["policy"] = get_obs(robot)[None, ...]
        inp["is_init"] = np.array([False])
        action, carry = policy(inp)
        if i > 50:
            robot.apply_action(action, 0.8)
            client.apply_action(action)
        inp = carry

        time.sleep(max(0, 0.02 - (time.perf_counter() - iter_start)))
        control_freq = i / (time.perf_counter() - t0)
        if i % 50 == 0:
            print("Control freq:", control_freq)
            # print(client.jpos_real[REAL2ISAAC])
            # print(robot.jpos[mjc2isaac])


if __name__ == "__main__":
    main()