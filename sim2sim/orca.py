import torch
import numpy as np
import time
import os
import itertools
import argparse
import math
import threading
import lcm
import json

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

isaac2mjc = [ISAAC_JOINTS.index(joint) for joint in MJC_JOINTS]
mjc2isaac = [MJC_JOINTS.index(joint) for joint in ISAAC_JOINTS]

MJC_QPOS = [0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.1, 0.0, 0.3, 0.0, 0.0]

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(FILE_PATH, "orca/orca_stable/mjcf/orca_description_stable.xml")

from torch.utils._pytree import tree_map
from sim2sim.robot import MJCRobot
from sim2sim.utils import *
from sim2sim.lcm_types.python.cyan_legged_cmd_lcmt import *
from sim2sim.lcm_types.python.cyan_armwaisthead_cmd_lcmt import *
from sim2sim.lcm_types.python.cyan_legged_data_lcmt import *
from sim2sim.lcm_types.python.cyan_armwaisthead_data_lcmt import *
from sim2sim.lcm_types.python.cyan_lower_bodyimu_data_lcmt import *

class OrcaReal:
    
    sub_topic_name_legs="robot2controller_legs"
    sub_topic_name_ahw="robot2controller_ahw"
    sub_topic_name_sensor="robot2controller_sensors"
    sub_topic_name_gamepad="gamepad2controller"

    def __init__(self):
        # buffers
        self.jpos_real = np.zeros(len(ISAAC_JOINTS), dtype=np.float32)
        self.jvel_real = np.zeros(len(ISAAC_JOINTS), dtype=np.float32)

        self.jpos_isaac = np.zeros(len(ISAAC_JOINTS), dtype=np.float32)
        self.jvel_isaac = np.zeros(len(ISAAC_JOINTS), dtype=np.float32)

        # setup lcm
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

        self.lc_sub = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))
        self.lc_sub.subscribe(self.sub_topic_name_legs, self._handler_legs)
        self.lc_sub.subscribe(self.sub_topic_name_ahw, self._handler_ahw)
        self.lc_sub.subscribe(self.sub_topic_name_sensor, self._handler_sensor)

    
    def lcm_send(self):
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


class OrcaMJC(MJCRobot):
    
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
        # self._update_state()
        super().run_async()
    
    def run_sync(self):
        self._setup_lcm()
        self._update_state()
        super().run_sync()
    
    def _update_state(self, dt: float = 0.005):
        super()._update_state(dt)
        jpos_real = self.jpos_mjc[MJC2REAL]
        jvel_real = self.jvel_mjc[MJC2REAL]
        self.__ll_msg_legs.q = jpos_real[:12]
        self.__ll_msg_legs.qd = jvel_real[:12]
        self.__ll_msg_legs.tauIq = np.zeros(12)
        
        self.__ll_msg_ahw.q[:13] = jpos_real[12:]
        self.__ll_msg_ahw.qd[:13] = jvel_real[12:]
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
        self.num_joints = len(ISAAC_JOINTS)
        
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
        
        self.jpos_multistep = np.zeros((4, self.num_joints))
        self.jvel_multistep = np.zeros((4, self.num_joints))

        self.rot = R.from_quat([1., 0., 0., 0.])
        
        self.action_buf = np.zeros((action_dim, self.action_buf_steps), dtype=np.float32)
        # self.action_scaling = np.array(ISAAC_ACTION_SCALE)
        self.action_joint_ids = np.array([
            ISAAC_JOINTS.index(joint) for joint in CTRL_JOINTS])
        self.jpos_default = np.array(MJC_QPOS)[mjc2isaac]
        self.jpos_des = self.jpos_default.copy()

        thread = threading.Thread(target=self._handle)
        thread.start()
        while not hasattr(self, "gyro"):
            time.sleep(0.1)
        print("Connected to robot!")

    def set_cycle(self, cycle: float):
        self.omega = np.pi * 2 / cycle
        self.offset = np.pi / 2

    def get_cmd(self, time: float):
        self.command[0] = 1.0
        target_yaw = 0 # (0.1 * time * math.pi) % (2 * math.pi)
        yaw_diff = target_yaw - self.rot.as_euler("xyz")[2]
        self.command[2] = 2.0 * wrap_to_pi(yaw_diff)
        self.command[3] = wrap_to_pi(yaw_diff)

        t = self.omega * time + self.offset
        sin_t = math.sin(t)
        cos_t = math.cos(t)
        self.phase = np.array([sin_t, self.omega * cos_t, cos_t, -self.omega * sin_t])

        return np.concatenate([self.command, self.phase], dtype=np.float32)

    def get_obs(self, t: float):
        self.jpos_multistep = np.roll(self.jpos_multistep, shift=1, axis=0)
        # self.jpos_multistep[0] = self.jpos_real[REAL2ISAAC]
        self.jpos_multistep[0] = robot.jpos_isaac
        self.jvel_multistep = np.roll(self.jvel_multistep, shift=1, axis=0)
        # self.jvel_multistep[0] = self.jvel_real[REAL2ISAAC]
        self.jvel_multistep[0] = robot.jvel_isaac

        jpos_multistep = self.jpos_multistep.copy()
        jpos_multistep[1:] = self.jpos_multistep[1:] - self.jpos_multistep[:-1]
        jvel_multistep = self.jvel_multistep.copy()
        jvel_multistep[1:] = self.jvel_multistep[1:] - self.jvel_multistep[:-1]

        # print(time.time() - self.legs_timestamp, time.time() - self.ahw_timestamp, time.time() - self.sensor_timestamp)
        obs = [
            # self.command[:4], # 4
            # self.gyro, # 3
            # self.gravity, # 3
            # robot.gyro,
            robot.gravity,
            # self.jpos_real[REAL2ISAAC], # 25
            # self.jvel_real[REAL2ISAAC], # 25
            jpos_multistep.reshape(-1),
            jvel_multistep.reshape(-1),
            robot.action_buf[:, :3].reshape(-1), # 4
            # self.jpos_des,
            # self.phase, # 4
        ]
        return np.concatenate(obs, dtype=np.float32)

    def apply_action(self, action: np.ndarray):
        self.action_buf = np.roll(self.action_buf, 1, axis=1)
        self.action_buf[:, 0] = action
        self.jpos_des = self.jpos_default.copy()
        self.jpos_des[self.action_joint_ids] += action * self.action_scaling

    def _handle(self):
        while True:
            self.lc_sub.handle()
            pass

    def _handler_legs(self, channel, data):
        msg = cyan_legged_data_lcmt.decode(data)
        self.jpos_real[:12] = msg.q
        self.jvel_real[:12] = msg.qd
        self.legs_timestamp = time.time()

    def _handler_ahw(self, channel, data):
        msg = cyan_armwaisthead_data_lcmt.decode(data)
        self.jpos_real[12:] = msg.q[:13]
        self.jvel_real[12:] = msg.qd[:13]
        self.ahw_timestamp = time.time()

    def _handler_sensor(self, channel, data):
        msg = cyan_lower_bodyimu_data_lcmt.decode(data)
        self.quat = msg.quat
        self.gyro = msg.gyro
        self.rot = R.from_quat(self.quat, scalar_first=True)
        self.gravity = self.rot.inv().apply([0., 0., -1.])
        self.sensor_timestamp = time.time()

ISAAC_QPOS = np.array(MJC_QPOS)[mjc2isaac]
robot = OrcaMJC(XML_PATH, ISAAC_JOINTS, ISAAC_QPOS, CTRL_JOINTS)

@set_exploration_type(ExplorationType.MODE)
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-s", "--sync", default=False, action="store_true")
    parser.add_argument("-d", "--dry", default=False, action="store_true")
    args = parser.parse_args()


    if args.sync:
        robot.run_sync()
    else:
        robot.run_async()
    print(robot.num_joints, robot.action_dim)

    path = args.path
    if path.endswith(".onnx"):
        backend = "onnx"

        policy_module = ONNXModule(path)
        try:
            meta = json.load(open(path.replace(".onnx", ".json"), "r"))
            _, _, action_scaling = robot.resolve_isaac(meta["action_scaling"])
            robot.action_scaling = np.array(action_scaling)
            ids, _, stiffness = robot.resolve_mjc(meta["stiffness"])
            robot.jnt_kp[ids] = np.array(stiffness) * 0.8
            ids, _, damping = robot.resolve_mjc(meta["damping"])
            robot.jnt_kd[ids] = np.array(damping)
            ids, _, effort_limit = robot.resolve_mjc(meta["effort_limit"])
            robot.effort_limit[ids] = np.array(effort_limit)
        except KeyError as e:
            print(f"Failed to load metadata: {e}")

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
    
    print(robot.get_jpos_offset("lleg_joint1"))
    print(robot.get_jpos_offset("lleg_joint4"))
    robot.set_jpos_offset("lleg_joint1", -0.1)
    robot.set_jpos_offset("rleg_joint1", -0.1)

    inp = {
        "is_init": np.array([True]),
        "context_adapt_hx": np.zeros((1, 128), dtype=np.float32),
        "adapt_hx": np.zeros((1, 128), dtype=np.float32),
    }

    client = PolicyClient(action_dim=robot.action_dim)
    client.set_cycle(1.1)
    
    t0 = time.perf_counter()
    for i in itertools.count():
        iter_start = time.perf_counter()
        inp["command"] = client.get_cmd(time=robot.data.time)[None, ...]
        inp["policy"] = client.get_obs(t=robot.data.time)[None, ...]
        inp["is_init"] = np.array([False])
        action, carry = policy(inp)
        robot.apply_action(action, 0.8)
        
        # client.apply_action(action)
        if i > 20:
            robot.enable_control = True
        inp = carry

        time.sleep(max(0, 0.02 - (time.perf_counter() - iter_start)))
        control_freq = i / (time.perf_counter() - t0)
        if i % 20 == 0:
            print("Control freq:", control_freq)
            # print(client.jpos_real[REAL2ISAAC])
            # print(robot.jpos[mjc2isaac])


if __name__ == "__main__":
    main()