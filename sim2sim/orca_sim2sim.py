import os
import torch
import signal
import numpy as np
import mujoco
import mujoco_viewer
import lcm
import threading
import itertools
import time
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from sim2sim.timer_fd import Timer
from omni.isaac.lab.utils.string import (
    resolve_matching_names, 
    resolve_matching_names_values
)
from sim2sim.lcm_types.python.cyan_legged_cmd_lcmt import *
from sim2sim.lcm_types.python.cyan_armwaisthead_cmd_lcmt import *
from sim2sim.lcm_types.python.cyan_legged_data_lcmt import *
from sim2sim.lcm_types.python.cyan_armwaisthead_data_lcmt import *
from sim2sim.lcm_types.python.cyan_lower_bodyimu_data_lcmt import *

np.set_printoptions(precision=2, suppress=True)

MJC_JOINTS = [
    'lleg_joint1', 'lleg_joint2', 'lleg_joint3', 'lleg_joint4', 'lleg_joint5', 'lleg_joint6', 
    'rleg_joint1', 'rleg_joint2', 'rleg_joint3', 'rleg_joint4', 'rleg_joint5', 'rleg_joint6', 
    'waist_yaw_joint', 
    'larm_joint1', 'larm_joint2', 'larm_joint3', 'larm_joint4', 'larm_joint5', 'larm_joint6', 
    'rarm_joint1', 'rarm_joint2', 'rarm_joint3', 'rarm_joint4', 'rarm_joint5', 'rarm_joint6'
]

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

MJC_TO_ISAAC = [MJC_JOINTS.index(i) for i in ISAAC_JOINTS]
ISAAC_TO_MJC = [ISAAC_JOINTS.index(i) for i in MJC_JOINTS]
ISAAC_CTRL_JOINTS = [ISAAC_JOINTS.index(i) for i in CTRL_JOINTS]
MJC_CTRL_JOINTS = [MJC_JOINTS.index(i) for i in CTRL_JOINTS]

@dataclass
class OrcaCfg:
    stiffness = {
        "waist_yaw_joint": 75.,
        "[l,r]arm_joint1": 75.,
        "[l,r]arm_joint2": 50.,
        "[l,r]arm_joint3": 30.,
        "[l,r]arm_joint4": 30.,
        "[l,r]arm_joint5": 15.,
        "[l,r]arm_joint6": 15.,
        "[l,r]leg_joint1": 75.,
        "[l,r]leg_joint2": 50.,
        "[l,r]leg_joint3": 50.,
        "[l,r]leg_joint4": 75.,
        "[l,r]leg_joint5": 30.,
        "[l,r]leg_joint6": 15.,
    }
    damping = {
        "waist_yaw_joint": 3.,
        "[l,r]arm_joint1": 6.,
        "[l,r]arm_joint2": 3.,
        "[l,r]arm_joint3": 0.5,
        "[l,r]arm_joint4": 1.,
        "[l,r]arm_joint5": 1.,
        "[l,r]arm_joint6": 1.,
        "[l,r]leg_joint1": 6., # 6.
        "[l,r]leg_joint2": 3.,
        "[l,r]leg_joint3": 3.,
        "[l,r]leg_joint4": 6., # 6.
        "[l,r]leg_joint5": 2.,
        "[l,r]leg_joint6": 1.,
    }
    joint_pos = {
        "waist_yaw_joint": 0.0,
        ".*arm_joint[1,3,5]": 0.0,
        ".*leg_joint[1,2,3,5,6]": 0.0,
        "[l,r]arm_joint2": 0.1,
        "[l,r]leg_joint4": -0.1,
        "[l,r]arm_joint4": 0.3,
    }

def lerp(a, b, t):
    return a + (b - a) * t

class OrcaMJC:
    def __init__(self, xml_path: str, cfg: OrcaCfg):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.sync_control = True
        
        self.joint_names = []
        self.joint_ranges = []
        for i in range(self.model.njnt):
            jnt = self.model.joint(i)
            if jnt.type.item() == 3:
                self.joint_names.append(jnt.name)
                self.joint_ranges.append(jnt.range)
        
        self.joint_ranges = np.array(self.joint_ranges)
        self.num_joints = len(self.joint_names)
        print(len(self.data.qpos), self.num_joints, self.model.njnt)
        print(self.joint_names)
        
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        ttl = 255
        self.lowlevel_legs_lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))
        self.lowlevel_ahw_lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))
        self.lowlevel_sensors_lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))

        self.leg_joint_ids, self.leg_joint_names = resolve_matching_names(".*leg_joint.*", self.joint_names)
        self.ahw_joint_ids, self.ahw_joint_names = resolve_matching_names(["waist_yaw_joint", ".*arm_joint.*"], self.joint_names)
        print(self.leg_joint_names, self.ahw_joint_names)

        self.default_joint_pos = self.resolve_joint_params(self.cfg.joint_pos)
        self.joint_pos_des = self.default_joint_pos.copy()
        self.joint_stiffness = self.resolve_joint_params(self.cfg.stiffness)
        self.joint_damping = self.resolve_joint_params(self.cfg.damping)
        
        def list_str(l: list):
            return "[" + ", ".join([str(i) for i in l]) + "]"
        print(list_str(self.joint_stiffness))
        print(list_str(self.joint_damping))
        print(list_str(self.default_joint_pos))

        # exit(0)
        T = 4
        self.joint_vel_buf = np.zeros((self.num_joints, T))
        self.base_ang_vel_buf = np.zeros((3, T))

        self._pub_topic_legs = 'robot2controller_legs'
        self._pub_topic_ahw = 'robot2controller_ahw'
        self._pub_topic_sensors = 'robot2controller_sensors'
        self.__ll_msg_legs = cyan_legged_data_lcmt()
        self.__ll_msg_ahw = cyan_armwaisthead_data_lcmt()
        self.__ll_msg_sensor = cyan_lower_bodyimu_data_lcmt()
        self.thread_pub = threading.Thread(target=self.thread_pub_fn)
        self.thread_control = threading.Thread(target=self.thread_control_fn)

        mujoco.mj_step(self.model, self.data)
        self.update()
    
    def resolve_joint_params(self, data):
        result = np.zeros(self.data.ctrl.shape)
        idx, name, value = resolve_matching_names_values(data, self.joint_names)
        result[idx] = value
        return result

    def update(self):
        self.qpos = self.data.qpos
        self.qvel = self.data.qvel
        
        self.joint_pos = self.data.qpos[-self.num_joints:]
        joint_vel = self.data.qvel[-self.num_joints:]
        self.joint_vel_buf = np.roll(self.joint_vel_buf, 1, axis=1)
        self.joint_vel_buf[:, 0] = joint_vel
        self.joint_vel = np.mean(self.joint_vel_buf, axis=1)

        self.base_pos = self.data.qpos[:3]
        # self.base_quat = self.data.qpos[3:7]
        self.base_quat = self.data.sensor("orientation").data

        self.base_lin_vel = self.data.qvel[:3]
        # self.base_ang_vel = self.data.qvel[3:6]
        base_ang_vel = self.data.sensor("angular-velocity").data
        self.base_ang_vel_buf = np.roll(self.base_ang_vel_buf, 1, axis=1)
        self.base_ang_vel_buf[:, 0] = base_ang_vel
        self.base_ang_vel = np.mean(self.base_ang_vel_buf, axis=1)
        
        self.base_rot = R.from_quat(self.base_quat, scalar_first=True)
        self.gravity = self.base_rot.inv().apply([0, 0, -1.])

    @torch.inference_mode()
    def spin(self):
        self.thread_pub.start()
        if not self.sync_control:
            self.thread_control.start()
        
        # threading.Thread(target=self.thread_policy_fn).start()
        from sim2sim.utils import ONNXModule
        import math
        from tensordict import TensorDict

        # policy = ONNXModule("/home/btx0424/lab/active-adaptation/scripts/policy-10-28_00-28.onnx")
        policy = torch.load("/home/btx0424/lab/active-adaptation/scripts/policy-10-28_00-28.pt")
        action_dim = len(CTRL_JOINTS)
        self.action_buf = np.zeros((action_dim, 3), dtype=np.float32)
        
        offset = self.default_joint_pos[MJC_TO_ISAAC]

        omega = 1.0
        t0 = time.perf_counter()

        inp = TensorDict({
            "is_init": np.zeros((1,), dtype=bool),
            "context_adapt_hx": np.zeros((1, 128), dtype=np.float32),
        }, [1])

        main_control_timer = Timer(self.model.opt.timestep)
        substeps = int(0.02 / self.model.opt.timestep)
        for i in itertools.count():
            try:
                # self.update()
                t = omega * i * 0.001 + torch.pi / 2

                sin_t = math.sin(omega * t)
                cos_t = math.cos(omega * t)
                obs = np.concatenate([
                    np.asarray([1., 0., 0., 0.0]),
                    self.base_ang_vel,  # 3
                    self.gravity,       # 3
                    self.joint_pos[MJC_TO_ISAAC], # 21
                    self.joint_vel[MJC_TO_ISAAC], # 21
                    self.action_buf.reshape(-1),
                    np.asarray([sin_t, omega * cos_t, cos_t, -omega * sin_t]),
                ])
                inp["policy"] = torch.from_numpy(obs.reshape(1, -1).astype(np.float32))
                out = policy(inp)
                action = out["action"].reshape(-1).numpy()
                inp = TensorDict({
                    "is_init": np.zeros((1,), dtype=bool),
                    "context_adapt_hx": out["next", "context_adapt_hx"],
                }, [1])
                # action = np.array([sin_t, cos_t, sin_t, cos_t])

                self.action_buf[:, 1:] = self.action_buf[:, :-1]
                self.action_buf[:, 0] = action
                
                joint_pos_des = offset.copy()
                joint_pos_des[ISAAC_CTRL_JOINTS] += action * 0.5
                self.joint_pos_des = joint_pos_des[ISAAC_TO_MJC]
                
                for _ in range(substeps):
                    # if self.sync_control:
                    mujoco.mj_step(self.model, self.data)
                    self.update()
                    self.pd_control()

                # if i % 20 == 0:
                self.viewer.render()
                main_control_timer.sleep()
            except KeyboardInterrupt:
                self.thread_pub.join()
                break
    
    def pd_control(self):
        pos_error = self.joint_pos_des - self.joint_pos
        vel_error = 0. - self.joint_vel
        tau = self.joint_stiffness * pos_error + self.joint_damping * vel_error
        tau = lerp(self.data.ctrl, tau, 0.5)
        self.data.ctrl[:] = np.clip(tau, -200., 200.)

    def thread_policy_fn(self):
        from sim2sim.utils import ONNXModule
        import math

        policy = ONNXModule("/home/btx0424/lab/active-adaptation/scripts/policy-10-28_00-28.onnx")
        action_dim = len(CTRL_JOINTS)
        self.action_buf = np.zeros((action_dim, 3), dtype=np.float32)
        
        offset = self.default_joint_pos[MJC_TO_ISAAC]

        omega = 1.0
        t0 = time.perf_counter()

        inp = {
            "is_init": np.zeros((1,), dtype=bool),
            "context_adapt_hx": np.zeros((1, 128), dtype=np.float32),
        }
        for i in itertools.count():
            start = time.perf_counter()
            
            t = omega * (start - t0) + torch.pi / 2

            sin_t = math.sin(omega * t)
            cos_t = math.cos(omega * t)
            obs = np.concatenate([
                np.asarray([1., 0., 0., 0.0]),
                self.base_ang_vel,  # 3
                self.gravity,       # 3
                self.joint_pos[MJC_TO_ISAAC], # 21
                self.joint_vel[MJC_TO_ISAAC], # 21
                self.action_buf.reshape(-1),
                np.asarray([sin_t, omega * cos_t, cos_t, -omega * sin_t]),
            ])
            inp["policy"] = obs.reshape(1, -1).astype(np.float32)
            out = policy(inp)
            action = out["action"].reshape(-1)
            # action = np.array([sin_t, cos_t, sin_t, cos_t])

            self.action_buf[:, 1:] = self.action_buf[:, :-1]
            self.action_buf[:, 0] = action
            
            joint_pos_des = offset.copy()
            joint_pos_des[ISAAC_CTRL_JOINTS] += action * 0.5
            self.joint_pos_des = joint_pos_des[ISAAC_TO_MJC]

            inp = {
                "is_init": np.zeros((1,), dtype=bool),
                "context_adapt_hx": out[("next", "context_adapt_hx")],
            }
            
            time.sleep(0.02 - max(0, 0.02 - (time.perf_counter() - start)))
            
    
    def thread_control_fn(self):
        """
        PD control of joint position
        """
        timer = Timer(1 / 500)
        for i in itertools.count():
            self.pd_control()
            timer.sleep()
    
    def thread_pub_fn(self):
        timer = Timer(1 / 100)
        for i in itertools.count():
            self.__ll_msg_legs.q = self.joint_pos[self.leg_joint_ids]
            self.__ll_msg_legs.qd = self.joint_vel[self.leg_joint_ids]
            self.__ll_msg_legs.tauIq = np.zeros(12)
            
            self.__ll_msg_ahw.q[:13] = self.joint_pos[self.ahw_joint_ids]
            self.__ll_msg_ahw.qd[:13] = self.joint_vel[self.ahw_joint_ids]
            self.__ll_msg_ahw.tauIq = np.zeros(18)

            self.__ll_msg_sensor.quat = self.base_quat
            self.__ll_msg_sensor.gyro = self.base_lin_vel

            self.lowlevel_legs_lcm.publish(self._pub_topic_legs, self.__ll_msg_legs.encode())
            self.lowlevel_ahw_lcm.publish(self._pub_topic_ahw, self.__ll_msg_ahw.encode())
            self.lowlevel_sensors_lcm.publish(self._pub_topic_sensors, self.__ll_msg_sensor.encode())

            if i % 50 == 0:
                print(self.gravity)
            timer.sleep()

def main():
    xml_path = "/home/btx0424/lab/active-adaptation/sim2sim/orca_stable/mjcf/orca_description_stable.xml"
    env = OrcaMJC(xml_path, OrcaCfg())
    env.spin()

if __name__ == "__main__":
    main()