import torch
import numpy as np
import mujoco, mujoco_viewer
import time

from tensordict import TensorDict
from torchrl.envs import set_exploration_type, ExplorationType
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=2, suppress=True)

# joint order transformation
ISAAC_JOINTS = [
    'larm_joint1', 'rarm_joint1', 
    'waist_yaw_joint', 
    'larm_joint2', 'rarm_joint2', 
    'lleg_joint1', 'rleg_joint1', 
    'larm_joint3', 'rarm_joint3', 
    'lleg_joint2', 'rleg_joint2', 
    'larm_joint4', 'rarm_joint4', 
    'lleg_joint3', 'rleg_joint3', 
    'larm_joint5', 'rarm_joint5', 
    'lleg_joint4', 'rleg_joint4', 
    'lleg_joint5', 'rleg_joint5', 
    'lleg_joint6', 'rleg_joint6'
]
MJC_JOINTS = [
    "larm_joint1", "larm_joint2", "larm_joint3", "larm_joint4", "larm_joint5",
    "rarm_joint1", "rarm_joint2", "rarm_joint3", "rarm_joint4", "rarm_joint5",
    "waist_yaw_joint",
    "lleg_joint1", "lleg_joint2", "lleg_joint3", "lleg_joint4", "lleg_joint5", "lleg_joint6",
    "rleg_joint1", "rleg_joint2", "rleg_joint3", "rleg_joint4", "rleg_joint5", "rleg_joint6"
]

isaac2mjc = [ISAAC_JOINTS.index(joint) for joint in MJC_JOINTS]
mjc2isaac = [MJC_JOINTS.index(joint) for joint in ISAAC_JOINTS]

ISAAC_KP = [50., 50., 50., 50., 50., 50., 50., 30., 30., 50., 50., 30., 30., 50.,
        50., 15., 15., 50., 50., 50., 50., 30., 30.]
ISAAC_KD = [2., 2., 3., 2., 2., 3., 3., 1., 1., 3., 3., 1., 1., 3., 3., 1., 1., 3.,
        3., 3., 3., 1., 1.]
ISAAC_QPOS = [ 0.0000,  0.0000,  0.0000,  0.2000,  0.2000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000, -0.2000, -0.2000,  0.0000,  0.0000,  0.0000,
         0.0000, -0.2000, -0.2000,  0.0000,  0.0000,  0.0000,  0.0000]

DEBUG = False

class MJCRobot:

    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.action_dim = self.data.ctrl.shape[0]
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 1
        print(self.viewer.cam)
        breakpoint()

        self.decimation = int(0.02 / self.model.opt.timestep)
        print(f"Decimation: {self.decimation}")
        print(f"Action dim: {self.action_dim}")
        
        # allocate buffers
        self.command    = np.zeros(4)
        self.action_buf = np.zeros((self.action_dim, 2))
        smoothing: int = 2
        self.qvel_buf   = np.zeros((self.action_dim, smoothing))
        self.angvel_buf = np.zeros((3, smoothing))
        self.smth_weight = np.flip(np.arange(1, smoothing + 1)).reshape(1, -1)
        self.smth_weight = self.smth_weight / self.smth_weight.sum()
        print(f"Smoothing weight: {self.smth_weight}")
        
        self.qpos_default = np.zeros(self.action_dim)

        if DEBUG:
            self.kp = np.ones(self.action_dim) * 50
            self.kp[[2, 3]] = 30
            self.kp[[7, 8]] = 30
            self.kp[[-1, -2, -7, -8]] = 15.0
            self.kd = np.ones(self.action_dim)
            self.kd[[2, 3]] = 0.5
            self.kd[[7, 8]] = 0.5
            self.kd[[-1, -2, -7, -8]] = 0.5
        else:
            self.qpos_default = np.array(ISAAC_QPOS)
            self.kp = np.array(ISAAC_KP)[isaac2mjc] * 0.8
            self.kd = np.array(ISAAC_KD)[isaac2mjc] * 0.8
            self.kp[[-1, -2, -7, -8]] = 20
            self.kd[[-1, -2, -7, -8]] = 0.6

        mujoco.mj_step(self.model, self.data)

    def set_cycle(self, cycle):
        self.omega = np.pi * 2 / cycle
        self.offset = np.pi / 2
    
    def reset(self):
        self.set_cycle(1.1)
        self.qpos = self.data.qpos[-self.action_dim:]
        self.qvel = self.data.qvel[-self.action_dim:]
        self.angvel = self.data.sensor("angular-velocity").data
        obs = self.compute_obs()
        return obs
    
    def step(self, action=None):
        if action is not None:
            self.action_buf = np.roll(self.action_buf, 1, axis=1)
            self.action_buf[:, 0] = action
            qpos_des = np.clip(action * 0.5 + self.qpos_default, -4, 4)
        else:
            qpos_des = self.qpos_default
        
        if DEBUG:
            t = self.data.time
            phase = self.omega * t + self.offset
            qpos_des[0] = 0.5 * np.sin(phase)
            qpos_des[1] = 0.2
            qpos_des[5] = 0.5 * -np.sin(phase)
            qpos_des[6] = 0.2
            qpos_des[11] = 0.5 * np.sin(phase)
            qpos_des[17] = 0.5 * -np.sin(phase)

        self.pd_control(qpos_des[isaac2mjc])
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
            self.qpos = self.data.qpos[-self.action_dim:]
            qvel = self.data.qvel[-self.action_dim:]
            self.qvel_buf = np.roll(self.qvel_buf, 1, axis=1)
            self.qvel_buf[:, 0] = qvel
            self.qvel = np.mean(self.qvel_buf, axis=1)

            angvel = self.data.sensor("angular-velocity").data
            self.angvel_buf = np.roll(self.angvel_buf, 1, axis=1)
            self.angvel_buf[:, 0] = angvel
            self.angvel = np.mean(self.angvel_buf, axis=1)
        
        self.command[0] = 1.0

        self.viewer.cam
        self.viewer.render()
        return self.compute_obs()
    
    def compute_obs(self):
        quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]
        rot = R.from_quat(quat)
        gravity_vec = rot.inv().apply([0, 0, -1.])

        t = self.data.time
        phase = self.omega * t + self.offset

        obs = [
            self.command,
            self.angvel,
            gravity_vec,
            self.qpos[mjc2isaac],
            self.qvel[mjc2isaac],
            self.action_buf.flatten(),
            np.array([np.sin(phase), np.cos(phase)]),
        ]
        return np.concatenate(obs, dtype=np.float32)

    def pd_control(self, qpos_des):
        qpos_err = qpos_des - self.qpos
        self.tau_des = qpos_err * self.kp - self.qvel_buf.mean(1) * self.kd
        self.tau_ctrl = lerp(self.data.ctrl, self.tau_des, 0.4)
        self.data.ctrl[:] = np.clip(self.tau_ctrl, -200, 200)


def lerp(a, b, t):
    return a + (b - a) * t


@torch.inference_mode()
def main():
    # xml_path = "orca/mjcf/orca_1.xml"
    # xml_path = "XBot/mjcf/XBot-L.xml"
    # xml_path = "orca1h/mjcf/orca_description_mj.xml"
    xml_path = "orca1h/mjcf/orca1h.xml"
    # xml_path = "orca/mjcf/orca_description_mj.xml"
    robot = MJCRobot(xml_path)

    obs = robot.reset()

    policy_path = "/home/btx0424/isaac_lab/active-adaptation/scripts/policy-07-05_11-51.pt"
    policy = torch.load(policy_path)
    policy.module[0].set_missing_tolerance(True)

    tensordict = TensorDict({
        "policy": torch.as_tensor(obs),
        "is_init": torch.tensor(1, dtype=bool),
        "context_adapt_hx": torch.zeros(128)
    }, []).unsqueeze(0)

    with set_exploration_type(ExplorationType.MODE):
        i = 0
        while True:
            start = time.perf_counter()
            policy(tensordict)
            action = tensordict["action"].squeeze().numpy()
            obs = torch.as_tensor(robot.step(action))
            # obs = torch.as_tensor(robot.step())
            
            tensordict["next", "policy"] = obs.unsqueeze(0)
            tensordict["next", "is_init"] = torch.tensor([0], dtype=bool)
            tensordict = tensordict["next"]

            time.sleep(max(0, 0.02 - (time.perf_counter() - start)))
            if i % 20 == 0:
                print("qpos:", robot.qpos)
                print("qvel:", robot.qvel)
                print("qvel:", robot.qvel_buf.mean(axis=1))
                print("tau:", robot.tau_ctrl)
            i += 1

if __name__ == "__main__":
    main()