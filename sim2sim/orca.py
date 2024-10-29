import torch
import numpy as np
import mujoco, mujoco_viewer
import time
import os
import itertools
import imageio

from tensordict import TensorDict
from torchrl.envs import set_exploration_type, ExplorationType
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

DEBUG = False

class MJCRobot:

    smoothing: int = 3

    def __init__(self, xml_path, headless=False):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        if headless:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, mode="offscreen", width=640, height=480)
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 1
        print(self.viewer.cam)
        
        # sometimes we only control/observe a subset of the joints
        self.joint_names = []
        self.joint_ranges = []
        for i in range(self.model.njnt):
            jnt = self.model.joint(i)
            if jnt.type.item() == 3:
                self.joint_names.append(jnt.name)
                self.joint_ranges.append(jnt.range)
        self.num_joints = len(self.joint_names)
        
        self.action_joint_ids_mjc = [self.joint_names.index(name) for name in CTRL_JOINTS]
        self.action_joint_ids_isaac = [ISAAC_JOINTS.index(name) for name in CTRL_JOINTS]

        self.action_dim = len(self.action_joint_ids_isaac)

        self.decimation = int(0.02 / self.model.opt.timestep)
        print(f"Decimation: {self.decimation}")
        print(f"Action dim: {self.action_dim}")
        
        # allocate buffers
        self.command    = np.zeros(4)
        self.action_buf = np.zeros((self.action_dim, 3))
        self.qpos_buf   = np.zeros((self.num_joints, self.smoothing))
        self.qvel_buf   = np.zeros((self.num_joints, self.smoothing))
        self.gravity_buf = np.zeros((3, self.smoothing))
        self.angvel_buf = np.zeros((3, self.smoothing))
        
        self.smth_weight = np.flip(np.arange(1, self.smoothing + 1)).reshape(1, -1)
        self.smth_weight = self.smth_weight / self.smth_weight.sum()
        print(f"Smoothing weight: {self.smth_weight}")
        
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
            self.qpos_default = np.array(MJC_QPOS)
            self.kp = np.array(MJC_KP)
            self.kd = np.array(MJC_KD)
            ankle_joint_ids = [self.joint_names.index(name) for name in ["lleg_joint6", "rleg_joint6"]]
            self.kp[ankle_joint_ids] = 15.
            self.kd[ankle_joint_ids] = 1.0
            self.action_scale = np.array(ISAAC_ACTION_SCALE)
            self.applied_action = np.zeros(self.action_dim)
            self.elbow_joint_ids = [self.joint_names.index(name) for name in ["larm_joint4", "rarm_joint4"]]

        mujoco.mj_step(self.model, self.data)

    def set_cycle(self, cycle):
        self.omega = np.pi * 2 / cycle
        self.offset = np.pi / 2
    
    def reset(self):
        self.set_cycle(1.0)
        self.update()
        obs = self.compute_obs()
        return obs
    
    def update(self):
        qpos = self.data.qpos[-self.num_joints:]
        qvel = self.data.qvel[-self.num_joints:]
        
        self.qpos_buf[:, 1:] = self.qpos_buf[:, :-1]
        self.qpos_buf[:, 0] = qpos
        self.qpos = np.mean(self.qpos_buf, axis=1)

        self.qvel_buf[:, 1:] = self.qvel_buf[:, :-1]
        self.qvel_buf[:, 0] = qvel
        self.qvel = np.mean(self.qvel_buf, axis=1)
        
        angvel = self.data.sensor("angular-velocity").data
        self.angvel_buf = np.roll(self.angvel_buf, 1, axis=1)
        self.angvel_buf[:, 0] = angvel
        self.angvel = np.mean(self.angvel_buf, axis=1)

        self.quat = self.data.sensor("orientation").data
        self.rot = R.from_quat(self.quat, scalar_first=True)
        gravity = self.rot.inv().apply([0, 0, -1.])
        self.gravity_buf[:, 1:] = self.gravity_buf[:, :-1]
        self.gravity_buf[:, 0] = gravity
        self.gravity_vec = normalize(np.mean(self.gravity_buf, axis=1))

    def step(self, action=None):
        if action is not None:
            self.action_buf[:, 1:] = self.action_buf[:, :-1]
            self.action_buf[:, 0] = action
            self.applied_action = lerp(self.applied_action, action, 0.7)

        for _ in range(self.decimation):
            self.pd_control(self.applied_action * self.action_scale)
            mujoco.mj_step(self.model, self.data)
            self.update()
        
        self.command[0] = 1.0
        self.command[3] = 0.3

        if self.viewer.render_mode == "window":
            self.viewer.render()
        else:
            self.img = self.viewer.read_pixels()
        return self.compute_obs()
    
    def compute_obs(self):
        t = self.data.time

        phase = self.omega * t + self.offset
        sin_t = np.sin(phase)
        cos_t = np.cos(phase)

        obs = [
            self.command,
            self.angvel,
            # rot.inv().apply(self.angvel),
            # np.array(sin(yaw_des), cos(yaw_des)),
            self.gravity_vec,
            # self.quat,
            self.qpos[mjc2isaac],
            self.qvel[mjc2isaac],
            self.action_buf.flatten(),
            np.array([sin_t, self.omega * cos_t, cos_t, -self.omega * sin_t]),
        ]
        return np.concatenate(obs, dtype=np.float32)

    def pd_control(self, qpos_des):
        qpos_des_isaac = self.qpos_default.copy()
        qpos_des_isaac[self.action_joint_ids_isaac] += qpos_des
        qpos_des_mjc = qpos_des_isaac[isaac2mjc]
        qpos_des_mjc[self.elbow_joint_ids] = 0.3

        qpos_err = qpos_des_mjc - self.qpos
        self.tau_des = qpos_err * self.kp - self.qvel * self.kd
        self.tau_ctrl = lerp(self.data.ctrl, self.tau_des, 1.0)
        self.data.ctrl[:] = np.clip(self.tau_ctrl * 0.98, -200, 200)


def lerp(a, b, t):
    return a + (b - a) * t

def normalize(x):
    return x / np.linalg.norm(x)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


@set_exploration_type(ExplorationType.MODE)
@torch.inference_mode()
def main():

    # xml_path = os.path.join(FILE_PATH, "orca1h/mjcf/orca1h.xml")
    xml_path = os.path.join(FILE_PATH, "orca_stable/mjcf/orca_description_stable.xml")
    robot = MJCRobot(xml_path)

    obs = robot.reset()

    policy_path = "/home/btx0424/lab/active-adaptation/scripts/policy-10-28_03-47.pt"
    policy = torch.load(policy_path)
    # policy.module[0].set_missing_tolerance(True)

    tensordict = TensorDict({
        "policy": torch.as_tensor(obs),
        "is_init": torch.tensor(1, dtype=bool),
        "context_adapt_hx": torch.zeros(128)
    }, []).unsqueeze(0)

    imgs = []
    
    try:
        for i in itertools.count():
            start = time.perf_counter()
            policy(tensordict)
            action = tensordict["action"].squeeze().numpy()
            obs = torch.as_tensor(robot.step(action))

            if hasattr(robot, "img") and i % 5 == 0:
                imgs.append(robot.img)
            
            tensordict["next", "policy"] = obs.unsqueeze(0)
            tensordict["next", "is_init"] = torch.tensor([0], dtype=bool)
            tensordict = tensordict["next"]

            time.sleep(max(0, 0.02 - (time.perf_counter() - start)))
            if i % 20 == 0:
                print("time:", robot.data.time)
                print("qpos:", robot.qpos)
                print("qvel:", robot.qvel)
                print("qvel:", robot.qvel_buf.mean(axis=1))
                print("tau:", robot.tau_ctrl)
            i += 1
    except KeyboardInterrupt:
        pass
    
    if len(imgs):
        print("Saving gif of length", len(imgs))
        imageio.mimsave("orca1h.gif", imgs, fps=50)


if __name__ == "__main__":
    main()