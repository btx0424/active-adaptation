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
ISAAC_JOINTS = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
MJC_JOINTS = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

isaac2mjc = [ISAAC_JOINTS.index(joint) for joint in MJC_JOINTS]
mjc2isaac = [MJC_JOINTS.index(joint) for joint in ISAAC_JOINTS]

# fmt: off
ISAAC_KP = [
    25., 25., 25., 25., 
    25., 25., 25., 25., 
    25., 25., 25., 25.,
]
ISAAC_KD = [
    0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5,
]
ISAAC_QPOS = [
    0.1, -0.1, 0.1, -0.1,
    0.78, 0.78, 0.78, 0.78,
    -1.5, -1.5, -1.5, -1.5,

    # 0.2, -0.2, 0.2, -0.2,
    # 1.0, 1.0, 1.0, 1.0,
    # -1.8, -1.8, -1.8, -1.8,
]
# fmt: on

DEBUG = False


class CommandManager:

    command_dim: int

    def __init__(self) -> None:
        self.command = np.zeros(self.command_dim)

    def update(self):
        pass


class FixedCommandForce(CommandManager):
    command_dim = 10

    setpoint_pos_b = np.array([1.0, 0.0, 0.0])
    yaw_diff = 0.0

    kp = 10.0
    kd = 3.0
    virtual_mass = 10.0

    def update(self):
        self.command[:2] = self.setpoint_pos_b[:2]
        self.command[2] = self.yaw_diff
        self.command[3:5] = self.kp * self.setpoint_pos_b[:2]
        self.command[5:8] = self.kd
        self.command[8] = self.kp * self.yaw_diff
        self.command[9] = self.virtual_mass


class MJCRobot:

    smoothing: int = 2

    def __init__(self, xml_path, command_manager: CommandManager, headless=False):
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
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.joint_observed_ids = [self.joint_names.index(joint) for joint in MJC_JOINTS]

        self.actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.actuator_controlled_ids = [self.actuator_names.index(joint) for joint in MJC_JOINTS]
        
        self.action_dim = len(self.actuator_controlled_ids)
        
        self.command_manager = command_manager

        self.decimation = int(0.02 / self.model.opt.timestep)
        print(f"Decimation: {self.decimation}")
        print(f"Action dim: {self.action_dim}")
        
        # allocate buffers
        self.action_buf_steps = 3
        self.action_buf = np.zeros((self.action_dim, self.action_buf_steps))
        self.applied_action = np.zeros(self.action_dim)
        self.qvel_buf   = np.zeros((self.action_dim, self.smoothing))
        self.angvel_buf = np.zeros((3, self.smoothing))
        self.smth_weight = np.flip(np.arange(1, self.smoothing + 1)).reshape(1, -1)
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
            self.kp = np.array(ISAAC_KP)[isaac2mjc]
            self.kd = np.array(ISAAC_KD)[isaac2mjc]
            # self.kp[[-1, -2, -7, -8]] = 20
            # self.kd[[-1, -2, -7, -8]] = 0.6

        mujoco.mj_step(self.model, self.data)

    def reset(self):
        self.update()
        obs = self.compute_obs()
        return obs
    
    def update(self):
        self.qpos = self.data.qpos[-self.model.njnt:][self.joint_observed_ids]
        qvel = self.data.qvel[-self.model.njnt:][self.joint_observed_ids]
        self.qvel_buf = np.roll(self.qvel_buf, 1, axis=1)
        self.qvel_buf[:, 0] = qvel
        self.qvel = np.mean(self.qvel_buf, axis=1)
        
        self.command_manager.update()
        
        # angvel = self.data.sensor("angular-velocity").data
        # self.angvel_buf = np.roll(self.angvel_buf, 1, axis=1)
        # self.angvel_buf[:, 0] = angvel
        # self.angvel = np.mean(self.angvel_buf, axis=1)

    def step(self, action=None):
        if action is not None:
            self.action_buf = np.roll(self.action_buf, 1, axis=1)
            self.action_buf[:, 0] = action
            self.applied_action[:] = action
            # self.applied_action[:] = lerp(self.applied_action, action, 0.8)
            qpos_des = np.clip(action * 0.5 + self.qpos_default, -6, 6)
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


        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
            self.update()
            self.pd_control(qpos_des[isaac2mjc])
        
        if self.viewer.render_mode == "window":
            self.viewer.render()
        else:
            self.img = self.viewer.read_pixels()
        return self.compute_obs()
    
    def compute_obs(self):
        quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]
        rot = R.from_quat(quat)
        gravity_vec = rot.inv().apply([0, 0, -1.])

        print(self.command_manager.command)
        obs = [
            self.command_manager.command,
            # self.angvel,
            gravity_vec,
            self.qpos[mjc2isaac],
            self.qvel[mjc2isaac],
            self.action_buf.flatten(),
        ]
        return np.concatenate(obs, dtype=np.float32)

    def pd_control(self, qpos_des):
        qpos_err = qpos_des - self.qpos
        self.tau_des = qpos_err * self.kp - self.qvel_buf.mean(1) * self.kd
        # self.tau_ctrl = lerp(self.data.ctrl, self.tau_des, 0.5)
        self.tau_ctrl = self.tau_des
        self.data.ctrl[self.actuator_controlled_ids] = np.clip(self.tau_ctrl, -200, 200)


def lerp(a, b, t):
    return a + (b - a) * t


FILE_PATH = os.path.dirname(os.path.abspath(__file__))


@set_exploration_type(ExplorationType.MODE)
@torch.inference_mode()
def main():

    xml_path = os.path.join(FILE_PATH, "go2/go2_plane.xml")
    command_manager = FixedCommandForce()
    robot = MJCRobot(xml_path, command_manager)

    obs = robot.reset()

    policy_path = "policy-go2force-639.pt"
    policy_path = os.path.join(FILE_PATH, policy_path)
    policy = torch.load(policy_path)
    policy.module[0].set_missing_tolerance(True)

    command, obs = obs[:command_manager.command_dim], obs[command_manager.command_dim:]


    tensordict = TensorDict(
        {
            "command": torch.as_tensor(command),
            "policy": torch.as_tensor(obs),
            "is_init": torch.tensor(1, dtype=bool),
            "adapt_hx": torch.zeros(128),
        },
        [],
    ).unsqueeze(0)

    imgs = []

    recorded_actions = []

    
    try:
        for i in itertools.count():
            start = time.perf_counter()

            policy(tensordict)
            action = tensordict["action"].squeeze().numpy()
            recorded_actions.append(action)

            obs = torch.as_tensor(robot.step(action))
            command, obs = obs.split([command_manager.command_dim, obs.shape[0] - command_manager.command_dim])
            tensordict["next", "command"] = command.unsqueeze(0)
            tensordict["next", "policy"] = obs.unsqueeze(0)
            tensordict["next", "is_init"] = torch.tensor(0, dtype=bool).unsqueeze(0)

            if hasattr(robot, "img"):
                imgs.append(robot.img)
            
            tensordict = tensordict["next"]

            time.sleep(max(0, 0.02 - (time.perf_counter() - start)))
            if i % 20 == 0:
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

    if recorded_actions:
        import matplotlib.pyplot as plt
        recorded_actions = np.array(recorded_actions)  # Convert to numpy array for easier handling
        time_steps = np.arange(len(recorded_actions))

        fig, axs = plt.subplots(12, 1, figsize=(10, 20))  # Create 12 subplots, one for each action dimension

        for i in range(12):
            axs[i].plot(time_steps, recorded_actions[:, i])
            axs[i].set_title(f'Action {i+1}')
            axs[i].set_xlabel('Time Step')
            axs[i].set_ylabel('Action Value')

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    main()