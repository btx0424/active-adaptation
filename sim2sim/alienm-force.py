import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import itertools
import imageio
import queue

from tensordict import TensorDict
from torchrl.envs import set_exploration_type, ExplorationType
from scipy.spatial.transform import Rotation as R
from liveplot import LivePlotClient

import zmq

liveplot = LivePlotClient()

np.set_printoptions(precision=2, suppress=True)

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
    "arm_joint1",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
]
MJC_JOINTS = [
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
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
]

isaac2mjc = [ISAAC_JOINTS.index(joint) for joint in MJC_JOINTS]
mjc2isaac = [MJC_JOINTS.index(joint) for joint in ISAAC_JOINTS]

# fmt: off
dog_joint_kp = 80.0
dog_joint_kd = 2.0
arm_joint_kp_1 = 80.0
arm_joint_kp_2 = 20.0
arm_joint_kd_1 = 2.0
arm_joint_kd_2 = 1.0
ISAAC_KP = [dog_joint_kp] * 8 + [arm_joint_kp_1] + [dog_joint_kp] * 4 + [arm_joint_kp_1] * 2 + [arm_joint_kp_2] * 2

ISAAC_KD = [dog_joint_kd] * 8 + [arm_joint_kd_1] + [dog_joint_kd] * 4 + [arm_joint_kd_1] * 2 + [arm_joint_kd_2] * 2

ISAAC_QPOS = [
    0.3, -0.3, 0.3, -0.3,
    1.0, 1.0, 1.1, 1.1,
    0.0,
    -2.0, -2.0, -2.1, -2.1,
    1.0, -1.0, 0.0, 0.0
]

hip_scale = 0.1
thigh_calf_scale = 0.5
arm_scale = 1.0

ISAAC_SCALE = [hip_scale] * 4 + [thigh_calf_scale] * 4 + [arm_scale] + [thigh_calf_scale] * 4 + [arm_scale] * 4
# fmt: on


import numpy as np
import scipy.signal as signal


class SecondOrderLowPassFilter:
    def __init__(self, cutoff: float, fs: float):
        """Initialize the low-pass filter with cutoff frequency and sample rate."""
        self.fs = fs
        self.cutoff = cutoff

        # Compute the filter coefficients (second-order, low-pass)
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        print("Normalized cutoff: ", normal_cutoff)
        self.b, self.a = signal.butter(2, normal_cutoff, btype="lowpass", analog=False)
        print(self.b, self.a)
        # Initialize previous inputs and outputs (n-1, n-2 terms)
        self.x1 = 0.0  # x[n-1]
        self.x2 = 0.0  # x[n-2]
        self.y1 = 0.0  # y[n-1]
        self.y2 = 0.0  # y[n-2]

    def update(self, x: np.ndarray) -> np.ndarray:
        """Update the filter with a new input sample `x`, and return the filtered output."""
        # Apply the difference equation to get the new output value y[n]
        y = (
            self.b[0] * x
            + self.b[1] * self.x1
            + self.b[2] * self.x2
            - self.a[1] * self.y1
            - self.a[2] * self.y2
        )

        # Update the state (shift the previous inputs and outputs)
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y


def yaw_quat(quat):
    quat = np.array(quat)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    yaw_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    return yaw_quat


class MJCCommandManager:
    command_dim: int = 30

    def __init__(self, mj_model, mj_data):
        self._mj_data = mj_data
        eef_body_name = "arm_seg6"
        base_body_name = "base"

        self.eef_body_id = mj_model.body(eef_body_name).id
        self.base_body_id = mj_model.body(base_body_name).id

        self.setpoint_base_pos_diff_b_target = np.array([0.5, 0.0])  # x, y
        self.setpoint_base_pos_diff_b = self.setpoint_base_pos_diff_b_target.copy()
        self.yaw_diff = 0.0

        self.max_pos_x_b = 2.0
        self.max_pos_y_b = 2.0

        self.kp_dog_xy = 10.0
        self.kd_dog_xy = 2.0 * np.sqrt(self.kp_dog_xy)
        self.kp_dog_yaw = 1.0
        self.kd_dog_yaw = 2.0 * np.sqrt(self.kp_dog_yaw)
        self.virtual_mass = 5.0
        self.virtual_inertia = 1.0

        self.arm_default_pos = np.array([0.2, 0.0, 0.5])
        self.setpoint_pos_ee_b_target = self.arm_default_pos.copy()
        self.setpoint_pos_ee_b = self.arm_default_pos.copy()
        self.setpoint_pos_ee_b_diff = np.zeros(3)

        self.arm_pos_max = np.array([0.6, 0.3, 0.7])
        self.arm_pos_min = np.array([0.2, -0.3, 0.35])

        self.kp_arm = 40.0
        self.kd_arm = 2 * np.sqrt(self.kp_arm)
        self.mass_arm = 1.0

        self.command = np.zeros(18, dtype=np.float32)

    def update_command(self):
        ee_pos_w = (
            self._mj_data.xpos[self.eef_body_id] - self._mj_data.xpos[self.base_body_id]
        )
        quat = yaw_quat(self._mj_data.xquat[self.base_body_id])
        rot = R.from_quat(quat[[1, 2, 3, 0]])
        ee_pos_b = rot.inv().apply(ee_pos_w)

        self.setpoint_pos_ee_b_diff = self.setpoint_pos_ee_b - ee_pos_b

        self.command[0:2] = self.setpoint_base_pos_diff_b[:2]
        self.command[2] = self.yaw_diff
        self.command[3:6] = self.setpoint_pos_ee_b_diff

        # print("setpoint_ee diff_ b", self.setpoint_pos_ee_b_diff)

        self.command[6:8] = self.kp_dog_xy * self.setpoint_base_pos_diff_b
        self.command[8] = self.kp_dog_yaw * self.yaw_diff
        self.command[9:12] = self.kp_arm * self.setpoint_pos_ee_b_diff

        self.command[12:13] = self.kd_dog_xy
        self.command[13:14] = self.kd_dog_yaw
        self.command[14] = self.kd_arm

        self.command[15] = self.virtual_mass
        self.command[16] = self.virtual_inertia
        self.command[17] = self.mass_arm

    def update_setpoint(self):
        pass

    def update(self):
        self.update_setpoint()
        self.update_command()


class ZMQCommand(MJCCommandManager):
    def __init__(self, mj_model, mj_data, zmq_addr="tcp://192.168.1.101:5556"):
        super().__init__(mj_model, mj_data)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(zmq_addr)

    def update_setpoint(self):
        data_dict = self.socket.recv_pyobj()

        self.setpoint_base_pos_diff_b = data_dict["setpoint_base_pos_diff_b"]
        self.yaw_diff = data_dict["yaw_diff"]
        self.setpoint_pos_ee_b = data_dict["setpoint_pos_ee_b"]

        self.kp_dog_xy = data_dict["kp_dog_xy"]
        self.kp_dog_yaw = data_dict["kp_dog_yaw"]
        self.kp_arm = data_dict["kp_arm"]

        self.kd_dog_xy = data_dict["kd_dog_xy"]
        self.kd_dog_yaw = data_dict["kd_dog_yaw"]
        self.kd_arm = data_dict["kd_arm"]


class Oscillator:
    def __init__(self, use_history: bool = False):
        self.use_history = use_history
        self.phi = np.zeros(4)
        self.phi[0] = np.pi
        self.phi[3] = np.pi
        self.phi_dot = np.zeros(4)
        self.phi_history = np.zeros((4, 4))

    def update_phis(self, command, dt=0.02):
        omega = np.pi * 4
        move = True
        if move:
            dphi = self._trot(self.phi) + omega
        else:
            dphi = self._stand(self.phi)
        self.phi_dot[:] = dphi
        self.phi = (self.phi + self.phi_dot * dt) % (2 * np.pi)
        self.phi_history = np.roll(self.phi_history, 1, axis=0)
        self.phi_history[0] = self.phi

    def _trot(self, phi: np.ndarray):
        dphi = np.zeros(4)
        dphi[0] = phi[3] - phi[0]
        dphi[1] = (phi[2] - phi[1]) + ((phi[0] + np.pi - phi[1]) % (2 * np.pi))
        dphi[2] = (phi[1] - phi[2]) + ((phi[0] + np.pi - phi[2]) % (2 * np.pi))
        dphi[3] = phi[0] - phi[3]
        return dphi

    def _stand(self, phi: np.ndarray, target=np.pi * 3 / 2):
        return 2.0 * ((target - phi) % (2 * np.pi))

    def get_osc(self):
        phi_sin = np.sin(self.phi)
        phi_cos = np.cos(self.phi)
        osc = np.concatenate([phi_sin, phi_cos, self.phi_dot], axis=-1)
        return osc


class MJCRobot:
    smoothing: int = 2
    jpos_steps = 3
    jvel_steps = 3
    gravity_steps = 1

    def __init__(
        self, mj_model, mj_data, command_manager: MJCCommandManager, shadow=True
    ):
        self.model = mj_model
        self.data = mj_data
        self.tau = 1.0

        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.joint_observed_ids = [
            self.joint_names.index(joint) for joint in MJC_JOINTS
        ]

        self.actuator_names = [
            self.model.actuator(i).name for i in range(self.model.nu)
        ]
        self.actuator_controlled_ids = [
            self.actuator_names.index(joint) for joint in MJC_JOINTS
        ]

        self.body_names = [self.model.body(i).name for i in range(self.model.nbody)]
        self.arm_base_body_id = self.body_names.index("arm_base")

        self.action_dim = len(self.actuator_controlled_ids)

        self.command_manager = command_manager

        self.decimation = int(0.02 / self.model.opt.timestep)
        print(f"Decimation: {self.decimation}")
        print(f"Action dim: {self.action_dim}")

        self.action_buf_steps = 3
        self.action_buf = np.zeros((self.action_dim, self.action_buf_steps))
        self.applied_action = np.zeros(self.action_dim)
        self.qpos_buf = np.zeros((self.action_dim, self.smoothing))
        self.qvel_buf = np.zeros((self.action_dim, self.smoothing))
        self.angvel_buf = np.zeros((3, self.smoothing))
        self.smth_weight = np.flip(np.arange(1, self.smoothing + 1)).reshape(1, -1)
        self.smth_weight = self.smth_weight / self.smth_weight.sum()
        print(f"Smoothing weight: {self.smth_weight}")

        self.qpos_default = np.zeros(self.action_dim)

        self.jpos_multistep = np.zeros((self.jpos_steps, self.action_dim + 3))
        self.jvel_multistep = np.zeros((self.jvel_steps, self.action_dim + 3))
        self.gravity_multistep = np.zeros((self.gravity_steps, 3))

        self.qpos_default = np.array(ISAAC_QPOS)
        self.kp = np.array(ISAAC_KP)[isaac2mjc]
        self.kd = np.array(ISAAC_KD)[isaac2mjc]
        self.action_scale = np.array(ISAAC_SCALE)

        self.oscillator = Oscillator()
        self.command = np.zeros(18 + 12, dtype=np.float32)

        self.action_filter = SecondOrderLowPassFilter(50, 200)

        mujoco.mj_step(self.model, self.data)

        self.shadow = shadow
        if shadow:
            tcp_addr = "tcp://192.168.1.101:5557"
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.setsockopt(zmq.CONFLATE, 1)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket.connect(tcp_addr)

    def reset(self):
        self.update()
        obs = self.compute_obs()
        return obs

    def update(self):
        qpos = self.data.qpos[-self.model.njnt :][self.joint_observed_ids]
        self.qpos_buf = np.roll(self.qpos_buf, 1, axis=1)
        self.qpos_buf[:, 0] = qpos
        self.qpos = np.mean(self.qpos_buf, axis=1)

        qvel = self.data.qvel[-self.model.njnt :][self.joint_observed_ids]
        self.qvel_buf = np.roll(self.qvel_buf, 1, axis=1)
        self.qvel_buf[:, 0] = qvel
        self.qvel = np.mean(self.qvel_buf, axis=1)

    def update_command(self):
        self.command[:18] = self.command_manager.command
        self.command[-12:] = self.oscillator.get_osc()

    def step(self, action=None):
        if action is not None:
            self.action_buf = np.roll(self.action_buf, 1, axis=1)
            self.action_buf[:, 0] = action
            self.applied_action[:] = lerp(self.applied_action, action, 0.9)
            qpos_des = np.clip(
                self.applied_action * self.action_scale + self.qpos_default, -6, 6
            )
        else:
            qpos_des = self.qpos_default

        if self.shadow:
            arm_pos = self.socket.recv_pyobj()
            self.data.qpos[0:3] = self.data.xpos[self.arm_base_body_id]
            self.data.qpos[3:7] = self.data.xquat[self.arm_base_body_id]
            self.data.qpos[7:12] = arm_pos

        for _ in range(self.decimation):
            self.update()
            # target = self.action_filter.update(qpos_des[isaac2mjc])
            target = qpos_des[isaac2mjc]
            # target[-5:] = 0.
            self.pd_control(target)

            arm_qpos = self.data.qpos[:12].copy()
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[:12] = arm_qpos

        # liveplot.send(target[-5:].tolist() + self.qvel[isaac2mjc][-5:].tolist())
        # liveplot.send(target[-5:].tolist())

        return self.compute_obs()

    def compute_obs(self):
        qpos = np.concatenate([self.qpos[mjc2isaac], np.zeros(3)])
        qvel = np.concatenate([self.qvel[mjc2isaac], np.zeros(3)])

        # read mujoco time stamp
        # time = self.data.time
        # print(time)

        self.command_manager.update()
        self.oscillator.update_phis(self.command_manager.command)
        self.update_command()

        self.jpos_multistep = np.roll(self.jpos_multistep, 1, axis=0)
        self.jpos_multistep[0] = qpos
        self.jvel_multistep = np.roll(self.jvel_multistep, 1, axis=0)
        self.jvel_multistep[0] = qvel

        quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]
        rot = R.from_quat(quat)
        gravity_vec = rot.inv().apply([0, 0, -1.0])
        self.gravity_multistep = np.roll(self.gravity_multistep, 1, axis=0)
        self.gravity_multistep[0] = gravity_vec

        obs = [
            self.command,
            self.gravity_multistep.reshape(-1),
            self.jpos_multistep.reshape(-1),
            self.jvel_multistep.reshape(-1),
            self.action_buf.flatten(),
        ]
        return np.concatenate(obs, dtype=np.float32)

    def set_torque_smoothing(self, tau):
        self.tau = tau

    def pd_control(self, qpos_des):
        qpos_err = qpos_des - self.qpos_buf[:, 0]
        self.tau_des = qpos_err * self.kp - self.qvel_buf[:, 0] * self.kd
        self.tau_ctrl = lerp(self.data.ctrl, self.tau_des, self.tau)
        self.data.ctrl[self.actuator_controlled_ids] = np.clip(self.tau_ctrl, -200, 200)


def lerp(a, b, t):
    return a + (b - a) * t


FILE_PATH = os.path.dirname(os.path.abspath(__file__))


@set_exploration_type(ExplorationType.MODE)
@torch.inference_mode()
def main():
    xml_path = os.path.join(FILE_PATH, "aliengo/urdf/aliengo_a1_mj.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    body_names = [mj_model.body(i).name for i in range(mj_model.nbody)]
    # command_manager = MJCCommandManager(mj_model, mj_data)
    command_manager = ZMQCommand(mj_model, mj_data)
    robot = MJCRobot(mj_model, mj_data, command_manager, shadow=True)

    obs = robot.reset()

    # policy_path = "/home/elijah/Documents/projects/active-adaptation/scripts/exports/AlienMForce/policy-01-15_21-44-alienmforce-2040.pt"
    # policy_path = '/home/elijah/Documents/projects/active-adaptation/scripts/exports/AlienMForce/policy-01-14_23-26-alienmforce-2031-multistep-3.pt'
    # policy_path = '/home/elijah/Documents/projects/active-adaptation/scripts/exports/AlienMForce/policy-01-15_15-56-alienmforce-2034.pt'
    # policy_path = '/home/elijah/Documents/projects/active-adaptation/scripts/exports/AlienMForce/policy-01-16_20-06-alienmforce-2050-small_osc.pt'
    # policy_path = '/home/elijah/Documents/projects/active-adaptation/scripts/exports/AlienMForce/policy-01-18_10-48.pt'
    # policy_path = "/home/elijah/Documents/projects/active-adaptation/scripts/exports/AlienMForce/policy-01-18_10-56-alienmforce-2081.pt"
    policy_path = "/home/elijah/Downloads/policy-01-18_19-50.pt"
    # policy_path = os.path.join(FILE_PATH, policy_path)
    policy = torch.load(policy_path)

    command, obs = (
        obs[: command_manager.command_dim],
        obs[command_manager.command_dim :],
    )

    tensordict = TensorDict(
        {
            "command_": torch.as_tensor(command),
            "policy": torch.as_tensor(obs),
            "is_init": torch.tensor(1, dtype=bool),
            "adapt_hx": torch.zeros(128),
        },
        [],
    ).unsqueeze(0)

    base_body_id = mj_model.body("base").id

    key_queue = queue.Queue()

    def key_callback(keycode):
        key_queue.put(keycode)

    def create_setpoint_geoms(viewer, base_pos, base_quat):
        # Clear previous geoms
        viewer.user_scn.ngeom = 0

        base_quat = yaw_quat(base_quat)
        base_setpoint_pos_b = np.concatenate(
            [command_manager.setpoint_base_pos_diff_b, np.array([0.2])]
        )
        base_setpoint_pos = base_pos + R.from_quat(base_quat[[1, 2, 3, 0]]).apply(
            base_setpoint_pos_b
        )
        ee_setpoint_pos = base_pos + R.from_quat(base_quat[[1, 2, 3, 0]]).apply(
            command_manager.setpoint_pos_ee_b
        )

        # Base setpoint geom
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.05, 0.05, 0.05]),
            pos=np.array(base_setpoint_pos),
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.0, 0.0, 0.5]),
        )
        viewer.user_scn.ngeom += 1

        # EE setpoint geom
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.05, 0.05, 0.05]),
            pos=ee_setpoint_pos,
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 0.0, 1.0, 0.5]),
        )
        viewer.user_scn.ngeom += 1

    try:
        with mujoco.viewer.launch_passive(
            mj_model,
            mj_data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=key_callback,
        ) as viewer:
            i = 0
            viewer.cam.distance = 2.0  # Set the camera distance from the base
            viewer.cam.elevation = -20  # Adjust the camera elevation (tilt)
            viewer.cam.azimuth = (
                45  # Adjust the camera azimuth (rotation around the base)
            )
            while viewer.is_running():
                start = time.perf_counter()

                # Handle keyboard input
                while not key_queue.empty():
                    keycode = key_queue.get()
                    step_size = 2.0
                    if (
                        keycode == mujoco.viewer.glfw.KEY_UP
                    ):  # Move base setpoint forward
                        command_manager.setpoint_base_pos_diff_b_target[0] += step_size
                    elif (
                        keycode == mujoco.viewer.glfw.KEY_DOWN
                    ):  # Move base setpoint backward
                        command_manager.setpoint_base_pos_diff_b_target[0] -= step_size
                    elif (
                        keycode == mujoco.viewer.glfw.KEY_LEFT
                    ):  # Move base setpoint left
                        command_manager.setpoint_base_pos_diff_b_target[1] += step_size
                    elif (
                        keycode == mujoco.viewer.glfw.KEY_RIGHT
                    ):  # Move base setpoint right
                        command_manager.setpoint_base_pos_diff_b_target[1] -= step_size
                    elif (
                        keycode == mujoco.viewer.glfw.KEY_W
                    ):  # Move EE setpoint forward
                        command_manager.setpoint_pos_ee_b_target[0] += 0.1
                    elif (
                        keycode == mujoco.viewer.glfw.KEY_S
                    ):  # Move EE setpoint backward
                        command_manager.setpoint_pos_ee_b_target[0] -= 0.1
                    elif keycode == mujoco.viewer.glfw.KEY_A:  # Move EE setpoint left
                        command_manager.setpoint_pos_ee_b_target[1] += 0.1
                    elif keycode == mujoco.viewer.glfw.KEY_D:  # Move EE setpoint right
                        command_manager.setpoint_pos_ee_b_target[1] -= 0.1
                    elif keycode == mujoco.viewer.glfw.KEY_E:  # Move EE setpoint up
                        command_manager.setpoint_pos_ee_b_target[2] += 0.1
                    elif keycode == mujoco.viewer.glfw.KEY_Q:  # Move EE setpoint down
                        command_manager.setpoint_pos_ee_b_target[2] -= 0.1

                command_manager.setpoint_base_pos_diff_b[:] += (
                    command_manager.setpoint_base_pos_diff_b_target
                    - command_manager.setpoint_base_pos_diff_b
                ) * 0.04
                command_manager.setpoint_pos_ee_b[:] += (
                    command_manager.setpoint_pos_ee_b_target
                    - command_manager.setpoint_pos_ee_b
                ) * 0.04

                policy(tensordict)
                action = tensordict["action"].squeeze().numpy()
                # recorded_actions.append(action)

                obs = torch.as_tensor(robot.step(action))
                command, obs = obs.split(
                    [
                        command_manager.command_dim,
                        obs.shape[0] - command_manager.command_dim,
                    ]
                )

                tensordict["next", "command_"] = command.unsqueeze(0)
                tensordict["next", "policy"] = obs.unsqueeze(0)
                tensordict["next", "is_init"] = torch.tensor(0, dtype=bool).unsqueeze(0)

                tensordict = tensordict["next"]

                # Update setpoint geoms
                base_pos = mj_data.xpos[base_body_id]
                base_quat = mj_data.xquat[base_body_id]
                create_setpoint_geoms(viewer, base_pos, base_quat)

                viewer.cam.lookat = mj_data.xpos[base_body_id]
                # Focus the camera on the base
                viewer.sync()

                time.sleep(max(0, 0.02 - (time.perf_counter() - start)))
                if i % 20 == 0:
                    print("setpoint diff:", command[:6])
                    # print("qpos:", robot.qpos)
                    # print("qvel:", robot.qvel)
                    # print("qvel:", robot.qvel_buf.mean(axis=1))
                    # print("tau:", robot.tau_ctrl)
                i += 1
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
