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
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
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
]

isaac2mjc = [ISAAC_JOINTS.index(joint) for joint in MJC_JOINTS]
mjc2isaac = [MJC_JOINTS.index(joint) for joint in ISAAC_JOINTS]

# fmt: off
ISAAC_KP = [
    60., 60., 60., 60., 
    60., 60., 60., 60., 
    60., 60., 60., 60.,
]
ISAAC_KD = [
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
]
ISAAC_QPOS = [
    0.2, -0.2, 0.2, -0.2,
    0.8, 0.8, 0.8, 0.8,
    -1.5, -1.5, -1.5, -1.5,

    # 0.2, -0.2, 0.2, -0.2,
    # 1.0, 1.0, 1.0, 1.0,
    # -1.8, -1.8, -1.8, -1.8,
]

ISAAC_SCALE = [
    0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5,
]
# fmt: on

DEBUG = False


def yaw_quat(quat):
    quat = np.array(quat)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    yaw_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    return yaw_quat


class MJCCommandManager:
    command_dim: int = 10

    setpoint_pos_b = np.array([1.0, 0.0, 0.0])
    yaw_diff = 0.0

    kp = 10.0
    kd = 3.0
    virtual_mass = 10.0

    def __init__(self):
        self.command = np.zeros(self.command_dim)
        self.setpoint_pos_b_target = np.array([1.0, 0.0, 0.0])

    def update(self):
        self.command[:2] = self.setpoint_pos_b[:2]
        self.command[2] = self.yaw_diff
        self.command[3:5] = self.kp * self.setpoint_pos_b[:2]
        self.command[5:8] = self.kd
        self.command[8] = self.kp * self.yaw_diff
        self.command[9] = self.virtual_mass


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
    gravity_steps = 3
    jpos_steps = 3
    jvel_steps = 3
    gravity_steps = 3

    def __init__(self, mj_model, mj_data, command_manager: MJCCommandManager):
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

        self.action_dim = len(self.actuator_controlled_ids)

        self.command_manager = command_manager

        self.decimation = int(0.02 / self.model.opt.timestep)
        print(f"Decimation: {self.decimation}")
        print(f"Action dim: {self.action_dim}")

        # allocate buffers
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

        self.jpos_multistep = np.zeros((self.jpos_steps, self.action_dim))
        self.jvel_multistep = np.zeros((self.jvel_steps, self.action_dim))
        self.gravity_multistep = np.zeros((self.gravity_steps, 3))

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

        self.oscillator = Oscillator()
        self.command = np.zeros(22, dtype=np.float32)

        mujoco.mj_step(self.model, self.data)

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
        
        self.command_manager.update()
        self.oscillator.update_phis(self.command_manager.command)
        self.update_command()
        
        # angvel = self.data.sensor("angular-velocity").data
        # self.angvel_buf = np.roll(self.angvel_buf, 1, axis=1)
        # self.angvel_buf[:, 0] = angvel
        # self.angvel = np.mean(self.angvel_buf, axis=1)

        # 更新多步 jpos, jvel
        self.jpos_multistep = np.roll(self.jpos_multistep, 1, axis=0)
        self.jpos_multistep[0] = self.qpos
        self.jvel_multistep = np.roll(self.jvel_multistep, 1, axis=0)
        self.jvel_multistep[0] = self.qvel

        # 更新多步 gravity
        quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]
        rot = R.from_quat(quat)
        gravity_vec = rot.inv().apply([0, 0, -1.0])
        self.gravity_multistep = np.roll(self.gravity_multistep, 1, axis=0)
        self.gravity_multistep[0] = gravity_vec

    def update_command(self):
        # 真实部署里前10维是基础力指令，后12维为振荡器数据
        self.command[:10] = self.command_manager.command
        self.command[10:22] = self.oscillator.get_osc()

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

        for _ in range(self.decimation):
            self.update()
            # target = self.action_filter.update(qpos_des[isaac2mjc])
            target = qpos_des[isaac2mjc]
            # target[-5:] = 0.
            self.pd_control(target)
            mujoco.mj_step(self.model, self.data)
        # liveplot.send(target[-5:].tolist() + self.qvel[isaac2mjc][-5:].tolist())
        # liveplot.send(target[-5:].tolist())

        return self.compute_obs()

    def compute_obs(self):
        qpos = self.qpos[mjc2isaac]
        qvel = self.qvel[mjc2isaac]

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

        # 用多步数据替代单步 gravity_vec、qpos、qvel
        obs = [
            self.command,
            self.gravity_multistep.reshape(-1),
            self.jpos_multistep.reshape(-1),
            self.jvel_multistep.reshape(-1),
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

    xml_path = os.path.join(FILE_PATH, "aliengo/urdf/aliengo_mj.xml")
    # xml_path = os.path.join(FILE_PATH, "aliengo/urdf/aliengo_mj_damped.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    command_manager = MJCCommandManager()
    robot = MJCRobot(mj_model, mj_data, command_manager)

    obs = robot.reset()

    policy_path = "/home/elijah/Downloads/policy-01-18_19-39.pt"
    # policy_path = "policy-alienforce-626.pt"
    policy_path = os.path.join(FILE_PATH, policy_path)
    policy = torch.load(policy_path)
    # policy.module[0].set_missing_tolerance(True)

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
        base_setpoint_pos = base_pos + R.from_quat(base_quat[[1, 2, 3, 0]]).apply(
            command_manager.setpoint_pos_b
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
                    step_size = 0.5
                    if (
                        keycode == mujoco.viewer.glfw.KEY_UP
                    ):  # Move base setpoint forward
                        command_manager.setpoint_pos_b_target[0] += step_size
                    elif (
                        keycode == mujoco.viewer.glfw.KEY_DOWN
                    ):  # Move base setpoint backward
                        command_manager.setpoint_pos_b_target[0] -= step_size
                    elif (
                        keycode == mujoco.viewer.glfw.KEY_LEFT
                    ):  # Move base setpoint left
                        command_manager.setpoint_pos_b_target[1] += step_size
                    elif (
                        keycode == mujoco.viewer.glfw.KEY_RIGHT
                    ):  # Move base setpoint right
                        command_manager.setpoint_pos_b_target[1] -= step_size

                command_manager.setpoint_pos_b[:] += (
                    command_manager.setpoint_pos_b_target
                    - command_manager.setpoint_pos_b
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
                    print("qpos:", robot.qpos)
                    print("qvel:", robot.qvel)
                    print("qvel:", robot.qvel_buf.mean(axis=1))
                    print("tau:", robot.tau_ctrl)
                i += 1
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
