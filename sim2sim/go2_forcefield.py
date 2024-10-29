import torch
import numpy as np
import mujoco
import mujoco.viewer
import mujoco_viewer
import time
import os
import itertools
import imageio
import queue
import glfw

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
ISAAC_ACT_SCALE = [
    0.25, 0.25, 0.25, 0.25,
    0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5,
]
# fmt: on

def quat_to_yaw(quat: np.ndarray):
    rot = R.from_quat(quat[[1, 2, 3, 0]])
    yaw = rot.as_euler("xyz")[2]
    return yaw


class MJCRobot:
    pass

class CommandManager:

    command_dim: int

    def __init__(self) -> None:
        self.command = np.zeros(self.command_dim, dtype=np.float32)

    def update(self, robot: MJCRobot):
        pass


class FixedCommandForce(CommandManager):
    command_dim = 10

    setpoint_pos_b = np.array([0.5, 0.0, 0.0])
    yaw_diff = 0.0

    kp = 10.0
    kd = 6.0
    virtual_mass = 10.0

    def update(self, robot: MJCRobot):
        self.command[:2] = self.setpoint_pos_b[:2]
        self.command[2] = self.yaw_diff
        self.command[3:5] = self.kp * self.setpoint_pos_b[:2]
        self.command[5:8] = self.kd
        self.command[8] = self.kp * self.yaw_diff
        self.command[9] = self.virtual_mass


class CommandForceKeyboard(FixedCommandForce):
    force = np.zeros(3)

    def __init__(self, key_queue):
        super().__init__()
        self.key_queue = key_queue

        self.movement_speed = 0.1
        self.rotation_speed = 0.1
        self.force_speed = 1.0
        
        self.target_yaw = 0.0

    def update(self, robot: MJCRobot):
        self._process_keyboard_input(robot)
        super().update(robot)

    def _process_keyboard_input(self, robot):
        while not self.key_queue.empty():
            key = self.key_queue.get()
            if key == glfw.KEY_W:
                self.setpoint_pos_b[0] += self.movement_speed
            elif key == glfw.KEY_S:
                self.setpoint_pos_b[0] -= self.movement_speed
            elif key == glfw.KEY_A:
                self.setpoint_pos_b[1] += self.movement_speed
            elif key == glfw.KEY_D:
                self.setpoint_pos_b[1] -= self.movement_speed
            elif key == glfw.KEY_Q:
                self.target_yaw += self.rotation_speed
            elif key == glfw.KEY_E:
                self.target_yaw -= self.rotation_speed
            elif key == glfw.KEY_UP:
                self.force[0] += self.force_speed
            elif key == glfw.KEY_DOWN:
                self.force[0] -= self.force_speed
            elif key == glfw.KEY_LEFT:
                self.force[1] += self.force_speed
            elif key == glfw.KEY_RIGHT:
                self.force[1] -= self.force_speed
            
        robot_quat = robot.data.qpos[robot.base_qpos_start_idx + 3 : robot.base_qpos_start_idx + 7]
        robot_yaw = quat_to_yaw(robot_quat)
        self.yaw_diff = self.target_yaw - robot_yaw


class MJCRobot:

    smoothing: int = 2

    def __init__(self, xml_path, command_manager: CommandManager):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.init_q = self.model.keyframe("home").qpos

        # sometimes we only control/observe a subset of the joints
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

        self.base_body = self.model.body("base_link")
        self.base_body_id = self.model.body("base_link").id
        self.base_qpos_start_idx = self.model.body_dofadr[self.base_body_id]

        self.action_dim = len(self.actuator_controlled_ids)

        self.command_manager = command_manager

        self.decimation = int(0.02 / self.model.opt.timestep)
        print(f"Decimation: {self.decimation}")
        print(f"Action dim: {self.action_dim}")

        # allocate buffers
        self.action_buf_steps = 3
        self.action_buf = np.zeros((self.action_dim, self.action_buf_steps))
        self.applied_action = np.zeros(self.action_dim)

        self.qvel_buf = np.zeros((self.action_dim, self.smoothing))
        self.angvel_buf = np.zeros((3, self.smoothing))
        self.smth_weight = np.flip(np.arange(1, self.smoothing + 1)).reshape(1, -1)
        self.smth_weight = self.smth_weight / self.smth_weight.sum()
        print(f"Smoothing weight: {self.smth_weight}")

        self.action_scale_isaac = np.array(ISAAC_ACT_SCALE)
        self.qpos_default_isaac = np.array(ISAAC_QPOS)
        self.kp_mjc = np.array(ISAAC_KP)[isaac2mjc]
        self.kd_mjc = np.array(ISAAC_KD)[isaac2mjc]

        self.spring_kp = 10.0  # Spring constant
        self.spring_target = np.array([0.0, 0.0, 0.0])  # Target position (origin)

    def reset(self):
        self.data.qpos[:] = self.init_q
        self.data.qvel[:] = 0.0
        mujoco.mj_step(self.model, self.data)

    def update(self):
        self.qpos = self.data.qpos[-self.model.njnt :][self.joint_observed_ids]
        qvel = self.data.qvel[-self.model.njnt :][self.joint_observed_ids]
        self.qvel_buf = np.roll(self.qvel_buf, 1, axis=1)
        self.qvel_buf[:, 0] = qvel
        self.qvel = np.mean(self.qvel_buf, axis=1)

        self.command_manager.update(self)

    def compute_obs(self):
        self.update()

        quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]
        rot = R.from_quat(quat)
        gravity_vec = rot.inv().apply([0, 0, -1.0])

        obs = [
            gravity_vec,
            self.qpos[mjc2isaac],
            self.qvel[mjc2isaac],
            self.action_buf.flatten(),
        ]
        return np.concatenate(obs, dtype=np.float32)

    def step(self, action=None):
        if action is not None:
            self.action_buf = np.roll(self.action_buf, 1, axis=1)
            self.action_buf[:, 0] = action
            self.applied_action[:] = lerp(self.applied_action, action, 0.8)
            qpos_des = np.clip(
                action * self.action_scale_isaac + self.qpos_default_isaac, -6, 6
            )
        else:
            qpos_des = self.qpos_default_isaac

        for _ in range(self.decimation):
            # Apply spring force before each simulation step
            base_pos = self.data.qpos[
                self.base_qpos_start_idx : 3
            ]
            spring_force = self.spring_kp * (self.spring_target - base_pos)
            spring_force[2] = 0

            # force = np.array([-10, 0, 0], dtype=np.float32)
            force = spring_force
            # force = self.command_manager.force
            torque = np.zeros(3)
            point = np.zeros(3)

            # self.data.xfrc_applied[self.base_body_id, :3] = force

            self.data.qfrc_applied[:] = 0
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force,
                torque,
                point,
                self.base_body_id,
                self.data.qfrc_applied,
            )

            mujoco.mj_step(self.model, self.data)
            self.update()
            self.pd_control(qpos_des[isaac2mjc])

    def pd_control(self, qpos_des):
        qpos_err = qpos_des - self.qpos
        self.tau_des = qpos_err * self.kp_mjc - self.qvel_buf.mean(1) * self.kd_mjc
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
    key_queue = queue.Queue()
    # command_manager = FixedCommandForce()
    command_manager = CommandForceKeyboard(key_queue)
    robot = MJCRobot(xml_path, command_manager)

    # policy_path = "policy-go2force-639.pt"
    policy_path = "policy-go2force-861.pt"
    policy_path = os.path.join(FILE_PATH, policy_path)
    policy = torch.load(policy_path)
    policy.module[0].set_missing_tolerance(True)

    imgs = []
    recorded_actions = []

    recorded_applied_forces = []
    recorded_commanded_forces = []
    recorded_predicted_forces = []

    try:
        with mujoco.viewer.launch_passive(
            model=robot.model,
            data=robot.data,
            key_callback=lambda key: key_queue.put(key),
        ) as viewer:
            robot.reset()
            tensordict = TensorDict(
                {
                    "is_init": torch.tensor(1, dtype=bool),
                    "adapt_hx": torch.zeros(128),
                },
                [],
            ).unsqueeze(0)

            i = 0
            while viewer.is_running():
                start = time.perf_counter()

                command = torch.as_tensor(command_manager.command)
                obs = torch.as_tensor(robot.compute_obs())
                tensordict["command"] = command.unsqueeze(0)
                tensordict["policy"] = obs.unsqueeze(0)
                policy(tensordict)
                action = tensordict["action"].squeeze().numpy()
                robot.step(action)
                recorded_actions.append(action)
                
                applied_force = robot.data.qfrc_applied[:3].copy()
                commanded_force = command_manager.setpoint_pos_b * command_manager.kp * command_manager.virtual_mass
                predicted_force = tensordict["ext_rec"][0, :3].cpu().numpy()
                recorded_applied_forces.append(applied_force)
                recorded_commanded_forces.append(commanded_force)
                recorded_predicted_forces.append(predicted_force)

                if i % 20 == 0:
                    print("applied force:", applied_force)
                    print("commanded force:", commanded_force)
                    print("predicted force:", predicted_force)
                    # print("qpos:", robot.qpos)
                    # print("qvel:", robot.qvel)
                    # print("qvel:", robot.qvel_buf.mean(axis=1))
                    # print("tau:", robot.tau_ctrl)

                i += 1
                tensordict["next", "is_init"] = torch.tensor(0, dtype=bool).unsqueeze(0)
                tensordict = tensordict["next"]

                viewer.cam.type = 1
                viewer.cam.trackbodyid = robot.base_body_id
                viewer.sync()
                time.sleep(max(0, 0.02 - (time.perf_counter() - start)))

    except KeyboardInterrupt:
        pass

    if len(imgs):
        print("Saving gif of length", len(imgs))
        imageio.mimsave("go2.mp4", imgs, fps=30)

    if recorded_actions:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        recorded_actions = np.array(
            recorded_actions
        )  # Convert to numpy array for easier handling
        time_steps = np.arange(len(recorded_actions))

        fig, axs = plt.subplots(
            12, 1, figsize=(10, 20)
        )  # Create 12 subplots, one for each action dimension

        for i in range(12):
            axs[i].plot(time_steps, recorded_actions[:, i])
            axs[i].set_title(f"Action {i+1}")
            axs[i].set_xlabel("Time Step")
            axs[i].set_ylabel("Action Value")

        plt.tight_layout()
        plt.show()

        # plot applied forces
        recorded_applied_forces = np.array(recorded_applied_forces)
        recorded_commanded_forces = np.array(recorded_commanded_forces)
        recorded_predicted_forces = np.array(recorded_predicted_forces)
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        for i in range(3):
            axs[i].plot(time_steps, recorded_applied_forces[:, i], label="applied")
            axs[i].plot(time_steps, -recorded_commanded_forces[:, i], label="commanded")
            axs[i].plot(time_steps, recorded_predicted_forces[:, i], label="predicted")
            axs[i].set_title(f"Force {i+1}")
            axs[i].set_xlabel("Time Step")
            axs[i].set_ylabel("Force (N)")
            axs[i].legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
