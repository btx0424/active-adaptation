import os
import json
from active_adaptation.envs.mujoco import MJArticulationCfg
import active_adaptation.utils.symmetry as symmetry_utils

ROBOTS = {}

PATH = os.path.dirname(__file__)

ROBOTS["sirius_wheel"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "sirius_wheel", "sirius_wheel.xml"),
    **json.load(open(os.path.join(PATH, "sirius_wheel", "sirius_wheel.json"))),
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "LF_HAA": (-1, "RF_HAA"),
        "LH_HAA": (-1, "RH_HAA"),
        "LF_HFE": (1, "RF_HFE"),
        "LH_HFE": (1, "RH_HFE"),
        "LF_KFE": (1, "RF_KFE"),
        "LH_KFE": (1, "RH_KFE"),
        "LF_WHEEL": (1, "RF_WHEEL"),
        "LH_WHEEL": (1, "RH_WHEEL"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "trunk": "trunk",
        "front": "front",
        "back": "back",
        "LF_hip": "RF_hip",
        "LH_hip": "RH_hip",
        "LF_calf": "RF_calf",
        "LH_calf": "RH_calf",
        "LF_thigh": "RF_thigh",
        "LH_thigh": "RH_thigh",
        "LF_FOOT": "RF_FOOT",
        "LH_FOOT": "RH_FOOT",
    })
)

ROBOTS["g1_23dof"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "g1_23dof", "g1_23dof.xml"),
    **json.load(open(os.path.join(PATH, "g1_23dof", "g1_23dof.json")))
)

ROBOTS["g1_27dof"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "g1_23dof", "g1_27dof.xml"),
    **json.load(open(os.path.join(PATH, "g1_23dof", "g1_27dof.json"))),
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
        "left_wrist_roll_joint": (1, "right_wrist_roll_joint"),
        "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
        "left_wrist_yaw_joint": (1, "right_wrist_yaw_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "pelvis": "pelvis",
        "torso_link": "torso_link",
        "torso_com_link": "torso_com_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_roll_link": "waist_roll_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
        "left_wrist_roll_link": "right_wrist_roll_link",
        "left_wrist_pitch_link": "right_wrist_pitch_link",
        "left_wrist_yaw_link": "right_wrist_yaw_link",
        "left_rubber_hand": "right_rubber_hand",
    })
)

ROBOTS["gr1"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "GR1T2", "GR1T2_nohand.xml"),
    **json.load(open(os.path.join(PATH, "GR1T2", "GR1T2_nohand.json"))),
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
        "left_knee_pitch_joint": (1, "right_knee_pitch_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "waist_roll_joint": (-1, "waist_roll_joint"),
        "waist_pitch_joint": (1, "waist_pitch_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
        "left_elbow_pitch_joint": (1, "right_elbow_pitch_joint"),
        "left_wrist_yaw_joint": (1, "right_wrist_yaw_joint"),
        "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
        "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
        "head_yaw_joint": (-1,  "head_yaw_joint"),
        "head_roll_joint": (-1, "head_roll_joint"),
        "head_pitch_joint": (1, "head_pitch_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "base_link": "base_link",
        "left_thigh_roll_link": "right_thigh_roll_link",
        "left_thigh_yaw_link": "right_thigh_yaw_link",
        "left_thigh_pitch_link": "right_thigh_pitch_link",
        "left_shank_pitch_link": "right_shank_pitch_link",
        "left_foot_pitch_link": "right_foot_pitch_link",
        "left_foot_roll_link": "right_foot_roll_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_pitch_link": "waist_pitch_link",
        "waist_roll_link": "waist_roll_link",
        "head_roll_link": "head_roll_link",
        "head_yaw_link": "head_yaw_link",
        "head_pitch_link": "head_pitch_link",
        "left_upper_arm_pitch_link": "right_upper_arm_pitch_link",
        "left_upper_arm_roll_link": "right_upper_arm_roll_link",
        "left_upper_arm_yaw_link": "right_upper_arm_yaw_link",
        "left_lower_arm_pitch_link": "right_lower_arm_pitch_link",
        "left_hand_yaw_link": "right_hand_yaw_link",
        "left_hand_roll_link": "right_hand_roll_link",
        "left_hand_pitch_link": "right_hand_pitch_link",
    })
)

ROBOTS["g1_waist_unlocked"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "g1_23dof", "g1_29dof_rev_1_0.xml"),
    **json.load(open(os.path.join(PATH, "g1_23dof", "g1_29dof_rev_1_0.json"))),
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "waist_roll_joint": (-1, "waist_roll_joint"),
        "waist_pitch_joint": (1, "waist_pitch_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "pelvis": "pelvis",
        "torso_link": "torso_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_roll_link": "waist_roll_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
    })
)

ROBOTS["g1_29dof"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "g1_23dof", "g1_29dof_nohand-feet_sphere.xml"),
    **json.load(open(os.path.join(PATH, "g1_23dof", "g1_29dof_nohand.json"))),
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "waist_roll_joint": (-1, "waist_roll_joint"),
        "waist_pitch_joint": (1, "waist_pitch_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
        "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
        "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
        "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "pelvis": "pelvis",
        "torso_link": "torso_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_roll_link": "waist_roll_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
        "left_wrist_yaw_link": "right_wrist_yaw_link",
        "left_wrist_roll_link": "right_wrist_roll_link",
        "left_wrist_pitch_link": "right_wrist_pitch_link",
        "pelvis_contour_link": "pelvis_contour_link",
        "imu_link": "imu_link",
        "d435_link": "d435_link",
        "head_link": "head_link",
        "logo_link": "logo_link",
        "mid360_link": "mid360_link",
        "waist_support_link": "waist_support_link",
        "left_hand_marker": "right_hand_marker",
    })
)

ROBOTS["h2"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "h1_2", "h1_2_handless.xml"),
    **json.load(open(os.path.join(PATH, "h1_2", "h1_2_handless.json")))
)

ROBOTS["go2"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "go2", "go2.xml"),
    **json.load(open(os.path.join(PATH, "go2", "go2.json"))),
    joint_symmetry_mapping = { 
        "FL_hip_joint": [-1, "FR_hip_joint"],
        "FR_hip_joint": [-1, "FL_hip_joint"],
        "RL_hip_joint": [-1, "RR_hip_joint"],
        "RR_hip_joint": [-1, "RL_hip_joint"],
        "FL_thigh_joint": [1, "FR_thigh_joint"],
        "FR_thigh_joint": [1, "FL_thigh_joint"],
        "RL_thigh_joint": [1, "RR_thigh_joint"],
        "RR_thigh_joint": [1, "RL_thigh_joint"],
        "FL_calf_joint": [1, "FR_calf_joint"],
        "FR_calf_joint": [1, "FL_calf_joint"],
        "RL_calf_joint": [1, "RR_calf_joint"],
        "RR_calf_joint": [1, "RL_calf_joint"]
    },
    spatial_symmetry_mapping = {
        "FL_hip": "FR_hip",
        "FR_hip": "FL_hip",
        "RL_hip": "RR_hip",
        "RR_hip": "RL_hip",
        "FL_thigh": "FR_thigh",
        "FR_thigh": "FL_thigh",
        "RL_thigh": "RR_thigh",
        "RR_thigh": "RL_thigh",
        "FL_calf": "FR_calf",
        "FR_calf": "FL_calf",
        "RL_calf": "RR_calf",
        "RR_calf": "RL_calf",
        "FL_foot": "FR_foot",
        "FR_foot": "FL_foot",
        "RL_foot": "RR_foot",
        "RR_foot": "RL_foot",
        "base": "base",
        "Head_upper": "Head_upper",
        "Head_lower": "Head_lower",
    }
)
