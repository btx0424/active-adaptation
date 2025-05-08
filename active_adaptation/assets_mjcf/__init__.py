import os
import json
from active_adaptation.envs.mujoco import MJArticulationCfg

ROBOTS = {}

PATH = os.path.dirname(__file__)

ROBOTS["sirius_wheel"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "sirius_mid_wheel", "sirius_mid_wheel.xml"),
    **json.load(open(os.path.join(PATH, "sirius_mid_wheel", "sirius_mid_wheel.json")))
)

ROBOTS["g1_23dof"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "g1_23dof", "g1_23dof.xml"),
    **json.load(open(os.path.join(PATH, "g1_23dof", "g1_23dof.json")))
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
