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
    mjcf_path=os.path.join(PATH, "go2", "go2_description.xml"),
    **json.load(open(os.path.join(PATH, "go2", "go2.json")))
)
