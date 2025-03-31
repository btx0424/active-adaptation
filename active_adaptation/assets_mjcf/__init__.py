import os
import json
from active_adaptation.envs.mujoco import MJArticulationCfg

ROBOTS = {}

PATH = os.path.dirname(__file__)

ROBOTS["sirius"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "sirius_mid_wheel", "sirius_mid_wheel.xml"),
    **json.load(open(os.path.join(PATH, "sirius_mid_wheel", "sirius_mid_wheel.json")))
)