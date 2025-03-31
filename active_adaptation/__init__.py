import os
from active_adaptation.learning import ALGOS

_BACKEND = "isaac"

ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets")

def set_backend(backend: str):
    if not backend in ("isaac", "mujoco"):
        raise ValueError(f"backend must be either 'isaac' or 'mujoco', got {backend}")
    global _BACKEND
    _BACKEND = backend


def get_backend():
    return _BACKEND

