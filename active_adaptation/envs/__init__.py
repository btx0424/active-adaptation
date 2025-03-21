from .base import Env
from .locomotion import LocomotionEnv
from .manipulation import QuadrupedManip, ManipulationEnv
from .humanoid import Humanoid

TASKS = {
    "QuadrupedManip": QuadrupedManip,
    "Manipulation": ManipulationEnv,
    "Humanoid": Humanoid,
}
