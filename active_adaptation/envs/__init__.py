from .quadruped import Quadruped
from .manipulation import QuadrupedManip, ManipulationEnv
from .humanoid import Humanoid

TASKS = {
    "Quadruped": Quadruped,
    "QuadrupedManip": QuadrupedManip,
    "Manipulation": ManipulationEnv,
    "Humanoid": Humanoid,
}
