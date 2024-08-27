from .quadruped import Quadruped
from .biped import Biped
from .manipulation import QuadrupedManip, ManipulationEnv
from .humanoid import Humanoid
from .dribble import Dribble

TASKS = {
    "Quadruped": Quadruped,
    "Biped": Biped,
    "QuadrupedManip": QuadrupedManip,
    "Manipulation": ManipulationEnv,
    "Humanoid": Humanoid,
    "Dribble": Dribble
}
