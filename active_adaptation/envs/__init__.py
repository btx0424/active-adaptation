from .quadruped import Quadruped
from .biped import Biped
from .manipulation import QuadrupedManip
from .humanoid import Humanoid
from .dribble import Dribble

TASKS = {
    "Quadruped": Quadruped,
    "Biped": Biped,
    "QuadrupedManip": QuadrupedManip,
    "Humanoid": Humanoid,
    "Dribble": Dribble
}
