from .quadruped import Quadruped
from .biped import Biped
# from .manipulation import QuadrupedManip
from .humanoid import Humanoid

TASKS = {
    "Quadruped": Quadruped,
    "Biped": Biped,
    # "QuadrupedManip": QuadrupedManip,
    "Humanoid": Humanoid
}
