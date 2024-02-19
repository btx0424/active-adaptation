from .quadruped import Quadruped
from .biped import Biped
from .manipulation import QuadrupedManip

TASKS = {
    "Quadruped": Quadruped,
    "Biped": Biped,
    "QuadrupedManip": QuadrupedManip,
}
