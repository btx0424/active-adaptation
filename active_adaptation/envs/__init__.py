from .velocity_v1 import LocomotionV1
from .velocity_v2 import LocomotionV2
from .quadruped import Quadruped
from .biped import Biped
from .recover import Recover

TASKS = {
    "Quadruped": Quadruped,
    "Biped": Biped,
    "Recover": Recover,
}
