from .velocity_v1 import LocomotionV1
from .velocity_v2 import LocomotionV2
from .velocity_v3 import LocomotionV3
from .recover import Recover

TASKS = {
    "Velocity": LocomotionV3,
    "Recover": Recover,
}
