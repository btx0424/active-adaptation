from .ppo import *
from .bc import BCPolicy
from .sac import SAC
# from .hbc import HBC

ALGOS = {
    "sac": SAC,
    # "hbc": HBC,
    "bc": BCPolicy
}