from .randomizations import *
from .observations import *
from .rewards import *
from .terminations import *

def get_obj_by_class(mapping, obj_class):
    return {
        k: v for k, v in mapping.items() 
        if isinstance(v, type) and issubclass(v, obj_class)
    }

OBS_FUNCS = get_obj_by_class(vars(observations), observations.Observation)
REW_FUNCS = get_obj_by_class(vars(rewards), rewards.Reward)
TERM_FUNCS = get_obj_by_class(vars(terminations), terminations.Termination)
RAND_FUNCS = get_obj_by_class(vars(randomizations), randomizations.Randomization)
