import torch
import inspect
import abc
from typing import Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from active_adaptation.envs.base import _Env


class Observation:
    """
    Base class for all observations.
    """
    registry = {}

    def __init__(self, env):
        self.env: _Env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self) ->  Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.compute()
        return tensor
    
    def startup(self):
        """Called once upon initialization of the environment"""
        pass
    
    def post_step(self, substep: int):
        """Called after each physics substep"""
        pass

    def update(self):
        """Called after all physics substeps are completed"""
        pass

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination"""

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled"""
        pass

    def __init_subclass__(cls) -> None:
        """Put the subclass in the global registry"""
        cls_name = cls.__name__
        cls._file = inspect.getfile(cls)
        cls._line = inspect.getsourcelines(cls)[1]
        if cls_name not in Observation.registry:
            Observation.registry[cls_name] = cls    
        else:
            conflicting_cls = Observation.registry[cls_name]
            location = f"{conflicting_cls._file}:{conflicting_cls._line}"
            raise ValueError(f"Observation {cls_name} already registered in {location}")


def reward(func):
    func.is_reward = True
    return func

def observation(func):
    func.is_observation = True
    return func

def termination(func):
    func.is_termination = True
    return func