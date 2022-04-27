from typing import Dict, Union

import numpy as np
from gym import spaces

from tesse.msgs import DataResponse


class Observer:
    def observe(self, info_dict: Dict[str, Union[DataResponse, np.ndarray]]):
        return NotImplementedError()

    def reset(self):
        pass


class EnvObserver:
    def __init__(self, observers: Union[Dict[str, Observer], Observer]):
        assert isinstance(observers, dict) or isinstance(observers, Observer)
        self.observers = observers

    @property
    def observation_space(self) -> Union[spaces.Dict, spaces.Box]:
        if isinstance(self.observers, dict):
            space = {}
            for observer in self.observers.values():
                space.update(observer.observation_space.spaces)
            return spaces.Dict(space)
        else:
            return self.observers.observation_space

    def reset(self):
        if isinstance(self.observers, dict):
            for observer in self.observers.values():
                observer.reset()
        else:
            self.observers.reset()

    def observe(self, env_info):
        if isinstance(self.observers, dict):
            observation = {}
            for observer in self.observers.values():
                observation.update(observer.observe(env_info))
            return observation
        else:
            return self.observers.observe(env_info)
