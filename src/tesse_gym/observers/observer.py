###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2022 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################


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
