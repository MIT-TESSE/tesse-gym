""" Base agent """
import numpy as np


class Agent:
    """ Abstract class to handle """

    def act(self, observation):
        """ Act upon an environment observation.

        Args:
            observation (np.ndarray): Observation.

        Returns:
            int: Agent's action.
        """
        raise NotImplementedError

    def reset(self):
        """ Called when environment resets. """
        raise NotImplementedError
