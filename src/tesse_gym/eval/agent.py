""" Base agent """
import numpy as np


class Agent:
    """ Abstract class for define agents that act within `TesseGym`.

    Designed to simplifying agent benchmarking.
    """

    def act(self, observation: np.ndarray) -> int:
        """ Act upon an environment observation.

        Args:
            observation (np.ndarray): Observation.

        Returns:
            int: Agent's action.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """ Called when environment resets. """
        raise NotImplementedError
