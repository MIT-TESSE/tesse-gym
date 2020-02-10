""" Base environment evaluation class """

from typing import Dict

from .agent import Agent


class Benchmark:
    """ Abstract class to handle agent evaluation """

    STEP_RATE = 20

    def evaluate(self, agent: Agent) -> Dict[str, Dict[str, float]]:
        """ Evaluate agent.

        Args:
            agent (tesse_gym.eval.agent.Agent): Agent to be evaluated.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing performance
                metrics over evaluated episodes.
        """
        raise NotImplementedError
