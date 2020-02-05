""" Base environment evaluation class """


class Benchmark:
    """ Abstract class to handle agent evaluation """

    STEP_RATE = 5

    def evaluate(self, agent):
        """ Evaluate agent.

        Args:
            agent (tesse_gym.eval.agent.Agent): Agent to be evaluated.
        """
        raise NotImplementedError
