""" Base environment evaluation class """


class Benchmark:
    """ Abstract class to handle agent evaluation """

    STEP_RATE = 5

    def evaluate(self, agent):
        raise NotImplementedError

