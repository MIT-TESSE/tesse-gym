from tesse_gym.eval.agent import Agent
from stable_baselines import PPO2
import numpy as np


class TreasureHuntAgentV1(Agent):
    def __init__(self, config):
        self.model = PPO2.load(config["weights"])
        self.state = None

    def act(self, observation):
        """

        args:
            observation (np.ndarray): observation of shape (240, 320, 7).

        returns:
            int: action in (forward, right, left, declare target)
        """
        # model was trained on 6 parallel environments so expects an observation
        # of shape (6, 240, 320, 7)
        observation = np.repeat(observation[np.newaxis], 6, 0)
        actions, state = self.model.predict(observation, state=self.state)
        self.state = state  # update model state
        return actions[0]

    def reset(self):
        """ Reset model state. """
        self.state = None

