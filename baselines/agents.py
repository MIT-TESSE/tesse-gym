from tesse_gym.eval.agent import Agent
from stable_baselines import PPO2
import numpy as np


class TreasureHuntAgentV1(Agent):
    def __init__(self, config):
        self.model = PPO2.load(config["weights"])
        self.state = None
        # Number of environments used to train model
        # to which stable-baselines input tensor size is fixed
        self.n_train_envs = self.model.initial_state.shape[0]

    def act(self, observation):
        """ Act upon an observation.

        args:
            observation (np.ndarray): observation.

        returns:
            int: action in (forward, right, left, declare target)
        """
        observation = np.repeat(observation[np.newaxis], self.n_train_envs, 0)
        actions, state = self.model.predict(observation, state=self.state)
        self.state = state  # update model state
        return actions[0]

    def reset(self):
        """ Reset model state. """
        self.state = None

