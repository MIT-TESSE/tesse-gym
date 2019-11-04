from .tesse_env import TesseEnv
import defusedxml.ElementTree as ET
from gym import spaces
from tesse.msgs import Transform


class Navigation(TesseEnv):
    @property
    def action_space(self):
        """ Agent can turn left, right, or move forward. """
        return spaces.Discrete(3)

    def _apply_action(self, action):
        """ Turn left, right, or move forward.
        Args:
            action: action from `action_space`.
        """
        if action == 0:
            self.env.send(Transform(0, 0.5, 0))  # forward
        elif action == 1:
            self.env.send(Transform(0, 0, 8))  # turn right
        elif action == 2:
            self.env.send(Transform(0, 0, -6))  # turn left
        else:
            pass

    def _compute_reward(self, observation, action):
        """ Reward agent for moving forward. Penalize agent for
        colliding with the environment.

        Args:
            observation: Environment observation.
            action: Agent's action.

        Returns:
             Computed reward.
        """
        reward = 0.0
        if action == 0:
            reward += 0.1  # reward for stepping forward

        self.steps += 1
        if self.steps > self.max_steps:
            self.done = True

        # check for collision
        if (
            ET.fromstring(observation.metadata)
            .find("collision")
            .attrib["status"]
            .lower()
            == "true"
        ):
            reward -= 1.0  # Reward for colliding
            self.done = True  # If colliding, the scenario ends

        return reward
