from .tesse_env import TesseEnv
import defusedxml.ElementTree as ET
from gym import spaces
from tesse.msgs import Transform


class NavigationEnv(TesseEnv):
    @property
    def action_space(self):
        return spaces.Discrete(3)

    def _apply_action(self, action):
        if action == 0:
            self.env.send(Transform(0, 0.5, 0))  # forward
        elif action == 1:
            self.env.send(Transform(0, 0, 8))  # turn right
        elif action == 2:
            self.env.send(Transform(0, 0, -6))  # turn left
        else:
            pass

    def _compute_reward(self, observation, action):
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
