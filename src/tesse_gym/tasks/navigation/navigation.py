###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
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

import defusedxml.ElementTree as ET
from gym import spaces

from tesse.msgs import Transform
from tesse_gym.core.tesse_gym import TesseGym


class Navigation(TesseGym):
    @property
    def action_space(self):
        """ Agent can turn left, right, or move forward. """
        return spaces.Discrete(3)

    def apply_action(self, action):
        """ Turn left, right, or move forward.

        Args:
            action (action_space): Make agent take `action`.
        """
        if action == 0:
            self.env.send(Transform(0, 0.5, 0))  # forward
        elif action == 1:
            self.env.send(Transform(0, 0, 8))  # turn right
        elif action == 2:
            self.env.send(Transform(0, 0, -6))  # turn left
        else:
            pass

    def compute_reward(self, observation, action):
        """ Reward agent for moving forward. Penalize agent for
        colliding with the environment.

        Args:
            observation (DataResponse): Images and metadata used to
                compute the reward.
            action (action_space): Action taken by agent.

        Returns:
            float: Computed reward.
            dict: Empty dictionary as required by `step`
        """
        reward = 0.0
        if action == 0:
            reward += 0.1  # reward for stepping forward

        self.steps += 1
        if self.steps > self.episode_length:
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

        return reward, {}
