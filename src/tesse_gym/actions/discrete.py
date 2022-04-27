###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2022 Massachusetts Institute of Technology.
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


import numpy as np
from gym import spaces

from .action_mapper import ActionMapper


class DiscreteNavigationMapper(ActionMapper):
    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(3)

    def apply_action(self, action: int) -> None:
        """Make agent take the specified action.

        Args:
            action (int): Make agent take `action`.
        """
        if action not in self.action_space:
            raise ValueError(f"Unexpected action {action}")
        if action == 0:  # move forward 0.5m
            self.transform(0, 0.5, 0)
        elif action == 1:  # turn right 8 degrees
            self.transform(0, 0, 8)
        elif action == 2:  # turn left 8 degrees
            self.transform(0, 0, -8)

    def transform(self, x: float, z: float, y: float) -> None:
        """Apply desired transform to agent. If in continuous mode, the
        agent is moved via force commands. Otherwise, a discrete transform
        is applied.

        Args:
            x (float): desired x translation.
            z (float): desired z translation.
            y (float): Desired rotation (in degrees).
        """
        if self.ground_truth_mode:
            self.env.send(self.TransformMessage(x, z, y))
        else:
            self.continuous_controller.transform(x, z, np.deg2rad(y))


class DiscreteCollectNavigationMapper(DiscreteNavigationMapper):
    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(4)
