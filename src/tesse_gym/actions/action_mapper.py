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

from typing import Any, Optional

from gym import spaces
from tesse.env import Env
from tesse.msgs import StepWithTransform, Transform
from tesse_gym.core.continuous_control import ContinuousController


class ActionMapper:
    def __init__(
        self,
        step_mode: Optional[bool] = True,
        ground_truth_mode: Optional[bool] = True,
        step_rate: Optional[int] = 20,
    ):
        self.TransformMessage = StepWithTransform if step_mode else Transform
        self.ground_truth_mode = ground_truth_mode
        self.step_rate = step_rate

        if not ground_truth_mode:
            self.continuous_controller = ContinuousController(
                env=self.env, framerate=step_rate
            )

    def register_env(self, env: Env) -> None:
        self.env = env
        if not self.ground_truth_mode:
            self.continuous_controller = ContinuousController(
                env=self.env, framerate=self.step_rate
            )

    @property
    def action_space(self) -> spaces.Space:
        return NotImplementedError()

    def apply_action(self, action: Any) -> None:
        raise NotImplementedError()
