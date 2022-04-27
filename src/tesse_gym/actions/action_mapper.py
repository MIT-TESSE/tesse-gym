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
