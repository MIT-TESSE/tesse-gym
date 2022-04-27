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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from gym import spaces

from tesse.msgs import DataResponse
from tesse_gym.core.logging import TESSEVideoWriter
from tesse_gym.core.observations import ObservationConfig
from tesse_gym.core.tesse_gym import TesseGym
from tesse_gym.core.utils import NetworkConfig, set_all_camera_params
from tesse_gym.tasks.exploration.logging import TESSEExplorationVideoWriter


class Exploration(TesseGym):
    def __init__(
        self,
        build_path: str,
        network_config: Optional[NetworkConfig] = NetworkConfig(),
        scene_id: Optional[Union[int, List[int]]] = None,
        episode_length: Optional[int] = 400,
        step_rate: Optional[int] = 20,
        restart_on_collision: Optional[bool] = False,
        init_function: Optional[Callable[[TesseGym], None]] = set_all_camera_params,
        ground_truth_mode: Optional[bool] = True,
        observation_config: Optional[ObservationConfig] = ObservationConfig(),
        video_log_path: str = None,
        video_writer_type: TESSEVideoWriter = TESSEExplorationVideoWriter,
        cell_size: int = 1,
        collision_penalty: float = 0,
    ):
        """Initialize the TESSE exploration environment.

        The agent is tasked with exploring the maximum possible space
        within the alloted episode length. The space is divided into 2D
        grids of a user specified size.

        Args:
            build_path (str): Path to TESSE executable.
            network_config (NetworkConfig): Network configuration parameters.
            scene_id (int): Scene id to load.
            episode_length (int): Maximum number of steps in the episode.
            step_rate (int): If specified, game time is fixed to
                `step_rate` FPS.
            init_function (callable): Method to adjust experiment specific parameters
                upon startup (e.g. camera parameters).
            ground_truth_mode (bool): Assumes gym is consuming ground truth data. Otherwise,
                assumes an external perception pipeline is running. In the latter mode, discrete
                steps will be translated to continuous control commands and observations will be
                explicitly synced with sim time.
            observation_config (Optional[ObservationConfig): Specifies observation (i.e., image
                modalities, pose).
            video_log_path (str): Write videos here. If `None` is given, no videos are written.
            video_writer_type (TESSEVideoWriter): Give a video writer type.
            cell_size (int): Size of explored/unexplored cell in meters.
            collision_penalty (int): Penalty subtracted from reward upon collision.
        """

        super().__init__(
            sim_path=build_path,
            network_config=network_config,
            scene_id=scene_id,
            episode_length=episode_length,
            step_rate=step_rate,
            init_function=init_function,
            ground_truth_mode=ground_truth_mode,
            observation_config=observation_config,
            video_log_path=video_log_path,
            video_writer_type=video_writer_type,
        )
        # 2d cells comprise the area to explore
        # below defines that cell size
        self.cell_size = cell_size

        # vector of cell coordinates
        self.visited_cells = []
        self.collision_penalty = collision_penalty
        self.n_collisions = 0  # collisions per episode

    @property
    def action_space(self) -> spaces.Discrete(3):
        """Agent can turn left, right, or move forward."""
        return spaces.Discrete(3)

    def apply_action(self, action: int) -> None:
        """Turn left, right, or move forward.

        Args:
            action (int): Take `action` in `self.action_space`.
        """
        if action == 0:
            self.transform(0, 0.5, 0)  # forward
        elif action == 1:
            self.transform(0, 0, 8)  # turn right
        elif action == 2:
            self.transform(0, 0, -8)  # turn left
        else:
            raise ValueError(f"Unexpected action {action}")

    def reset(
        self, scene_id: Optional[int] = None, random_seed: Optional[int] = None
    ) -> np.ndarray:
        """Reset environment upon episode completion."""
        observation = super().reset(scene_id, random_seed)

        self.visited_cells = []
        self.n_collisions = 0

        return observation

    def compute_reward(
        self, observation: DataResponse, action: int
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute exploration reward.

        Small time penalty. Give reward for entering
        a new cell.

        observation (DataResponse): tesse_interface
            DataResponse object containing a combination
            of image and pose data

        action (int): The agent's action in
            `self.action_space`. See `apply_action`
            for mapping from action id to agent
            behavior.

        Returns:
            Tuple[float, Dict[str, Any]]
                - reward
                - dict containing metadata.
        """
        time_penalty = -0.01
        reward = time_penalty

        # check for falls out of scene
        if self._get_agent_position(observation.metadata)[1] < 0:
            self.done = True
            reward += time_penalty * (self.episode_length - self.steps)  # min reward
            return reward, {"env_changed": False}

        # check for collisions
        if self._collision(observation.metadata):
            reward += self.collision_penalty
            self.n_collisions += 1

        # center grid around agent's initial position
        visited_cell = (
            int((self.relative_pose[0] - self.cell_size / 2) // self.cell_size),
            int((self.relative_pose[1] - self.cell_size / 2) // self.cell_size),
        )

        if visited_cell not in self.visited_cells:
            self.visited_cells.append(visited_cell)

            # don't reward initial cell
            if len(self.visited_cells) > 1:
                reward += 1

        return reward, {"env_changed": False}
