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

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from gym import spaces

from tesse.msgs import DataResponse, Transform
from tesse_gym.core.observations import ObservationConfig, setup_observations
from tesse_gym.core.tesse_gym import TesseGym
from tesse_gym.core.utils import NetworkConfig, set_all_camera_params
from tesse_gym.core.logging import TESSEVideoWriter
from tesse_gym.tasks.pointgoal.logging import TESSEPointGoalVideoWriter


class PointGoal(TesseGym):
    def __init__(
        self,
        build_path: str,
        network_config: Optional[NetworkConfig] = NetworkConfig(),
        scene_id: Optional[int] = None,
        episode_length: Optional[int] = 400,
        step_rate: Optional[int] = 20,
        success_dist: Optional[float] = 2,
        restart_on_collision: Optional[bool] = False,
        init_hook: Optional[Callable[[TesseGym], None]] = set_all_camera_params,
        ground_truth_mode: Optional[bool] = True,
        observation_config: Optional[ObservationConfig] = ObservationConfig(),
        video_log_path: str = None,
        video_writer_type: TESSEVideoWriter = TESSEPointGoalVideoWriter,
    ):
        """ Initialize the TESSE PointGoal environment.

        Args:
            build_path (str): Path to TESSE executable.
            network_config (NetworkConfig): Network configuration parameters.
            scene_id (int): Scene id to load.
            episode_length (int): Maximum number of steps in the episode.
            step_rate (int): If specified, game time is fixed to
                `step_rate` FPS.
            success_dist (float): Distance target must be from agent to
                be considered found. Target must also be in agent's
                field of view.
            init_hook (callable): Method to adjust any experiment specific parameters
                upon startup (e.g. camera parameters).
            ground_truth_mode (bool): Assumes gym is consuming ground truth data. Otherwise,
                assumes an external perception pipeline is running. In the latter mode, discrete
                steps will be translated to continuous control commands and observations will be
                explicitly synced with sim time.
            observation_config (Optional[ObservationConfig): Specifies observation (i.e., image 
                modalities, pose).
            video_log_path (str): Write videos here. If `None` is given, no videos are written.
            video_writer_type (TESSEVideoWriter): Give a video writer type.
        """
        super().__init__(
            build_path,
            network_config,
            scene_id,
            episode_length,
            step_rate,
            init_hook=init_hook,
            ground_truth_mode=ground_truth_mode,
            observation_config=observation_config,
            video_log_path=video_log_path,
            video_writer_type=video_writer_type,
        )

        self.success_dist = success_dist
        self.restart_on_collision = restart_on_collision

        self.scene_id = scene_id if scene_id is not None else 1
        self.spawn_points = self._load_spawn_points(build_path)

        self.goal_point = np.zeros(2)
        self.prev_dist_from_goal = None

        self._observation_space = spaces.Box(
            -np.Inf, np.Inf, shape=(self.observation_space.shape[0] + 2,)
        )
        self.reached_goal = False

    @property
    def action_space(self) -> spaces.Discrete:
        """ Agent can turn left, right, move forward, or signal `done`. """
        return spaces.Discrete(4)

    def apply_action(self, action: int) -> None:
        """ Turn left, right, move forward, or signal done.

        Args:
            action (int): Take `action` in `self.action_space`.
        """
        if action == 0:
            self.transform(0, 0.5, 0)  # forward
        elif action == 1:
            self.transform(0, 0, 8)  # turn right
        elif action == 2:
            self.transform(0, 0, -8)  # turn left
        elif action != 3:
            raise ValueError(f"Unexpected action {action}")

    def reset(
        self, scene_id: Optional[int] = None, random_seed: Optional[int] = None
    ) -> np.ndarray:
        observation = super().reset(scene_id, random_seed)

        if scene_id is not None:
            self.scene_id = scene_id

        # select random goal point
        scene_points = self.spawn_points[self.scene_id]

        # make point relative to agent's initial pose
        self.goal_point = scene_points[np.random.randint(scene_points.shape[0])]
        self.goal_point = np.matmul(self.initial_rotation, self.goal_point)
        self.goal_point -= self.initial_pose[:2]

        self.prev_dist_from_goal = None
        self.reached_goal = False

        # initial pose is set in `super().reset()` and is needed to
        # compute goal point
        observation[-2:] = self.goal_point

        return observation

    def compute_reward(
        self, observation: DataResponse, action: int
    ) -> Tuple[float, Dict[str, Any]]:
        """ Compute PointGoal reward. """
        reward = 0.0  # no time penalty

        self.steps += 1

        if self.steps >= self.episode_length:
            self.done = True

        dist_from_goal = np.linalg.norm(self.relative_pose[:2] - self.goal_point, ord=2)

        if self.prev_dist_from_goal is not None:
            reward += self.prev_dist_from_goal - dist_from_goal
            self.prev_dist_from_goal = dist_from_goal
        else:
            self.prev_dist_from_goal = dist_from_goal

        if dist_from_goal <= self.success_dist and action == 3:
            self.done = True
            self.reached_goal = True
            reward += 10

        return reward, {"env_changed": False}

    def form_agent_observation(self, tesse_data: DataResponse) -> np.ndarray:
        """ Adds goal point to the default TesseGym observation. """
        observation = super().form_agent_observation(tesse_data)

        if isinstance(self._observation_space, spaces.Box):
            if len(observation.shape) == 1:
                observation = np.concatenate((observation, self.goal_point))
            else:
                observation = np.concatenate((observation.reshape(-1), self.goal_point))
        elif isinstance(self._observation_space, spaces.Dict):
            observation["goal"] = self.goal_point

        return observation

    def _read_spawn_point_file(self, points_file: str) -> np.ndarray:
        """ Read points from TESSE spawn points file.

        Args:
            points_file (str): Path to TESSE spawn points file.
        
        Returns:
            np.ndarray, shape=(N, 2)
                `N` (x, z) spawn points, corresponding to location
                on the horizontal plane.
        """
        with open(points_file) as f:
            data = json.load(f)

        # parse TESSE spawn points json formatted as
        # {"spawnPoints": [{"name": "", "points" []}]}
        return np.array(
            [[v["x"], v["z"]] for s in data["spawnPoints"] for v in list(s.values())[1]]
        )

    def _load_spawn_points(self, sim_path: str) -> Dict[int, np.ndarray]:
        """ Get TESSE spawn points for each scene.

        Args:
            sim_path (str): Path to TESSE build. Path to spawn
                points is inferred form there.

        Returns:
            Dict[int, np.ndarray]: Scene -> spawn points dict.
                Spawn points are of shape (N, 2). `N` is the number 
                of points. Each point is in the (x, z) coord on the 
                horizontal plane. 
        """
        sim_path = Path(sim_path)
        spawn_point_root = (
            sim_path.parent
            / sim_path.name.replace(".x86_64", "_Data")
            / "StreamingAssets"
        )

        spawn_points = sorted(spawn_point_root.glob("*points"))
        return {
            i + 1: self._read_spawn_point_file(p) for i, p in enumerate(spawn_points)
        }
