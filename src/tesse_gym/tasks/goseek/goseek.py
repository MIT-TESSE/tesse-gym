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

from typing import Callable, Dict, List, Optional, Tuple, Union

import defusedxml.ElementTree as ET
import numpy as np

from tesse.msgs import (
    DataResponse,
    MetadataMessage,
    ObjectSpawnMethod,
    ObjectsRequest,
    RemoveObjectsRequest,
    SpawnObjectRequest,
)
from tesse_gym.actions.discrete import ActionMapper
from tesse_gym.core.logging import TESSEVideoWriter
from tesse_gym.core.observations import ObservationConfig
from tesse_gym.core.tesse_gym import TesseGym
from tesse_gym.core.utils import NetworkConfig, set_all_camera_params
from tesse_gym.observers.observer import Observer
from tesse_gym.tasks.goseek.logging import TESSEGoSeekVideoWriter


# define custom message to signal episode reset
# Used for resetting external perception pipelines
class EpisodeResetSignal(MetadataMessage):
    __tag__ = "sRES"


class GoSeek(TesseGym):
    TARGET_COLOR = (10, 138, 80)
    CAMERA_HFOV = 80
    CAMERA_REL_AGENT = np.array([-0.05, 0])

    def __init__(
        self,
        build_path: str,
        observers: List[Observer],
        action_mapper: ActionMapper,
        network_config: Optional[NetworkConfig] = NetworkConfig(),
        scene_id: Optional[int] = None,
        episode_length: Optional[int] = 400,
        step_rate: Optional[int] = 20,
        n_targets: Optional[int] = 30,
        success_dist: Optional[float] = 2,
        restart_on_collision: Optional[bool] = False,
        init_function: Optional[Callable[[TesseGym], None]] = set_all_camera_params,
        target_found_reward: Optional[int] = 1,
        ground_truth_mode: Optional[bool] = True,
        n_target_types: Optional[int] = 5,
        collision_reward: Optional[int] = 0,
        false_positive_reward: Optional[int] = 0,
        observation_config: Optional[ObservationConfig] = ObservationConfig(),
        video_writer_type: TESSEVideoWriter = TESSEGoSeekVideoWriter,
        video_log_path=None,
        video_save_freq: int = 5,
        no_collisions=False,
        query_image_data=True,
        visible_targets=True,
        eval_env: Optional[bool] = False,
        eval_config: Optional[List[Tuple[int, int]]] = None,
    ):
        """Initialize the TESSE treasure hunt environment.

        Args:
            build_path (str): Path to TESSE executable.
            network_config (NetworkConfig): Network configuration parameters.
            scene_id (int): Scene id to load.
            episode_length (int): Maximum number of steps in the episode.
            step_rate (int): If specified, game time is fixed to
                `step_rate` FPS.
            n_targets (int): Number of targets to spawn in the scene.
            success_dist (float): Distance target must be from agent to
                be considered found. Target must also be in agent's
                field of view.
            init_hook (callable): Method to adjust any experiment specific parameters
                upon startup (e.g. camera parameters).
            ground_truth_mode (bool): Assumes gym is consuming ground truth data. Otherwise,
                assumes an external perception pipeline is running. In the latter mode, discrete
                steps will be translated to continuous control commands and observations will be
                explicitly synced with sim time.
            n_target_types (int): Number of target types available to spawn. GOSEEK challenge
                has 5 target types by default.
            collision_reward: (int): Added to total step reward upon collision. Default is 0.
            false_positive_reward (int): Added total step reward when agent incorrectly
                declares a target found (action 3). Default is 0.
            observation_modalities: (Optional[Tuple[Camera]]): Input modalities to be used.
                Defaults to (RGB_LEFT, SEGMENTATION, DEPTH).
            no_collisions (bool): Disable scene colliders.
            query_image_data (bool): True to query image data. Otherwise, use zero images.
            visible_targets (bool): True to render targets. Otherwise, targets are invisible.
            eval_config (Optional[List[Tuple[int, int]]]): Pairs of [scene, random seed]
                to cycle through during evaluation. Episodes are initialized randomly if
                no config is given.
            eval_env (Optional[bool]): True if environment is used for evaluation. If
                True, `env_config` parameters will be used to initialize episodes.

        """
        super().__init__(
            build_path,
            observers,
            action_mapper,
            network_config,
            scene_id,
            episode_length,
            step_rate,
            init_function=init_function,
            ground_truth_mode=ground_truth_mode,
            observation_config=observation_config,
            video_writer_type=video_writer_type,
            video_log_path=video_log_path,
            video_save_freq=video_save_freq,
            no_collisions=no_collisions,
            query_image_data=query_image_data,
            eval_env=eval_env,
            eval_config=eval_config,
        )
        self.n_targets = n_targets
        self.success_dist = success_dist
        self.restart_on_collision = restart_on_collision
        self.target_found_reward = target_found_reward
        self.n_target_types = n_target_types
        self.n_found_targets = 0
        self.collision_reward = collision_reward
        self.false_positive_reward = false_positive_reward
        self.visible_targets = visible_targets

        if self.visible_targets:
            self.target_positions = None
            self.target_ids = None

    def reset(
        self, scene_id: Optional[int] = None, random_seed: Optional[int] = None
    ) -> np.ndarray:
        """Reset environment and respawn agent.

        Args:
            scene_id (int): If given, change to this scene.
            random_seed (int): If give, set simulator random seed.

        Returns:
            np.ndarray: Agent's observation."""
        self.env.send(EpisodeResetSignal())
        super().reset(scene_id, random_seed)

        self.env.request(RemoveObjectsRequest())
        self.n_found_targets = 0

        for i in range(self.n_targets):
            self.env.request(
                SpawnObjectRequest(i % self.n_target_types, ObjectSpawnMethod.RANDOM)
            )

        if not self.visible_targets:
            self.init_invisible_targets()

        # respawn doesn't advance game time
        # if running an external perception server, advance game time to refresh
        if not self.ground_truth_mode:
            self.advance_game_time(1)

        observation = self.get_synced_observation()
        obs = self.form_observation(observation)
        return obs

    def init_invisible_targets(self):
        """Record target positions and remove them from scene.

        This keeps logic consistant with normal goseek, except that
        the targets are not visible.
        """
        targets = self._data_request(ObjectsRequest())
        self.target_ids, self.target_positions = self._get_target_id_and_positions(
            targets.metadata
        )
        self.env.request(RemoveObjectsRequest())

    def remove_invisible_targets(self, inds):
        """In remove targets in invisible mode."""
        keep_inds = ~np.in1d(self.target_ids, inds)
        self.target_ids = self.target_ids[keep_inds]
        self.target_positions = self.target_positions[keep_inds]

    def compute_reward(
        self, observation: DataResponse, action: int
    ) -> Tuple[float, Dict[str, Union[int, bool]]]:
        """Compute reward.

        Reward consists of:
            - Small time penalty
            - n_targets_found * `target_found_reward` if `action` == 3.
                n_targets_found is the number of targets that are
                (1) within `success_dist` of agent and (2) within
                a bearing of `CAMERA_FOV` degrees.

        Args:
            observation (DataResponse): TESSE DataResponse object containing images
                and metadata.
            action (int): Action taken by agent.

        Returns:
            Tuple[float, dict[str, [bool, int]]
                Reward
                Dictionary with the following keys
                    - env_changed: True if agent changed the environment.
                    - collision: True if there was a collision
                    - n_found_targets: Number of targets found during step.
        """
        if self.visible_targets:
            targets = self.env.request(ObjectsRequest())
            target_ids, target_position = self._get_target_id_and_positions(
                targets.metadata
            )
        else:
            target_ids, target_position = self.target_ids, self.target_positions

        # If not in ground truth mode, metadata will only provide position estimates
        # In that case, get ground truth metadata from the controller
        agent_metadata = (
            observation.metadata
            if self.ground_truth_mode
            else self.continuous_controller.get_broadcast_metadata()
        )
        reward_info = {"env_changed": False, "collision": False, "n_found_targets": 0}

        # compute agent's distance from targets
        agent_position = self._get_agent_position(agent_metadata)
        reward = -0.01  # small time penalty

        # check for found targets
        if target_position.shape[0] > 0 and (action == 3 or not self.visible_targets):
            found_targets = self.get_found_targets(
                agent_position, target_position, target_ids, agent_metadata
            )

            # if targets are found, update reward and related episode info
            if len(found_targets):
                self.n_found_targets += len(found_targets)
                reward += self.target_found_reward * len(found_targets)
                reward_info["env_changed"] = True
                reward_info["n_found_targets"] += len(found_targets)

                if self.visible_targets:
                    self.env.request(RemoveObjectsRequest(ids=found_targets))
                else:
                    self.remove_invisible_targets(found_targets)

                # if all targets have been found, restart the episode
                if self.n_found_targets == self.n_targets:
                    self.done = True

        # False positive
        elif target_position.shape[0] == 0 and True:  # action == 3:
            reward += self.false_positive_reward

        # collision information isn't provided by the controller metadata
        if self._collision(observation.metadata):
            reward_info["collision"] = True
            reward += self.collision_reward

            if self.restart_on_collision:
                self.done = True

        return reward, reward_info

    def get_found_targets(
        self,
        agent_position: np.ndarray,
        target_position: np.ndarray,
        target_ids: np.ndarray,
        agent_metadata: str,
    ) -> List[int]:
        """Get IDs of all found targets

        Targets are considered found when they are:
            (1) within `success_dist` of the agent.
            (2) Within a bearing of `CAMERA_FOV` degrees.

        Args:
            agent_position (np.ndarray): Agent position in (x, y, z) as a shape (3,) array.
            target_position (np.ndarray): Target positions in (x, y, z) as a shape (n, 3) array.
            target_ids (np.ndarray): Target IDS corresponding to position
            agent_metadata (str): Agent metadata from TESSE.

        Returns:
            List[int]: IDs of found targets.
        """
        found_targets = []

        # only compare (x, z) coordinates
        agent_position = agent_position[np.newaxis, (0, 2)]

        # get bearing and distance of targets w.r.t the left camera
        # get left camera position in world coordinates
        agent_orientation = self._get_agent_rotation(agent_metadata)[-1]
        left_camera_position = agent_position + np.matmul(
            self.get_2d_rotation_mtrx(agent_orientation), self.CAMERA_REL_AGENT
        )

        target_position = target_position[:, (0, 2)]
        dists = np.linalg.norm(target_position - left_camera_position, axis=-1)

        if dists.min() < self.success_dist:
            # get positions of targets
            targets_in_range = target_ids[dists < self.success_dist]
            found_target_positions = target_position[dists < self.success_dist]

            if self.visible_targets:
                target_bearing = self.get_target_bearing(
                    agent_orientation, found_target_positions, left_camera_position
                )

                # targets that meet distance and bearing requirements
                found_targets = targets_in_range[
                    np.where(target_bearing < self.CAMERA_HFOV / 2)
                ]
            else:
                found_targets = targets_in_range

        return found_targets

    @staticmethod
    def get_target_bearing(
        agent_orientation: float,
        target_positions: np.ndarray,
        agent_position: np.ndarray,
    ) -> np.ndarray:
        """Get orientation of targets relative to agents given the agent position, orientation,
        and target positions.

        Args:
            agent_orientation (float): Orientation of agent (y rotation) in radians.
            target_positions (np.ndarray): Array of target (x, z) positions of shape (n, 2)
                where n is the number of targets.
            agent_position (np.ndarray): Array of agent (x, z) position of shape (1, 2)

        Returns:
            np.ndarray: Array of target orientations relative to agent.
        """
        heading = np.array([[np.sin(agent_orientation), np.cos(agent_orientation)]])
        target_relative_to_agent = target_positions - agent_position
        target_orientation = np.arccos(
            np.dot(heading, target_relative_to_agent.T)
            / (
                np.linalg.norm(target_relative_to_agent, axis=-1)
                * np.linalg.norm(heading)
            )
        )
        return np.rad2deg(target_orientation).reshape(-1)

    def _get_target_id_and_positions(
        self, target_metadata: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get target positions from metadata.

        Args:
            target_metadata (str): Metadata string.

        Returns:
            Tuple[np.ndarray, np.array]: shape (n, 3) of (x, y, z) target positions and
                shape (n,) target ids.
        """
        position, obj_ids = [], []
        for obj in ET.fromstring(target_metadata).findall("object"):
            position.append(self._read_position(obj.find("position")))
            obj_ids.append(obj.find("id").text)
        return np.array(obj_ids, dtype=np.uint32), np.array(position, dtype=np.float32)
