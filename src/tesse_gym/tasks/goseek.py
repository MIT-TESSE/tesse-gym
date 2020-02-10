###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2019 Massachusetts Institute of Technology.
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

from typing import Dict, Tuple, Union, List
import defusedxml.ElementTree as ET
import numpy as np
from gym import spaces

from tesse.msgs import (
    Camera,
    Channels,
    Compression,
    DataRequest,
    DataResponse,
    ObjectSpawnMethod,
    ObjectsRequest,
    RemoveObjectsRequest,
    Respawn,
    SpawnObjectRequest,
)
from tesse_gym.core.tesse_gym import TesseGym
from tesse_gym.core.utils import NetworkConfig


class GoSeek(TesseGym):
    TARGET_COLOR = (10, 138, 80)
    CAMERA_FOV = 60

    def __init__(
        self,
        build_path: str,
        network_config: NetworkConfig = NetworkConfig(),
        scene_id: int = None,
        max_steps: int = 300,
        step_rate: int = -1,
        n_targets: int = 50,
        success_dist: float = 2,
        restart_on_collision: bool = False,
        init_hook: callable = None,
        target_found_reward: int = 1,
        continuous_control: bool = False,
        launch_tesse: bool = True,
        n_target_types: int = 1,
    ):
        """ Initialize the TESSE treasure hunt environment.

        Args:
            build_path (str): Path to TESSE executable.
            network_config (NetworkConfig): Network configuration parameters.
            scene_id (int): Scene id to load.
            max_steps (int): Maximum number of steps in the episode.
            step_rate (int): If specified, game time is fixed to
                `step_rate` FPS.
            n_targets (int): Number of targets to spawn in the scene.
            success_dist (float): Distance target must be from agent to
                be considered found. Target must also be in agent's
                field of view.
            init_hook (callable): Method to adjust any experiment specific parameters
                upon startup (e.g. camera parameters).
            continuous_control (bool): True to use a continuous controller to move the
                agent. False to use discrete transforms.
            launch_tesse (bool): True to start tesse instance. Otherwise, assume another
                instance is running.
            n_target_types (int): Number of target types available to spawn.
        """
        super().__init__(
            build_path,
            network_config,
            scene_id,
            max_steps,
            step_rate,
            init_hook=init_hook,
            continuous_control=continuous_control,
            launch_tesse=launch_tesse,
        )
        self.n_targets = n_targets
        self.success_dist = success_dist
        self.max_steps = max_steps
        self.restart_on_collision = restart_on_collision
        self.target_found_reward = target_found_reward
        self.n_found_targets = 0
        self.n_target_types = n_target_types

    @property
    def action_space(self) -> spaces.Discrete:
        """ Actions available to agent. """
        return spaces.Discrete(4)

    @property
    def observation_space(self) -> spaces.Box:
        """ Space observed by the agent """
        return spaces.Box(-np.Inf, np.Inf, dtype=np.uint8, shape=self.shape)

    def observe(self) -> DataResponse:
        """ Observe the state.

        Returns:
            DataResponse: The `DataResponse` object. """
        cameras = [
            (Camera.RGB_LEFT, Compression.OFF, Channels.THREE),
            (Camera.SEGMENTATION, Compression.OFF, Channels.THREE),
        ]
        agent_data = self.env.request(DataRequest(metadata=True, cameras=cameras))
        return agent_data

    def reset(self) -> np.ndarray:
        """ Reset the sim, randomly respawn agent and targets.

        Returns:
            np.ndarray: Agent's observation. """
        self.done = False
        self.steps = 0
        self.n_found_targets = 0

        self.env.send(Respawn())
        self.env.request(RemoveObjectsRequest())

        for i in range(self.n_targets):
            self.env.request(
                SpawnObjectRequest(i % self.n_target_types, ObjectSpawnMethod.RANDOM)
            )

        if self.step_mode:
            self.advance_game_time(1)  # respawn doesn't advance game time

        self._init_pose()

        return self.form_agent_observation(self.observe())

    def apply_action(self, action: int) -> None:
        """ Make agent take the specified action.

        Args:
            action (int): Make agent take `action`.
        """
        if action == 0:  # move forward 0.5m
            self.transform(0, 0.5, 0)
        elif action == 1:
            self.transform(0, 0, 8)  # turn right 8 degrees
        elif action == 2:
            self.transform(0, 0, -8)  # turn left 8 degrees
        elif action != 3:
            raise ValueError(f"Unexpected action {action}")

    def compute_reward(
        self, observation: DataResponse, action: int
    ) -> Tuple[float, Dict[str, Union[int, bool]]]:
        """ Compute reward.

        Reward consists of:
            - Small time penalty
            - If the agent is (1) within `success_dist`, (2) has the target in its FOV,
                and (3) executes action 3, it receives a reward equal to the
                number of targets found times `self.target_found_reward`

        Args:
            observation (DataResponse): Images and metadata used to
                compute the reward.
            action (action_space): Action taken by agent.

        Returns:
            Tuple[float, dict[str, [bool, int]]
                Reward,
                Dictionary with the following keys
                    - env_changed: True if agent changed the environment.
                    - collision: True if there was a collision

        """
        targets = self.env.request(ObjectsRequest())
        agent_data = observation
        reward_info = {"env_changed": False, "collision": False, "n_found_targets": 0}

        # compute agent's distance from targets
        agent_position = self._get_agent_position(agent_data.metadata)
        target_ids, target_position = self._get_target_id_and_positions(
            targets.metadata
        )

        reward = -0.01  # small time penalty

        # check for found targets
        if target_position.shape[0] > 0 and action == 3:
            found_targets = self.get_found_targets(
                agent_position, target_position, target_ids, agent_data
            )

            # if targets are found, update reward and related episode info
            if len(found_targets):
                self.n_found_targets += len(found_targets)
                reward += self.target_found_reward * len(found_targets)
                self.env.request(RemoveObjectsRequest(ids=found_targets))
                reward_info["env_changed"] = True
                reward_info["n_found_targets"] += len(found_targets)

                # if all targets have been found, restart the episode
                if self.n_found_targets == self.n_targets:
                    self.done = True

        self.steps += 1
        if self.steps > self.max_steps:
            self.done = True

        if self._collision(agent_data.metadata):
            reward_info["collision"] = True

            if self.restart_on_collision:
                self.done = True

        return reward, reward_info

    def get_found_targets(
        self,
        agent_position: np.ndarray,
        target_position: np.ndarray,
        target_ids: np.ndarray,
        agent_data: DataResponse,
    ) -> List[int]:
        """ Get targets that are within `self.success_dist` of agent and in FOV.

        Args:
            agent_position (np.ndarray): Agent position in (x, y, z) as a shape (3,) array.
            target_position (np.ndarray): Target positions in (x, y, z) as a shape (n, 3) array.
            target_ids (np.ndarray): Target IDS corresponding to position
            agent_data (DataResponse): Agent observation data.

        Returns:
            List[int]: IDs of found targets.
        """
        found_targets = []

        # only compare (x, z) coordinates
        agent_position = agent_position[np.newaxis, (0, 2)]
        target_position = target_position[:, (0, 2)]
        dists = np.linalg.norm(target_position - agent_position, axis=-1)

        # can we see the target?
        seg = agent_data.images[1]
        target_in_fov = np.all(seg == self.TARGET_COLOR, axis=-1)

        # if the agent is within `success_dist` of target, can see it,
        # and gives the `found` action, count as found
        if dists.min() < self.success_dist and target_in_fov.any():
            targets_in_range = target_ids[dists < self.success_dist]
            found_target_positions = target_position[dists < self.success_dist]
            agent_orientation = self._get_agent_rotation(agent_data.metadata)[-1]
            target_heading_relative_agent = self.get_target_orientation(
                agent_orientation, found_target_positions, agent_position
            )
            found_targets = targets_in_range[
                np.where(target_heading_relative_agent < self.CAMERA_FOV)
            ]

        return found_targets

    @staticmethod
    def get_target_orientation(
        agent_orientation: float,
        target_positions: np.ndarray,
        agent_position: np.ndarray,
    ) -> np.ndarray:
        """ Get orientation of targets relative to agents given the agent position, orientation,
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

    @staticmethod
    def _collision(metadata: str) -> bool:
        """ Check for collision with environment.

        Args:
            metadata (str): Metadata string.

        Returns:
            bool: True if agent has collided with the environment. Otherwise, false.
        """
        return (
            ET.fromstring(metadata).find("collision").attrib["status"].lower() == "true"
        )

    def _get_target_id_and_positions(
        self, target_metadata: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Get target positions from metadata.

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


class MultiModalGoSeek(GoSeek):
    """ Define a custom TESSE gym environment to provide RGB, depth,
    segmentation, and pose data.
    """

    DEPTH_SCALE = 5
    N_CLASSES = 11
    WALL_CLS = 2

    @property
    def observation_space(self) -> spaces.Box:
        """ Define an observation space for RGB, depth, segmentation, and pose.

        Because Stables Baselines (the baseline PPO library) does not support dictionary spaces,
        the observation images and pose vector will be combined into a vector. The RGB image
        is of shape (240, 320, 3), depth and segmentation are both (240, 320), ose is (3,), thus
        the total shape is (240 * 320 * 5 + 3).
        """
        return spaces.Box(np.Inf, np.Inf, shape=(240 * 320 * 5 + 3,))

    def form_agent_observation(self, tesse_data: DataResponse) -> np.ndarray:
        """ Create the agent's observation from a TESSE data response.

        Args:
            tesse_data (DataResponse): TESSE DataResponse object containing
                RGB, depth, segmentation, and pose.

        Returns:
            np.ndarray: The agent's observation.
        """
        eo, seg, depth = tesse_data.images
        seg = seg[..., 0].copy()  # get segmentation as one-hot encoding

        # TESSE assigns areas without a material (e.g. outside windows) the maximum possible
        # image value. It will be classified as wall.
        seg[seg > (self.N_CLASSES - 1)] = self.WALL_CLS
        observation = np.concatenate(
            (
                eo / 255.0,
                seg[..., np.newaxis] / (self.N_CLASSES - 1),
                (depth[..., np.newaxis] * self.DEPTH_SCALE).clip(0, 1),
            ),
            axis=-1,
        ).reshape(-1)
        pose = self.get_pose().reshape((3))

        if (np.abs(pose) > 100).any():
            raise ValueError("Pose is out of observation space")
        return np.concatenate((observation, pose))

    def observe(self) -> DataResponse:
        """ Get observation data from TESSE.

        Returns:
            DataResponse: TESSE DataResponse object.
        """
        cameras = [
            (Camera.RGB_LEFT, Compression.OFF, Channels.THREE),
            (Camera.SEGMENTATION, Compression.OFF, Channels.THREE),
            (Camera.DEPTH, Compression.OFF, Channels.THREE),
        ]
        agent_data = self.env.request(DataRequest(metadata=True, cameras=cameras))
        return agent_data
