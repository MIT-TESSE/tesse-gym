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

from enum import Enum
import time

import defusedxml.ElementTree as ET
import numpy as np
from gym import spaces

from .tesse_gym import TesseGym, NetworkConfig
from tesse.msgs import (
    Camera,
    DataRequest,
    Compression,
    Channels,
    ObjectsRequest,
    Respawn,
    SpawnObjectRequest,
    ObjectType,
    ObjectSpawnMethod,
    RemoveObjectsRequest,
)


class HuntMode(Enum):
    SINGLE = 0
    MULTIPLE = 1


class TreasureHunt(TesseGym):
    # TARGET_COLOR = (245, 231, 50)
    TARGET_COLOR = (10, 138, 80)  # for tesse v5.1 and above
    CAMERA_FOV = 45

    def __init__(
        self,
        environment_file: str,
        network_config: NetworkConfig = NetworkConfig(),
        scene_id: int = None,
        max_steps: int = 300,
        step_rate: int = -1,
        n_targets: int = 50,
        success_dist: float = 2,
        restart_on_collision: bool = False,
        init_hook: callable = None,
        hunt_mode: HuntMode = HuntMode.MULTIPLE,
        target_found_reward: int = 1,
        continuous_control: bool = False,
        launch_tesse: bool = True,
    ):
        """ Initialize the TESSE treasure hunt environment.

        Args:
            environment_file (str): Path to TESSE executable.
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
        """
        super().__init__(
            environment_file,
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
        self.hunt_mode = hunt_mode
        self.n_found_targets = 0

    @property
    def action_space(self):
        """ Actions available to agent. """
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        """ Space observed by the agent """
        return spaces.Box(0, 255, dtype=np.uint8, shape=self.shape)

    def observe(self):
        """ Observe the state.

        Returns
            DataResponse: The `DataResponse` object. """
        cameras = [
            (Camera.RGB_LEFT, Compression.OFF, Channels.THREE),
            (Camera.SEGMENTATION, Compression.OFF, Channels.THREE),
        ]
        agent_data = self.env.request(DataRequest(metadata=True, cameras=cameras))
        return agent_data

    def reset(self):
        """ Reset the sim, randomly respawn agent and targets.

        Returns:
            np.ndarray: The observed image. """
        self.done = False
        self.steps = 0
        self.n_found_targets = 0

        self.env.send(Respawn())
        self.env.request(RemoveObjectsRequest())

        for i in range(self.n_targets):
            self.env.request(
                SpawnObjectRequest(ObjectType.CUBE, ObjectSpawnMethod.RANDOM)
            )

        if self.step_mode:
            self.advance_game_time(1)  # respawn doesn't advance game time

        self._init_pose()

        return self.form_agent_observation(self.observe())

    def apply_action(self, action):
        """ Make agent take the specified action.

        Args:
            action (action_space): Make agent take `action`.
        """
        if action == 0:
            self.transform(0, 0.5, 0)
        elif action == 1:
            self.transform(0, 0, 8)
        elif action == 2:
            self.transform(0, 0, -8)
        elif action != 3:
            raise ValueError(f"Unexpected action {action}")

    def forward_transform(self, x, z, y):
        """ Move forward in 5 small increments. This a bit of a
        hack to accommodate thin colliders, the small steps ensure
        the agent doesn't pass through them. """
        x /= 5.0
        z /= 5.0
        y /= 5.0

        for _ in range(4):
            self.transform(x, z, y)
            time.sleep(0.02)  # so messages don't get dropped
        self.transform(x, z, y)

    def compute_reward(self, observation, action):
        """ Compute reward consisting of
            - Reward if the agent has completed its task
              of being within `success_dist` of a target in its FOV
               and has given the 'done' signal (action == 3).
            - Small time penalty

        Args:
            observation (DataResponse): Images and metadata used to
            compute the reward.
            action (action_space): Action taken by agent.

        Returns:
            float: Computed reward.
        """
        targets = self.env.request(ObjectsRequest())
        agent_data = observation
        # track if agent changes environment (e.g. collects reward) so it can reobserve
        env_changed = False

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

            if len(found_targets):
                # if in `MULTIPLE` mode, remove found targets
                if self.hunt_mode is HuntMode.MULTIPLE:
                    self.n_found_targets += len(found_targets)
                    reward += self.target_found_reward * len(found_targets)
                    self.env.request(RemoveObjectsRequest(ids=found_targets))
                    env_changed = True

                    # if all targets have been found, restart the  episode
                    if self.n_found_targets == self.n_targets:
                        self._success_action()
                        self.done = True

                # if in `SINGLE` mode, reset the episode
                elif self.hunt_mode == HuntMode.SINGLE:
                    self._success_action()  # signal task was successful
                    reward += self.target_found_reward
                    self.done = True

        self.steps += 1
        if self.steps > self.max_steps:
            self.done = True

        if self.restart_on_collision and self._collision(agent_data.metadata):
            self.done = True

        return reward, env_changed

    def get_found_targets(
        self, agent_position, target_position, target_ids, agent_data
    ):
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
            targets_within_dist = target_ids[dists < self.success_dist]
            found_target_positions = target_position[dists < self.success_dist]
            agent_orientation = self._get_agent_rotation(agent_data.metadata)[-1]
            angles_rel_agent = self.get_target_orientation(
                agent_orientation, found_target_positions, agent_position
            )
            found_targets = targets_within_dist[
                np.where(angles_rel_agent < self.CAMERA_FOV)
            ]

        return found_targets

    @staticmethod
    def get_target_orientation(agent_orientation, target_positions, agent_position):
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

    def _success_action(self):
        """ Simple indicator that the agent has achieved the goal. """
        for i in range(0, 360, 360 // 5):
            self.env.send(self.TransformMessage(0, 0, 360 // 5))
            time.sleep(0.1)

    @staticmethod
    def _collision(metadata):
        """ Check for collision with environment.

        Args:
            metadata (str): Metadata string.

        Returns:
            bool: True if agent has collided with the environment. Otherwise, false.
        """
        return (
            ET.fromstring(metadata).find("collision").attrib["status"].lower() == "true"
        )

    def _get_target_id_and_positions(self, target_metadata):
        """ Get target positions from metadata.
        Args:
            target_metadata (str): Metadata string.
        Returns:
            np.ndarray: shape (n, 3) of (x, y, z) positions for the
                n targets.
        """
        position, obj_ids = [], []
        for obj in ET.fromstring(target_metadata).findall("object"):
            position.append(self._read_position(obj.find("position")))
            obj_ids.append(obj.find("id").text)
        return np.array(obj_ids, dtype=np.uint32), np.array(position, dtype=np.float32)


class RGBSegDepthInput(TreasureHunt):
    """ Legacy environment used for benchmarking """
    DEPTH_SCALE = 10

    @property
    def observation_space(self):
        """ This must be defined for custom observations. """
        return spaces.Box(0, 255, dtype=np.float32, shape=(240, 320, 7))

    def form_agent_observation(self, tesse_data):
        """ Create the agent's observation from a TESSE data response. """
        eo, seg, depth = tesse_data.images
        observation = np.concatenate((eo / 255.0,
                                      seg / 255.0,
                                      depth[..., np.newaxis] * self.DEPTH_SCALE), axis=-1)
        return observation

    def observe(self):
        cameras = [
            (Camera.RGB_LEFT, Compression.OFF, Channels.THREE),
            (Camera.SEGMENTATION, Compression.OFF, Channels.THREE),
            (Camera.DEPTH, Compression.OFF, Channels.THREE)
        ]
        agent_data = self.env.request(DataRequest(metadata=True, cameras=cameras))
        return agent_data
