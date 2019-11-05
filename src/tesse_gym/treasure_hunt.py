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

from .tesse_gym import TesseGym
import defusedxml.ElementTree as ET
import numpy as np
import time
from gym import spaces
from scipy.spatial.transform import Rotation
from enum import Enum
from tesse.msgs import (
    Transform,
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
    StepWithTransform,
)


class HuntMode(Enum):
    SINGLE = 0
    MULTIPLE = 1


class TreasureHunt(TesseGym):
    TARGET_COLOR = (245, 231, 50)

    def __init__(
        self,
        environment_file: str,
        simulation_ip: str,
        own_ip: str,
        worker_id: int = 0,
        base_port: int = 9000,
        scene_id: int = None,
        max_steps: int = 100,
        step_rate: int = -1,
        n_targets: int = 50,
        success_dist: float = 2,
        restart_on_collision: bool = False,
        init_hook: callable = None,
        hunt_mode: HuntMode = HuntMode.MULTIPLE,
        target_found_reward: int = 1
    ):
        """ Initialize the TESSE treasure hunt environment.

        Args:
            environment_file (str): Path to TESSE executable.
            simulation_ip (str): TESSE IP address
            own_ip (str): Interface IP address.
            worker_id (int): Subprocess worker id.
            base_port (int): Ports are assigned as follows:
                position_port = `base_port`
                metadata_port = `base_port` + 1
                image_port = `base_port` + 2
                step_port = `base_port` + 5
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
        """
        super().__init__(
            environment_file,
            simulation_ip,
            own_ip,
            worker_id,
            base_port,
            scene_id,
            max_steps,
            step_rate,
            init_hook=init_hook
        )
        self.n_targets = n_targets
        self.success_dist = success_dist
        self.max_steps = max_steps
        self.restart_on_collision = restart_on_collision

        self.TransformMessage = StepWithTransform if self.step_mode else Transform

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

        return self.form_agent_observation(self.observe())

    def apply_action(self, action):
        """ Make agent take the specified action.

        Args:
            action (action_space): Make agent take `action`.
        """
        if action == 0:
            # move forward, a bit of a hack to accommodate thin colliders
            for _ in range(4):
                self.env.send(self.TransformMessage(0, 0.1, 0))
                time.sleep(0.02)  # so messages don't get dropped
            self.env.send(self.TransformMessage(0, 0.1, 0))  # don't need a final sleep call
        elif action == 1:
            self.env.send(self.TransformMessage(0, 0, 8))  # turn right
        elif action == 2:
            self.env.send(self.TransformMessage(0, 0, -8))  # turn left
        elif action != 3:
            raise ValueError(f"Unexpected action {action}")

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

        # compute agent's distance from targets
        agent_position = self._get_agent_position(agent_data.metadata)
        target_ids, target_position = self._get_target_id_and_positions(targets.metadata)

        # Agent can fall out of scenes
        if agent_position[1] < 1:
            reward = -1  # discourage falling out of windows, etc
            self.done = True
            return reward

        reward = -0.01  # small time penalty
        if target_position.shape[0] > 0:
            # only compare (x, z) coordinates
            agent_position = agent_position[np.newaxis, (0, 2)]
            target_position = target_position[:, (0, 2)]
            dists = np.linalg.norm(target_position - agent_position, axis=-1)

            # can we see the target?
            seg = agent_data.images[1]
            target_in_fov = np.all(seg == self.TARGET_COLOR, axis=-1)

            # if the agent is within `success_dist` of target, can see it,
            # and gives the `found` action, count as found
            if dists.min() < self.success_dist and target_in_fov.any() and action == 3:
                # if in `MULTIPLE` mode, remove found targets
                if self.hunt_mode.value == HuntMode.MULTIPLE.value:  # TODO: need to compare values?
                    found_targets = target_ids[dists < self.success_dist]
                    self.n_found_targets += len(found_targets)
                    reward += self.target_found_reward * len(found_targets)
                    self.env.request(RemoveObjectsRequest(ids=found_targets))

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

        return reward

    def _success_action(self):
        """ Simple indicator that the agent has achieved the goal. """
        for i in range(0, 360, 360 // 5):
            self.env.send(self.TransformMessage(0, 0, 360 // 5))
            time.sleep(0.1)

    def _collision(self, metadata):
        """ Check for collision with environment.

        Args:
            metadata (str): Metadata string.

        Returns:
            bool: True if agent has collided with the environment. Otherwise, false.
        """
        return (
            ET.fromstring(metadata).find("collision").attrib["status"].lower() == "true"
        )

    def _get_agent_position(self, agent_metadata):
        """ Get the agent's position from metadata.

        Args:
            agent_metadata (str): Metadata string.

        Returns:
            np.ndarray: shape (3,) containing the agents (x, y, z) position.
        """
        return (
            np.array(
                self._read_position(ET.fromstring(agent_metadata).find("position"))
            )
            .astype(np.float32)
            .reshape(-1)
        )

    def _get_agent_rotation(self, agent_metadata):
        """ Get the agent's rotation.

        Args:
            agent_metadata (str): Metadata string.

        Returns:
            np.ndarray: shape (3,) containing (z, x, y)
                euler angles.
        """
        root = ET.fromstring(agent_metadata)
        x = float(root.find('quaternion').attrib['x'])
        y = float(root.find('quaternion').attrib['y'])
        z = float(root.find('quaternion').attrib['z'])
        w = float(root.find('quaternion').attrib['w'])
        return Rotation((x, y, z, w)).as_euler('zxy')

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

    def _read_position(self, pos):
        """ Get (x, y, z) coordinates from metadata.

        Args:
            pos (str): XML element from metadata string.

        Returns:
            np.ndarray: shape (3, ), or (x, y, z) positions.
        """
        return np.array(
            [pos.attrib["x"], pos.attrib["y"], pos.attrib["z"]], dtype=np.float32
        )
