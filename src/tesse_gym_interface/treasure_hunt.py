from .tesse_env import TesseEnv
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


class TreasureHunt(TesseEnv):
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
        n_targets: int = 25,
        success_dist: int = 5,
        restart_on_collision: bool = True,
        init_hook: callable = None,
        hunt_mode: HuntMode = HuntMode.MULTIPLE,
        target_found_reward: int = 10
    ):
        """ Initialize the TESSE treasure hunt environment.

            Args:
                environment_file (str): path to TESSE executable.
                simulation_ip (str): TESSE IP address
                own_ip (str): Interface IP address.
                worker_id: subprocess worker id.
                base_port:
                scene_id: int of scene id to load.
                max_steps: Number of steps in the episode.
                step_rate:
                n_targets: Number of targets to spawn in the scene.
                success_dist: Distance target must be from agent to
                    be considered found. Target must also be in agent's
                    field of view.
                init_hook: Method to adjust any experiment specific parameters
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
            step_rate
        )
        self.n_targets = n_targets
        self.success_dist = success_dist
        self.max_steps = max_steps
        self.restart_on_collision = restart_on_collision

        self.TransformMessage = StepWithTransform if self.step_mode else Transform

        #  any experiment specific settings go here
        if init_hook:
            init_hook(self)

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
            A `DataResponse` object. """
        cameras = [
            (Camera.RGB_LEFT, Compression.OFF, Channels.THREE),
            (Camera.SEGMENTATION, Compression.OFF, Channels.THREE),
        ]
        agent_data = self.env.request(DataRequest(metadata=True, cameras=cameras))
        return agent_data

    def reset(self):
        """ Reset the sim, randomly respawn agent and targets.

        Returns:
            Observed image. """
        self.done = False
        self.steps = 0
        self.n_found_targets = 0

        self.env.send(Respawn())
        self.env.request(RemoveObjectsRequest())
        for i in range(self.n_targets):
            self.env.request(
                SpawnObjectRequest(ObjectType.CUBE, ObjectSpawnMethod.RANDOM)
            )

        return self.observe().images[0]

    def _apply_action(self, action):
        """ Make agent take the specified action. """
        if action == 0:
            # forward, a bit of a hack to accommodate thin colliders
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

    def _success_action(self):
        """ Simple indicator that the agent has achieved the goal. """
        for i in range(0, 360, 360 // 5):
            self.env.send(self.TransformMessage(0, 0, 360 // 5))
            time.sleep(0.1)

    def _compute_reward(self, observation, action):
        """ Compute reward consisting of
            - Reward if the agent has completed its task
              of being within `success_dist` of a target in its FOV
               and has given the 'done' signal (action == 3).
            - Small time penalty
            - taken from https://arxiv.org/pdf/1609.05143.pdf

        Args:
            observation: `DataResponse` object containing images
                and metadata used to compute the reward.
            action: action  taken by agent.

        Returns:
            Computed reward.
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

    def _collision(self, metadata):
        """ Check for collision with environment.

        Args:
            metadata: metadata string.

        Returns:
            boolean, true if agent has collided with the environment. Otherwise, false.
        """
        return (
            ET.fromstring(metadata).find("collision").attrib["status"].lower() == "true"
        )

    def _get_agent_position(self, agent_metadata):
        """ Get the agent's position from metadata.

        Args:
            agent_metadata: metadata string.

        Returns:
            ndarray of shape (3,) containing the agents (x, y, z) position.
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
            agent_metadata: metadata string.

        Returns:
            ndarray of shape (3,) containing (z, x, y)
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
            target_metadata: metadata string.
        Returns:
            ndarray, shape (n, 3), of (x, y, z) positions for the
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
            pos: XML element from metadata string.

        Returns:
            ndarray, shape (3, ), or (x, y, z) positions.
        """
        return np.array(
            [pos.attrib["x"], pos.attrib["y"], pos.attrib["z"]], dtype=np.float32
        )
