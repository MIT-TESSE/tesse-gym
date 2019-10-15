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
)


class SpawnMethod(Enum):
    RANDOM = 0
    NEAR_AGENT = 1


class TreasureHuntEnv(TesseEnv):
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
        n_targets: int = 25,
        success_dist: int = 5,
        spawn_method: SpawnMethod = SpawnMethod.RANDOM,
        restart_on_collision = True
    ):
        super().__init__(
            environment_file,
            simulation_ip,
            own_ip,
            worker_id,
            base_port,
            scene_id,
            max_steps,
        )
        self.n_targets = n_targets
        self.success_dist = success_dist
        self.max_steps = max_steps
        self.spawn_method = spawn_method
        self.restart_on_collision = restart_on_collision

    @property
    def action_space(self):
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        return spaces.Box(0, 255, dtype=np.uint8, shape=self.shape)

    def observe(self):
        """ Observe state. """
        cameras = [
            (Camera.RGB_LEFT, Compression.OFF, Channels.THREE),
            (Camera.SEGMENTATION, Compression.OFF, Channels.THREE),
            (Camera.DEPTH, Compression.OFF, Channels.THREE),
        ]
        agent_data = self.env.request(DataRequest(metadata=True, cameras=cameras))
        return agent_data

    def reset(self):
        """ Reset the sim, respawn agent, and spawn targets. """
        self.done = False
        self.steps = 0

        if self.spawn_method == SpawnMethod.RANDOM:
            self._randomly_spawn_agent_and_targets()
        elif self.spawn_method == SpawnMethod.NEAR_AGENT:
            radius_range = (3, 5)
            angle_range = (170, 190)
            self._spawn_targets_near_agent(radius_range, angle_range)
        else:
            raise ValueError("Invalid spawn method")

        return self.observe().images[0]

    def _randomly_spawn_agent_and_targets(self):
        """ Randomly spawn agents and targets around the scene. """
        self.env.send(Respawn())
        self.env.request(RemoveObjectsRequest())

        for i in range(self.n_targets):
            self.env.request(
                SpawnObjectRequest(ObjectType.CUBE, ObjectSpawnMethod.RANDOM)
            )

    def _spawn_targets_near_agent(self, radius_range, angle_range):
        """ Randomly spawn agent then spawn targets within `radius_range`
        and `angle_range` of target. """
        self.env.request(Respawn())
        self.env.request(RemoveObjectsRequest())

        response = self.env.request(
            DataRequest(cameras=[(Camera.RGB_LEFT, Compression.OFF, Channels.THREE)])
        )
        x, y, z = self._get_agent_position(response.metadata)
        rot_x, rot_z, rot_y = self._get_agent_rotation(response.metadata)

        radii = self._sample_range(self.n_targets, *radius_range)
        angles = self._sample_angles(self.n_targets, *angle_range)
        orientation = [0.4619398, 0.1913417, 0.4619398, 0.7325378]

        for radius, angle in zip(radii, np.deg2rad(angles)):
            angle = (angle + rot_y) % (2 * np.pi)  # recenter rotation on agent
            response = self.env.request(
                SpawnObjectRequest(
                    ObjectType.CUBE,
                    ObjectSpawnMethod.USER,
                    x + radius * np.sin(angle),
                    y,
                    z + radius * np.cos(angle),
                    *orientation,
                )
            )

    def _sample_range(self, n_samples, low, high):
        """ Randomly sample value within `low` and `high`"""
        return np.random.random(n_samples) * (high - low) + low

    def _sample_angles(self, n_samples, low, high):
        """ Sample an angle, in degrees, between `low` and `high`. """
        high = high + 360 if high < low else high
        return (np.random.random(n_samples) * (high - low) + low) % 360

    def _apply_action(self, action):
        """ Make agent take the specified action. """
        if action == 0:
            # forward, a bit of a hack to accommodate thin colliders
            for _ in range(4):
                self.env.send(Transform(0, 0.1, 0))
                time.sleep(0.02)  # so messages don't get dropped
            self.env.send(Transform(0, 0.1, 0))  # don't need a final sleep call
        elif action == 1:
            self.env.send(Transform(0, 0, 8))  # turn right
        elif action == 2:
            self.env.send(Transform(0, 0, -8))  # turn left
        elif action != 3:
            raise ValueError(f"Unexpected action {action}")

    def _success_action(self):
        """ Simple indicator that the agent has achieved the goal. """
        for i in range(0, 360, 360 // 5):
            self.env.send(Transform(0, 0, 360 // 5))
            time.sleep(0.1)

    def _compute_reward(self, observation, action):
        """ Compute reward consisting of
            - Reward if the agent has completed its task
              of being within `success_dist` of a target in its FOV
               and has given the 'done' signal (action == 3).
            - Small time penalty
            - taken from https://arxiv.org/pdf/1609.05143.pdf
        """
        targets = self.env.request(ObjectsRequest())
        agent_data = observation

        # compute agent's distance from targets
        agent_position = self._get_agent_position(agent_data.metadata)
        target_position = self._get_target_positions(targets.metadata)

        # Agent can fall out of scenes
        # TODO fix this
        if agent_position[1] < 1:
            reward = 1  # discourage falling out of windows, etc
            self.done = True
            return reward

        reward = -0.01  # small time penalty
        if target_position.shape[0] > 0:
            # only compare (x, z) coords
            agent_position = agent_position[np.newaxis, (0, 2)]
            target_position = target_position[:, (0, 2)]
            dists = np.linalg.norm(target_position - agent_position, axis=-1)

            # can we see the target
            seg, depth = agent_data.images[1], agent_data.images[2]
            target_in_fov = np.all(seg == self.TARGET_COLOR, axis=-1)

            # if the agent is within `success_dist` of target, can see it,
            # and gives the `found` action, count as found
            if dists.min() < self.success_dist and target_in_fov.any() and action == 3:
                self._success_action()  # signal task was successful
                reward += 10
                self.done = True

        self.steps += 1
        if self.steps > self.max_steps:
            self.done = True

        if self.restart_on_collision and self._collision(agent_data.metadata):
            self.done = True

        return reward

    def _distance_from_target_reward(self, target_in_fov, depth):
        """ Give a reward based on distance from the closest target. """
        depth_on_target = depth[target_in_fov] / 255.0
        return max(0.01 * (1 - depth_on_target.min()), 0)

    def _collision(self, metadata):
        return (
            ET.fromstring(metadata).find("collision").attrib["status"].lower() == "true"
        )

    def _get_agent_position(self, agent_metadata):
        """ Get the agent's position from metadata. """
        return (
            np.array(
                self._read_position(ET.fromstring(agent_metadata).find("position"))
            )
            .astype(np.float32)
            .reshape(-1)
        )

    def _get_agent_rotation(self, agent_metadata):
        root = ET.fromstring(agent_metadata)
        x = float(root.find('quaternion').attrib['x'])
        y = float(root.find('quaternion').attrib['y'])
        z = float(root.find('quaternion').attrib['z'])
        w = float(root.find('quaternion').attrib['w'])
        return Rotation((x, y, z, w)).as_euler('zxy')

    def _get_target_positions(self, target_metadata):
        """ Get target positions from metadata. """
        return np.array(
            [
                self._read_position(o.find("position"))
                for o in ET.fromstring(target_metadata).findall("object")
            ]
        ).astype(np.float32)

    def _read_position(self, pos):
        """ Get (x, z) coordinates from metadata. """
        return np.array(
            [pos.attrib["x"], pos.attrib["y"], pos.attrib["z"]], dtype=np.float32
        )
