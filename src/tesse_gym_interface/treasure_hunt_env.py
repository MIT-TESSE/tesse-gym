from .tesse_env import TesseEnv
import defusedxml.ElementTree as ET
import numpy as np
import time
from gym import spaces
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

    @property
    def action_space(self):
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        return spaces.Box(0, 255, dtype=np.uint8, shape=self.shape)

    def set_n_targets(self, n_targets):
        self.n_targets = n_targets

    def set_success_dist(self, dist):
        self.success_dist = dist

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
            self.env.send(Transform(0, 0.5, 0))  # forward
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
            - Intermediate reward of distance from nearest target
             see:
             https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch_env.py#L53
             for similar idea.
        """
        targets = self.env.request(ObjectsRequest())
        agent_data = observation

        # compute agent's distance from targets
        agent_position = self._get_agent_position(agent_data.metadata)
        target_position = self._get_target_positions(targets.metadata)

        reward = 0.0
        if target_position.shape[0] > 0:
            dists = np.linalg.norm(target_position - agent_position, axis=-1)

            seg, depth = agent_data.images[1], agent_data.images[2]
            target_in_fov = np.all(seg == self.TARGET_COLOR, axis=-1)

            # if agent is within 1m of agent and can see the
            # target, count as found
            # otherwise, give intermediary reward
            if dists.min() < self.success_dist and target_in_fov.any() and action == 3:
                self._success_action()  # signal task was successful
                reward += 1
                self.done = True
            elif target_in_fov.any():
                depth_on_target = depth[target_in_fov] / 255.0
                intermediary_reward = max(0.01 * (1 - depth_on_target.min()), 0)
                reward += intermediary_reward

        self.steps += 1
        if self.steps > self.max_steps:
            self.done = True

        return reward

    def _get_agent_position(self, agent_metadata):
        """ Get the agent's position from metadata. """
        return (
            np.array(
                self._read_position(
                    list(ET.fromstring(agent_metadata).iter("position"))[0]
                )
            )
            .astype(np.float32)
            .reshape(1, -1)
        )

    def _get_target_positions(self, target_metadata):
        """ Get target positions from metadata. """
        return np.array(
            [
                self._read_position(list(o.iter("position"))[0])
                for o in ET.fromstring(target_metadata).findall("object")
            ]
        ).astype(np.float32)

    def _read_position(self, pos):
        """ Get (x, z) coordinates from metadata. """
        return pos.attrib["x"], pos.attrib["z"]
