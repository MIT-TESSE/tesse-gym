import atexit
import subprocess
import numpy as np
from gym import Env as GymEnv, logger, spaces
from tesse.env import Env
from tesse.msgs import Camera, Compression, Channels, DataRequest, Respawn, SetFrameRate, SceneRequest

import time


class TesseEnv(GymEnv):
    N_PORTS = 4
    DONE_WARNING = (
        "You are calling 'step()' even though this environment "
        "has already returned done = True. You should always call "
        "'reset()' once you receive 'done = True' -- any further "
        "steps are undefined behavior. "
    )
    shape = (240, 320, 3)  # TODO make this adjustable?

    def __init__(
        self,
        environment_file: str,
        simulation_ip: str,
        own_ip: str,
        worker_id: int = 0,
        base_port: int = 9000,
        scene_id: int = None,
        max_steps: int = 40,
        step_rate: int = -1
    ):
        """
        Args:
            environment_file (str): Path to TESS executable.
            simulation_ip (str): TESS IP address.
            own_ip (int): Local IP address.
            worker_id (int): Simulation ID (for running as subprocess).
                Defaults to 0.
            base_port (int): Interface Base port.
            scene_id (int): Scene to use.
            max_steps (int): Max steps per episode
        """
        atexit.register(self.close)

        # Launch Unity
        self.proc = subprocess.Popen(
            [
                environment_file,
                "--listen_port",
                str(int(base_port + worker_id * self.N_PORTS)),
                "--send_port",
                str(int(base_port + worker_id * self.N_PORTS)),
                "--set_resolution",
                str(self.shape[1]),
                str(self.shape[0]),
            ]
        )

        # setup environment
        self.env = Env(
            simulation_ip=simulation_ip,
            own_ip=own_ip,
            position_port=base_port + worker_id * self.N_PORTS,
            metadata_port=base_port + worker_id * self.N_PORTS + 1,
            image_port=base_port + worker_id * self.N_PORTS + 2,
            step_port=base_port + worker_id * self.N_PORTS + 5,
        )

        if scene_id:
            time.sleep(10)  # wait for sim to initialize
            self.env.send(SceneRequest(scene_id))

        self.step_mode = False
        if step_rate > 0:
            self.env.request(SetFrameRate(step_rate))
            self.step_mode = True

        self.metadata = {"render.modes": ["rgb_array"]}
        self.reward_range = (-float("inf"), float("inf"))

        self.max_steps = max_steps
        self.done = False
        self.steps = 0

    @property
    def observation_space(self):
        """ Space observed by the agent. """
        return spaces.Box(0, 255, dtype=np.uint8, shape=self.shape)

    def step(self, action):
        """ Take a training step consisting of an action, observation, and
        reward.

        Args:
            action: An action defined in `self.action_space`.

        Returns:
            Observation, reward, done, info.
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        if self.done:
            logger.warn(self.DONE_WARNING)

        self._apply_action(action)
        response = self.observe()
        reward = self._compute_reward(response, action)

        return response.images[0], reward, self.done, {}

    def observe(self):
        """ Observe state. """
        cameras = [(Camera.RGB_LEFT, Compression.OFF, Channels.THREE)]
        return self.env.request(DataRequest(metadata=True, cameras=cameras))

    def reset(self):
        """ Reset environment and respawn agent.

        Returns:
            Observation.
        """
        self.done = False
        self.steps = 0
        self.env.send(Respawn())
        return self.observe().images[0]

    def render(self, mode="rgb_array"):
        """ Get observation. """
        return self.observe().images[0]

    def close(self):
        """ Kill simulation. """
        self.proc.kill()

    @property
    def action_space(self):
        """ Defines space of valid action. """
        raise NotImplementedError

    def _apply_action(self, action):
        """ Apply the given action to the sim. """
        raise NotImplementedError

    def _compute_reward(self, observation, action):
        """ Compute the reward based on the agent's observation and action. """
        raise NotImplementedError
