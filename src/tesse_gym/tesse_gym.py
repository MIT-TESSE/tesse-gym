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

import atexit
import subprocess
import numpy as np
from collections import namedtuple
from gym import Env as GymEnv, logger, spaces
from tesse.env import Env
from tesse.msgs import *
from .continuous_control import ContinuousController

import time


NetworkConfig = namedtuple("NetworkConfig",
                           ['simulation_ip', 'own_ip', 'position_port', 'metadata_port', 'image_port', 'step_port'],
                           defaults=('localhost', 'localhost', 9000, 9001, 9002, 9005))


def default_network_config(simulation_ip='localhost', own_ip='localhost', base_port=9000, worker_id=0, n_ports=6):
    return NetworkConfig(simulation_ip=simulation_ip,
                         own_ip=own_ip,
                         position_port=base_port + worker_id * n_ports,
                         metadata_port=base_port + worker_id * n_ports + 1,
                         image_port=base_port + worker_id * n_ports + 2,
                         step_port=base_port + worker_id * n_ports + 5,
                         )


class TesseGym(GymEnv):
    metadata = {"render.modes": ["rgb_array"]}
    reward_range = (-float("inf"), float("inf"))
    N_PORTS = 6
    DONE_WARNING = (
        "You are calling 'step()' even though this environment "
        "has already returned done = True. You should always call "
        "'reset()' once you receive 'done = True' -- any further "
        "steps are undefined behavior. "
    )
    shape = (240, 320, 3)
    hover_height = 0.5

    def __init__(
        self,
        environment_file: str,
        network_config: NetworkConfig = NetworkConfig(),
        scene_id: int = None,
        max_steps: int = 40,
        step_rate: int = -1,
        init_hook: callable = None,
        continuous_control: bool = False,
        launch_tesse: bool = True
    ):
        """
        Args:
            environment_file (str): Path to TESS executable.
            network_config (NetworkConfig): Network configuration parameters.
            scene_id (int): Scene to use.
            max_steps (int): Max steps per episode.
            step_rate (int): If specified, game time is fixed to
                `step_rate` FPS.
            init_hook (callable): Method to adjust any experiment specific parameters
                upon startup (e.g. camera parameters).
            continuous_control (bool): True to use a continuous controller to move the
                agent. False to use discrete transforms.
        """
        atexit.register(self.close)

        # launch Unity if in training mode
        # otherwise, assume Unity is already running (e.g. for Kimera)
        if launch_tesse:
            self.proc = subprocess.Popen(
                [
                    environment_file,
                    "--listen_port",
                    str(int(network_config.position_port)),
                    "--send_port",
                    str(int(network_config.position_port)),
                    "--set_resolution",
                    str(self.shape[1]),
                    str(self.shape[0]),
                ]
            )
        self.launch_tesse = launch_tesse

        # setup environment
        self.env = Env(
            simulation_ip=network_config.simulation_ip, #simulation_ip,
            own_ip=network_config.own_ip, # own_ip,
            position_port=network_config.position_port, # base_port + worker_id * self.N_PORTS,
            metadata_port=network_config.metadata_port, # 9007, #base_port + worker_id * self.N_PORTS + 1,
            image_port=network_config.image_port, # 9008, #base_port + worker_id * self.N_PORTS + 2,
            step_port=network_config.step_port, # base_port + worker_id * self.N_PORTS + 5,
        )

        if scene_id:
            time.sleep(10)  # wait for sim to initialize
            self.env.send(SceneRequest(scene_id))

        # if specified, set step mode parameters
        self.step_mode = False
        if step_rate > 0:
            self.env.request(SetFrameRate(step_rate))
            self.step_mode = True

        self.TransformMessage = StepWithTransform if self.step_mode else Transform

        # if specified, set continuous control
        self.continuous_control = continuous_control
        if self.continuous_control and step_rate < 1:
            raise ValueError(f"A step rate must be given to run the continuous controller")

        if self.continuous_control:
            self.continuous_controller = ContinuousController(self.env, framerate=step_rate)

        self.max_steps = max_steps
        self.done = False
        self.steps = 0
        self.env.request(SetHoverHeight(self.hover_height))

        #  any experiment specific settings go here
        if init_hook:
            init_hook(self)

    def advance_game_n_steps(self, n_steps):
        """ Advance game time by sending step forces of 0 to TESSE. """
        for i in range(n_steps):
            #  TODO look into making
            self.env.send(StepWithForce(0, 0, 0))  # move game time to update observation

    def transform(self, x, z, y):
        """ Apply desired transform to agent. If in continuous mode, the
        agent is moved via force commands. Otherwise, a discrete transform
        is applied.

        Args:
            x (float): desired x translation.
            z (float): desired z translation.
            y (float): Desired rotation (in degrees).
        """
        if self.continuous_control:
            self.continuous_controller.transform(x, z, np.deg2rad(y))
        else:
            self.env.send(self.TransformMessage(x, z, y))

    @property
    def observation_space(self):
        """ Space observed by the agent. """
        return spaces.Box(0, 255, dtype=np.uint8, shape=self.shape)

    def step(self, action):
        """ Take a training step consisting of an action, observation, and
        reward.

        Args:
            action (action_space): An action defined in `self.action_space`.

        Returns:
            (np.ndarray, float, bool, dict): Observation, reward, done, info.
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        if self.done:
            logger.warn(self.DONE_WARNING)

        self.apply_action(action)
        response = self.observe()
        reward, env_changed = self.compute_reward(response, action)

        if env_changed and not self.done:
            response = self.observe()

        return self.form_agent_observation(response), reward, self.done, {}

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
        return self.form_agent_observation(self.observe())

    def render(self, mode="rgb_array"):
        """ Get observation.

        Args:
            mode (str): Render mode.

        Returns:
            np.ndarray: Current observation.
         """
        return self.observe().images[0]

    def close(self):
        """ Kill simulation if running. """
        if self.launch_tesse:
            self.proc.kill()

    def form_agent_observation(self, scene_observation):
        """ Form agent's observation from a part
        of all of the information received from TESSE.
        This is useful if some information is required to compute
        a reward (e.g. segmentation for finding targets), but
        that information should not go to the agent.

        Args:
            scene_observation (DataResponse): tesse_interface
                `DataResponse` object.

        Returns:
            np.ndarray: Observation given to the agent.
        """
        return scene_observation.images[0]

    @property
    def action_space(self):
        """ Defines space of valid action. """
        raise NotImplementedError

    def apply_action(self, action):
        """ Make agent take the specified action.

        Args:
            action (action_space): Make agent take `action`.
        """
        raise NotImplementedError

    def compute_reward(self, observation, action):
        """ Compute the reward based on the agent's observation and action.

        Args:
            observation (DataResponse): Images and metadata used to
            compute the reward.
            action (action_space): Action taken by agent.

        Returns:
            float: Computed reward.
        """
        raise NotImplementedError
