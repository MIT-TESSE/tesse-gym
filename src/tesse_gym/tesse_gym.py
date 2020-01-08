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
import defusedxml.ElementTree as ET
from scipy.spatial.transform import Rotation
from gym import Env as GymEnv, logger, spaces
from tesse.env import Env
from tesse.msgs import *
from .continuous_control import ContinuousController

import time


NetworkConfig = namedtuple("NetworkConfig",
                           ['simulation_ip', 'own_ip', 'position_port', 'metadata_port', 'image_port', 'step_port'],
                           defaults=('localhost', 'localhost', 9000, 9001, 9002, 9005))


def get_network_config(simulation_ip='localhost', own_ip='localhost', base_port=9000, worker_id=0, n_ports=6):
    """ Get a TESSE network configuration instance.

    Args:
        simulation_ip (str): TESSE IP address.
        own_ip (str): Local IP address.
        base_port (int): Starting connection port. It is assumed the rest of the ports
            follow sequentially.
        worker_id (int): Worker ID of this Gym instance. Ports are staggered by ID.
        n_ports (int): Number of ports allocated to each TESSE instance.

    Returns:
        NetworkConfig: NetworkConfig object.
    """
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
        network_config: NetworkConfig = get_network_config(),
        scene_id: int = None,
        max_steps: int = 300,
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
                agent. False to use discrete transforms..
            launch_tesse (bool): True to start tesse instance. Otherwise, assume another
                instance is running.
        """
        atexit.register(self.close)

        # launch Unity if in training mode
        # otherwise, assume Unity is already running (e.g. for Kimera)
        self.launch_tesse = launch_tesse
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

            time.sleep(10)  # wait for sim to initialize

        # setup environment
        self.env = Env(
            simulation_ip=network_config.simulation_ip,
            own_ip=network_config.own_ip,
            position_port=network_config.position_port,
            metadata_port=network_config.metadata_port,
            image_port=network_config.image_port,
            step_port=network_config.step_port,
        )

        if scene_id is not None:
            self.env.request(SceneRequest(scene_id))

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
        self.env.send((ColliderRequest(1)))

        #  any experiment specific settings go here
        if init_hook:
            init_hook(self)

        # track relative pose throughout episode
        self.initial_pose = np.zeros((3,))  # (x, z, yaw) pose from starting point in agent frame
        self.initial_rotation = np.eye(2)
        self.relative_pose = np.zeros((3,))

    def advance_game_time(self, n_steps):
        """ Advance game time in step mode by sending step forces of 0 to TESSE. """
        for i in range(n_steps):
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

        self._update_pose(response.metadata)

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
        self._init_pose()
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

    def get_pose(self):
        """ Get agent pose relative to start location. """
        return self.relative_pose

    def _init_pose(self):
        """ Initialize agent's starting pose """
        metadata = self.env.request(MetadataRequest()).metadata
        position = self._get_agent_position(metadata)
        rotation = self._get_agent_rotation(metadata)

        # initialize position in in agent frame
        initial_yaw = rotation[2]
        self.initial_rotation = self.get_2d_rotation_mtrx(initial_yaw)
        initial_position = np.array([position[0], position[2]])
        initial_position = np.matmul(self.initial_rotation, initial_position)
        self.initial_pose = np.array([*initial_position, rotation[2]])

        self.relative_pose = np.zeros((3,))

    def _update_pose(self, metadata):
        """ Update current pose.

        Args:
            metadata (str): TESSE metadata message.
        """
        position = self._get_agent_position(metadata)
        rotation = self._get_agent_rotation(metadata)

        x = position[0]
        z = position[2]
        yaw = rotation[2]

        # Get pose from start in agent frame
        position = np.array([x, z])
        position = np.matmul(self.initial_rotation, position)
        position -= self.initial_pose[:2]
        yaw -= self.initial_pose[2]
        self.relative_pose = np.array([position[0], position[1], yaw])

        # keep yaw in range [-pi, pi]
        if self.relative_pose[2] < -np.pi:
            self.relative_pose[2] = self.relative_pose[2] % np.pi
        elif self.relative_pose[2] > np.pi:
            self.relative_pose[2] = self.relative_pose[2] % (-1*np.pi)

    @staticmethod
    def get_2d_rotation_mtrx(rad):
        """ Get 2d rotation matrix.

        Args:
            rad (float): Angle in radians

        Returns:
            np.ndarray: Rotation matrix
                [[cos(rad) -sin(rad)]
                 [sin(rad)  cos(rad)]]
        """
        return np.array([[np.cos(rad), -1*np.sin(rad)],
                         [np.sin(rad),    np.cos(rad)]])

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

    @staticmethod
    def _get_agent_rotation(agent_metadata, as_euler=True):
        """ Get the agent's rotation.

        Args:
            agent_metadata (str): Metadata string.
            as_euler (bool): True to return zxy euler angles.
                Otherwise, return quaternion.

        Returns:
            np.ndarray: shape (3,) containing (z, x, y)
                euler angles.
        """
        root = ET.fromstring(agent_metadata)
        x = float(root.find('quaternion').attrib['x'])
        y = float(root.find('quaternion').attrib['y'])
        z = float(root.find('quaternion').attrib['z'])
        w = float(root.find('quaternion').attrib['w'])
        return Rotation((x, y, z, w)).as_euler('zxy') if as_euler else (x, y, z, w)

    @staticmethod
    def _read_position(pos):
        """ Get (x, y, z) coordinates from metadata.

        Args:
            pos (ET.Element): XML element from metadata string.

        Returns:
            np.ndarray: shape (3, ), or (x, y, z) positions.
        """
        return np.array(
            [pos.attrib["x"], pos.attrib["y"], pos.attrib["z"]], dtype=np.float32
        )
