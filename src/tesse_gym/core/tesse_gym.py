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

import atexit
import subprocess
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
from xml.etree.cElementTree import Element

import defusedxml.ElementTree as ET
import numpy as np
from gym import Env as GymEnv
from gym import logger, spaces
from scipy.spatial.transform import Rotation

from tesse.env import Env
from tesse.msgs import *

from .continuous_control import ContinuousController
from .utils import NetworkConfig, get_network_config, set_all_camera_params


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
    sim_query_timeout = 10

    def __init__(
        self,
        sim_path: Union[str, None],
        network_config: Optional[NetworkConfig] = get_network_config(),
        scene_id: Optional[int] = None,
        episode_length: Optional[int] = 400,
        step_rate: Optional[int] = -1,
        init_hook: Optional[Callable[["TesseGym"], None]] = set_all_camera_params,
        ground_truth_mode: Optional[bool] = True,
    ) -> None:
        """
        Args:
            sim_path (str): Path to simulator executable. If `None` is given, assume
                the simulator is running externally.
            network_config (NetworkConfig): Network configuration parameters.
            scene_id (int): Scene to use.
            episode_length (int): Max steps per episode.
            step_rate (int): If specified, game time is fixed to
                `step_rate` FPS.
            init_hook (callable): Method to adjust simulation upon startup
                (e.g. camera parameters). Note, this will only be run in the
                simulator is launched internally.
            ground_truth_mode (bool): Assumes gym is consuming ground truth data. Otherwise,
                assumes an external perception pipeline is running. In the latter mode, discrete
                steps will be translated to continuous control commands and observations will be
                explicitly synced with sim time.
        """
        atexit.register(self.close)

        # launch Unity if in training mode
        # otherwise, assume Unity is already running (e.g. for Kimera)
        self.launch_tesse = isinstance(sim_path, str) and sim_path is not ""
        if self.launch_tesse:
            self.proc = subprocess.Popen(
                [
                    sim_path,
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
        self.step_rate = step_rate
        if step_rate > 0:
            self.env.request(SetFrameRate(step_rate))
            self.step_mode = True

        self.TransformMessage = StepWithTransform if self.step_mode else Transform

        # if specified, set continuous control
        self.ground_truth_mode = ground_truth_mode
        if not self.ground_truth_mode and step_rate < 1:
            raise ValueError(
                f"A step rate must be given to run the continuous controller"
            )

        if not self.ground_truth_mode:
            self.continuous_controller = ContinuousController(
                env=self.env, framerate=step_rate
            )

        self.episode_length = episode_length
        self.done = False
        self.steps = 0

        self.env.request(SetHoverHeight(self.hover_height))
        self.env.send((ColliderRequest(1)))

        #  any experiment specific settings go here
        if init_hook and self.launch_tesse:
            init_hook(self)

        # track relative pose throughout episode
        # (x, z, yaw) pose from starting point in agent frame
        self.initial_pose = np.zeros((3,))
        self.initial_rotation = np.eye(2)
        self.relative_pose = np.zeros((3,))
        self._init_pose()

    def advance_game_time(self, n_steps: int) -> None:
        """ Advance game time in step mode by sending step forces of 0 to TESSE. """
        for i in range(n_steps):
            self.env.send(
                StepWithForce(0, 0, 0)
            )  # move game time to update observation

    def transform(self, x: float, z: float, y: float) -> None:
        """ Apply desired transform to agent. If in continuous mode, the
        agent is moved via force commands. Otherwise, a discrete transform
        is applied.

        Args:
            x (float): desired x translation.
            z (float): desired z translation.
            y (float): Desired rotation (in degrees).
        """
        if not self.ground_truth_mode:
            self.continuous_controller.transform(x, z, np.deg2rad(y))
        else:
            self.env.send(self.TransformMessage(x, z, y))

    @property
    def observation_space(self) -> spaces.Box:
        """ Space observed by the agent. """
        return spaces.Box(0, 255, dtype=np.uint8, shape=self.shape)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
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
        reward, reward_info = self.compute_reward(response, action)

        if reward_info["env_changed"] and not self.done:
            # environment changes will not advance game time
            # advance here so the perception server will be up to date
            self.advance_game_time(1)
            response = self.get_synced_observation()

        self._update_pose(response.metadata)

        return self.form_agent_observation(response), reward, self.done, reward_info

    def observe(self) -> DataResponse:
        """ Observe state. """
        cameras = [(Camera.RGB_LEFT, Compression.OFF, Channels.THREE)]
        return self.env.request(DataRequest(metadata=True, cameras=cameras))

    def reset(
        self, scene_id: Optional[int] = None, random_seed: Optional[int] = None
    ) -> np.ndarray:
        """ Reset environment and respawn agent.

        Args:
            scene_id (int): If given, change to this scene.
            random_seed (int): If give, set simulator random seed.

        Returns:
            Observation.
        """
        if random_seed:
            self.env.request(SetRandomSeed(random_seed))

        if scene_id:
            self.env.request(SceneRequest(scene_id))

        self.env.request(Respawn())
        self.done = False
        self.steps = 0
        self._init_pose()
        return self.form_agent_observation(self.observe())

    def render(self, mode: str = "rgb_array") -> np.ndarray:
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
        if not self.ground_truth_mode:
            self.continuous_controller.close()

    def get_synced_observation(self) -> DataResponse:
        """ Get observation synced with sim time.

        Compares current sim time to latest image message and queries
        new images until they are up to date.

        Returns:
            DataResponse
        """
        response = self.observe()
        if self.launch_tesse or self.ground_truth_mode:
            return response
        else:
            # Ensure observations are current with sim by comparing timestamps
            requery_limit = 10
            time_advance_frequency = 5
            for attempts in range(requery_limit):
                # heuristic to account for dropped messages
                if (attempts + 1) % time_advance_frequency == 0:
                    self.advance_game_time(1)

                sim_time = self.continuous_controller.get_current_time()
                observation_time = float(ET.fromstring(response.metadata).find("time").text)

                timediff = np.round(sim_time - observation_time, 2)

                # if observation is synced with sim time, break
                # otherwise, requery
                if timediff < 1 / self.step_rate:
                    break
                else:
                    response = self.observe()
        return response

    def form_agent_observation(self, scene_observation: DataResponse) -> np.ndarray:
        """ Create agent's observation from `DataResponse` message.

        Creates agent's observation from a part
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

    def compute_reward(
        self, observation: DataResponse, action: int
    ) -> Tuple[float, Dict[str, Any]]:
        """ Compute the reward based on the agent's observation and action.

        Args:
            observation (DataResponse): Images and metadata used to
            compute the reward.
            action (action_space): Action taken by agent.

        Returns:
            Tuple[float, Dict[str, Any]]: Computed reward and dictionary with task relevant information
        """
        raise NotImplementedError

    def get_pose(self) -> np.ndarray:
        """ Get agent pose relative to start location.

        Returns:
            np.ndarray: (3,) array containing (x, z, heading). Heading is in degrees from [-180, 180].
        """
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

    def _update_pose(self, metadata: str):
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
            self.relative_pose[2] = self.relative_pose[2] % (-1 * np.pi)

    @staticmethod
    def get_2d_rotation_mtrx(rad: float) -> np.ndarray:
        """ Get 2d rotation matrix.

        Args:
            rad (float): Angle in radians

        Returns:
            np.ndarray: Rotation matrix
                [[cos(rad) -sin(rad)]
                 [sin(rad)  cos(rad)]]
        """
        return np.array([[np.cos(rad), -1 * np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    def _get_agent_position(self, agent_metadata: str) -> np.ndarray:
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
    def _get_agent_rotation(agent_metadata: str, as_euler: bool = True) -> np.ndarray:
        """ Get the agent's rotation.

        Args:
            agent_metadata (str): Metadata string.
            as_euler (bool): True to return zxy euler angles.
                Otherwise, return quaternion.

        Returns:
            np.ndarray: shape (3,) array containing (z, x, y) euler angles.
        """
        root = ET.fromstring(agent_metadata)
        x = float(root.find("quaternion").attrib["x"])
        y = float(root.find("quaternion").attrib["y"])
        z = float(root.find("quaternion").attrib["z"])
        w = float(root.find("quaternion").attrib["w"])
        return Rotation((x, y, z, w)).as_euler("zxy") if as_euler else (x, y, z, w)

    @staticmethod
    def _read_position(pos: Element) -> np.ndarray:
        """ Get (x, y, z) coordinates from metadata.

        Args:
            pos (Element): XML element from metadata string.

        Returns:
            np.ndarray: shape (3, ), of (x, y, z) positions.
        """
        return np.array(
            [pos.attrib["x"], pos.attrib["y"], pos.attrib["z"]], dtype=np.float32
        )
