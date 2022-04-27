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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from xml.etree.cElementTree import Element

import defusedxml.ElementTree as ET
import numpy as np
import numpy.random as rnd
from gym import Env as GymEnv
from gym import logger, spaces
from scipy.spatial.transform import Rotation

from tesse.env import Env
from tesse.msgs import *
from tesse_gym.actions.discrete import ActionMapper
from tesse_gym.observers.observer import EnvObserver, Observer

from .continuous_control import ContinuousController
from .logging import TESSEVideoWriter
from .observations import ObservationConfig, setup_observations
from .utils import (
    NetworkConfig,
    TesseConnectionError,
    get_network_config,
    response_nonetype_check,
    set_all_camera_params,
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
    sim_query_timeout = 10

    def __init__(
        self,
        sim_path: Union[str, None],
        observers: Union[Dict[str, Observer], Observer],
        action_mapper: ActionMapper,
        network_config: Optional[NetworkConfig] = get_network_config(),
        scene_id: Optional[Union[int, List[int]]] = None,
        episode_length: Optional[int] = 400,
        step_rate: Optional[int] = -1,
        init_function: Optional[Callable[["TesseGym"], None]] = set_all_camera_params,
        ground_truth_mode: Optional[bool] = True,
        observation_config=ObservationConfig(),
        video_log_path: str = None,
        video_save_freq: int = 5,
        video_writer_type: TESSEVideoWriter = TESSEVideoWriter,
        no_collisions: bool = False,
        query_image_data: bool = True,
        eval_config: Optional[List[Tuple[int, int]]] = None,
        eval_env: Optional[bool] = False,
    ) -> None:
        """
        Args:
            sim_path (str): Path to simulator executable. If `None` is given, assume
                the simulator is running externally.
            network_config (NetworkConfig): Network configuration parameters.
            scene_id (int): Scene to use. If a list is given, scenes will be randomly
                chosen upon reset.
            episode_length (int): Max steps per episode.
            step_rate (int): If specified, game time is fixed to
                `step_rate` FPS.
            init_function (callable): Method to adjust simulation upon startup
                (e.g. camera parameters). Note, this will only be run in the
                simulator is launched internally.
            ground_truth_mode (bool): Assumes gym is consuming ground truth data.
                Otherwise, assumes an external perception pipeline is running.
                In the latter mode, discrete steps will be translated to continuous
                control commands and observations will be explicitly synced with sim time.
            observation_modalities: (Optional[Tuple[Camera]]): Input modalities to be used.
                Defaults to (RGB_LEFT, SEGMENTATION, DEPTH).
            video_log_path (str): Write episode videos to this directory.
            video_writer_type (TesseVideoWriter): Customizable video writer class.
            no_collisiosn (bool): Disable scene colliders.
            query_image_data (bool): True to query image data. Otherwise, use zero images.
            eval_config (Optional[List[Tuple[int, int]]]): Pairs of [scene, random seed]
                to cycle through during evaluation. Episodes are initialized randomly if
                no config is given.
            eval_env (Optional[bool]): True if environment is used for evaluation. If
                True, `env_config` parameters will be used to initialize episodes.
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
        self.action_mapper = action_mapper
        self.action_mapper.register_env(self.env)

        self.scene_id = scene_id
        self.current_scene = scene_id if isinstance(scene_id, int) else None
        self._adjust_scene()

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
        self.env.send(ColliderRequest(0 if no_collisions else 1))

        # optionally adjust parameters on startup
        if init_function and self.launch_tesse:
            init_function(self)

        # track relative pose throughout episode
        # (x, z, yaw) pose from starting point in agent frame
        self.initial_pose = np.zeros((3,))
        self.world_to_spawn_rot = np.eye(2)
        self.relative_pose = np.zeros((3,))
        self.episode_trajectory = np.empty((0, 3))

        # setup observation
        self.observation_modalities, self._observation_space = setup_observations(
            observation_config
        )
        self.observer = EnvObserver(observers)

        if video_log_path:
            self.video_writer = video_writer_type(
                video_log_path, self.env, gym=self, save_freq=video_save_freq
            )
        else:
            self.video_writer = None

        self.no_collisions = no_collisions
        self.current_response = None
        self.query_image_data = query_image_data
        self.n_collisions = 0

        self.eval_config = eval_config
        self.eval_env = eval_env
        self.eval_itr = 0

        self._init_visited_space_tracking()

    def get_observers(self) -> Dict[str, Observer]:
        return self.observer.observers

    def _adjust_scene(self):
        """Set scene as determined by `self.scene_id`."""
        if self.scene_id is None:
            return
        elif isinstance(self.scene_id, int):
            self.env.request(SceneRequest(self.scene_id))
        elif isinstance(self.scene_id, list):
            scene = rnd.choice(self.scene_id)
            self.current_scene = scene
            self.env.request(SceneRequest(scene))

    def advance_game_time(self, n_steps: int) -> None:
        """Advance game time in step mode by sending step forces of 0 to
        TESSE."""
        for i in range(n_steps):
            self.env.send(
                StepWithForce(0, 0, 0)
            )  # move game time to update observation

    @property
    def observation_space(self) -> Union[spaces.Dict, spaces.Box]:
        """Space observed by the agent."""
        return self.observer.observation_space

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a training step consisting of an action, observation, and
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

        self.action_mapper.apply_action(action)
        response = self.observe()
        self.current_response = response
        self._update_pose(response.metadata)
        self._update_visited_space()
        reward, reward_info = self.compute_reward(response, action)

        if reward_info["env_changed"] and not self.done:
            # environment changes will not advance game time
            # advance here so the perception server will be up to date
            if not self.ground_truth_mode:
                self.advance_game_time(1)
            response = self.get_synced_observation()

        observation = self.form_observation(response)

        if self.video_writer != None:
            self.video_writer.step()

        self.steps += 1
        if self.steps >= self.episode_length:
            self.done = True

        return observation, reward, self.done, reward_info

    def observe(self) -> DataResponse:
        """Observe state."""
        if self.query_image_data:
            return self._data_request(
                DataRequest(metadata=True, cameras=self.observation_modalities)
            )
        # if we don't need images, use 0 valued images to reduce latency
        else:
            response = self._data_request(MetadataRequest())
            response.images.append(np.zeros((120, 160, 3)))
            response.images.append(np.zeros((120, 160, 3)))
            response.images.append(np.zeros((120, 160)))
        return response

    def reset(
        self, scene_id: Optional[int] = None, random_seed: Optional[int] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Reset environment and respawn agent.

        Args:
            scene_id (int): If given, change to this scene.
            random_seed (int): If give, set simulator random seed.

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]: Observation.
        """
        if self.video_writer != None:
            self.video_writer.reset()

        if self.eval_env:
            scene_id, random_seed = self.eval_config[
                self.eval_itr % len(self.eval_config)
            ]
            self.eval_itr += 1
            print(
                f"Eval mode: {self.eval_env}, iteration: {self.eval_itr}"
                f" setting scene to {scene_id} and random seed "
                f"to {random_seed}"
            )

        if random_seed:
            self.env.request(SetRandomSeed(random_seed))

        if scene_id:
            self.env.request(SceneRequest(scene_id))
            self.current_scene = scene_id
        elif isinstance(self.scene_id, list):
            self._adjust_scene()  # randomly choose a scene

        self.env.request(Respawn())
        self.done = False
        self.steps = 0
        self.episode_trajectory = np.empty((0, 3))
        self.n_collisions = 0

        self.observer.reset()

        observation = self.get_synced_observation()
        self.current_response = observation

        self._init_pose(observation.metadata)
        self._reset_visited_space()

        return self.form_observation(observation)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Get observation.

        Args:
            mode (str): Render mode.

        Returns:
            np.ndarray: Current observation.
        """
        return self.observe().images[0]

    def close(self):
        """Kill simulation if running."""
        if self.launch_tesse:
            self.proc.kill()
        if not self.ground_truth_mode:
            self.continuous_controller.close()

    def get_synced_observation(self) -> DataResponse:
        """Get observation synced with sim time.

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
                observation_time = float(
                    ET.fromstring(response.metadata).find("time").text
                )
                timediff = np.round(sim_time - observation_time, 2)

                # if observation is synced with sim time, break otherwise, requery
                if timediff < 1 / self.step_rate:
                    break

                response = self.observe()
        return response

    def form_observation(self, tesse_data: DataResponse) -> np.ndarray:
        """Create the agent's observation from a TESSE data response.

        Args:
            tesse_data (DataResponse): TESSE DataResponse object containing
                data specified in `self.observation_modalities`.

        Returns:
            np.ndarray: The agent's observation consisting of data
                specified in `self.observation_modalities`.

        Notes:
            If pose is included, the observation will be given
            as a vector. Otherwise, the observation will preserve
            the image shape.
        """
        pose = self.get_pose().reshape((3))
        env_info = {
            "relative_pose": pose,
            "initial_pose": self.initial_pose,
            "tesse_data": tesse_data,
            "episode_trajectory": self.episode_trajectory,
            "scene": self.current_scene,
        }
        return self.observer.observe(env_info)

    @property
    def action_space(self) -> spaces.Space:
        """Defines space of valid action."""
        return self.action_mapper.action_space

    def apply_action(self, action):
        """Make agent take the specified action.

        Args:
            action (action_space): Make agent take `action`.
        """
        raise NotImplementedError

    def compute_reward(
        self, observation: DataResponse, action: int
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute the reward based on the agent's observation and action.

        Args:
            observation (DataResponse): Images and metadata used to
            compute the reward.
            action (action_space): Action taken by agent.

        Returns:
            Tuple[float, Dict[str, Any]]: Computed reward and dictionary
                with task relevant information
        """
        raise NotImplementedError

    def get_pose(self) -> np.ndarray:
        """Get agent pose relative to start location.

        Returns:
            np.ndarray: (3,) array containing (x, z, heading).
                Heading is in degrees from [-180, 180].
        """
        return self.relative_pose

    def _data_request(self, request_type: DataRequest, n_attempts: int = 20):
        """Make a data request while handling potential network limitations.

        If during the request, a `TesseConnectionError` is throw, this assumes
        there is a spurrious bandwidth issue and re-requests `n_attempts` times.
        If, after `n_attempts`, data cannot be recieved, a `TesseConnectionError`
        is thrown.

        Args:
            request_type (DataRequest): Data request type.
            n_attempts (int): Number of times to request data from TESSE.
                Default is 20.

        Returns:
            DataResponse: Response from TESSE.

        Raises:
            TesseConnectionError: Raised if data cannot be read from TESSE.
        """
        for _ in range(n_attempts):
            try:
                return response_nonetype_check(self.env.request(request_type))
            except TesseConnectionError:
                pass

        raise TesseConnectionError()

    def _collision(self, metadata: str) -> bool:
        """Check for collision with environment.

        Args:
            metadata (str): Metadata string.

        Returns:
            bool: True if agent has collided with the environment. Otherwise, false.
        """
        had_collision = (
            ET.fromstring(metadata).find("collision").attrib["status"].lower() == "true"
        )
        if had_collision:
            self.n_collisions += 1
        return had_collision

    def _init_pose(self, metadata: str = None) -> None:
        """Initialize agent's starting pose"""
        if metadata is None:
            metadata = self._data_request(MetadataRequest()).metadata

        position = self._get_agent_position(metadata)
        rotation = self._get_agent_rotation(metadata)

        # initialize position in in agent frame
        initial_yaw = rotation[2]

        # rotation for global -> relative to spawn point
        self.world_to_spawn_rot = self.get_2d_rotation_mtrx(-1 * initial_yaw)
        initial_position = np.array([position[0], position[2]])
        self.initial_pose = np.array([*initial_position, rotation[2]])
        self.episode_trajectory = np.append(
            self.episode_trajectory, self.initial_pose.reshape(1, 3), axis=0
        )

        self.relative_pose = np.zeros((3,))

    def _update_pose(self, metadata: str):
        """Update current pose.

        Args:
            metadata (str): TESSE metadata message.
        """
        position = self._get_agent_position(metadata)
        rotation = self._get_agent_rotation(metadata)

        x = position[0]
        y = position[1]
        z = position[2]
        yaw = rotation[2]

        # record trajectory in global frame
        self.episode_trajectory = np.append(
            self.episode_trajectory, np.array([[x, z, yaw]]), axis=0
        )

        # Restart episode if agent falls out of scene.
        # Workaround for what  seems to be a
        # bug in the spawn points
        if y < -0.5:
            self.done = True

        # Get pose relative to spawn point
        position = np.array([x, z])
        position -= self.initial_pose[:2]
        position = np.matmul(self.world_to_spawn_rot, position)
        yaw -= self.initial_pose[2]
        self.relative_pose = np.array([position[0], position[1], yaw])

        # keep yaw in range [-pi, pi]
        if self.relative_pose[2] < -np.pi:
            self.relative_pose[2] = self.relative_pose[2] % np.pi
        elif self.relative_pose[2] > np.pi:
            self.relative_pose[2] = self.relative_pose[2] % (-1 * np.pi)

        # TODO(ZR) feature specific - reposition to a height of 0.5m
        x_rot, y_rot, z_rot, w_rot = self._get_agent_rotation(metadata, as_euler=False)
        self.env.send(Reposition(x, self.hover_height, z, x_rot, y_rot, z_rot, w_rot))

    @staticmethod
    def get_2d_rotation_mtrx(rad: float) -> np.ndarray:
        """Get 2d rotation matrix.

        Args:
            rad (float): Angle in radians

        Returns:
            np.ndarray: Rotation matrix
                [[cos(rad) -sin(rad)]
                 [sin(rad)  cos(rad)]]
        """
        return np.array([[np.cos(rad), -1 * np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    def _get_agent_position(self, agent_metadata: str) -> np.ndarray:
        """Get the agent's position from metadata.

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

    def _init_visited_space_tracking(self, cell_size: Optional[int] = 1):
        """Record explored space.

        The world is binned into square cell with side length `cell_size`.
        Each cell visited during an episode is logged."""
        self.cell_size = cell_size  # m
        self.visited_cells = []

    def _reset_visited_space(self):
        self.visited_cells = []

    def _update_visited_space(self):
        # center grid around agent's initial position
        visited_cell = (
            int((self.relative_pose[0] - self.cell_size / 2) // self.cell_size),
            int((self.relative_pose[1] - self.cell_size / 2) // self.cell_size),
        )

        if visited_cell not in self.visited_cells:
            self.visited_cells.append(visited_cell)

    @staticmethod
    def _get_agent_rotation(agent_metadata: str, as_euler: bool = True) -> np.ndarray:
        """Get the agent's rotation.

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
        """Get (x, y, z) coordinates from metadata.

        Args:
            pos (Element): XML element from metadata string.

        Returns:
            np.ndarray: shape (3, ), of (x, y, z) positions.
        """
        return np.array(
            [pos.attrib["x"], pos.attrib["y"], pos.attrib["z"]], dtype=np.float32
        )
