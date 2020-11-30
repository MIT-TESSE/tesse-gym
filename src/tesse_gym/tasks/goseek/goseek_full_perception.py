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

from typing import Callable, Optional, Tuple

import numpy as np
from gym import spaces

from tesse.msgs import Camera, Channels, Compression, DataRequest, DataResponse
from tesse_gym.core.tesse_gym import TesseGym
from tesse_gym.core.utils import NetworkConfig, set_all_camera_params
from tesse_gym.tasks.goseek.goseek import GoSeek


class GoSeekFullPerception(GoSeek):
    """ Define a custom TESSE gym environment to provide RGB, depth, segmentation, and pose data.
    """

    # Unity scales depth in range [0, `FAR_CLIP_FIELD`] to [0, 1].
    # Values above `FAR_CLIP_FIELD` are, as the name implies, clipped.
    # Multiply depth by `FAR_CLIP_FIELD` to recover values in meters.
    FAR_CLIP_FIELD = 50
    N_CLASSES = 11  # Number of classes used for GOSEEK

    # Unity assigns areas without a material (e.g. outside windows) the maximum possible
    # image value. For convenience, this will be changed to the wall class.
    WALL_CLS = 2

    def __init__(
        self,
        build_path: str,
        network_config: Optional[NetworkConfig] = NetworkConfig(),
        scene_id: Optional[int] = None,
        episode_length: Optional[int] = 400,
        step_rate: Optional[int] = 20,
        n_targets: Optional[int] = 30,
        success_dist: Optional[float] = 2,
        restart_on_collision: Optional[bool] = False,
        init_hook: Optional[Callable[[TesseGym], None]] = set_all_camera_params,
        target_found_reward: Optional[int] = 1,
        ground_truth_mode: Optional[bool] = True,
        n_target_types: Optional[int] = 5,
        collision_reward: Optional[float] = 0,
        false_positive_reward: Optional[float] = 0,
        observation_modalities: Optional[Tuple[Camera]] = (
            Camera.RGB_LEFT,
            Camera.SEGMENTATION,
            Camera.DEPTH,
        ),
    ):
        """ Initialize the TESSE treasure hunt environment.

        Args:
            build_path (str): Path to TESSE executable.
            network_config (NetworkConfig): Network configuration parameters.
            scene_id (int): Scene id to load.
            episode_length (int): Maximum number of steps in the episode.
            step_rate (int): If specified, game time is fixed to
                `step_rate` FPS.
            n_targets (int): Number of targets to spawn in the scene.
            success_dist (float): Distance target must be from agent to
                be considered found. Target must also be in agent's
                field of view.
            init_hook (callable): Method to adjust any experiment specific parameters
                upon startup (e.g. camera parameters).
            ground_truth_mode (bool): Assumes gym is consuming ground truth data. Otherwise,
                assumes an external perception pipeline is running. In the latter mode, discrete
                steps will be translated to continuous control commands and observations will be
                explicitly synced with sim time.
            n_target_types (int): Number of target types available to spawn. GOSEEK challenge 
                has 5 target types by default. 
            collision_reward: (int): Added to total step reward upon collision. Default is 0.
            false_positive_reward (int): Added tot total step reward when agent incorrectly 
                declares a target found (action 3). Default is 0.
            observation_modalities: (Optional[Tuple[Camera]]): Input modalities to be used.
                Defaults to (RGB_LEFT, SEGMENTATION, DEPTH).
        """
        super().__init__(
            build_path,
            network_config,
            scene_id,
            episode_length,
            step_rate,
            init_hook=init_hook,
            ground_truth_mode=ground_truth_mode,
            n_targets=n_targets,
            success_dist=success_dist,
            restart_on_collision=restart_on_collision,
            target_found_reward=target_found_reward,
            n_target_types=n_target_types,
            collision_reward=collision_reward,
            false_positive_reward=false_positive_reward, 
            observation_modalities=observation_modalities,
        )
        assert np.alltrue(
            [isinstance(camera, Camera) for camera in observation_modalities]
        )
        self.observation_modalities = [
            (camera, Compression.OFF, Channels.THREE)
            for camera in observation_modalities
        ]
        self._observation_space = self._get_observation_space()

    def _get_observation_space(self) -> spaces.Box:
        """ TODO(ZR) Docs """
        n_channels = 0  # total image channels
        for camera in self.observation_modalities:
            if camera[0] in (Camera.RGB_RIGHT, Camera.RGB_LEFT):
                n_channels += 3
            elif camera[0] == Camera.SEGMENTATION:
                n_channels += 1
            elif camera[0] == Camera.DEPTH:
                n_channels += 1
            else:
                raise ValueError(f"Unrecognized camera: {camera[0]}")
        return spaces.Box(-np.Inf, np.Inf, shape=(240 * 320 * n_channels + 3,))

    @property
    def observation_space(self) -> spaces.Box:
        """ Define an observation space for RGB, depth, segmentation, and pose.

        Because Stables Baselines (the baseline PPO library) does not support dictionary spaces,
        the observation images and pose vector will be combined into a vector. The RGB image
        is of shape (240, 320, 3), depth and segmentation are both (240, 320), ose is (3,), thus
        the total shape is (240 * 320 * 5 + 3).
        """
        return self._observation_space

    def form_agent_observation(self, tesse_data: DataResponse) -> np.ndarray:
        """ Create the agent's observation from a TESSE data response.

        Args:
            tesse_data (DataResponse): TESSE DataResponse object containing
                RGB, depth, segmentation, and pose.

        Returns:
            np.ndarray: The agent's observation consisting of flatted RGB,
                segmentation, and depth images concatenated with the relative
                pose vector. To recover images and pose, see `decode_observations` below.
        """
        observation_imgs = []
        for i, camera_info in enumerate(self.observation_modalities):
            if camera_info[0] == Camera.RGB_LEFT:
                observation_imgs.append(tesse_data.images[i] / 255.0)
            elif camera_info[0] == Camera.SEGMENTATION:
                # get segmentation as one-hot encoding
                seg = tesse_data.images[i][..., 0].copy()
                seg[seg > (self.N_CLASSES - 1)] = self.WALL_CLS  # See WALL_CLS comment
                seg = seg[..., np.newaxis] / (self.N_CLASSES - 1)
                observation_imgs.append(seg)
            elif camera_info[0] == Camera.DEPTH:
                observation_imgs.append(tesse_data.images[i][..., np.newaxis])

        observation = np.concatenate(observation_imgs, axis=-1).reshape(-1)
        pose = self.get_pose().reshape((3))
        return np.concatenate((observation, pose))

    def observe(self) -> DataResponse:
        """ Get observation data from TESSE.

        Returns:
            DataResponse: TESSE DataResponse object.
        """
        return self._data_request(
            DataRequest(metadata=True, cameras=self.observation_modalities)
        )


def decode_observations(
    observation: np.ndarray, img_shape: Tuple[int, int, int, int] = (-1, 240, 320, 5)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Decode observation vector into images and poses.

    Args:
        observation (np.ndarray): Shape (N,) observation array of flattened
            images concatenated with a pose vector. Thus, N is equal to N*H*W*C + N*3.
        img_shape (Tuple[int, int, int, int]): Shapes of all observed images stacked across
            the channel dimension, resulting in a shape of (N, H, W, C).
             Default value is (-1, 240, 320, 5).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays with the following information
            - RGB image(s) of shape (N, H, W, 3)
            - Segmentation image(s) of shape (N, H, W), in range [0, C) where C is the number of classes.
            - Depth image(s) of shape (N, H, W), in range [0, 1]. To get true depth, multiply by the
                Unity far clipping plane (default 50).
            - Pose array of shape (N, 3) containing (x, y, heading) relative to starting point.
                (x, y) are in meters, heading is given in degrees in the range [-180, 180].
    """
    imgs = observation[:, :-3].reshape(img_shape)
    rgb = imgs[..., :3]
    segmentation = imgs[..., 3]
    depth = imgs[..., 4]

    pose = observation[:, -3:]

    return rgb, segmentation, depth, pose
