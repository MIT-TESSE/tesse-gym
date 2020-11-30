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

from collections import namedtuple
from typing import Optional, Tuple

import gym.spaces as spaces
import numpy as np

from tesse.msgs import Camera, Channels, Compression, DataResponse

ObservationConfig = namedtuple(
    "ObservationConfig",
    ["modalities", "pose", "height", "width", "min", "max"],
    defaults=((Camera.RGB_LEFT,), False, 240, 320, -np.Inf, np.Inf),
)


def setup_observations(
    observation_config: ObservationConfig,
) -> Tuple[Tuple[Camera, Compression, Channels], spaces.Box]:
    """ Creates observation configuration for `TesseGym`.

    Parameters:
        observation_config (ObservationConfig): Object 
            containing info to specify gym observations.
    
    Returns:
        Tuple[Tuple[Camera, Compression, Channels], spaces.Box]
            - Tuple listing TESSE cameras to query when getting an 
                observation.
            - spaces.Box object describing the observation space.
    """
    observation_modalities = [
        (camera, Compression.OFF, Channels.THREE)
        for camera in observation_config.modalities
    ]
    observation_space = get_observation_space(
        observation_modalities,
        height_in_pixels=observation_config.height,
        width_in_pixels=observation_config.width,
        pose=observation_config.pose,
        obs_min=observation_config.min,
        obs_max=observation_config.max,
    )
    return observation_modalities, observation_space


def get_observation_space(
    observation_modalities: Optional[Tuple[Camera, ...]],
    height_in_pixels: Optional[int] = 240,
    width_in_pixels: Optional[int] = 320,
    pose: Optional[bool] = True,
    obs_min: Optional[float] = -np.Inf,
    obs_max: Optional[float] = np.Inf,
) -> spaces.Box:
    """ Get observation space.
    
    Parameters:
        observation_modalities (Tuple[Camera, ...]): Tuple of observation 
            modalities (e.g., RGB_LEFT, DEPTH, SEGMENTATION, ...)
        height_in_pixels (int): Observation image height.
        width_in_pixels (int): Observation image width.
        pose (bool): True if pose is included in observation.
        obs_min (float): Min possible observation value.
        obs_max (float): Max possible observation value. 
    
    Returns:
        spaces.Box: OpenAI Gym spaces.Box object describing the 
            observation given as arguments.
    """
    n_channels = 0  # total image channels
    for camera in observation_modalities:
        if camera[0] in (Camera.RGB_RIGHT, Camera.RGB_LEFT):
            n_channels += 3
        elif camera[0] == Camera.SEGMENTATION:
            n_channels += 1
        elif camera[0] == Camera.DEPTH:
            n_channels += 1
        else:
            raise ValueError(f"Unrecognized camera: {camera[0]}")
    if pose:
        observation_shape = (height_in_pixels * width_in_pixels * n_channels + 3,)
    else:
        observation_shape = (height_in_pixels, width_in_pixels, n_channels)

    return spaces.Box(obs_min, obs_max, shape=observation_shape)


def decode_observations(
    observation: np.ndarray,
    img_shape: Tuple[int, int, int, int] = (240, 320),
    img_channels: Tuple[int, ...] = (3, 1, 1),
    pose_shapes: Tuple[int, ...] = (3,),
) -> Tuple[np.ndarray, ...]:
    """ Decode observation vector into images and poses.

    Args:
        observation (np.ndarray): Shape (N,) observation array of flattened
            images concatenated with a pose vector. Thus, N is equal to N*H*W*C + N*3.
        img_shape (Tuple[int, int]): (H, W) of all images.
        img_channels (Tuple[int, ...]): Number of channles of each image in `observation`.
        pose_shapes (Tuple[int, ...]): Shape of all poses stacked 

    Returns:
        Tuple[np.ndarray, ...]: Arrays of images and poses.

    Notes:
        Assumes images are flattened and stacked first, followed by poses. 
    """
    decoded_obs = []

    added_dim = False
    if len(observation.shape) == 1:
        observation = observation[np.newaxis]
        added_dim = True

    # decode images
    len_img_values = img_shape[0] * img_shape[1] * np.sum(img_channels)
    imgs = observation[:, :len_img_values].reshape(
        (-1,) + img_shape + (np.sum(img_channels),)
    )
    img_channels = np.concatenate(([0], np.cumsum(img_channels)))
    for i in range(len(img_channels) - 1):
        decoded_obs.append(imgs[..., img_channels[i] : img_channels[i + 1]])

    # decode poses
    pose_inds = np.cumsum([len_img_values, *pose_shapes])

    for i in range(len(pose_inds) - 1):
        decoded_obs.append(observation[:, pose_inds[i] : pose_inds[i + 1]])

    if added_dim:
        decoded_obs = [v[0] for v in decoded_obs]

    return decoded_obs
