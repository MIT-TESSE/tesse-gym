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

from typing import Tuple

import numpy as np
from gym import spaces

from tesse.msgs import Camera, Channels, Compression, DataRequest, DataResponse

from tesse_gym.tasks.goseek.goseek import GoSeek


class GoSeekFullPerception(GoSeek):
    """ Define a custom TESSE gym environment to provide RGB, depth, segmentation, and pose data.
    """

    # Unity scales depth in range [0, FAR_CLIP_FIELD] to [0, 1].
    # Values above FAR_CLIP_FIELD are, as the name implies, clipped.
    # Multiply depth by `DEPTH_SCALE` to recover values in meters.
    FAR_CLIP_FIELD = 50
    N_CLASSES = 11  # Number of classes used for GOSEEK

    # Unity assigns areas without a material (e.g. outside windows) the maximum possible
    # image value. For convenience, this will be changed to the wall class.
    WALL_CLS = 2

    @property
    def observation_space(self) -> spaces.Box:
        """ Define an observation space for RGB, depth, segmentation, and pose.

        Because Stables Baselines (the baseline PPO library) does not support dictionary spaces,
        the observation images and pose vector will be combined into a vector. The RGB image
        is of shape (240, 320, 3), depth and segmentation are both (240, 320), ose is (3,), thus
        the total shape is (240 * 320 * 5 + 3).
        """
        return spaces.Box(np.Inf, np.Inf, shape=(240 * 320 * 5 + 3,))

    def form_agent_observation(self, tesse_data: DataResponse) -> np.ndarray:
        """ Create the agent's observation from a TESSE data response.

        Args:
            tesse_data (DataResponse): TESSE DataResponse object containing
                RGB, depth, segmentation, and pose.

        Returns:
            np.ndarray: The agent's observation.
        """
        eo, seg, depth = tesse_data.images
        seg = seg[..., 0].copy()  # get segmentation as one-hot encoding

        # See WALL_CLS comment
        seg[seg > (self.N_CLASSES - 1)] = self.WALL_CLS
        observation = np.concatenate(
            (
                eo / 255.0,
                seg[..., np.newaxis] / (self.N_CLASSES - 1),
                depth[..., np.newaxis],
            ),
            axis=-1,
        ).reshape(-1)
        pose = self.get_pose().reshape((3))

        if (np.abs(pose) > 100).any():
            raise ValueError("Pose is out of observation space")
        return np.concatenate((observation, pose))

    def observe(self) -> DataResponse:
        """ Get observation data from TESSE.

        Returns:
            DataResponse: TESSE DataResponse object.
        """
        cameras = [
            (Camera.RGB_LEFT, Compression.OFF, Channels.THREE),
            (Camera.SEGMENTATION, Compression.OFF, Channels.THREE),
            (Camera.DEPTH, Compression.OFF, Channels.THREE),
        ]
        agent_data = self.env.request(DataRequest(metadata=True, cameras=cameras))
        return agent_data


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
