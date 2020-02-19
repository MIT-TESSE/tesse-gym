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

import numpy as np
from gym import spaces

from tesse.msgs import Camera, Channels, Compression, DataRequest, DataResponse

from tesse_gym.tasks.goseek.goseek import GoSeek


class GoSeekFullPerception(GoSeek):
    """ Define a custom TESSE gym environment to provide RGB, depth, segmentation, and pose data.
    """

    DEPTH_SCALE = 5
    N_CLASSES = 11
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

        # TESSE assigns areas without a material (e.g. outside windows) the maximum possible
        # image value. It will be classified as wall.
        seg[seg > (self.N_CLASSES - 1)] = self.WALL_CLS
        observation = np.concatenate(
            (
                eo / 255.0,
                seg[..., np.newaxis] / (self.N_CLASSES - 1),
                (depth[..., np.newaxis] * self.DEPTH_SCALE).clip(0, 1),
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
