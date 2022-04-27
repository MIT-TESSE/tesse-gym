from typing import Dict, List, Tuple, Union

import numpy as np
from gym import spaces

from tesse.msgs import Camera, Channels, Compression
from tesse_gym.observers.observer import Observer


class TesseImageObserver(Observer):
    N_CLASSES = 11
    WALL_CLS = 2

    def __init__(
        self,
        observation_modalities: List[Tuple[Camera, Compression, Channels]],
        observation_space: Union[spaces.Dict, spaces.Box],
    ):
        self.observation_modalities = observation_modalities
        self.observation_space = observation_space

    def observe(
        self,
        env_info: Dict[
            str, np.ndarray
        ],  # tesse_data: DataResponse, pose_dict: Dict[str, np.ndarray]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        tesse_data = env_info["tesse_data"]
        pose = env_info["relative_pose"]
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
                observation_imgs.append(
                    tesse_data.images[i][..., np.newaxis].astype(np.float64)
                )

        pose = env_info["relative_pose"]

        # make observations either Box or Dict type
        if isinstance(self.observation_space, spaces.Box):
            observation_imgs = np.concatenate(observation_imgs, axis=-1)
            # flattened observation space means we use pose
            if len(self.observation_space.shape) == 1:
                # pose = self.get_pose().reshape((3))
                observation = np.concatenate((observation_imgs.reshape(-1), pose))
            else:
                observation = observation_imgs
        elif isinstance(self.observation_space, spaces.Dict):
            names = [c.name for c, _, _ in self.observation_modalities]
            observation = dict(zip(names, observation_imgs))
            if "POSE" in self.observation_space.spaces.keys():
                observation["POSE"] = pose  # self.get_pose().reshape((3))

        return observation

    def reset(self):
        pass
