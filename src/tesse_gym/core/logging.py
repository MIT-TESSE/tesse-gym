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


import hashlib
import logging
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from tesse.env import Env
from tesse.msgs import *
from tesse.utils import UdpListener

from .continuous_control import parse_metadata


VideoLoggingConfig = namedtuple("VideoLoggingConfig", ["save_root", "save_freq"])


class TesseLogger:
    def __init__(
        self,
        udp_port: int,
        log_dir: Optional[str] = "./tesse_gym_logs/",
        udp_rate: Optional[int] = 200,
    ):
        """ Logger for tesse_gym.

        Args:
            udp_port (int): Port used for high rate UDP metadata
                broadcasts.
            log_dir (Optional[str]): Root log directory.
            udp_rate (Optional[int]): UDP metadata listener rate in Hz.
        """
        # for now, assume log dirs are named sequentially
        n_log_sets = len([p.is_dir() for p in sorted(Path(log_dir).glob("*"))])
        self.log_dir = (
            Path(log_dir) / f"evaluation_{n_log_sets + 1}"
        )  # TODO(ZR) make this configurable
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.episode_count = 0

        self.logger = logging.getLogger("tesse_gym")
        self.logger.setLevel(logging.INFO)

        self.udp_listener = UdpListener(port=udp_port, rate=udp_rate)
        self.udp_listener.subscribe("catch_metadata", self.catch_udp_broadcast)
        self.udp_listener.start()
        self.last_metadata = None

    def catch_udp_broadcast(self, udp_metadata: Callable[[str], None]) -> None:
        """ Catch UDP metadata broadcast from TESSE. """
        self.last_metadata = udp_metadata

    def log_step(self) -> None:
        """ Log one step in an episode.

        Logs in the following format
            <X>,<Z>,<YAW>
        """
        if self.last_metadata == None:
            logging.warn("Metadata is none")
        else:
            state = parse_metadata(self.last_metadata)
            x, z, yaw = state.position.x, state.position.z, state.rotation.yaw
            self.logger.info(f"{x},{z},{yaw}")

    def set_next_episode(self) -> None:
        """ Increment log file and update file handler. """
        fileh = logging.FileHandler(self.get_next_log(), "a")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fileh.setFormatter(formatter)
        fileh.setLevel(logging.INFO)

        for hdlr in self.logger.handlers[:]:
            self.logger.removeHandler(hdlr)

        self.logger.addHandler(fileh)

    def get_next_log(self) -> str:
        """ Get next log file's name

        Follows the convention
            `<TIME>-episode-<EPISODE_COUNT>`
        """
        self.episode_count += 1
        return (
            f"{self.log_dir}/"
            f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%s')}-episode-{self.episode_count}.log"
        )

    def close(self) -> None:
        """ Called upon destruction, join UDP listener. """
        self.udp_listener.join()


class TESSEVideoWriter:
    def __init__(
        self, save_root: str, env: Env, save_freq: int = 100, gym = None
    ) -> None:
        """ Store tesse-gym observations as a video

        Args:
            name (str): Video name (including extension)
            env (tesse.env.Env): TESSE env object.
            save_freq (int): Write video at this frequency.
            gym (TesseGym): Gym environment to access any
                data required for logging.

        Notes:
            Instead of directly writing to a video, we store
            the frames in a buffer and write everything at once.
            This prevents corruption by reducing the time a video
            writer is open.
        """
        self.env = env

        self.buffer = []
        self.write_frames = False
        self.save_freq = save_freq

        video_hash = hashlib.sha1()
        video_hash.update(str(time.time()).encode())
        self.video_hash = video_hash.hexdigest()[:5]
        self.episode_num = 0

        self.save_root = save_root
        i = 0
        while True:
            p = Path(self.save_root)
            p = p / f"run-{i}"
            i += 1
            if not p.exists():
                p.mkdir(parents=True, exist_ok=False)
                break

        self.gym = gym

    def reset(self) -> None:
        if self.episode_num % self.save_freq == 0:
            self.write_frames = True
        elif self.episode_num % self.save_freq != 0 and self.write_frames == True:
            self.release()
            self.write_frames = False
        self.episode_num += 1

    def step(self) -> None:
        """ Create frame from observation. """
        if self.write_frames:
            third_person, first_person, seg, depth = self.get_images(self.env)
            scale = third_person.shape[0] // first_person.shape[0]
            show_img = self.tile_imgs(
                self.resize_img(third_person, 2),
                self.resize_img(first_person, scale * 1 / 1.5),
                self.resize_img(seg, scale * 1 / 1.5),
                self.get_show_img(self.resize_img(depth, scale * 1 / 1.5)),
            )[..., (2, 1, 0)]
            self.buffer.append(show_img)

    def release(self) -> None:
        """ Write video. """
        if len(self.buffer) == 0:
            return

        writer = self.get_video_writer(
            f"{self.save_root}"
            f"/TESSE_gym_episode_{self.episode_num:07d}_env_{self.video_hash}.avi",
            shape=self.buffer[0].shape[:2][::-1],
        )
        for frame in self.buffer:
            writer.write(frame)

        self.buffer = []
        writer.release()

    @staticmethod
    def get_images(env) -> None:
        image_settings = (Compression.OFF, Channels.THREE)
        response = env.request(
            DataRequest(
                metadata=False,
                cameras=[
                    (Camera.THIRD_PERSON,) + image_settings,
                    (Camera.RGB_LEFT,) + image_settings,
                    (Camera.SEGMENTATION,) + image_settings,
                    (Camera.DEPTH,) + image_settings,
                ],
            )
        )
        return response.images

    @staticmethod
    def get_video_writer(name: str, shape: Tuple[int, int] = (853, 480)):
        return cv2.VideoWriter(name, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 20.0, shape) #*XVID

    @staticmethod
    def resize_img(img: np.ndarray, scale: float):
        return cv2.resize(
            img, tuple((np.array(img.shape[1::-1]) * scale).astype(np.uint16))
        )

    @staticmethod
    def get_show_img(img):
        img -= img.min()
        img /= img.max()
        return (255 * np.repeat(img[..., np.newaxis], 3, -1)).astype(np.uint8)

    @staticmethod
    def tile_imgs(
        main_img: np.ndarray,
        lower_img: np.ndarray,
        mid_img: np.ndarray,
        top_img: np.ndarray,
    ):
        h_main, w_main, c_main = main_img.shape
        h_lower, w_lower, c_lower = lower_img.shape
        assert lower_img.shape == mid_img.shape == top_img.shape
        assert c_main == c_lower

        canvas_shape = (h_main, w_main + w_lower, c_main)
        canvas = np.zeros(canvas_shape, dtype=np.uint8)

        canvas[0:h_main, 0:w_main, :] = main_img
        canvas[-h_lower:, -w_lower:, :] = lower_img
        canvas[-2 * h_lower : -h_lower, -w_lower:, :] = mid_img
        canvas[: -2 * h_lower, -w_lower:, :] = top_img

        return canvas
