import hashlib

import cv2

from tesse.msgs import *


class TESSEVideoWriter:
    def __init__(self, name, env):
        """ Store tesse-gym observations as a video 
        
        Args:
            name (str): Video name (including extension)
            env (tesse.env.Env): TESSE env object.

        Notes:
            Instead of directly writing to a video, we store
            the frames in a buffer and write everything at once.
            This prevents corruption by reducing the time a video 
            writer is open. 
        """
        self.name = name
        # self.writer = get_video_writer(name)
        self.env = env
        self.buffer = []

        video_hash = hashlib.sha1()
        video_hash.update(str(time.time()).encode())
        self.video_hash = video_hash.hexdigest()[:5]
        self.episode_num = 0
        self.video_writer = None

    def step(self):
        save_freq = 50
        # TODO(ZR) for debugging
        if self.episode_num % save_freq == 0:
            self.video_writer = TESSEVideoWriter(
                f"/home/za27933/tess/tesse-gym/debugging-videos/TESSE_gym_episode_{self.episode_num:05d}_env_{self.video_hash}.mp4",
                self.env,
            )
        elif self.episode_num % save_freq != 0 and self.video_writer != None:
            self.video_writer.release()
            self.video_writer = None
        print(f"On Episode: {self.episode_num}")
        self.episode_num += 1

    def write_frame(self):
        """ Create frame from observation. """
        third_person, first_person, seg, depth = get_images(self.env)
        show_img = tile_imgs(
            resize_img(third_person, 2),
            resize_img(first_person, 1 / 1.5),
            resize_img(seg, 1 / 1.5),
            get_show_img(resize_img(depth, 1 / 1.5)),
        )[..., (2, 1, 0)]
        # self.writer.write(show_img)
        self.buffer.append(show_img)

    def release(self):
        """ Write video. """
        writer = get_video_writer(self.name)
        for frame in self.buffer:
            writer.write(frame)
        writer.release()
        # self.writer.release()


def get_video_writer(name, shape=(853, 480)):
    return cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"XVID"), 20.0, shape)


def write_frame(env, video_writer):
    third_person, first_person, seg, depth = get_images(env)
    show_img = tile_imgs(
        resize_img(third_person, 2),
        resize_img(first_person, 1 / 1.5),
        resize_img(seg, 1 / 1.5),
        get_show_img(resize_img(depth, 1 / 1.5)),
    )
    video_writer.write(show_img[..., (2, 1, 0)])


def resize_img(img, scale):
    return cv2.resize(
        img, tuple((np.array(img.shape[1::-1]) * scale).astype(np.uint16))
    )


def get_show_img(img):
    img -= img.min()
    img /= img.max()
    return (255 * np.repeat(img[..., np.newaxis], 3, -1)).astype(np.uint8)


def get_images(env):
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


def tile_imgs(main_img, lower_img, mid_img, top_img):
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


def eval_agent(env, episode_length=None):
    episode_length = env.envs[0].max_steps if None else episode_length
    obs = env.reset()
    for _ in range(episode_length):
        action, _ = model.predict(np.repeat(obs, 4, axis=0))
        obs, step_reward, done, _ = env.step(action)
        if step_reward > 9:
            return True
        elif done:
            return False
    return False


def compute_average_success(env, episode_length=None):
    episode_length = env.envs[0].max_steps if None else episode_length
    success = []
    for i in range(100):
        success.append(eval_agent(episode_length))
    return np.array(success).mean()
