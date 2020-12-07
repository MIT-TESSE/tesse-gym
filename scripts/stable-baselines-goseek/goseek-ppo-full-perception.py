import yaml
import argparse

from pathlib import Path

import tensorflow as tf
from gym import spaces
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnLstmPolicy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from tesse.msgs import *
from tesse_gym import get_network_config
from tesse_gym.tasks.goseek import GoSeekFullPerception, decode_observations


IMG_SHAPE = (-1, 240, 320, 5)


def make_unity_env(filename, num_env):
    """ Create a wrapped Unity environment. """

    def make_env(rank):
        def _thunk():
            env = GoSeekFullPerception(
                str(filename),
                network_config=get_network_config(worker_id=rank),
                n_targets=n_targets,
                episode_length=episode_length,
                scene_id=scene_id[rank],
                target_found_reward=target_found_reward,
                observation_modalities=modalities,
            )
            env.HFOV = 80
            return env

        return _thunk

    return SubprocVecEnv([make_env(i) for i in range(num_env)])


def decode_tensor_observations(observation, img_shape=IMG_SHAPE):
    """ Decode observation vector into images and poses.

    Args:
        observation (np.ndarray): Shape (N,) observation array of flattened
            images concatenated with a pose vector. Thus, N is equal to N*H*W*C + N*3.
        img_shape (Tuple[int, int, int, int]): Shapes of all images stacked in (N, H, W, C).
            Default value is (-1, 240, 320, 5).

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Tensors with the following information
            - Tensor of shape (N, `img_shape[1:]`) containing RGB,
                segmentation, and depth images stacked across the channel dimension.
            - Tensor of shape (N, 3) containing (x, y, heading) relative to starting point.
                (x, y) are in meters, heading is given in degrees in the range [-180, 180].
    """
    imgs = tf.reshape(observation[:, :-3], img_shape)
    pose = observation[:, -3:]

    return imgs, pose


def custom_cnn(scaled_images, **kwargs):
    """
    Modified CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    # tmp until refactor
    import tensorflow as tf
    from stable_baselines.a2c.utils import (
        conv,
        linear,
        conv_to_fc,
        batch_to_seq,
        seq_to_batch,
        lstm,
    )

    activ = tf.nn.relu
    layer_1 = activ(
        conv(
            scaled_images,
            "c1",
            n_filters=32,
            filter_size=8,
            stride=4,
            init_scale=np.sqrt(2),
            **kwargs,
        )
    )
    layer_2 = activ(
        conv(
            layer_1,
            "c2",
            n_filters=64,
            filter_size=4,
            stride=2,
            init_scale=np.sqrt(2),
            **kwargs,
        )
    )
    layer_3 = activ(
        conv(
            layer_2,
            "c3",
            n_filters=64,
            filter_size=3,
            stride=1,
            init_scale=np.sqrt(2),
            **kwargs,
        )
    )
    layer_4 = activ(
        conv(
            layer_3,
            "c4",
            n_filters=128,
            filter_size=3,
            stride=1,
            init_scale=np.sqrt(2),
            **kwargs,
        )
    )
    layer_5 = activ(
        conv(
            layer_4,
            "c5",
            n_filters=128,
            filter_size=3,
            stride=1,
            init_scale=np.sqrt(2),
            **kwargs,
        )
    )
    layer_6 = conv_to_fc(layer_5)
    return activ(linear(layer_6, "fc1", n_hidden=512, init_scale=np.sqrt(2)))


def image_and_pose_network(observation, **kwargs):
    """ Network to process image and pose data.

    Use the stable baselines nature_cnn to process images. The resulting
    feature vector is then combined with the pose estimate and given to an
    LSTM (LSTM defined in PPO2 below).

    Args:
        raw_observations (tf.Tensor): 1D tensor containing image and
            pose data.

    Returns:
        tf.Tensor: Feature vector.
    """
    imgs, pose = decode_tensor_observations(observation)
    image_features = custom_cnn(imgs)
    return tf.concat((image_features, pose), axis=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train GOSEEK ppo agent with stable-baselines")
    parser.add_argument("--config", type=str, help="Config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    filename = Path(config["FILENAME"])
    assert filename.exists(), f"Must set a valid path!"

    n_environments = config["N_ENVIRONMENTS"]
    total_timesteps = config["TOTAL_TIMESTEPS"]
    scene_id = config["SCENE_IDS"]
    n_targets = config["N_TARGETS"]
    target_found_reward = config["TARGET_FOUND_REWARD"]
    episode_length = config["EPISODE_LENGTH"]

    modalities = []
    n_channels = 0
    for modality in config["MODALITIES"]:
        if modality == "RGB_LEFT":
            modalities.append(Camera.RGB_LEFT)
            n_channels += 3
        elif modality == "SEGMENTATION":
            modalities.append(Camera.SEGMENTATION)
            n_channels += 1
        elif modality == "DEPTH":
            modalities.append(Camera.DEPTH)
            n_channels += 1
        else:
            raise ValueError(f"Modality {modality} not recognized")

    IMG_SHAPE = (-1, *config["IMG_SHAPE"], n_channels)

    TB_LOG_NAME = config["TENSORBOARD_LOG_NAME"]
    log_dir = Path(config["LOG_DIR"])

    LR = config["LEARNING_RATE"]
    GAMMA = config["GAMMA"]

    env = make_unity_env(filename, n_environments)

    N_LSTM = 512

    # Create model
    policy_kwargs = {"cnn_extractor": image_and_pose_network, "n_lstm": N_LSTM}
    model = PPO2(
        CnnLstmPolicy,
        env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        nminibatches=2,
        gamma=GAMMA,
        learning_rate=LR,
        policy_kwargs=policy_kwargs,
    )

    # define logging
    log_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint_callback(local_vars, global_vars):
        total_updates = local_vars["update"]
        if total_updates % 1000 == 0:
            local_vars["self"].save(str(log_dir / f"{total_updates:06d}.pkl"))

    # train
    model.learn(
        total_timesteps=total_timesteps,
        callback=save_checkpoint_callback,
        tb_log_name=TB_LOG_NAME,
    )
