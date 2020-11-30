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


import argparse
import itertools
import subprocess

import numpy as np
import ray
import yaml
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune import grid_search
from ray.tune.registry import register_env

from tesse.msgs import Camera
from tesse_gym import ObservationConfig, get_network_config
from tesse_gym.core.utils import set_all_camera_params
from tesse_gym.rllib.networks import NatureCNNRNNActorCritic
from tesse_gym.tasks.goseek import GoSeek


class GOSEEKGoalCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        mean_found_targets = np.array(
            [env.n_found_targets for env in base_env.get_unwrapped()]
        ).mean()
        episode.custom_metrics["found_targets"] = mean_found_targets


def init_function(env):
    set_all_camera_params(
        env, height_in_pixels=cnn_shape[1], width_in_pixels=cnn_shape[2]
    )


def make_goseek_env(config, video_log_path):
    observation_config = ObservationConfig(
        modalities=(Camera.RGB_LEFT, Camera.SEGMENTATION, Camera.DEPTH),
        height=cnn_shape[1],
        width=cnn_shape[2],
        pose=True,
    )

    worker_index = config.worker_index
    vector_index = config.vector_index
    N_ENVS_PER_WORKER = 3
    rank = worker_index + N_ENVS_PER_WORKER * vector_index
    scene = config["SCENES"][rank % len(config["SCENES"])]

    print(
        f"MAKING GOSEEK ENV w/ rank: {rank}, inds: ({worker_index}, {vector_index}, scene: {scene})"
    )
    env = GoSeek(
        str(config["FILENAME"]),
        network_config=get_network_config(
            simulation_ip="localhost", own_ip="localhost", worker_id=rank
        ),
        scene_id=scene,
        target_found_reward=2,
        observation_config=observation_config,
        init_hook=init_function,
        video_log_path=video_log_path,
    )
    return env


def populate_rllib_config(default_config, user_config):
    with open(user_config) as f:
        user_config = yaml.load(f)

    for key, value in user_config.items():
        if isinstance(value, str) and "grid_search" in value:
            parsed_value = [
                float(x) for x in value.split("([")[1].split("])")[0].split(",")
            ]
            default_config[key] = grid_search(parsed_value)
        else:
            default_config[key] = user_config[key]

    return default_config


def check_for_tesse_instances():
    if all(
        s in subprocess.run(["ps", "aux"], capture_output=True).stdout.decode("utf-8")
        for s in ["goseek-", ".x86_64"]
    ):
        raise EnvironmentError("TESSE is already running")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--name")
    return parser.parse_args()


if __name__ == "__main__":
    check_for_tesse_instances()
    args = get_args()

    ray.init()
    ModelCatalog.register_custom_model("nature_cnn_rnn", NatureCNNRNNActorCritic)
    ModelCatalog.register_custom_model("nature_cnn", NatureCNNRNNActorCritic)
    cnn_shape = (5, 120, 160)

    def make_logging_goseek_env(env_config):
        return make_goseek_env(
            env_config, f"/home/za27933/tess/tesse-gym/goseek-videos/{args.name}",
        )

    register_env("goseek", make_logging_goseek_env)

    config = ppo.DEFAULT_CONFIG.copy()
    config["callbacks"] = GOSEEKGoalCallbacks
    config = populate_rllib_config(config, args.config)

    search_exp = tune.Experiment(
        name=args.name,
        run="PPO",
        config=config,
        stop={"timesteps_total": 5000000},
        checkpoint_freq=500,
        checkpoint_at_end=True,
    )

    tune.run_experiments([search_exp])
