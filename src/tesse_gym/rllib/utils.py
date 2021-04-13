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
import subprocess
from typing import Any, Dict

import yaml
from ray.tune import grid_search


def populate_rllib_config(
    default_config: Dict[str, Any], user_config: Dict[str, str]
) -> Dict[str, Any]:
    """ Add or edit items from an rllib config.

    If `user_config` contains a value wrapped in the string
    `grid_search([])`, the value will be given as and
    rllib.tune `grid_search` option.

    Args:
        default_config (Dict[str, Any]): A default rllib config 
            (e.g., for ppo).
        user_config (Dict[str, str]): Configuration given by user.
    
    Returns:
        Dict[str, Any]: `default_config` with `user_config` items 
            added. If there are matching keys, `user_config` takes
            precedence.
    """
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
    """ Raise exception if TESSE instances are already running. """
    if all(
        s in subprocess.run(["ps", "aux"], capture_output=True).stdout.decode("utf-8")
        for s in ["goseek-", ".x86_64"]
    ):
        raise EnvironmentError("TESSE is already running")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--name")
    parser.add_argument("--timesteps", default=5000000, type=int)
    return parser.parse_args()
