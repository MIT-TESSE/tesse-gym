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

from yacs.config import CfgNode as CN

_C = CN()

_C.ENV = CN()
_C.ENV.sim_path = "simulator/goseek-v0.1.0.x86_64"
_C.ENV.position_port = 9000
_C.ENV.metadata_port = 9001
_C.ENV.image_port = 9002
_C.ENV.step_port = 9005
_C.ENV.ground_truth_mode = True

_C.EPISODE = CN()
_C.EPISODE.scenes = [1, 2, 3, 4, 5]
_C.EPISODE.success_dist = 2
_C.EPISODE.n_targets = [30, 30, 30, 30, 30]
_C.EPISODE.episode_length = [400, 400, 400, 400, 400]
_C.EPISODE.random_seeds = [10, 100, 1000, 10000, 100000]


def get_goseek_cfg_defaults():
    return _C.clone()
