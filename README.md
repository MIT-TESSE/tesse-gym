# tesse-gym

Provides a Python interface for reinforcement learning using the TESSE Unity environment and the OpenAI Gym toolkit. 

## Content
* [Overview](#overview)
* [Installation](#installation)
* [Supported Projects](#supported-projects)


## Overview
To facility extendability, this repository organizes the reinforcement learning environment into the following components:
* **Core**: Base class for all RL tasks. Interfaces with the [TESSE simulator](https://github.com/MIT-TESSE/tesse-core)
* **Observer**: Forms an observation from TESSE simulator data. Observations my include images (RGB, depth, segmentation), pose, and Dynamic Scene Graphs.
* **Action Mapper**: Updates the agents state within the simulator given some policy action. 
* **Tasks**: Derives from **Core** to implement an RL task. Tasks include GOSEEK, exploration, and point navigation.

## Installation

Using [Anaconda](https://www.anaconda.com/distribution/#download-section) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is highly recommended.

1. If using conda, create a new environment
```sh
conda create -n goseek python=3.7 ipython jupyter numpy scipy
conda activate goseek
```

2. Clone and install this repository. 
```sh
git clone https://github.com/MIT-TESSE/tesse-gym.git
cd tesse-gym

python setup.py install

cd ..
```

## Supported Projects

### RL with Dynamic Scene Graphs

This repo was used in paper 
[*Hierarchical representations and Explicit Memory: Learning Effective Navigation Polices on 3D Scene Graphs*](https://arxiv.org/abs/2108.01176)
 at the International Conference on Robotics and Automation 2022. Please see this [repository](https://github.mit.edu/TESS/dsg-rl) for paper-specific code.


### GOSEEK Challenge

The objective of this task is to navigate an agent through an office environment to collect randomly-spawned fruit as quickly as possible. 

[![GOSEEK Teaser Trailer](https://img.youtube.com/vi/KXTag0xsg28/0.jpg)](https://www.youtube.com/watch?v=KXTag0xsg28)

More specifically, the agent can select from one of four actions at each decision epoch: move forward 0.5 meters, turn left 8 degrees, turn right 8 degrees, and collect fruit within 2.0 meters of the agent's current position. Our robot is equiped with stereo cameras and an Inertial Measurement Unit (IMU), from which a state-of-the-art perception pipeline estimates three pieces of information that make up the agent's observation at each decision epoch: localization information (position and heading relative to start position), pixel-wise semantic labels for objects in the robot's field of view, and pixel-wise depth in the robot's field of view.

#### Quick Start

To run the example PPO baseline, install [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/) and a version of [Tensorflow](https://www.tensorflow.org/) between v0.8.0 and v1.14.0 (see the [Stable Baselines docs](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites) for details).

__Note__: Currently, we've tested Python 3.7 Tensorflow installation with Anaconda against Cuda 10.0-10.2 (run `nvcc -V` to check your Cuda version).

For Cuda 10.0, we'd recommend installing `tensorflow-gpu v1.13.1`:

```sh
conda activate goseek 
conda install tensorflow-gpu==1.13.1
```

For Cuda 10.1 and 10.2, we'd recommend installing `tensorflow-gpu v1.14`:

```sh
conda activate goseek 
conda install tensorflow-gpu==1.14
```

Then, install [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/)

```sh
conda activate goseek 
pip install stable-baselines
```

Then, see the [example notebook](baselines/goseek-ppo.ipynb) to train an agent.

#### Challenge Details 

See the [GOSEEK Challenge](https://github.com/MIT-TESSE/goseek-challenge.git) landing page for details on setup, evaluation, and submission.



## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2020 Massachusetts Institute of Technology.

MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
