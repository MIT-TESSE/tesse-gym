# tesse-gym

Provides a Python interface for reinforcement learning using the TESSE Unity environment and the OpenAI Gym toolkit.

## Setup

### Install dependencies

#### Stable Baselines
See the [Stable Baselines installation instructions](https://stable-baselines.readthedocs.io/en/master/guide/install.html) for details on how to install this on your system.

#### tesse-interface

```sh
git clone git@github.mit.edu:TESS/tesse-interface.git
cd tesse-interface/python
python setup.py develop
```

### Install tess-gym

```sh
git clone https://llcad-github.llan.ll.mit.edu/tess-open-source-staging/tess-gym.git
cd tesse-gym
python setup.py develop
```

## Getting started

This package provides environments for the following tasks:

### 1. Treasure Hunt

#### Objective 
Treasures (yellow cubes) are randomly placed throughout a TESSE environment. The agent must collect as many of these treasures as possible within the alloted time (default is 100 timesteps). A treasure is considered found when it is within `success_dist` (default is 2m) of the agent and within it's feild of view. 

#### Observation space
The agent acts on a first-person RGB image. We may add depth for the challange and provide a semantic segmentation model.

See the [example notebook](notebooks/treasure-hunt-training.ipynb) to get started.

### 2. Navigation

The agent must move throughout it's environment without collisions. See  the [example notebook](notebooks/navigation-training.ipynb) to get started.


Navigation | Treasure Hunt
:----------:|:---------------:
![](docs/nav-1.gif) | ![](docs/hunt-1.gif)


### New tasks
At a minimum, a new task will inherit `tess_gym.TesseGym` and impliment the following:

```python
class CustomTask(TesseGym):
    @property
    def action_space(self):
        pass
    
    def apply_action(self, action):
        pass
    
    def compute_reward(self, observation, action):
        pass
```
  


### Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2019 Massachusetts Institute of Technology.

MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
