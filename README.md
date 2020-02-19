# tesse-gym

Provides a Python interface for reinforcement learning using the TESSE Unity environment and the OpenAI Gym toolkit.


Treasure Hunt |  Navigation
:----------:|:---------------:
![](docs/hunt-1.gif) | ![](docs/nav-1.gif)

## Installation

### From Source
Using [Anaconda](https://www.anaconda.com/distribution/#download-section) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is highly recommended. Python 3.7 is required.

1. If using conda, create a new environment
```sh
conda create -n tesse_gym python=3.7
conda activate tesse_gym
```

2. Clone this repository, install dependencies, then install tesse-gym. *NOTE*: This requires access to [tesse-interface](https://github.mit.edu/TESS/tesse-interface). 
```sh
git clone git@github.mit.edu:TESS/tesse-gym.git
cd tesse-gym

pip install -r requirements.txt
python setup.py develop

cd ..
```


## GOSEEK Challenge

The objective of this task is to navigate an agent through an office environment to collect randomly-spawned fruit as quickly as possible. 

[![GOSEEK Teaser Trailer](https://img.youtube.com/vi/KXTag0xsg28/0.jpg)](https://www.youtube.com/watch?v=KXTag0xsg28)

More specifically, the agent can select from one of four actions at each decision epoch: move forward 0.5 meters, turn left 8 degrees, turn right 8 degrees, and collect fruit within 2.0 meters of the agent's current position. Our robot is equiped with stereo cameras and an Inertial Measurement Unit (IMU), from which a state-of-the-art perception pipeline estimates three pieces of information that make up the agent's observation at each decision epoch: localization information (position and heading relative to start position), pixel-wise semantic labels for objects in the robot's field of view, and pixel-wise depth in the robot's field of view.

### Quick Start

See the [example notebook](baselines/goseek-ppo.ipynb) to train an agent.

### Challenge Details 

See the [GOSEEK Challenge](https://github.mit.edu/TESS/goseek-challenge) landing page for details on setup, evaluation, and submission.

## Other Tasks

### Navigation

The agent must move throughout it's environment without collisions. See  the [example notebook](baselines/navigation.ipynb) to get started.

### New Tasks
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

© 2020 Massachusetts Institute of Technology.

MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
