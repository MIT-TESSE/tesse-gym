# tesse-gym

Provides a Python interface for reinforcement learning using the TESSE Unity environment and the OpenAI Gym toolkit.


Navigation | Treasure Hunt
:----------:|:---------------:
![](docs/hunt-1.gif) | ![](docs/nav-1.gif)

## Installation

### From Source
Using [Anaconda](https://www.anaconda.com/distribution/#download-section) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is highly recommended. Python 3.6 is required.

1. Clone this repository
```sh
git clone git@github.mit.edu:TESS/tesse-gym.git
cd tesse-gym
```

2. Install Dependencies

```sh
conda create -n tess_gym python=3.6
conda activate tess_gym
pip install -r requirements.txt
```

3. Install tesse-interface

Note: the branch `feature/objects` is currently required.

```sh
cd ..
git clone git@github.mit.edu:TESS/tesse-interface.git
cd tesse-interface/python
git checkout feature/objects
python setup.py install
cd ../../tesse-gym
```

3. Install tesse-gym

```sh
python setup.py install
```

## Getting started

### Treasure Hunt Training

Treasures (yellow cubes) are randomly placed throughout a TESSE environment. The agent must collect as many of these treasures as possible within the alloted time (default is 100 timesteps). A treasure is considered found when it is within `success_dist` (default is 2m) of the agent and within it's feild of view. The agent acts on a first-person RGB, depth, and semantic segmentation images as well as pose.

See the [example notebook](notebooks/stable-baselines-ppo.ipynb) to get started.

### Evaluation

To evaluate an agent:

1. Update the TESSE build path in `./evaluation/config/treasure-hunt-challenge.yaml`

2. Define the agent in `./baselines/agents.py`

3. Add a configuration file to `./baselines/config`

4. Edit `run_treasure_hunt_eval.sh` to include the following

```sh
python eval.py --env-config config/treasure-hunt-challenge.yaml --agent-config YOUR_CONFIG
```  
  
#### Evaluate the [example notebook](notebooks/stable-baselines-ppo.ipynb) Agent

To run this evaluation, make sure that:

1. TESSE path from step 1 is correct.

2. You have the proper weight file, `./baselines/config/stable-baselines-ppo-1.pkl`. If your on the LLAN, the weight file can be found at `//group104/users/RavichandranZachary/public/tess/icra-2020-ws/rl-models/stable-baselines-ppo-1.pkl`

Then, run the evaluation script 

```sh 
./run_treasure_hunt_eval.sh
```


## Other Tasks

### Navigation

The agent must move throughout it's environment without collisions. See  the [example notebook](notebooks/navigation-training.ipynb) to get started.

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
