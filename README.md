# TESSE_gym_interface

Provides a Python interface for reinforcement learning using the TESSE Unity environment and the OpenAI Gym toolkit.

## Notes

Expect some name changes shortly.

## Setup

Ensure the following dependencies are installed:
- [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/)
- [TESSE_interface](https://github.mit.edu/TESS/TESSE_interface)

Then, clone the repo and setup the `TESSE_gym_interface` package.

```sh
git clone git@github.mit.edu:TESS/TESSE_gym_interface.git
cd TESSE_gym_interface
python setup.py develop
```

### Usage 

This package provides environments for the following tasks
- Navigation: The agent must move throughout it's environment without collisions.

- Treasure Hunt: The agent must find 'treasures' placed throughout it's environment. 
    
    See the [example notebook](https://github.mit.edu/TESS/TESSE_gym_interface/blob/master/agent-training.ipynb) to get this up and running.
   
__TODO (?)__: add some gifs and an example notebook for the navigation task.


### Disclaimer

Distribution authorized to U.S. Government agencies and their contractors. Other requests for this document shall be referred to the MIT Lincoln Laboratory Technology Office.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2019 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
