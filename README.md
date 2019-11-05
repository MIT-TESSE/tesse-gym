# tesse-gym

Provides a Python interface for reinforcement learning using the TESSE Unity environment and the OpenAI Gym toolkit.

## Install dependencies

### Stable Baselines
See the [Stable Baselines installation instructions](https://stable-baselines.readthedocs.io/en/master/guide/install.html) for details on how to install this on your system.

### tesse-interface

```python
git clone git@github.mit.edu:TESS/tesse-interface.git
cd tesse-interface/python
python setup.py developo
```

### Usage and Examples

This package provides environments for the following tasks
- Navigation: The agent must move throughout it's environment without collisions. See  the [example notebook](notebooks/navigation-training.ipynb) to get started.

- Treasure Hunt: The agent must find 'treasures' placed throughout it's environment. See the [example notebook](notebooks/treasure-hunt-training.ipynb) to get started.

Navigation | Treasure Hunt
:----------:|:---------------:
![](docs/nav-1.gif) | ![](docs/hunt-1.gif)
  


### Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2019 Massachusetts Institute of Technology.

MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
