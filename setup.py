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

from setuptools import setup, find_packages
import versioneer

setup(
    name="tesse_gym",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="TESSE OpenAI Gym interface",
    packages=find_packages("src"),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    python_requires=">=3.7",
    package_dir={"": "src"},
    install_requires=[
        "numpy >= 1.17.3",
        "scipy >= 1.4.1",
        "gym >= 0.15.3",
        "defusedxml >= 0.6.0",
        "pillow >= 6.2.1",
        "yacs >= 0.1.6",
        "tqdm >= 4.42.1",
        "tesse@git+https://git@github.com/MIT-TESSE/tesse-interface.git@0.1.2#egg=tesse",
    ],
    extras_require={"rllib": ["ray[rllib]>=0.8.0", "torch>=1.4.0"],},
    dependency_links=[
        "git+https://git@github.com/MIT-TESSE/tesse-interface.git@0.1.2#egg=tesse"
    ],
)
