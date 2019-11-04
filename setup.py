from setuptools import setup, find_packages

setup(
    name='tesse_gym',
    version='0.0.1',
    description='TESSE OpenAI Gym python interface',
    packages=find_packages('src'),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={'': 'src'},
)
