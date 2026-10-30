# Installed by catkin (CMakeLists.txt: catkin_python_setup); do not run directly.
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup(**generate_distutils_setup(
    packages=['panda_calibration'],
    package_dir={'': '.'},
))
