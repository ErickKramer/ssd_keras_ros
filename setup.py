from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['ssd_keras_ros'],
    package_dir={'': 'ros/src'}
)

setup(**d)
