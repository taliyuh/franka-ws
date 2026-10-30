from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'fr3_calibration'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bartek',
    maintainer_email='bartlomiej.hryniewski@gmail.com',
    description='Camera calibration captures, hand-eye solves and board probing for the Franka FR3.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'run_calibration_capture = fr3_calibration.capture_node:main',
            'run_taught_capture = fr3_calibration.taught_capture:main',
            'capture_stereo_rgbd = fr3_calibration.capture_stereo_rgbd:main',
            'solve_handeye = fr3_calibration.solve_handeye:main',
            'solve_joint_handeye = fr3_calibration.solve_handeye:main_joint',
            'probe_board = fr3_calibration.board_manipulation:main',
        ],
    },
)
