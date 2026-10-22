from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'calibration_claude'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bartek',
    maintainer_email='bartlomiej.hryniewski@gmail.com',
    description='Hand-eye calibration pose driver for the Franka FR3.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'preview_poses = calibration_claude.preview_poses:main',
            'run_calibration_capture = calibration_claude.capture_node:main',
            'run_taught_capture = calibration_claude.taught_capture:main',
            'tune_cameras = calibration_claude.camera_tuning:main',
            'capture_stereo_rgbd = calibration_claude.capture_stereo_rgbd:main',
            'solve_handeye = calibration_claude.solve_handeye:main',
            'solve_joint_handeye = calibration_claude.solve_handeye:main_joint',
            'probe_board = calibration_claude.board_manipulation:main',
            'make_tag_assets = calibration_claude.tag:make_assets_main',
            'repeatability_diagnostic = calibration_claude.repeatability_diagnostic:main',
        ],
    },
)
