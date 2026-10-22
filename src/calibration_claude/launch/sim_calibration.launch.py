"""Gazebo rehearsal scene for FR3 hand-eye calibration, with the tag on the gripper.

This deliberately re-implements franka_gazebo_bringup's
``gazebo_franka_arm_example_controller.launch.py`` instead of including it, for
two reasons:

* the robot has to be spawned from *our* xacro, which welds the AprilTag to
  ``fr3_hand_tcp``; and
* that launch always spawns one of the franka example controllers, and the
  default (``gravity_compensation_example_controller``) claims the same effort
  interfaces the trajectory controller needs.

The world SDF and the camera bridges are reused from franka_gazebo_bringup
unchanged.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import xacro

from calibration_claude.cameras import load_cameras
from calibration_claude.kinematics import matrix_to_rpy

CAMERA_NAMES = ('femto_bolt', 'femto_mega')


def ground_truth_camera_frames(package_share):
    """Publish the *correct* world->optical transforms for the two cameras.

    franka_gazebo_bringup's fr3_dual_camera_calibration.launch.py publishes the
    camera *body* poses (X forward, Z up) under names ending in
    ``_color_optical_frame`` (Z forward, X right, Y down), so those frames are
    rotated ~90/90 deg away from where the images are actually taken.  Nothing in
    this package reads TF for the cameras, so that mismatch is harmless here --
    these ``gt_`` frames exist purely so ground truth can be eyeballed in RViz
    next to a calibration result.
    """
    cameras = load_cameras(os.path.join(package_share, 'config', 'cameras_sim.yaml'))
    nodes = []
    for name, camera in sorted(cameras.items()):
        xyz = camera.pose[:3, 3]
        rpy = matrix_to_rpy(camera.pose[:3, :3])
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'gt_{name}_tf',
            arguments=[
                '--x', f'{xyz[0]:.6f}', '--y', f'{xyz[1]:.6f}', '--z', f'{xyz[2]:.6f}',
                '--roll', f'{rpy[0]:.6f}', '--pitch', f'{rpy[1]:.6f}',
                '--yaw', f'{rpy[2]:.6f}',
                '--frame-id', 'world',
                '--child-frame-id', f'gt_{name}_color_optical_frame',
            ],
            condition=IfCondition(LaunchConfiguration('publish_ground_truth')),
            output='log',
        ))
    return nodes


def _resource_path_actions(*packages):
    parents = []
    for package in packages:
        parent = os.path.dirname(get_package_share_directory(package))
        if parent not in parents:
            parents.append(parent)
    joined = os.pathsep.join(parents)
    return [AppendEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', joined),
            AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', joined)]


def generate_launch_description():
    package_share = get_package_share_directory('calibration_claude')
    franka_share = get_package_share_directory('franka_gazebo_bringup')
    world_file = os.path.join(franka_share, 'worlds', 'fr3_dual_camera_calibration.sdf')
    robot_xacro = os.path.join(package_share, 'urdf', 'fr3_with_apriltag.gazebo.xacro')

    robot_description = xacro.process_file(robot_xacro, mappings={
        'robot_type': 'fr3',
        'hand': 'true',
        'gazebo': 'true',
        'ee_id': 'franka_hand',
        'gazebo_effort': 'true',
    }).toprettyxml(indent='  ')

    controller_params = PathJoinSubstitution([
        FindPackageShare('calibration_claude'), 'config',
        'fr3_arm_controller_effort.yaml',
    ])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': LaunchConfiguration('gz_args')}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/robot_description', '-name', 'fr3'],
        output='screen',
    )

    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '60'],
        output='screen',
    )

    arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'fr3_arm_controller',
            '-t', 'joint_trajectory_controller/JointTrajectoryController',
            '-p', controller_params,
            '--controller-manager-timeout', '60',
        ],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='calibration_camera_info_bridge',
        arguments=(
            ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
            + [f'/{name}/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
               for name in CAMERA_NAMES]
        ),
        output='screen',
    )
    image_bridges = [
        Node(package='ros_gz_image', executable='image_bridge',
             name=f'{name}_image_bridge', arguments=[f'/{name}/color/image_raw'],
             output='screen')
        for name in CAMERA_NAMES
    ]

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-f', 'world'],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='log',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'gz_args', default_value=f'-r {world_file}',
            description='Gazebo arguments. For headless runs use '
                        f'"-r -s --headless-rendering {world_file}".'),
        DeclareLaunchArgument(
            'publish_ground_truth', default_value='true',
            description='Publish gt_*_color_optical_frame TFs for the known camera poses.'),
        DeclareLaunchArgument('rviz', default_value='false'),

        # URDF package:// URIs become model:// once ros_gz_sim converts the robot
        # to SDF, and gz then resolves them against these paths.  colcon installs
        # each package under its own prefix, so every package whose meshes are
        # referenced needs its share *parent* on the path -- franka_description for
        # the arm and hand, calibration_claude for the tag.
        *_resource_path_actions('calibration_claude', 'franka_description'),

        gazebo,
        robot_state_publisher,
        spawn,
        bridge,
        *image_bridges,
        rviz,
        RegisterEventHandler(OnProcessExit(target_action=spawn,
                                           on_exit=[joint_state_broadcaster])),
        RegisterEventHandler(OnProcessExit(target_action=joint_state_broadcaster,
                                           on_exit=[arm_controller])),
        *ground_truth_camera_frames(package_share),
    ])
