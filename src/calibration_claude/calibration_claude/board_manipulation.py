"""Probe a board-relative point using Franka's native PTP motion action.

The board pose is deliberately an input artifact.  Calibration and board-pose
refinement can run independently, then publish a JSON file with ``T_base_board``
for this program to consume. Every approach and descent move is sent through
``/action_server/ptp_motion``; a generic joint trajectory controller is not
accepted for probing.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Callable

import numpy as np
import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String

from .capture_node import LATCHED
from .context import build_chain, default_config_path, load_config
from .board_pose import (
    board_target_pose,
    load_board_pose,
    signed_board_clearance,
)
from .kinematics import (
    make_transform,
    matrix_to_quaternion,
    pose_error,
    quaternion_to_matrix,
)
from .trajectory_client import ArmDriver


NATIVE_PTP_ACTION_TYPE = 'ptp_motion'
NATIVE_PTP_ACTION = '/action_server/ptp_motion'


def compensate_pose_for_measured_error(desired: np.ndarray,
                                      actual: np.ndarray) -> np.ndarray:
    """Return a first-order command correction for a measured TCP residual."""
    corrected = desired.copy()
    corrected[:3, 3] += desired[:3, 3] - actual[:3, 3]
    corrected[:3, :3] = (
        desired[:3, :3] @ actual[:3, :3].T @ desired[:3, :3])
    return corrected


def validate_probe_motion_config(config: dict) -> None:
    """Require the native Franka PTP action for real board probing.

    A one-point FollowJointTrajectory goal can be accepted by a generic ROS 2
    controller while still producing a discontinuous command at the libfranka
    boundary. Board probing must therefore never silently fall back to that
    controller.
    """
    motion = config.get('motion')
    if not isinstance(motion, dict):
        raise ValueError('robot configuration is missing the motion section')
    action_type = motion.get('action_type')
    action_name = motion.get('controller_action')
    if action_type != NATIVE_PTP_ACTION_TYPE or action_name != NATIVE_PTP_ACTION:
        raise ValueError(
            'probe_board requires native Franka PTP motion: '
            f'action_type={NATIVE_PTP_ACTION_TYPE!r}, '
            f'controller_action={NATIVE_PTP_ACTION!r}; '
            f'got action_type={action_type!r}, controller_action={action_name!r}')


class BoardManipulation(Node):
    def __init__(self, config: dict, args):
        super().__init__('board_manipulation')
        validate_probe_motion_config(config)
        self.config = config
        self.args = args
        self.joint_names = config['robot']['joint_names']
        self.driver = ArmDriver(self, self.joint_names, config['motion'])
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self,
                                                      spin_thread=False)
        self.local_chain = None
        self.kinematic_chain = None
        self.use_moveit_ik = False
        self.robot_description = None
        self.create_subscription(
            String,
            '/robot_description',
            self._on_robot_description,
            LATCHED,
        )

    def _on_robot_description(self, message):
        self.robot_description = message.data

    def wait_for_robot_description(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while self.robot_description is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.robot_description

    def wait_for_robot(self) -> bool:
        if not self.args.dry_run:
            motion_action = self.config['motion']['controller_action']
            self.get_logger().info(f'waiting for native Franka PTP action {motion_action}')
            if not self.driver.wait_for_controller(timeout=10.0):
                self.get_logger().error(
                    f'native Franka PTP action {motion_action} is unavailable; '
                    'start franka_bringup without fr3_arm_controller')
                return False

        self.get_logger().info('waiting for /joint_states')
        if not self.driver.wait_for_state(timeout=10.0):
            self.get_logger().error(
                'no /joint_states received; start the robot MoveIt stack first')
            return False
        self.use_moveit_ik = self.ik_client.wait_for_service(timeout_sec=2.0)
        urdf_xml = self.wait_for_robot_description()
        if urdf_xml:
            self.get_logger().info(
                'using live /robot_description for local TCP IK/FK')
            self.kinematic_chain = build_chain(self.config, urdf_xml)
        else:
            self.get_logger().warning(
                'live /robot_description unavailable; falling back to packaged '
                'xacro for local TCP IK/FK')
            self.kinematic_chain = build_chain(self.config)
        if not self.use_moveit_ik:
            self.get_logger().warn('MoveIt IK unavailable; using local FR3 IK')
            self.local_chain = self.kinematic_chain
        self.get_logger().info('robot state and IK are ready')
        return True

    def current_tcp_position(self, base_frame: str) -> np.ndarray:
        """Return the live TCP position in *base_frame* from TF."""
        source_frame = self.args.ik_link
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            if self.tf_buffer.can_transform(base_frame, source_frame, Time()):
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        else:
            raise RuntimeError(
                f'no TF transform from {base_frame!r} to {source_frame!r}')
        transform = self.tf_buffer.lookup_transform(base_frame, source_frame, Time())
        translation = transform.transform.translation
        return np.array([translation.x, translation.y, translation.z], dtype=float)

    def current_tcp_pose(self, base_frame: str) -> np.ndarray:
        """Return the latest live TCP pose in *base_frame* from TF."""
        source_frame = self.args.ik_link
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            if self.tf_buffer.can_transform(base_frame, source_frame, Time()):
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        else:
            raise RuntimeError(
                f'no TF transform from {base_frame!r} to {source_frame!r}')
        transform = self.tf_buffer.lookup_transform(base_frame, source_frame, Time())
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return make_transform(
            quaternion_to_matrix([rotation.x, rotation.y, rotation.z, rotation.w]),
            [translation.x, translation.y, translation.z],
        )

    def solve_ik(self, pose: np.ndarray, frame_id: str) -> list[float]:
        if not self.use_moveit_ik:
            motion = self.config['motion']
            joints, converged = self.local_chain.ik_with_restarts(
                pose,
                seed=self.driver.positions,
                position_tolerance=float(motion.get('max_cartesian_position_error_m', 1e-4)),
                rotation_tolerance=float(motion.get('max_cartesian_rotation_error_rad', 8.0e-4)),
            )
            if not converged:
                raise RuntimeError('local IK could not solve this pose')
            return [float(value) for value in joints]

        request = GetPositionIK.Request()
        request.ik_request.group_name = self.args.move_group
        request.ik_request.ik_link_name = self.args.ik_link
        request.ik_request.avoid_collisions = True
        current_positions = self.driver.positions
        if current_positions is None:
            raise RuntimeError('current joint state is unavailable')
        request.ik_request.robot_state.joint_state.name = list(self.joint_names)
        request.ik_request.robot_state.joint_state.position = [
            float(value) for value in current_positions
        ]
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = frame_id
        target = request.ik_request.pose_stamped.pose
        target.position.x, target.position.y, target.position.z = pose[:3, 3]
        target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = \
            matrix_to_quaternion(pose[:3, :3])
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done():
            raise RuntimeError('timed out waiting for MoveIt IK')
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'MoveIt IK failed with error code {response.error_code.val}')
        positions = dict(zip(response.solution.joint_state.name,
                             response.solution.joint_state.position))
        missing = [name for name in self.joint_names if name not in positions]
        if missing:
            raise RuntimeError(f'MoveIt IK response is missing joints: {missing}')
        return [positions[name] for name in self.joint_names]

    def move_cartesian_pose(
            self,
            pose: np.ndarray,
            frame_id: str,
            speed_scale: float,
            pose_validator: Callable[[np.ndarray], None] | None = None) -> None:
        """Move to a Cartesian pose, correcting only a bounded measured residual."""
        motion = self.config['motion']
        maximum_translation_error = float(motion['max_cartesian_position_error_m'])
        maximum_rotation_error = float(motion['max_cartesian_rotation_error_rad'])
        correction_attempts = int(motion.get('cartesian_correction_attempts', 0))
        correction_window_position = float(
            motion.get('cartesian_correction_window_m', maximum_translation_error))
        correction_window_rotation = float(
            motion.get('cartesian_correction_window_rad', maximum_rotation_error))
        desired_pose = pose.copy()
        command_pose = desired_pose.copy()

        for attempt in range(correction_attempts + 1):
            if pose_validator is not None:
                pose_validator(command_pose)
            joints = self.solve_ik(command_pose, frame_id)
            if self.args.dry_run:
                self.get_logger().info(
                    f'dry-run IK: {np.array2string(np.asarray(joints), precision=4)}')
                return

            # Check the measured pose against the original desired pose, not
            # the corrected command. This makes the correction converge on the
            # task pose instead of merely moving the acceptance gate.
            ok, _, message = self.driver.move_to(
                joints,
                speed_scale,
                cartesian_target=desired_pose,
                kinematic_chain=self.kinematic_chain,
                cartesian_frame=frame_id,
            )
            actual_pose = self.driver.last_measured_tcp_pose
            if actual_pose is None:
                actual_pose = self.current_tcp_pose(frame_id)
            translation_error, rotation_error = pose_error(desired_pose, actual_pose)
            self.get_logger().info(
                f'achieved TCP error: {translation_error:.6f} m, '
                f'{rotation_error:.6f} rad'
                + (f' (correction {attempt}/{correction_attempts})'
                   if correction_attempts else ''))

            if ok and (translation_error <= maximum_translation_error
                       and rotation_error <= maximum_rotation_error):
                if message:
                    self.get_logger().warn(message)
                return

            if not ok and 'settled Cartesian error is' not in message:
                raise RuntimeError(message)
            if (translation_error <= maximum_translation_error
                    and rotation_error <= maximum_rotation_error):
                return
            if translation_error > correction_window_position:
                raise RuntimeError(
                    f'Cartesian position residual is {translation_error * 1000.0:.3f} mm; '
                    f'correction window is {correction_window_position * 1000.0:.3f} mm')
            if rotation_error > correction_window_rotation:
                raise RuntimeError(
                    f'Cartesian rotation residual is {np.degrees(rotation_error):.4f} deg; '
                    f'correction window is {np.degrees(correction_window_rotation):.4f} deg')
            if attempt >= correction_attempts:
                raise RuntimeError(
                    f'achieved TCP error exceeds limits: {translation_error:.6f} m '
                    f'(limit {maximum_translation_error:.6f} m), {rotation_error:.6f} rad '
                    f'(limit {maximum_rotation_error:.6f} rad)')

            command_pose = compensate_pose_for_measured_error(desired_pose, actual_pose)
            self.get_logger().warn(
                'measured TCP residual is within the correction window; '
                'retrying with a compensated Cartesian command')

        raise RuntimeError('Cartesian correction loop ended unexpectedly')


def wait_for_enter(stop_event: threading.Event) -> None:
    """Set *stop_event* when the operator presses Enter during descent."""
    try:
        input('Descending. Press Enter to stop at the current height... ')
    except EOFError:
        return
    stop_event.set()


def load_manipulation_config(path: str | None) -> dict:
    with open(path or default_config_path('board_manipulation.yaml'), 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board-pose', required=True, help='JSON artifact containing T_base_board')
    parser.add_argument('--offset', nargs=3, type=float, required=True, metavar=('X', 'Y', 'Z'),
                        help='pen-tip offset in board axes, in metres')
    parser.add_argument(
        '--config',
        help=('robot calibration YAML; defaults to calibration_hardware.yaml, which '
              'selects /action_server/ptp_motion'))
    parser.add_argument('--manipulation-config', help='board probing YAML')
    parser.add_argument('--move-group', default='fr3_arm')
    parser.add_argument('--ik-link', default='fr3_hand_tcp')
    parser.add_argument('--staging-z', type=float, default=None,
                        help='fixed TCP Z in the base frame for the XY phase')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate artifact and IK without sending trajectories')
    parser.add_argument('--yes', action='store_true',
                        help='skip the approach confirmation (never skip physical e-stop checks)')
    args = parser.parse_args(argv)

    try:
        base_frame, board_transform, surface_z = load_board_pose(args.board_pose)
        config = load_config(args.config or default_config_path('calibration_hardware.yaml'))
        validate_probe_motion_config(config)
        manipulation = load_manipulation_config(args.manipulation_config)
        probe = manipulation['probe']
        orientation = np.asarray(probe['tcp_orientation_board_rpy'], dtype=float)
        pen_tip_offset = np.asarray(probe['pen_tip_offset_tcp_m'], dtype=float)
        offset = np.asarray(args.offset, dtype=float)
        if orientation.shape != (3,) or pen_tip_offset.shape != (3,):
            raise ValueError('tool orientation and pen_tip_offset_tcp_m must each have three values')
        if not (np.all(np.isfinite(orientation)) and np.all(np.isfinite(pen_tip_offset))
                and np.all(np.isfinite(offset))):
            raise ValueError('offset and tool configuration values must be finite')
        approach_height = float(probe['approach_height_m'])
        segment = float(probe['descent_segment_m'])
        maximum = float(probe['max_descent_m'])
        if approach_height <= 0 or segment <= 0 or maximum <= 0:
            raise ValueError('approach_height_m, descent_segment_m, and max_descent_m must be positive')
        target, approach = board_target_pose(
            board_transform, surface_z, offset, approach_height, orientation, pen_tip_offset)
        board_point = board_transform @ np.array(
            [offset[0], offset[1], surface_z + offset[2], 1.0], dtype=float)
        board_normal = board_transform[:3, 2]
        board_normal /= np.linalg.norm(board_normal)
    except (OSError, ValueError, KeyError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2

    print(f'base frame: {base_frame}')
    print(f'board offset [m]: {args.offset}')
    print(f'pen-tip target in base [m]: {board_point[:3].tolist()}')
    print(f'TCP target in base [m]: {target[:3, 3].tolist()}')
    if args.dry_run:
        print('Dry-run: no Franka PTP goals will be sent.')
    elif not args.yes:
        input('Press Enter to approach the point, or Ctrl-C to abort... ')

    rclpy.init()
    node = BoardManipulation(config, args)
    try:
        if not node.wait_for_robot():
            return 1
        configured_staging_z = probe.get('staging_z_m')
        if args.staging_z is not None:
            staging_z = float(args.staging_z)
        elif configured_staging_z is not None:
            staging_z = float(configured_staging_z)
        else:
            staging_z = None

        if staging_z is None:
            # The configured approach height is measured from the board plane,
            # so it remains safe when the TCP-to-tip offset is nonzero.
            staging = approach.copy()
            staging_source = 'board-normal approach'
        else:
            if not np.isfinite(staging_z):
                raise RuntimeError('staging Z must be finite')
            staging = target.copy()
            staging[2, 3] = staging_z
            staging_source = f'explicit TCP z={staging_z:.3f} m'

        staging_clearance = signed_board_clearance(
            staging, pen_tip_offset, board_point[:3], board_normal)
        required_clearance = approach_height
        if staging_clearance + 1e-9 < required_clearance:
            raise RuntimeError(
                f'staging pen-tip clearance is {staging_clearance * 1000:.1f} mm; '
                f'requires at least {required_clearance * 1000:.1f} mm above the board '
                f'(TCP-to-tip offset or staging height is unsafe)')

        def validate_pen_clearance(command_pose: np.ndarray) -> None:
            clearance = signed_board_clearance(
                command_pose, pen_tip_offset, board_point[:3], board_normal)
            if clearance < -1e-8:
                raise RuntimeError(
                    f'commanded pen tip would be {abs(clearance) * 1000:.2f} mm '
                    'below the board plane; refusing to move')

        descent_vector = target[:3, 3] - staging[:3, 3]
        available_descent = float(np.linalg.norm(descent_vector))
        if available_descent <= 1e-9:
            raise RuntimeError('staging pose and target pose are indistinguishable')
        descent_direction = descent_vector / available_descent
        descent_limit = min(maximum, available_descent)
        node.get_logger().info(
            f'phase 1: moving to board-relative approach pose '
            f'({staging_source}, pen-tip clearance '
            f'{staging_clearance * 1000:.1f} mm)')
        node.move_cartesian_pose(
            staging, base_frame, float(probe['approach_speed_scale']),
            pose_validator=validate_pen_clearance)
        if args.dry_run:
            descent_target = target.copy()
            node.move_cartesian_pose(
                descent_target, base_frame, float(probe['descent_speed_scale']),
                pose_validator=validate_pen_clearance)
            return 0

        node.get_logger().info(
            f'phase 2: descending along the approach path at '
            f'{float(probe["descent_speed_scale"]):.3f} speed scale '
            f'(limit {descent_limit * 1000:.1f} mm)')
        stop_event = threading.Event()
        threading.Thread(target=wait_for_enter, args=(stop_event,), daemon=True).start()
        descended = 0.0
        while descended < descent_limit and not stop_event.is_set():
            step = min(segment, descent_limit - descended)
            next_pose = staging.copy()
            next_pose[:3, 3] = (
                staging[:3, 3] + descent_direction * (descended + step))
            next_clearance = signed_board_clearance(
                next_pose, pen_tip_offset, board_point[:3], board_normal)
            if next_clearance < -1e-8:
                raise RuntimeError(
                    f'descent would place the pen tip {abs(next_clearance) * 1000:.2f} mm '
                    'below the board plane; refusing to continue')
            node.move_cartesian_pose(
                next_pose,
                base_frame,
                float(probe['descent_speed_scale']),
                pose_validator=validate_pen_clearance)
            descended += step
            node.get_logger().info(f'descended {descended * 1000:.1f} mm')

        if not stop_event.is_set():
            node.get_logger().error('maximum descent reached without an operator stop')
            node.get_logger().warn('retracting to the configured approach pose')
            node.move_cartesian_pose(
                staging,
                base_frame,
                float(probe['approach_speed_scale']),
                pose_validator=validate_pen_clearance)
            return 1
        node.get_logger().info(f'stopped after {descended * 1000:.1f} mm; holding position')
        input('Press Enter to retract along the approach path, or Ctrl-C to hold... ')
        node.move_cartesian_pose(
            staging,
            base_frame,
            float(probe['approach_speed_scale']),
            pose_validator=validate_pen_clearance)
        return 0
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        if rclpy.ok():
            node.get_logger().error(f'aborted: {error}')
        else:
            print(f'aborted: {error}', file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
