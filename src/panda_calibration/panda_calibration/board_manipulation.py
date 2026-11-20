"""Probe a board-relative point with the Panda.

The board pose is deliberately an input artifact.  Calibration and board-pose
refinement can run independently, then publish a JSON file with ``T_base_board``
for this program to consume. Every approach and descent move is sent to
franka_ros' ``position_joint_trajectory_controller`` (libfranka joint position
interface) as a single-point quintic; an effort-mode trajectory controller is
not accepted for probing, because it settles with a load-dependent residual.

With ``--record FILE`` (``pipeline accuracy``) every attempt of every move is
written to FILE: the commanded TCP pose and joints and where the arm settled
(see probe_accuracy.py). The arm then also holds at the bottom when the descent
reaches the target, for photos, and asks for a note before it retracts.

The Panda is redundant, so one TCP pose has many joint solutions. ``--ik-seed``
names the one the caller planned for (``pipeline check/probe`` pass the joints
their board target was corrected at): every IK starts there and must stay
within ``--max-posture-change`` of the previous solution, so the arm cannot
flip its shoulder or wind joints 1 and 3 to their limits on the way to a hole.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Callable

import numpy as np
import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

from . import probe_accuracy
from .capture_node import robot_description
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
from .trajectory_client import ArmDriver, spin_once


PROBE_ACTION_TYPE = 'follow_joint_trajectory'
PROBE_ACTION = '/position_joint_trajectory_controller/follow_joint_trajectory'


def compensate_pose_for_measured_error(command: np.ndarray,
                                      desired: np.ndarray,
                                      actual: np.ndarray) -> np.ndarray:
    """Return *command* shifted by the measured TCP residual *desired* - *actual*.

    The shift accumulates on the previous command: restarting from *desired*
    would undo the last correction, so a repeatable sag would alternate between
    corrected and uncorrected poses instead of converging.
    """
    corrected = command.copy()
    corrected[:3, 3] += desired[:3, 3] - actual[:3, 3]
    # O_T_EE is orthonormal only to about 3e-5 on the Panda. Project the result
    # back onto a rotation: MoveIt's KDL IK fails (NO_IK_SOLUTION) on the
    # non-unit quaternion the skewed matrix would give.
    u, _, vt = np.linalg.svd(desired[:3, :3] @ actual[:3, :3].T @ command[:3, :3])
    corrected[:3, :3] = u @ vt
    return corrected


def pen_tip_report(tcp_pose: np.ndarray, pen_tip_offset, board_transform: np.ndarray,
                   target_point) -> str:
    """Where the pen tip is (TCP pose + TCP->tip offset) and how far from the target, in board axes."""
    tip = tcp_pose[:3, :3] @ np.asarray(pen_tip_offset, dtype=float) + tcp_pose[:3, 3]
    rel = board_transform[:3, :3].T @ (tip - np.asarray(target_point, dtype=float)[:3])
    rel = np.round(rel, 5) + 0.0  # to the printed 0.01 mm, without -0.00
    height = 'above' if rel[2] >= 0 else 'below'
    return (f'pen tip in base [m]: {np.round(tip, 5).tolist()}; from the target (board axes): '
            f'x {rel[0] * 1000:+.2f}, y {rel[1] * 1000:+.2f} mm '
            f'({np.hypot(rel[0], rel[1]) * 1000:.2f} mm beside), '
            f'{abs(rel[2]) * 1000:.2f} mm {height} it')


def posture_change(joints, reference) -> float:
    """Largest single-joint difference (rad) between two joint solutions."""
    return float(np.max(np.abs(np.asarray(joints, dtype=float)
                               - np.asarray(reference, dtype=float))))


def validate_probe_motion_config(config: dict) -> None:
    """Require the joint *position* trajectory controller for real board probing.

    franka_control also offers ``effort_joint_trajectory_controller``, whose PD
    gains leave a load-dependent steady-state error far above the 0.1 mm probe
    tolerance. Board probing must therefore never silently use it.
    """
    motion = config.get('motion')
    if not isinstance(motion, dict):
        raise ValueError('robot configuration is missing the motion section')
    action_type = motion.get('action_type')
    action_name = motion.get('controller_action')
    if action_type != PROBE_ACTION_TYPE or action_name != PROBE_ACTION:
        raise ValueError(
            'probe_board requires the joint position trajectory controller: '
            f'action_type={PROBE_ACTION_TYPE!r}, '
            f'controller_action={PROBE_ACTION!r}; '
            f'got action_type={action_type!r}, controller_action={action_name!r}')


class BoardManipulation:
    def __init__(self, config: dict, args):
        validate_probe_motion_config(config)
        self.config = config
        self.args = args
        self.joint_names = config['robot']['joint_names']
        self.driver = ArmDriver(self.joint_names, config['motion'])
        self.ik_service = config['robot'].get('ik_service', '/compute_ik')
        self.ik_client = rospy.ServiceProxy(self.ik_service, GetPositionIK)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.local_chain = None
        self.kinematic_chain = None
        self.use_moveit_ik = False
        self.recorder = None   # probe_accuracy.Recorder with --record
        # The joint solution every IK must stay near: --ik-seed at first, then
        # the last accepted solution. None (a plain run without --ik-seed)
        # leaves the first IK free, seeded at the current joints as before.
        seed = getattr(args, 'ik_seed', None)
        self.posture = None if seed is None else np.asarray(seed, dtype=float)
        self.max_posture_change = float(getattr(args, 'max_posture_change', 0.5))

    def shutdown(self) -> None:
        self.driver.shutdown()

    def wait_for_robot_description(self, timeout=10.0):
        return robot_description(timeout)

    def ik_available(self, timeout: float) -> bool:
        try:
            rospy.wait_for_service(self.ik_service, timeout=timeout)
            return True
        except rospy.ROSException:
            return False

    def wait_for_robot(self) -> bool:
        # --where only reads the robot state, so it also works with the
        # controller stopped (hand guiding).
        moving = not (self.args.dry_run or self.args.where)
        if moving:
            motion_action = self.config['motion']['controller_action']
            rospy.loginfo(f'waiting for trajectory action {motion_action}')
            if not self.driver.wait_for_controller(timeout=10.0):
                rospy.logerr(
                    f'trajectory action {motion_action} is unavailable; start '
                    'panda_calibration robot.launch (it runs position_joint_trajectory_controller)')
                return False

        rospy.loginfo('waiting for /joint_states')
        if not self.driver.wait_for_state(timeout=10.0):
            rospy.logerr(
                'no /joint_states received; start the robot MoveIt stack first')
            return False
        self.use_moveit_ik = self.ik_available(2.0)
        urdf_xml = self.wait_for_robot_description()
        if urdf_xml:
            rospy.loginfo(
                'using live /robot_description for local TCP IK/FK')
            self.kinematic_chain = build_chain(self.config, urdf_xml)
        else:
            rospy.logwarn(
                'live /robot_description unavailable; falling back to packaged '
                'xacro for local TCP IK/FK')
            self.kinematic_chain = build_chain(self.config)
        if not self.use_moveit_ik:
            rospy.logwarn('MoveIt IK unavailable; using local Panda IK')
            self.local_chain = self.kinematic_chain
        frame_problem = self.driver.check_measured_tcp_frame(self.kinematic_chain)
        if frame_problem:
            if moving:
                rospy.logerr(frame_problem)
                return False
            rospy.logwarn(frame_problem)
        rospy.loginfo('robot state and IK are ready')
        return True

    def _lookup(self, base_frame: str):
        source_frame = self.args.ik_link
        deadline = time.monotonic() + 2.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.tf_buffer.can_transform(base_frame, source_frame, rospy.Time(0)):
                break
            spin_once(0.05)
        else:
            raise RuntimeError(
                f'no TF transform from {base_frame!r} to {source_frame!r}')
        return self.tf_buffer.lookup_transform(base_frame, source_frame, rospy.Time(0))

    def current_tcp_position(self, base_frame: str) -> np.ndarray:
        """Return the live TCP position in *base_frame* from TF."""
        translation = self._lookup(base_frame).transform.translation
        return np.array([translation.x, translation.y, translation.z], dtype=float)

    def current_tcp_pose(self, base_frame: str) -> np.ndarray:
        """Return the latest live TCP pose in *base_frame* from TF."""
        transform = self._lookup(base_frame)
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return make_transform(
            quaternion_to_matrix([rotation.x, rotation.y, rotation.z, rotation.w]),
            [translation.x, translation.y, translation.z],
        )

    def measured_tcp(self, base_frame: str) -> np.ndarray:
        """The robot's measured TCP pose (O_T_EE), or the TF one without it."""
        pose = self.driver.last_measured_tcp_pose
        return pose if pose is not None else self.current_tcp_pose(base_frame)

    def solve_ik(self, pose: np.ndarray, frame_id: str) -> list[float]:
        """Joint solution for a TCP pose, kept near the reference posture.

        The reference is ``--ik-seed`` for the first solve and the previous
        solution afterwards. MoveIt's KDL solver restarts from random joints
        when it fails from its seed and then returns any branch (2026-11-05:
        joint 1 on its +2.897 limit, where the planned solution had -0.35), so
        a solution farther than ``--max-posture-change`` from the reference is
        not used: the local IK finds the nearby one and MoveIt, seeded with it,
        has to confirm it (joint limits, collisions).
        """
        reference = self.posture
        seed = self.driver.positions if reference is None else reference
        if seed is None:
            raise RuntimeError('current joint state is unavailable')

        if not self.use_moveit_ik:
            joints = self._local_ik(pose, seed, restarts=reference is None)
        else:
            joints = self._moveit_ik(pose, frame_id, seed)
            if reference is not None and \
                    posture_change(joints, reference) > self.max_posture_change:
                rospy.logwarn(
                    f'MoveIt IK left the planned posture (a joint differs by '
                    f'{posture_change(joints, reference):.2f} rad); solving next to it instead')
                nearby = self._local_ik(pose, reference, restarts=False)
                joints = self._moveit_ik(pose, frame_id, nearby)
        if reference is not None:
            change = posture_change(joints, reference)
            if change > self.max_posture_change:
                raise RuntimeError(
                    f'no usable IK solution near the planned posture: the nearest one moves a '
                    f'joint by {change:.2f} rad (limit {self.max_posture_change:.2f} rad); the '
                    'nearby solution is at a joint limit or in collision. Move the arm closer '
                    'to the hole (or the board away from the limit) and run it again')
            rospy.loginfo(f'IK: largest joint change from the planned posture {change:.3f} rad')
        self.posture = np.asarray(joints, dtype=float)
        return [float(value) for value in joints]

    def _local_ik(self, pose: np.ndarray, seed, restarts: bool) -> list[float]:
        chain = self.local_chain or self.kinematic_chain
        if restarts:
            motion = self.config['motion']
            joints, converged = chain.ik_with_restarts(
                pose,
                seed=seed,
                position_tolerance=float(motion.get('max_cartesian_position_error_m', 1e-4)),
                rotation_tolerance=float(motion.get('max_cartesian_rotation_error_rad', 8.0e-4)),
            )
        else:
            # Next to a planned posture: no restarts, and the redundant degree
            # of freedom is pulled towards that posture, not towards READY.
            joints, converged = chain.ik(
                pose, seed, nullspace_target=seed,
                position_tolerance=2.0e-5, rotation_tolerance=1.0e-4)
        if not converged:
            raise RuntimeError('local IK could not solve this pose'
                               + ('' if restarts else ' near the planned posture'))
        return [float(value) for value in joints]

    def _moveit_ik(self, pose: np.ndarray, frame_id: str, seed) -> list[float]:
        request = GetPositionIKRequest()
        request.ik_request.group_name = self.args.move_group
        request.ik_request.timeout = rospy.Duration(0.5)
        request.ik_request.ik_link_name = self.args.ik_link
        request.ik_request.avoid_collisions = True
        request.ik_request.robot_state.joint_state.name = list(self.joint_names)
        request.ik_request.robot_state.joint_state.position = [float(value) for value in seed]
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = frame_id
        target = request.ik_request.pose_stamped.pose
        target.position.x, target.position.y, target.position.z = pose[:3, 3]
        target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = \
            matrix_to_quaternion(pose[:3, :3])
        try:
            response = self.ik_client(request)
        except rospy.ServiceException as error:
            raise RuntimeError(f'MoveIt IK call failed: {error}') from error
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
            pose_validator: Callable[[np.ndarray], None] | None = None,
            record: dict | None = None) -> None:
        """Move to a Cartesian pose, correcting only a bounded measured residual.

        *record* names the move for the recorder (``phase`` and, while descending,
        ``depth_mm``); every attempt is then recorded, the uncorrected first one too.
        """
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
        move = (self.recorder.start_move(**record)
                if self.recorder is not None and record is not None and not self.args.dry_run else None)

        for attempt in range(correction_attempts + 1):
            if pose_validator is not None:
                pose_validator(command_pose)
            joints = self.solve_ik(command_pose, frame_id)
            if self.args.dry_run:
                rospy.loginfo(
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
            rospy.loginfo(
                f'achieved TCP error: {translation_error:.6f} m, '
                f'{rotation_error:.6f} rad'
                + (f' (correction {attempt}/{correction_attempts})'
                   if correction_attempts else ''))
            if move is not None:
                entry = self.recorder.attempt(move, attempt + 1, desired_pose, command_pose, joints,
                                              self.driver.positions, actual_pose, ok, message)
                rospy.loginfo(probe_accuracy.describe(record['phase'], entry))

            if ok and (translation_error <= maximum_translation_error
                       and rotation_error <= maximum_rotation_error):
                if message:
                    rospy.logwarn(message)
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

            command_pose = compensate_pose_for_measured_error(
                command_pose, desired_pose, actual_pose)
            rospy.logwarn(
                'measured TCP residual is within the correction window; '
                'retrying with a compensated Cartesian command')

        raise RuntimeError('Cartesian correction loop ended unexpectedly')


def wait_for_enter(stop_event: threading.Event, typed: list | None = None) -> None:
    """Set *stop_event* when the operator presses Enter during descent; keep the line in *typed*."""
    try:
        line = input('Descending. Press Enter to stop at the current height... ')
    except EOFError:
        return
    if typed is not None:
        typed.append(line)
    stop_event.set()


PHOTO_PROMPT = ('Take the photos, then type what you see (optional, e.g. "in", "touches the +x edge") '
                'and press Enter to retract. Ctrl-C leaves the arm here and exits.')


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
              'selects position_joint_trajectory_controller'))
    parser.add_argument('--manipulation-config', help='board probing YAML')
    parser.add_argument('--move-group', default='panda_manipulator',
                        help='MoveIt group whose tip is --ik-link (panda_moveit_config)')
    parser.add_argument('--ik-link', default='panda_hand_tcp')
    parser.add_argument('--staging-z', type=float, default=None,
                        help='fixed TCP Z in the base frame for the XY phase')
    parser.add_argument('--ik-seed', nargs=7, type=float, default=None,
                        metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7'),
                        help='joint solution (rad) planned for the target; the IK starts '
                             'there and stays near it instead of at the current joints')
    parser.add_argument('--max-posture-change', type=float, default=0.5, metavar='RAD',
                        help='largest joint difference accepted between an IK solution and '
                             'the previous one (--ik-seed for the first), default 0.5')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate artifact and IK without sending trajectories')
    parser.add_argument('--yes', action='store_true',
                        help='skip the approach confirmation (never skip physical e-stop checks)')
    parser.add_argument('--where', action='store_true',
                        help='print where the pen tip is relative to the target and exit; no motion, '
                             'works with the controller stopped (hand guiding)')
    parser.add_argument('--watch', action='store_true',
                        help='with --where: keep printing twice a second until Ctrl-C')
    parser.add_argument('--record', metavar='FILE',
                        help='write the commanded and measured TCP pose and joints of every move to FILE '
                             '(YAML, see probe_accuracy.py); the arm then also holds at the bottom when '
                             'the descent reaches the target, for photos, and asks for a note before '
                             'it retracts')
    args = parser.parse_args(argv if argv is not None else rospy.myargv(sys.argv)[1:])
    if args.watch and not args.where:
        parser.error('--watch needs --where')
    if args.record and (args.dry_run or args.where):
        parser.error('--record needs a probe that moves (not --dry-run or --where)')

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
        if args.ik_seed is not None and not np.all(np.isfinite(args.ik_seed)):
            raise ValueError('--ik-seed joint angles must be finite')
        if not (np.isfinite(args.max_posture_change) and args.max_posture_change > 0):
            raise ValueError('--max-posture-change must be positive')
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
        print('Dry-run: no trajectory goals will be sent.')
    elif not args.yes and not args.where:
        input('Press Enter to approach the point, or Ctrl-C to abort... ')

    rospy.init_node('board_manipulation', anonymous=True, disable_signals=True)
    # The node log is the only record of a run: keep what was asked for in it.
    rospy.loginfo(f'board pose {args.board_pose}, board offset [m] {list(args.offset)}')
    rospy.loginfo(f'pen-tip target in base [m]: {np.round(board_point[:3], 5).tolist()}; '
                  f'TCP target in base [m]: {np.round(target[:3, 3], 5).tolist()}')
    if args.ik_seed is not None:
        rospy.loginfo(f'planned posture (--ik-seed): {np.round(args.ik_seed, 4).tolist()}')
    node = BoardManipulation(config, args)

    def pen_tip() -> str:
        return pen_tip_report(node.measured_tcp(base_frame), pen_tip_offset,
                              board_transform, board_point)

    try:
        if not node.wait_for_robot():
            return 1
        if args.where:
            while True:
                line = f'current {pen_tip()}'
                age = time.monotonic() - node.driver.state_received_monotonic_s
                if age > 1.0:
                    line += f' (no robot state for {age:.0f} s)'
                print(line, flush=True)
                if not args.watch:
                    return 0
                time.sleep(0.5)  # the subscriber threads keep the measured pose fresh
        print(f'current {pen_tip()}', flush=True)
        if args.record:
            node.recorder = probe_accuracy.Recorder(
                args.record, base_frame, board_transform, board_point, target, pen_tip_offset)
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
        rospy.loginfo(
            f'phase 1: moving to board-relative approach pose '
            f'({staging_source}, pen-tip clearance '
            f'{staging_clearance * 1000:.1f} mm)')
        node.move_cartesian_pose(
            staging, base_frame, float(probe['approach_speed_scale']),
            pose_validator=validate_pen_clearance, record={'phase': 'approach'})
        if not args.dry_run:
            rospy.loginfo(f'at the approach pose; {pen_tip()}')
        if args.dry_run:
            descent_target = target.copy()
            node.move_cartesian_pose(
                descent_target, base_frame, float(probe['descent_speed_scale']),
                pose_validator=validate_pen_clearance)
            return 0

        rospy.loginfo(
            f'phase 2: descending along the approach path at '
            f'{float(probe["descent_speed_scale"]):.3f} speed scale '
            f'(limit {descent_limit * 1000:.1f} mm)')
        stop_event = threading.Event()
        typed = []
        reader = threading.Thread(target=wait_for_enter, args=(stop_event, typed), daemon=True)
        reader.start()
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
                pose_validator=validate_pen_clearance,
                record={'phase': 'descent', 'depth_mm': (descended + step) * 1000})
            descended += step
            rospy.loginfo(f'descended {descended * 1000:.1f} mm; {pen_tip()}')

        recorder = node.recorder
        reached = not stop_event.is_set()
        if reached and recorder is None:
            rospy.logerr('maximum descent reached without an operator stop')
            rospy.logwarn('retracting to the configured approach pose')
            node.move_cartesian_pose(
                staging,
                base_frame,
                float(probe['approach_speed_scale']),
                pose_validator=validate_pen_clearance)
            rospy.loginfo(f'retracted; {pen_tip()}')
            return 1
        rospy.loginfo(f'{"reached the target" if reached else "stopped"} after {descended * 1000:.1f} mm; '
                      f'holding position; {pen_tip()}')
        if recorder is None:
            input('Press Enter to retract along the approach path, or Ctrl-C to hold... ')
        else:
            recorder.hold(probe_accuracy.TARGET_REACHED if reached else probe_accuracy.OPERATOR_STOP,
                          descended * 1000, node.driver.positions, node.measured_tcp(base_frame))
            print(f"\n>>> {recorder.data['hole']}: holding for photos, {descended * 1000:.1f} mm down, "
                  f"{time.strftime('%H:%M:%S')} <<<\n{PHOTO_PROMPT}", flush=True)
            if reached:
                # The descent's Enter reader still waits for a line: it takes the note.
                while reader.is_alive():
                    reader.join(0.2)
                note = typed[0] if typed else ''
            else:
                note = input('> ')
            recorder.note(note)
        node.move_cartesian_pose(
            staging,
            base_frame,
            float(probe['approach_speed_scale']),
            pose_validator=validate_pen_clearance,
            record={'phase': 'retract'})
        rospy.loginfo(f'retracted; {pen_tip()}')
        if recorder is not None:
            recorder.finish('done')
        return 0
    except KeyboardInterrupt:
        if node.recorder is not None:
            node.recorder.finish('interrupted', 'Ctrl-C')
        return 0 if args.where else 130
    except RuntimeError as error:
        if node.recorder is not None:
            node.recorder.finish('aborted', str(error))
        if not rospy.is_shutdown():
            rospy.logerr(f'aborted: {error}')
        else:
            print(f'aborted: {error}', file=sys.stderr)
        return 1
    finally:
        node.shutdown()
        rospy.signal_shutdown('done')


if __name__ == '__main__':
    sys.exit(main())
