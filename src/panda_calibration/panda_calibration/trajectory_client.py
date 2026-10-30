"""Drive the arm to joint configurations through the configured action.

Board probing uses Franka's native ``/action_server/ptp_motion`` action. The
legacy ``FollowJointTrajectory`` path remains available for simulation and
calibration workflows that explicitly select it in their configuration.

Everything runs on the calling thread: goals are awaited with
``spin_until_future_complete`` and settling is polled with ``spin_once``, so the
capture sequence reads as a straight-line script.

Timing never uses the wall clock for correctness.  Nothing in this workspace
sets ``use_sim_time`` and ``robot_state_publisher`` runs on system time while
Gazebo publishes ``/clock``, so wall-clock sleeps would be wrong whenever the
real-time factor is not 1.  Settling is decided from joint velocity and position
error instead; wall-clock values are only ever used as give-up timeouts.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from franka_msgs.action import PTPMotion
from franka_msgs.msg import TargetStatus
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ArmDriver:
    """Point-to-point joint moves with velocity-based settle detection."""

    def __init__(self, node, joint_names, motion_config: dict, timeout_scale: float = 1.0):
        self.node = node
        self.joint_names = list(joint_names)
        self.config = motion_config
        self.timeout_scale = float(timeout_scale)

        self._positions = None
        self._velocities = None
        self._stamp = 0.0
        self._state_stamp_s = None
        self._joint_history = deque(maxlen=20000)
        self._last_settled_state = None
        self._preferred_state_seen = False
        self._measured_tcp_pose = None
        self._measured_tcp_pose_frame = None
        self._measured_tcp_pose_received_monotonic_s = 0.0

        self.state_topic = motion_config.get('measured_joint_state_topic', '/joint_states')
        if self.state_topic == '/joint_states':
            self.node.create_subscription(JointState, self.state_topic,
                                          self._on_joint_state, 10)
        else:
            self.node.create_subscription(
                JointState,
                self.state_topic,
                self._on_preferred_joint_state,
                qos_profile_sensor_data,
            )
            # Keep the aggregated topic as a startup fallback. Once a valid
            # message from the preferred high-rate source arrives, it no
            # longer updates the driver state.
            self.node.create_subscription(
                JointState,
                '/joint_states',
                self._on_fallback_joint_state,
                10,
            )
        self.measured_tcp_pose_topic = motion_config.get('measured_tcp_pose_topic')
        self.measured_tcp_pose_frame = motion_config.get('measured_tcp_pose_frame')
        if self.measured_tcp_pose_topic:
            self.node.create_subscription(
                PoseStamped,
                self.measured_tcp_pose_topic,
                self._on_measured_tcp_pose,
                qos_profile_sensor_data,
            )
        self.action_type = str(motion_config.get('action_type', 'follow_joint_trajectory'))
        if self.action_type == 'ptp_motion':
            action_message = PTPMotion
        elif self.action_type == 'follow_joint_trajectory':
            action_message = FollowJointTrajectory
        else:
            raise ValueError(f'unsupported motion action type: {self.action_type}')
        self.client = ActionClient(self.node, action_message,
                                   motion_config['controller_action'])
        self.node.get_logger().info(
            f'using measured joint states from {self.state_topic}')
        if self.measured_tcp_pose_topic:
            self.node.get_logger().info(
                f'using measured TCP poses from {self.measured_tcp_pose_topic}')

    # -- state ------------------------------------------------------------- #

    def _on_joint_state(self, msg: JointState) -> bool:
        index = {name: i for i, name in enumerate(msg.name)}
        if not all(name in index for name in self.joint_names):
            return False
        self._positions = np.array([msg.position[index[n]] for n in self.joint_names])
        if len(msg.velocity) == len(msg.position):
            self._velocities = np.array([msg.velocity[index[n]] for n in self.joint_names])
        else:
            self._velocities = np.zeros(len(self.joint_names))
        self._stamp = time.monotonic()
        stamp = msg.header.stamp
        self._state_stamp_s = stamp.sec + stamp.nanosec * 1e-9
        self._joint_history.append({
            'header_stamp_s': float(self._state_stamp_s),
            'received_monotonic_s': float(self._stamp),
            'q_rad': self._positions.copy(),
        })
        return True

    def _on_preferred_joint_state(self, msg: JointState) -> None:
        if self._on_joint_state(msg):
            self._preferred_state_seen = True

    def _on_fallback_joint_state(self, msg: JointState) -> None:
        if not self._preferred_state_seen:
            self._on_joint_state(msg)

    def _on_measured_tcp_pose(self, msg: PoseStamped) -> None:
        from .kinematics import make_transform, quaternion_to_matrix

        position = msg.pose.position
        orientation = msg.pose.orientation
        try:
            self._measured_tcp_pose = make_transform(
                quaternion_to_matrix([
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                ]),
                [position.x, position.y, position.z],
            )
        except ValueError:
            self._measured_tcp_pose = None
            return
        self._measured_tcp_pose_frame = msg.header.frame_id
        self._measured_tcp_pose_received_monotonic_s = time.monotonic()

    @property
    def last_measured_tcp_pose(self):
        return (None if self._measured_tcp_pose is None
                else self._measured_tcp_pose.copy())

    @property
    def last_measured_tcp_pose_frame(self):
        return self._measured_tcp_pose_frame

    @property
    def positions(self):
        return None if self._positions is None else self._positions.copy()

    @property
    def velocities(self):
        return None if self._velocities is None else self._velocities.copy()

    @property
    def state_stamp_s(self):
        return self._state_stamp_s

    @property
    def state_received_monotonic_s(self):
        return self._stamp

    @property
    def last_settled_state(self):
        if self._last_settled_state is None:
            return None
        return {
            'header_stamp_s': self._last_settled_state['header_stamp_s'],
            'received_monotonic_s': self._last_settled_state['received_monotonic_s'],
            'q_rad': self._last_settled_state['q_rad'].copy(),
            'window_joint_spread_rad': self._last_settled_state['window_joint_spread_rad'].copy(),
        }

    def joint_bracket(self, received_monotonic_s):
        """Return measured joint samples immediately around an event."""
        before = None
        after = None
        for sample in reversed(self._joint_history):
            if sample['received_monotonic_s'] <= received_monotonic_s:
                before = sample
                break
        for sample in self._joint_history:
            if sample['received_monotonic_s'] > received_monotonic_s:
                after = sample
                break
        return before, after

    def wait_for_joint_after(self, received_monotonic_s, timeout: float):
        """Wait until a measured joint-state sample follows an image callback."""
        deadline = time.monotonic() + timeout * self.timeout_scale
        while time.monotonic() < deadline:
            before, after = self.joint_bracket(received_monotonic_s)
            if before is not None and after is not None:
                return before, after
            rclpy.spin_once(self.node, timeout_sec=0.01)
        before, after = self.joint_bracket(received_monotonic_s)
        return (before, after) if before is not None and after is not None else None

    def wait_for_state(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout * self.timeout_scale
        while self._positions is None and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return self._positions is not None

    def wait_for_controller(self, timeout: float = 60.0) -> bool:
        return self.client.wait_for_server(timeout_sec=timeout * self.timeout_scale)

    # -- motion ------------------------------------------------------------ #

    def duration_for(self, target, speed_scale: float = 1.0) -> float:
        current = self.positions
        # No joint state yet -> assume the worst-case joint delta.
        delta = (float(np.max(np.abs(np.asarray(target) - current)))
                 if current is not None else math.pi)
        speed = float(self.config['max_joint_speed']) * max(speed_scale, 1e-3)
        return max(float(self.config['min_duration']), delta / speed)

    def move_to(self, target, speed_scale: float = 1.0,
                joint_goal_tolerances=None, cartesian_target=None,
                kinematic_chain=None, cartesian_frame=None) -> tuple:
        """Move to *target*.  Returns ``(ok, position_error, message)``.

        When ``joint_goal_tolerances`` is supplied, it is sent with the action
        goal and a controller tolerance violation fails the move.

        ``cartesian_target`` is checked after the joints have settled using
        ``kinematic_chain``.  For native PTP moves, its Jacobian also derives
        a per-target joint goal bound from the configured Cartesian limits. The
        post-move Cartesian check remains authoritative. The legacy trajectory
        controller path still uses only its explicit joint tolerances.

        A goal that the controller does not report as SUCCESSFUL is not fatal on
        its own: under effort control a joint can sit a few hundredths of a
        radian off and trip the controller's goal tolerance while the arm is
        perfectly stationary and perfectly usable for calibration -- what matters
        is the *measured* pose, which is recorded either way.  So the arm's own
        settled state is the deciding evidence, and only an error larger than
        ``motion.max_position_error`` fails the move.
        """
        target = np.asarray(target, dtype=float)
        if cartesian_target is not None and kinematic_chain is None:
            raise ValueError('kinematic_chain is required for a Cartesian target check')
        if self.action_type == 'ptp_motion':
            return self._move_to_ptp(
                target,
                speed_scale,
                cartesian_target,
                kinematic_chain,
                cartesian_frame,
            )

        duration = self.duration_for(target, speed_scale)

        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in target]
        point.velocities = [0.0] * len(target)
        point.time_from_start = Duration(sec=int(duration),
                                         nanosec=int((duration % 1.0) * 1e9))

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points = [point]
        if joint_goal_tolerances is not None:
            tolerances = np.asarray(joint_goal_tolerances, dtype=float)
            if tolerances.shape != (len(self.joint_names),) or not np.isfinite(tolerances).all():
                raise ValueError('joint goal tolerances must match the joint list and be finite')
            if np.any(tolerances <= 0.0):
                raise ValueError('joint goal tolerances must be positive')
            goal.goal_tolerance = [
                JointTolerance(name=name, position=float(tolerance))
                for name, tolerance in zip(self.joint_names, tolerances)
            ]
        # Leave header.stamp at zero: "start now", in the controller's own clock.

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future,
                                         timeout_sec=15.0 * self.timeout_scale)
        if not send_future.done():
            return False, None, 'timed out submitting the trajectory goal'
        handle = send_future.result()
        if not handle.accepted:
            return False, None, 'trajectory goal rejected by the controller'

        budget = (duration + float(self.config['goal_timeout_margin'])) * self.timeout_scale
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=budget)

        note = ''
        if not result_future.done():
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self.node, cancel_future, timeout_sec=5.0)
            return False, None, (
                f'controller did not report a result within {budget:.1f} s; '
                'cancellation requested'
            )
        else:
            result = result_future.result().result
            controller_succeeded = (
                result.error_code == FollowJointTrajectory.Result.SUCCESSFUL)
            if not controller_succeeded:
                note = (f'controller error {result.error_code}'
                        f'{": " + result.error_string if result.error_string else ""}')

        if not self.wait_until_stationary(budget):
            return False, None, f'joints never stopped moving ({note})' if note else \
                'joints never stopped moving'

        if joint_goal_tolerances is not None and not controller_succeeded:
            return False, None, (
                f'{note}; controller rejected the requested joint goal tolerance')

        error = float(np.max(np.abs(self.positions - target)))
        limit = float(self.config['max_position_error'])
        if error > limit:
            return False, error, (f'settled {error:.4f} rad from the target '
                                  f'(limit {limit:.4f})'
                                  + (f'; {note}' if note else ''))

        if cartesian_target is not None:
            from .kinematics import pose_error

            actual_pose = kinematic_chain.fk(self.positions)
            position_error, rotation_error = pose_error(cartesian_target, actual_pose)
            position_limit = float(self.config['max_cartesian_position_error_m'])
            rotation_limit = float(self.config['max_cartesian_rotation_error_rad'])
            if position_error > position_limit or rotation_error > rotation_limit:
                return False, error, (
                    f'settled Cartesian error is {position_error * 1000.0:.3f} mm '
                    f'and {math.degrees(rotation_error):.4f} deg '
                    f'(limits {position_limit * 1000.0:.3f} mm and '
                    f'{math.degrees(rotation_limit):.4f} deg)'
                    + (f'; {note}' if note else ''))
        return True, error, note

    def _move_to_ptp(self, target, speed_scale: float,
                     cartesian_target=None, kinematic_chain=None,
                     cartesian_frame=None) -> tuple:
        """Move through Franka's native smooth point-to-point action.

        The native action uses libfranka's ``AsyncPositionControlHandler``.  It
        avoids the single-point spline generated by the generic ROS 2
        ``JointTrajectoryController``, which can trip libfranka's motion
        generator discontinuity reflex on real hardware.
        """
        motion_started = time.monotonic()
        duration = self.duration_for(target, speed_scale)
        speed = float(self.config['max_joint_speed']) * max(speed_scale, 1e-3)
        current = self.positions
        if current is not None:
            joint_delta = float(np.max(np.abs(np.asarray(target) - current)))
            if joint_delta > 0.0:
                # ``min_duration`` is a real motion constraint, not only a
                # timeout floor.  Native PTP otherwise finishes small moves
                # faster than the old trajectory-based implementation.
                speed = min(speed, joint_delta / duration)
        # A tiny move (e.g. a sub-mm Cartesian correction) would otherwise get
        # a velocity limit near zero, which the native PTP cannot finish with.
        speed = max(speed, float(self.config.get('min_joint_speed', 0.02)))

        goal_tolerance = float(self.config.get('ptp_goal_tolerance', 0.002))
        if cartesian_target is not None:
            from .kinematics import conservative_joint_tolerances

            position_limit = float(self.config['max_cartesian_position_error_m'])
            rotation_limit = float(self.config['max_cartesian_rotation_error_rad'])
            derived_tolerance = conservative_joint_tolerances(
                kinematic_chain.jacobian(target),
                position_limit,
                rotation_limit,
            )
            # PTP exposes one scalar joint bound.  Use the configuration as a
            # ceiling and the Jacobian-derived bound as the task-specific
            # limit; the post-move Cartesian check remains authoritative.
            goal_tolerance = min(goal_tolerance, float(np.min(derived_tolerance)))

        goal = PTPMotion.Goal()
        goal.goal_joint_configuration = [float(value) for value in target]
        goal.maximum_joint_velocities = [speed] * len(target)
        goal.goal_tolerance = goal_tolerance
        self.node.get_logger().info(
            f'PTP limits: max joint speed {speed:.4f} rad/s, '
            f'goal tolerance {goal_tolerance:.7f} rad, '
            f'minimum move duration {duration:.2f} s')

        budget = (duration + float(self.config['goal_timeout_margin'])) * self.timeout_scale
        # The action server aborts a goal the robot never starts executing
        # (robot_mode stays idle; see franka_action_server.cpp). That happens
        # occasionally on real hardware and a resubmitted goal then runs.
        for attempt in range(3):
            ok, message = self._run_ptp_goal(goal, budget)
            if ok or 'never started the motion' not in message or attempt == 2:
                break
            self.node.get_logger().warn(f'{message}; resubmitting (attempt {attempt + 2}/3)')
            time.sleep(1.0)
        if not ok:
            return False, None, message

        if not self.wait_until_stationary(budget):
            return False, None, 'joints never stopped moving after Franka PTP motion'

        error = float(np.max(np.abs(self.positions - target)))
        limit = float(self.config['max_position_error'])
        if error > limit:
            return False, error, (
                f'settled {error:.4f} rad from the target (limit {limit:.4f})')

        if cartesian_target is not None:
            from .kinematics import pose_error

            actual_pose, pose_source, pose_error_message = self._pose_for_cartesian_check(
                motion_started,
                cartesian_frame,
                kinematic_chain,
            )
            if actual_pose is None:
                return False, error, pose_error_message
            position_error, rotation_error = pose_error(cartesian_target, actual_pose)
            position_limit = float(self.config['max_cartesian_position_error_m'])
            rotation_limit = float(self.config['max_cartesian_rotation_error_rad'])
            if position_error > position_limit or rotation_error > rotation_limit:
                return False, error, (
                    f'settled Cartesian error is {position_error * 1000.0:.3f} mm '
                    f'and {math.degrees(rotation_error):.4f} deg '
                    f'(limits {position_limit * 1000.0:.3f} mm and '
                    f'{math.degrees(rotation_limit):.4f} deg); '
                    f'max joint error {error:.6f} rad '
                    f'(PTP tolerance {goal_tolerance:.6f} rad); '
                    f'pose source: {pose_source}')
            self._last_cartesian_pose = actual_pose
            self._last_cartesian_pose_source = pose_source
        return True, error, ''

    def _run_ptp_goal(self, goal, budget: float) -> tuple:
        """Submit one PTP goal and wait for it; returns ``(ok, message)``."""
        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self.node, send_future, timeout_sec=15.0 * self.timeout_scale)
        if not send_future.done():
            return False, 'timed out submitting the Franka PTP goal'
        handle = send_future.result()
        if not handle.accepted:
            return False, 'Franka PTP goal rejected'

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=budget)
        if not result_future.done():
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self.node, cancel_future, timeout_sec=5.0)
            return False, (f'Franka PTP action did not finish within {budget:.1f} s; '
                           'cancellation requested')

        result_response = result_future.result()
        result = result_response.result
        if (result_response.status != GoalStatus.STATUS_SUCCEEDED or
                result.target_status.status != TargetStatus.TARGET_REACHED):
            detail = result.error_message or f'target status {result.target_status.status}'
            return False, f'Franka PTP motion failed: {detail}'
        return True, ''

    def _pose_for_cartesian_check(self, motion_started, expected_frame, kinematic_chain):
        if self.measured_tcp_pose_topic:
            if (self._measured_tcp_pose is None
                    or self._measured_tcp_pose_received_monotonic_s <= motion_started):
                return None, None, (
                    'no fresh measured TCP pose received from '
                    f'{self.measured_tcp_pose_topic}')
            if expected_frame and self._measured_tcp_pose_frame != expected_frame:
                return None, None, (
                    f'measured TCP pose frame is {self._measured_tcp_pose_frame!r}, '
                    f'expected {expected_frame!r}')
            if (self.measured_tcp_pose_frame
                    and self._measured_tcp_pose_frame != self.measured_tcp_pose_frame):
                return None, None, (
                    f'measured TCP pose frame is {self._measured_tcp_pose_frame!r}, '
                    f'configured {self.measured_tcp_pose_frame!r}')
            return self._measured_tcp_pose.copy(), 'Franka measured TCP pose', None

        return kinematic_chain.fk(self.positions), 'FK of measured joints', None

    def wait_until_stationary(self, timeout: float) -> bool:
        """Wait until the measured joint positions stop changing.

        Deliberately position-based rather than velocity-based: fr3_joint7's
        velocity state interface reports a constant ~2.93 rad/s in Gazebo while
        the joint is standing still, so anything keyed on reported velocity never
        concludes.  Watching the positions themselves needs no such trust, and
        behaves the same on hardware.
        """
        tolerance = float(self.config['settle_motion'])
        hold = float(self.config['settle_hold'])

        deadline = time.monotonic() + timeout
        history = deque()
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if self._positions is None:
                continue
            now = time.monotonic()
            history.append((now, self._positions.copy()))
            # Drop samples only while the *next* one is still old enough, so the
            # window keeps exactly one sample at or beyond the hold horizon.
            while len(history) > 2 and now - history[1][0] >= hold:
                history.popleft()
            if now - history[0][0] < hold:
                continue
            window = np.array([q for _, q in history])
            if float(np.max(window.max(axis=0) - window.min(axis=0))) < tolerance:
                latest = self._joint_history[-1]
                self._last_settled_state = {
                    'header_stamp_s': latest['header_stamp_s'],
                    'received_monotonic_s': latest['received_monotonic_s'],
                    'q_rad': latest['q_rad'].copy(),
                    'window_joint_spread_rad': window.max(axis=0) - window.min(axis=0),
                }
                return True
        return False
