"""Drive the Panda to joint configurations through franka_ros' trajectory controller.

Every move is one ``FollowJointTrajectory`` goal to
``position_joint_trajectory_controller`` (franka_control, libfranka joint
position interface): a single point with zero velocity and acceleration, so the
controller interpolates a quintic from the current state. This is the same
path franka_ros' own ``move_to_start`` uses.

rospy delivers subscription callbacks on their own threads, so state is
guarded by a lock and the waits below poll with short sleeps; the capture
sequence still reads as a straight-line script.

Timing never uses the wall clock for correctness.  Settling is decided from
the measured joint positions; wall-clock values are only ever used as give-up
timeouts.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

import actionlib
import numpy as np
import rospy
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import (FollowJointTrajectoryAction, FollowJointTrajectoryGoal,
                              FollowJointTrajectoryResult, JointTolerance)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ERROR_RECOVERY_COMMAND = ('rostopic pub -1 /franka_control/error_recovery/goal '
                          'franka_msgs/ErrorRecoveryActionGoal "{}"')
ERROR_RECOVERY_HINT = (
    f'if franka_control reports a reflex/error, recover with `{ERROR_RECOVERY_COMMAND}`')

# franka_msgs/FrankaState robot_mode. The arm is in MOVE whenever franka_control
# runs a command controller. In any other mode (after a reflex, a user stop)
# franka_control does not command the arm, but the trajectory controller keeps
# running: it accepts goals and plays them out while the arm stands still.
ROBOT_MODES = {0: 'OTHER', 1: 'IDLE', 2: 'MOVE', 3: 'GUIDING', 4: 'REFLEX',
               5: 'USER_STOPPED', 6: 'AUTOMATIC_ERROR_RECOVERY'}

GOAL_STATES = {getattr(GoalStatus, name): name for name in (
    'PENDING', 'ACTIVE', 'PREEMPTED', 'SUCCEEDED', 'ABORTED', 'REJECTED',
    'PREEMPTING', 'RECALLING', 'RECALLED', 'LOST')}


def spin_once(timeout_sec: float = 0.01) -> None:
    """rospy runs callbacks on its own threads; just yield for a moment."""
    if rospy.is_shutdown():
        raise KeyboardInterrupt('ROS shut down')
    time.sleep(timeout_sec)


def stamp_seconds(stamp) -> float:
    return stamp.secs + stamp.nsecs * 1e-9


def column_major_transform(values) -> np.ndarray:
    """franka_msgs/FrankaState O_T_EE (16 values, column-major) as a 4x4 matrix."""
    return np.asarray(values, dtype=float).reshape(4, 4).T


class ArmDriver:
    """Point-to-point joint moves with position-based settle detection."""

    def __init__(self, joint_names, motion_config: dict, timeout_scale: float = 1.0):
        self.joint_names = list(joint_names)
        self.config = motion_config
        self.timeout_scale = float(timeout_scale)

        self._lock = threading.Lock()
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
        self._robot_mode = None
        self._motion_errors = ()
        self._subscribers = []

        self.state_topic = motion_config.get('measured_joint_state_topic', '/joint_states')
        if self.state_topic == '/joint_states':
            self._subscribers.append(rospy.Subscriber(
                self.state_topic, JointState, self._on_joint_state, queue_size=10))
        else:
            self._subscribers.append(rospy.Subscriber(
                self.state_topic, JointState, self._on_preferred_joint_state,
                queue_size=100, tcp_nodelay=True))
            # Keep the aggregated topic as a startup fallback. Once a valid
            # message from the preferred high-rate source arrives, it no
            # longer updates the driver state.
            self._subscribers.append(rospy.Subscriber(
                '/joint_states', JointState, self._on_fallback_joint_state, queue_size=10))
        # franka_msgs/FrankaState: O_T_EE is the robot's own measured pose of
        # the end-effector frame configured in Desk (Franka Hand: the hand TCP).
        self.measured_tcp_pose_topic = motion_config.get('measured_tcp_pose_topic')
        self.measured_tcp_pose_frame = motion_config.get('measured_tcp_pose_frame')
        if self.measured_tcp_pose_topic:
            from franka_msgs.msg import FrankaState
            self._subscribers.append(rospy.Subscriber(
                self.measured_tcp_pose_topic, FrankaState, self._on_franka_state,
                queue_size=1, tcp_nodelay=True))
        self.action_type = str(motion_config.get('action_type', 'follow_joint_trajectory'))
        if self.action_type != 'follow_joint_trajectory':
            raise ValueError(f'unsupported motion action type: {self.action_type} '
                             '(the Panda driver supports follow_joint_trajectory)')
        self.client = actionlib.SimpleActionClient(motion_config['controller_action'],
                                                   FollowJointTrajectoryAction)
        rospy.loginfo(f'using measured joint states from {self.state_topic}')
        if self.measured_tcp_pose_topic:
            rospy.loginfo(f'using measured TCP poses from {self.measured_tcp_pose_topic}')

    def shutdown(self) -> None:
        for subscriber in self._subscribers:
            subscriber.unregister()

    # -- state ------------------------------------------------------------- #

    def _on_joint_state(self, msg: JointState) -> bool:
        index = {name: i for i, name in enumerate(msg.name)}
        if not all(name in index for name in self.joint_names):
            return False
        positions = np.array([msg.position[index[n]] for n in self.joint_names])
        if len(msg.velocity) == len(msg.position):
            velocities = np.array([msg.velocity[index[n]] for n in self.joint_names])
        else:
            velocities = np.zeros(len(self.joint_names))
        received = time.monotonic()
        header_stamp = stamp_seconds(msg.header.stamp)
        with self._lock:
            self._positions = positions
            self._velocities = velocities
            self._stamp = received
            self._state_stamp_s = header_stamp
            self._joint_history.append({
                'header_stamp_s': float(header_stamp),
                'received_monotonic_s': float(received),
                'q_rad': positions.copy(),
            })
        return True

    def _on_preferred_joint_state(self, msg: JointState) -> None:
        if self._on_joint_state(msg):
            self._preferred_state_seen = True

    def _on_fallback_joint_state(self, msg: JointState) -> None:
        if not self._preferred_state_seen:
            self._on_joint_state(msg)

    def _on_franka_state(self, msg) -> None:
        # franka_msgs/Errors is all bool flags: keep the names of the set ones.
        errors = tuple(name for name in msg.last_motion_errors.__slots__
                       if getattr(msg.last_motion_errors, name))
        with self._lock:
            self._robot_mode = int(msg.robot_mode)
            self._motion_errors = errors
        pose = column_major_transform(msg.O_T_EE)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return
        with self._lock:
            self._measured_tcp_pose = pose
            # franka_state_controller stamps O_T_EE in the arm's link0 frame.
            self._measured_tcp_pose_frame = msg.header.frame_id or self.measured_tcp_pose_frame
            self._measured_tcp_pose_received_monotonic_s = time.monotonic()

    @property
    def last_measured_tcp_pose(self):
        with self._lock:
            return (None if self._measured_tcp_pose is None
                    else self._measured_tcp_pose.copy())

    @property
    def last_measured_tcp_pose_frame(self):
        return self._measured_tcp_pose_frame

    def robot_mode_problem(self):
        """Why franka_control cannot move the arm right now, or None.

        Known only once a FrankaState has arrived (``measured_tcp_pose_topic``).
        """
        with self._lock:
            mode = self._robot_mode
            errors = self._motion_errors
        name = ROBOT_MODES.get(mode, str(mode))
        if mode is None or name == 'MOVE':
            return None
        if name == 'REFLEX':
            cause = f' after {", ".join(errors)}' if errors else ''
            return (f'the robot is in reflex mode{cause}; franka_control does not move the arm '
                    'until it is recovered: check that nothing presses on the arm or its '
                    f'cables, then run `{ERROR_RECOVERY_COMMAND}`')
        if name == 'USER_STOPPED':
            return f'the user stop is pressed; release it, then run `{ERROR_RECOVERY_COMMAND}`'
        return (f'robot mode is {name}, not MOVE, so franka_control is not commanding the arm; '
                f'check Desk (joints unlocked, FCI active), then run `{ERROR_RECOVERY_COMMAND}`')

    @property
    def positions(self):
        with self._lock:
            return None if self._positions is None else self._positions.copy()

    @property
    def velocities(self):
        with self._lock:
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
        with self._lock:
            history = list(self._joint_history)
        before = None
        after = None
        for sample in reversed(history):
            if sample['received_monotonic_s'] <= received_monotonic_s:
                before = sample
                break
        for sample in history:
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
            spin_once(0.005)
        before, after = self.joint_bracket(received_monotonic_s)
        return (before, after) if before is not None and after is not None else None

    def wait_for_state(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout * self.timeout_scale
        while self._positions is None and time.monotonic() < deadline:
            spin_once(0.05)
        return self._positions is not None

    def wait_for_controller(self, timeout: float = 60.0) -> bool:
        return self.client.wait_for_server(rospy.Duration(timeout * self.timeout_scale))

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

        ``cartesian_target`` is checked after the joints have settled, against
        the robot's measured TCP pose when ``measured_tcp_pose_topic`` is
        configured, else against the FK of the measured joints through
        ``kinematic_chain``.

        A goal that the controller does not report as SUCCESSFUL is not fatal on
        its own: franka_control's default goal tolerance is coarse and a joint
        can sit marginally outside it while the arm is stationary and perfectly
        usable for calibration -- what matters is the *measured* pose, which is
        recorded either way.  So the arm's own settled state is the deciding
        evidence, and only an error larger than ``motion.max_position_error``
        fails the move.

        A robot that is not in MOVE mode (reflex, user stop) fails the move
        before the goal is sent, and a reflex during the move is reported as
        such rather than as a large settle error.
        """
        target = np.asarray(target, dtype=float)
        if cartesian_target is not None and kinematic_chain is None:
            raise ValueError('kinematic_chain is required for a Cartesian target check')
        if target.shape != (len(self.joint_names),) or not np.isfinite(target).all():
            raise ValueError('joint target must match the joint list and be finite')
        problem = self.robot_mode_problem()
        if problem:
            return False, None, problem
        motion_started = time.monotonic()
        duration = self.duration_for(target, speed_scale)

        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in target]
        # Zero end velocity and acceleration: the controller fits a quintic,
        # so the command starts and ends without a jump in acceleration
        # (libfranka rejects acceleration discontinuities with a reflex).
        point.velocities = [0.0] * len(target)
        point.accelerations = [0.0] * len(target)
        point.time_from_start = rospy.Duration.from_sec(duration)

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = rospy.Duration.from_sec(
            float(self.config.get('goal_time_tolerance', 0.5)))
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
        rospy.loginfo(f'trajectory goal: {duration:.2f} s')

        self.client.send_goal(goal)
        budget = (duration + float(self.config['goal_timeout_margin'])) * self.timeout_scale
        finished = self.client.wait_for_result(rospy.Duration(budget))

        note = ''
        controller_succeeded = False
        if not finished:
            self.client.cancel_goal()
            return False, None, (
                f'controller did not report a result within {budget:.1f} s; '
                'cancellation requested')
        # A reflex during the move stops the arm, not the goal: the controller
        # plays it out and then fails it on the goal tolerance.
        problem = self.robot_mode_problem()
        if problem:
            return False, None, f'the arm stopped during the move: {problem}'
        state = self.client.get_state()
        result = self.client.get_result()
        if state != GoalStatus.SUCCEEDED and (
                result is None or result.error_code == FollowJointTrajectoryResult.SUCCESSFUL):
            # Aborted/rejected without a controller error code: the controller
            # refused the goal or was stopped.
            text = self.client.get_goal_status_text()
            return False, None, (
                f'trajectory goal ended in state {GOAL_STATES.get(state, state)}'
                f'{": " + text if text else ""}; {ERROR_RECOVERY_HINT}')
        controller_succeeded = (
            result is not None and result.error_code == FollowJointTrajectoryResult.SUCCESSFUL)
        if not controller_succeeded and result is not None:
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
                                  + (f'; {note}' if note else '') + f'; {ERROR_RECOVERY_HINT}')

        if cartesian_target is not None:
            from .kinematics import pose_error

            actual_pose, pose_source, pose_error_message = self._pose_for_cartesian_check(
                motion_started, cartesian_frame, kinematic_chain)
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
                    f'max joint error {error:.6f} rad; pose source: {pose_source}'
                    + (f'; {note}' if note else ''))
        return True, error, note

    def _pose_for_cartesian_check(self, motion_started, expected_frame, kinematic_chain):
        if self.measured_tcp_pose_topic:
            with self._lock:
                pose = self._measured_tcp_pose
                frame = self._measured_tcp_pose_frame
                received = self._measured_tcp_pose_received_monotonic_s
            if pose is None or received <= motion_started:
                return None, None, (
                    'no fresh measured TCP pose received from '
                    f'{self.measured_tcp_pose_topic}')
            if expected_frame and frame != expected_frame:
                return None, None, (
                    f'measured TCP pose frame is {frame!r}, expected {expected_frame!r}')
            if self.measured_tcp_pose_frame and frame != self.measured_tcp_pose_frame:
                return None, None, (
                    f'measured TCP pose frame is {frame!r}, '
                    f'configured {self.measured_tcp_pose_frame!r}')
            return pose.copy(), 'Franka measured TCP pose (O_T_EE)', None

        return kinematic_chain.fk(self.positions), 'FK of measured joints', None

    def check_measured_tcp_frame(self, kinematic_chain, position_limit_m: float = 0.002,
                                 rotation_limit_rad: float = 0.01, timeout: float = 5.0):
        """Compare O_T_EE with the FK of the measured joints; returns a message or None.

        O_T_EE is the end effector configured in Desk. With the Franka Hand
        selected there it coincides with ``panda_hand_tcp``. Any other Desk
        end-effector setting makes every Cartesian check compare different
        frames, so this is verified once before moving.
        """
        if not self.measured_tcp_pose_topic:
            return None
        deadline = time.monotonic() + timeout * self.timeout_scale
        while self.last_measured_tcp_pose is None and time.monotonic() < deadline:
            spin_once(0.05)
        measured = self.last_measured_tcp_pose
        positions = self.positions
        if measured is None or positions is None:
            return f'no FrankaState received on {self.measured_tcp_pose_topic}'
        from .kinematics import pose_error
        position_error, rotation_error = pose_error(kinematic_chain.fk(positions), measured)
        if position_error > position_limit_m or rotation_error > rotation_limit_rad:
            return (f'O_T_EE on {self.measured_tcp_pose_topic} differs from the FK of '
                    f'{kinematic_chain.tip_link} by {position_error * 1000:.2f} mm / '
                    f'{math.degrees(rotation_error):.3f} deg; set the end effector in Desk '
                    'to the Franka Hand (or remove measured_tcp_pose_topic from the config)')
        return None

    def wait_until_stationary(self, timeout: float) -> bool:
        """Wait until the measured joint positions stop changing.

        Deliberately position-based rather than velocity-based: velocity
        estimates are noisy at rest and in simulation can report a constant
        offset on a standing joint, so anything keyed on reported velocity may
        never conclude.  Watching the positions themselves needs no such trust,
        and behaves the same on hardware.
        """
        tolerance = float(self.config['settle_motion'])
        hold = float(self.config['settle_hold'])

        deadline = time.monotonic() + timeout
        history = deque()
        while time.monotonic() < deadline:
            spin_once(0.02)
            positions = self.positions
            if positions is None:
                continue
            now = time.monotonic()
            history.append((now, positions))
            # Drop samples only while the *next* one is still old enough, so the
            # window keeps exactly one sample at or beyond the hold horizon.
            while len(history) > 2 and now - history[1][0] >= hold:
                history.popleft()
            if now - history[0][0] < hold:
                continue
            window = np.array([q for _, q in history])
            if float(np.max(window.max(axis=0) - window.min(axis=0))) < tolerance:
                with self._lock:
                    latest = self._joint_history[-1]
                self._last_settled_state = {
                    'header_stamp_s': latest['header_stamp_s'],
                    'received_monotonic_s': latest['received_monotonic_s'],
                    'q_rad': latest['q_rad'].copy(),
                    'window_joint_spread_rad': window.max(axis=0) - window.min(axis=0),
                }
                return True
        return False
