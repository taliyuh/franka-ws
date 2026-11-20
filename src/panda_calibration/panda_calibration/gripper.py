"""Close the Franka Hand on a peg, or open it again, through franka_gripper's actions.

`pipeline grasp SHAPE` runs `grasp_peg --until-contact` with the peg's grasp
width from pipeline/config/grasps.yaml (see below); `pipeline grasp --epsilon MM`
runs the plain check instead. The grasp action succeeds when the fingers stop
within width +- epsilon; the hand then keeps squeezing with the given force
until a move or stop goal releases it.

`pipeline release` runs `release_peg`: a move action to a wider width, which
ends the grasp and opens the fingers.

`grasp_peg --until-contact` (the peg swap of `pipeline accuracy`, and
`pipeline repeatability`) closes until the fingers clamp whatever is between
them, accepting any width up to the CAD width plus 1 mm: a peg can sit deeper in
the printed V than the CAD says.
"""

from __future__ import annotations

import argparse
import sys

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus, GoalStatusArray
from franka_gripper.msg import (GraspAction, GraspEpsilon, GraspGoal, MoveAction, MoveGoal,
                                StopAction, StopGoal)
from sensor_msgs.msg import JointState

GRASP_ACTION = '/franka_gripper/grasp'
MOVE_ACTION = '/franka_gripper/move'
STOP_ACTION = '/franka_gripper/stop'
# Goal states in which franka_gripper is still inside libfranka's blocking call.
# Cancelling does not interrupt that call: a timed-out grasp stays PREEMPTING
# and every later grasp queues behind it until a stop goal ends it.
BUSY_STATES = (GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.PREEMPTING, GoalStatus.RECALLING)
GRIPPER_STATE = '/franka_gripper/joint_states'
MAX_WIDTH = 0.08  # Franka Hand stroke, m
RELEASE_HINT = ('release with `pipeline release`; do not run homing while holding a peg '
                '(it closes and reopens the fingers fully)')

# Close until contact: the grasp target sits below any peg, so the hand closes until it
# clamps whatever the real width in the printed fingers is. Anything up to the CAD width
# plus the margin counts.
CLOSE_TO_M = 0.0005
WIDTH_MARGIN_M = 0.001
EMPTY_HAND_M = 0.0005   # fingers closer than this have nothing between them


def finger_width(timeout: float = 2.0) -> float | None:
    """Current distance between the fingers, m (the sum of both finger joints)."""
    try:
        state = rospy.wait_for_message(GRIPPER_STATE, JointState, timeout=timeout)
    except rospy.ROSException:
        return None
    return float(sum(state.position))


def _connect(name: str, action) -> actionlib.SimpleActionClient | None:
    client = actionlib.SimpleActionClient(name, action)
    if not client.wait_for_server(rospy.Duration(5.0)):
        print(f'error: no {name} action; start panda_calibration robot.launch '
              '(load_gripper:=true, the default)', file=sys.stderr)
        return None
    return client


def busy(action: str, timeout: float = 2.0) -> bool:
    """True when *action* (e.g. GRASP_ACTION) still runs a goal, possibly another node's."""
    try:
        status = rospy.wait_for_message(action + '/status', GoalStatusArray, timeout=timeout)
    except rospy.ROSException:
        return False
    return any(s.status in BUSY_STATES for s in status.status_list)


def stop_hand(timeout: float = 5.0) -> bool:
    """End whatever grasp or move the hand is running (the hand stops squeezing); True on success."""
    client = _connect(STOP_ACTION, StopAction)
    if client is None:
        return False
    client.send_goal(StopGoal())
    if not client.wait_for_result(rospy.Duration(timeout)):
        client.cancel_goal()
        print(f'error: the hand did not stop within {timeout:g} s; restart robot.launch',
              file=sys.stderr)
        return False
    result = client.get_result()
    if result is None or not result.success:
        print(f'error: stopping the hand failed ({getattr(result, "error", "") or "no result"})',
              file=sys.stderr)
        return False
    return True


def _run(client: actionlib.SimpleActionClient, goal, timeout: float, what: str):
    """Send the goal and return its result, None on timeout (after stopping the hand)."""
    client.send_goal(goal)
    if not client.wait_for_result(rospy.Duration(timeout)):
        client.cancel_goal()
        # The cancel alone leaves franka_gripper blocked in the libfranka call.
        stopped = stop_hand()
        print(f'error: the {what} did not finish within {timeout:g} s; '
              + ('cancelled and stopped the hand' if stopped else 'cancelled'), file=sys.stderr)
        return None
    return client.get_result()


def _measured() -> str:
    width = finger_width()
    return 'unknown' if width is None else f'{width * 1000:.2f} mm'


def grasp(width: float, epsilon: float = 0.001, force: float = 20.0, speed: float = 0.02,
          timeout: float = 10.0, epsilon_outer: float | None = None) -> bool:
    """Close on a peg (needs an initialised ROS node); prints what happened, True on success.

    The hand succeeds between width - epsilon and width + epsilon_outer (default: epsilon).
    """
    outer = epsilon if epsilon_outer is None else epsilon_outer
    client = _connect(GRASP_ACTION, GraspAction)
    if client is None:
        return False
    if busy(GRASP_ACTION) or busy(MOVE_ACTION):
        # A grasp left over from an earlier, timed-out run: the new goal would only queue.
        print('the hand is still running an earlier grasp or move; stopping it first')
        if not stop_hand():
            return False
    result = _run(client, GraspGoal(width=width, epsilon=GraspEpsilon(inner=epsilon, outer=outer),
                                    speed=speed, force=force), timeout, 'grasp')
    print(f'fingers at {_measured()} (accepted {(width - epsilon) * 1000:.2f} to '
          f'{(width + outer) * 1000:.2f} mm)')
    if result is None or not result.success:
        reason = result.error if result is not None and result.error else 'no result'
        print(f'error: grasp failed ({reason}); is the peg seated in the V-block?', file=sys.stderr)
        return False
    print(f'grasped; the hand keeps squeezing with {force:g} N')
    return True


def clamp(cad_width: float, force: float = 20.0, speed: float = 0.02,
          timeout: float = 10.0) -> float | None:
    """Close until the fingers clamp the peg (needs an initialised ROS node).

    Returns the finger width it ended at, m, or None when the grasp failed or the
    fingers closed on nothing. Any width up to *cad_width* + 1 mm counts.
    """
    if not grasp(CLOSE_TO_M, epsilon=CLOSE_TO_M, force=force, speed=speed, timeout=timeout,
                 epsilon_outer=cad_width + WIDTH_MARGIN_M - CLOSE_TO_M):
        return None
    width = finger_width()
    if width is None:
        print(f'error: no finger reading on {GRIPPER_STATE}', file=sys.stderr)
        return None
    if width < EMPTY_HAND_M:
        print('error: the fingers closed on nothing; put the peg in the V and try again',
              file=sys.stderr)
        return None
    return width


def open_hand(width: float = 0.03, speed: float = 0.05, timeout: float = 10.0) -> bool:
    """Open the fingers to *width* (needs an initialised ROS node); True on success."""
    # A move to a smaller width would close the hand instead of opening it.
    current = finger_width()
    if current is not None and current >= width:
        print(f'fingers already open at {current * 1000:.2f} mm (>= {width * 1000:.2f} mm); '
              'nothing to do')
        return True
    client = _connect(MOVE_ACTION, MoveAction)
    if client is None:
        return False
    if busy(GRASP_ACTION) or busy(MOVE_ACTION):
        print('the hand is still running an earlier grasp or move; stopping it first')
        if not stop_hand():
            return False
    result = _run(client, MoveGoal(width=width, speed=speed), timeout, 'release')
    print(f'fingers at {_measured()} (target {width * 1000:.2f} mm)')
    if result is None or not result.success:
        reason = result.error if result is not None and result.error else 'no result'
        print(f'error: release failed ({reason}); is something between the fingers blocking them?',
              file=sys.stderr)
        return False
    print('released')
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Close the Franka Hand on a peg '
                                                 '(franka_gripper grasp action).')
    parser.add_argument('--width', type=float, required=True,
                        help='finger width with the peg seated in the V, m')
    parser.add_argument('--epsilon', type=float, default=0.001,
                        help='accepted deviation from --width, m (default 0.001)')
    parser.add_argument('--force', type=float, default=20.0, help='grip force, N (default 20)')
    parser.add_argument('--speed', type=float, default=0.02,
                        help='closing speed, m/s (default 0.02)')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='give up after this many seconds (default 10)')
    parser.add_argument('--until-contact', action='store_true',
                        help='close until the fingers clamp the peg and accept any width up to '
                             '--width + 1 mm (--epsilon is then not used)')
    args = parser.parse_args(argv if argv is not None else rospy.myargv(sys.argv)[1:])

    rospy.init_node('grasp_peg', anonymous=True, disable_signals=True)
    if args.until_contact:
        width = clamp(args.width, args.force, args.speed, args.timeout)
        if width is None:
            return 1
        print(f'peg held at {width * 1000:.2f} mm (CAD {args.width * 1000:.2f} mm)')
    elif not grasp(args.width, args.epsilon, args.force, args.speed, args.timeout):
        return 1
    print(RELEASE_HINT)
    return 0


def release_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Open the Franka Hand to let go of the held peg '
                                                 '(franka_gripper move action).')
    parser.add_argument('--width', type=float, default=0.03,
                        help=f'open the fingers to this width, m (default 0.03, max {MAX_WIDTH:g})')
    parser.add_argument('--speed', type=float, default=0.05,
                        help='opening speed, m/s (default 0.05)')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='give up after this many seconds (default 10)')
    args = parser.parse_args(argv if argv is not None else rospy.myargv(sys.argv)[1:])
    if not 0.0 < args.width <= MAX_WIDTH:
        print(f'error: --width must be between 0 and {MAX_WIDTH:g} m', file=sys.stderr)
        return 1

    rospy.init_node('release_peg', anonymous=True, disable_signals=True)
    return 0 if open_hand(args.width, args.speed, args.timeout) else 1
