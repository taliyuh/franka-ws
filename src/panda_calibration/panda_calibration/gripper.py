"""Close the Franka Hand on a peg through franka_gripper's grasp action.

`pipeline grasp --hole NAME` runs this with the peg's grasp width from
pipeline/config/grasps.yaml. The action succeeds when the fingers stop within
width +- epsilon; the hand then keeps squeezing with the given force until a
move or stop goal releases it.
"""

from __future__ import annotations

import argparse
import sys

import actionlib
import rospy
from franka_gripper.msg import GraspAction, GraspEpsilon, GraspGoal
from sensor_msgs.msg import JointState

GRASP_ACTION = '/franka_gripper/grasp'
GRIPPER_STATE = '/franka_gripper/joint_states'
RELEASE_HINT = ('release with `rostopic pub -1 /franka_gripper/move/goal '
                'franka_gripper/MoveActionGoal "goal: {width: 0.03, speed: 0.05}"`; '
                'do not run homing while holding a peg (it closes and reopens the fingers fully)')


def finger_width(timeout: float = 2.0) -> float | None:
    """Current distance between the fingers, m (the sum of both finger joints)."""
    try:
        state = rospy.wait_for_message(GRIPPER_STATE, JointState, timeout=timeout)
    except rospy.ROSException:
        return None
    return float(sum(state.position))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=float, required=True,
                        help='finger width with the peg seated in the V, m')
    parser.add_argument('--epsilon', type=float, default=0.001,
                        help='accepted deviation from --width, m (default 0.001)')
    parser.add_argument('--force', type=float, default=20.0, help='grip force, N (default 20)')
    parser.add_argument('--speed', type=float, default=0.02,
                        help='closing speed, m/s (default 0.02)')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='give up after this many seconds (default 10)')
    args = parser.parse_args(argv if argv is not None else rospy.myargv(sys.argv)[1:])

    rospy.init_node('grasp_peg', anonymous=True, disable_signals=True)
    client = actionlib.SimpleActionClient(GRASP_ACTION, GraspAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        print(f'error: no {GRASP_ACTION} action; start panda_calibration robot.launch '
              '(load_gripper:=true, the default)', file=sys.stderr)
        return 1

    client.send_goal(GraspGoal(width=args.width,
                               epsilon=GraspEpsilon(inner=args.epsilon, outer=args.epsilon),
                               speed=args.speed, force=args.force))
    if not client.wait_for_result(rospy.Duration(args.timeout)):
        client.cancel_goal()
        print(f'error: the grasp did not finish within {args.timeout:g} s; cancelled',
              file=sys.stderr)
        return 1
    result = client.get_result()

    width = finger_width()
    measured = 'unknown' if width is None else f'{width * 1000:.2f} mm'
    print(f'fingers at {measured} (expected {args.width * 1000:.2f} '
          f'+- {args.epsilon * 1000:.2f} mm)')
    if result is None or not result.success:
        reason = result.error if result is not None and result.error else 'no result'
        print(f'error: grasp failed ({reason}); is the peg seated in the V-block?', file=sys.stderr)
        return 1
    print(f'grasped; the hand keeps squeezing with {args.force:g} N')
    print(RELEASE_HINT)
    return 0
