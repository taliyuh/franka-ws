#!/usr/bin/env python3
"""Move the TCP (panda_hand_tcp) straight up in panda_link0, keeping its orientation.

Run in the franka_ros container with robot.launch up (it uses move_group):

    python3 /franka_ws/pipeline/ros_nodes/lift.py            # 0.5 m up
    python3 /franka_ws/pipeline/ros_nodes/lift.py 0.2 --speed 0.05

It plans the whole straight line first and refuses to move unless all of it is
reachable, then asks for Enter. Negative distances move down.
"""
import argparse
import copy
import sys

import moveit_commander
import rospy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("distance", nargs="?", type=float, default=0.5, help="metres up (default 0.5)")
    parser.add_argument("--speed", type=float, default=0.1,
                        help="velocity and acceleration scaling, 0-1 (default 0.1)")
    parser.add_argument("--yes", action="store_true", help="do not ask before moving")
    args = parser.parse_args(rospy.myargv(sys.argv)[1:])
    if not 0.0 < args.speed <= 0.3:
        parser.error("--speed must be in (0, 0.3]")

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("lift", anonymous=True, disable_signals=True)
    group = moveit_commander.MoveGroupCommander("panda_manipulator")
    start = group.get_current_pose().pose
    goal = copy.deepcopy(start)
    goal.position.z += args.distance

    plan, fraction = group.compute_cartesian_path([goal], 0.005, True)
    if fraction < 0.999:
        print(f"error: only {fraction * 100:.0f}% of the {args.distance * 100:.0f} cm line is reachable; "
              "nothing moved", file=sys.stderr)
        return 1
    plan = group.retime_trajectory(group.get_current_state(), plan,
                                   velocity_scaling_factor=args.speed,
                                   acceleration_scaling_factor=args.speed)
    points = plan.joint_trajectory.points
    print(f"TCP z {start.position.z:.3f} -> {goal.position.z:.3f} m "
          f"(x {start.position.x:.3f}, y {start.position.y:.3f} kept), "
          f"{points[-1].time_from_start.to_sec():.1f} s")
    print("end joints [rad]:", [round(v, 2) for v in points[-1].positions])
    if not args.yes:
        input("Keep the e-stop at hand. Press Enter to move, Ctrl-C to abort...")
    if not group.execute(plan, wait=True):
        print("error: execution failed (reflex or user stop? check the robot mode)", file=sys.stderr)
        return 1
    group.stop()
    end = group.get_current_pose().pose.position
    print(f"done: TCP at [{end.x:.3f}, {end.y:.3f}, {end.z:.3f}] m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
