"""Hand-guided touch of one board hole, for `pipeline diag 3-6` (runs in the container).

    python3 touch_hole.py SPEC.json OUT.json

SPEC (written by board_pipeline/diagnose.py) gives the hole, the held peg's
TCP -> tip offset and the probe's target in the nominal frame. The script stops
position_joint_trajectory_controller so the arm can be guided, prints a live
line (where the tip is relative to the target, the hand yaw), and on Enter
averages the robot state for a second while the arm is still and writes OUT:
measured joints q and O_T_EE (R, t). It restarts the controller at the end.
Typing q + Enter stops without writing OUT.
"""

import datetime
import json
import math
import sys
import threading
import time

import numpy as np
import rospy
from franka_msgs.msg import FrankaState

from panda_calibration.pivot_calibration import (MOTION_CONTROLLER, STILL_RAD, capture_pose,
                                                 controller_running, switch_controller)

SAMPLE_SECONDS = 1.0


class Live:
    """Prints one refreshing status line from the robot state until paused."""

    def __init__(self, spec):
        self.tip = np.asarray(spec['tip_offset_tcp_m'], float)
        self.target = np.asarray(spec['target_nominal_m'], float)
        self.board = np.asarray(spec['board_rotation'], float)
        self.reference_yaw = spec.get('reference_yaw_deg')
        self.state = None
        self.running = threading.Event()
        rospy.Subscriber(spec['topic'], FrankaState, self._on_state, queue_size=1)
        threading.Thread(target=self._loop, daemon=True).start()

    def _on_state(self, msg):
        self.state = (time.monotonic(), np.array(msg.O_T_EE).reshape(4, 4).T)

    def line(self):
        if self.state is None:
            return 'waiting for the robot state ...'
        stamp, pose = self.state
        tip = pose[:3, :3] @ self.tip + pose[:3, 3]
        rel = self.board.T @ (tip - self.target)
        yaw = math.degrees(math.atan2(pose[1, 0], pose[0, 0]))
        text = (f'tip vs target (board axes): x {rel[0] * 1e3:+6.2f}  y {rel[1] * 1e3:+6.2f} mm '
                f'({math.hypot(rel[0], rel[1]) * 1e3:5.2f} beside), {rel[2] * 1e3:+6.1f} mm height | '
                f'hand yaw {yaw:+6.1f} deg')
        if self.reference_yaw is not None:
            text += f' (turned {math.degrees(math.remainder(math.radians(yaw - self.reference_yaw), 2 * math.pi)):+.0f})'
        if time.monotonic() - stamp > 1.0:
            text += '  [no robot state]'
        return text

    def _loop(self):
        while not rospy.is_shutdown():
            if self.running.is_set():
                sys.stdout.write('\r\033[K' + self.line())
                sys.stdout.flush()
            time.sleep(0.25)

    def ask(self, prompt):
        """Show the live line until the operator presses Enter; returns the typed text."""
        print(prompt)
        self.running.set()
        try:
            line = sys.stdin.readline()
        finally:
            self.running.clear()
        sys.stdout.write('\r\033[K')
        return line.strip().lower()


def record(spec, live):
    print(f"\n=== {spec['instructions']} ===")
    while True:
        command = live.ask('Enter = record (arm still, buttons released), q + Enter = stop the test')
        if command == 'q':
            return None
        q, rotation, translation, moved = capture_pose(spec['topic'], SAMPLE_SECONDS)
        if moved > STILL_RAD:
            print(f'  the arm moved {moved * 1e3:.2f} mrad while sampling; release the buttons, '
                  'let it settle and record again')
            continue
        tip = rotation @ live.tip + translation
        rel = live.board.T @ (tip - live.target)
        print(f'  recorded: tip {rel[0] * 1e3:+.2f} / {rel[1] * 1e3:+.2f} mm from the target in board x / y')
        if input('Enter = keep this touch, r = record again > ').strip().lower() != 'r':
            return {'hole': spec['hole'], 'label': spec['label'], 'q': q.tolist(),
                    'R': rotation.tolist(), 't': translation.tolist(), 'joint_spread_rad': moved,
                    'time': datetime.datetime.now().isoformat(timespec='seconds')}


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1]) as handle:
        spec = json.load(handle)
    rospy.init_node('touch_hole', anonymous=True, disable_signals=True)
    was_running = controller_running(MOTION_CONTROLLER)
    if was_running is None:
        print('error: no controller manager; is panda_calibration robot.launch running?', file=sys.stderr)
        return 1
    if was_running:
        input(f'{MOTION_CONTROLLER} will be stopped so you can guide the arm by hand. '
              'Enter = continue, Ctrl-C = abort > ')
        if not switch_controller(stop=[MOTION_CONTROLLER]):
            print(f'error: could not stop {MOTION_CONTROLLER}', file=sys.stderr)
            return 1
    live = Live(spec)
    try:
        result = record(spec, live)
    except KeyboardInterrupt:
        result = None
    finally:
        if was_running:
            input('\nRelease the guiding buttons and let go of the arm, then press Enter to hold it '
                  'again (the peg can stay in the hole) > ')
            if switch_controller(start=[MOTION_CONTROLLER]):
                print(f'{MOTION_CONTROLLER} running again')
            else:
                print(f'warning: could not restart {MOTION_CONTROLLER}; run\n  rosservice call '
                      '/controller_manager/switch_controller '
                      f'"{{start_controllers: [{MOTION_CONTROLLER}], stop_controllers: [], strictness: 2}}"',
                      file=sys.stderr)
    if result is None:
        return 3
    with open(sys.argv[2], 'w') as handle:
        json.dump(result, handle, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
