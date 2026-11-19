"""Hand-guided touches of several AprilGrid points, for calibrate_joint_offsets.py (container).

    python3 touch_grid.py SPEC.json OUT.json

SPEC (written by pipeline/tools/calibrate_joint_offsets.py) lists the targets
(label, board point, rough base-frame position, how to hold the pen), the
pen's TCP -> tip offset and the FrankaState topic. The script stops
position_joint_trajectory_controller so the arm can be guided, shows a live
line (tip vs target in board axes, from the robot's nominal O_T_EE), and on
Enter averages the robot state for a second while the arm is still. OUT is
rewritten after every kept touch, so an interrupted run keeps what it has and
a rerun skips the targets already in OUT. The controller is restarted at the end.
"""

import datetime
import json
import math
import os
import sys
import threading
import time

import numpy as np
import rospy
from franka_msgs.msg import FrankaState

from panda_calibration.pivot_calibration import (MOTION_CONTROLLER, STILL_RAD, capture_pose,
                                                 controller_running, switch_controller)

SAMPLE_SECONDS = 1.0
HELP = ("Hold the guiding buttons on the wrist and put the pen tip exactly on the marked X (where the\n"
        "tag corner meets the small black square), holding the pen as asked. Release the buttons so\n"
        "the arm holds still, then press Enter.")


class Live:
    """Prints one refreshing status line from the robot state while a prompt is open."""

    def __init__(self, spec):
        self.tip = np.asarray(spec['tip_offset_tcp_m'], float)
        self.board = np.asarray(spec['board_rotation'], float)
        self.target = None
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
        tilt = math.degrees(math.acos(min(1.0, abs(pose[2, 2]))))
        text = (f'tip vs target (board axes, uncalibrated): x {rel[0] * 1e3:+6.1f}  y {rel[1] * 1e3:+6.1f}  '
                f'z {rel[2] * 1e3:+6.1f} mm | pen tilt {tilt:4.1f} deg')
        if time.monotonic() - stamp > 1.0:
            text += '  [no robot state]'
        return text

    def _loop(self):
        while not rospy.is_shutdown():
            if self.running.is_set() and self.target is not None:
                sys.stdout.write('\r\033[K' + self.line())
                sys.stdout.flush()
            time.sleep(0.25)

    def ask(self, prompt):
        print(prompt)
        self.running.set()
        try:
            line = sys.stdin.readline()
        finally:
            self.running.clear()
        sys.stdout.write('\r\033[K')
        return line.strip().lower()


def save(path, records):
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(records, handle, indent=2)
    os.replace(tmp, path)


def touch_all(spec, live, out_path, records):
    done = {r['target'] for r in records}
    print(HELP)
    print(f"Targets: {spec['image']} (the numbers in that picture)")
    for index, target in enumerate(spec['targets']):
        if index in done:
            continue
        live.target = np.asarray(target['target_base_m'], float)
        print(f"\n=== point {target['label']} of {len(spec['targets'])}: {target['instruction']} ===")
        while True:
            command = live.ask('Enter = record, s = skip this point, q = stop touching')
            if command == 'q':
                return
            if command == 's':
                break
            q, rotation, translation, moved = capture_pose(spec['topic'], SAMPLE_SECONDS)
            if moved > STILL_RAD:
                print(f'  the arm moved {moved * 1e3:.2f} mrad while sampling; release the buttons, '
                      'let it settle and record again')
                continue
            tip = rotation @ live.tip + translation
            rel = live.board.T @ (tip - live.target)
            print(f'  recorded; uncalibrated tip is {np.hypot(rel[0], rel[1]) * 1e3:.1f} mm beside and '
                  f'{rel[2] * 1e3:+.1f} mm above the target (expected: a few mm before calibration)')
            if input('Enter = keep, r = record again > ').strip().lower() == 'r':
                continue
            records.append({'target': index, 'label': target['label'], 'q': q.tolist(),
                            'R': rotation.tolist(), 't': translation.tolist(),
                            'joint_spread_rad': moved,
                            'time': datetime.datetime.now().isoformat(timespec='seconds')})
            save(out_path, records)
            break


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1]) as handle:
        spec = json.load(handle)
    out_path = sys.argv[2]
    records = []
    if os.path.exists(out_path):
        with open(out_path) as handle:
            records = json.load(handle)
        print(f'{len(records)} touches already recorded in {out_path}; continuing')
    rospy.init_node('touch_grid', anonymous=True, disable_signals=True)
    was_running = controller_running(MOTION_CONTROLLER)
    if was_running is None:
        print('error: no controller manager; is panda_calibration robot.launch running?', file=sys.stderr)
        return 1
    if was_running:
        input(f'\n{MOTION_CONTROLLER} will be stopped so you can guide the arm by hand. '
              'Enter = continue, Ctrl-C = abort > ')
        if not switch_controller(stop=[MOTION_CONTROLLER]):
            print(f'error: could not stop {MOTION_CONTROLLER}', file=sys.stderr)
            return 1
    live = Live(spec)
    try:
        touch_all(spec, live, out_path, records)
    except KeyboardInterrupt:
        pass
    finally:
        if was_running:
            input('\nGuide the pen at least 10 cm up, away from the board, release the buttons and let go '
                  'of the arm, then press Enter to hold it again (the next motion starts from there) > ')
            if switch_controller(start=[MOTION_CONTROLLER]):
                print(f'{MOTION_CONTROLLER} running again')
            else:
                print(f'warning: could not restart {MOTION_CONTROLLER}; run\n  rosservice call '
                      '/controller_manager/switch_controller '
                      f'"{{start_controllers: [{MOTION_CONTROLLER}], stop_controllers: [], strictness: 2}}"',
                      file=sys.stderr)
    print(f'{len(records)} of {len(spec["targets"])} touches recorded')
    return 0


if __name__ == '__main__':
    sys.exit(main())
