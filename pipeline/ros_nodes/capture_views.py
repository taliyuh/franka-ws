"""Resumable capture of planned views for calibrate_joint_offsets.py (runs in the container).

    python3 capture_views.py POSES.yaml OUT_DIR [--speed-scale 0.3] [--timeout-scale 3]
                             [--config calibration_hardware.yaml] [--ns /azure_camera]

Moves the arm through POSES (pose 0 = the look pose) and saves every view the
moment it is taken: OUT_DIR/images/pose_NNN_azure.png and OUT_DIR/poses.yaml
(rewritten after each view, in the layout run_calibration_capture writes, so
its partial results are picked up too). Rerun with the same arguments after
any stop (lost robot connection, reflex, Ctrl-C): views already in poses.yaml
are skipped.

The arm only moves along the planned path READY -> pose 0 -> pose 1 -> ...:
from where it stands it rejoins that path at the nearest segment (the stop of
a joint-space move leaves it on one) and follows it, without capturing, to the
first missing view. If it stands off the path it asks before moving to READY.

A failed move stops the capture (moving on to the next view would leave the
planned path); recover and rerun to try it again.
Exit codes: 0 all views done, 3 stopped (robot error, failed move, no frame;
rerun to continue), 1 setup error or the operator declined to move, 130 Ctrl-C.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import rospy
import yaml

from panda_calibration.capture_node import CameraStream
from panda_calibration.context import load_config
from panda_calibration.kinematics import PANDA_READY
from panda_calibration.trajectory_client import ArmDriver

ON_PATH_RAD = 0.035          # the arm counts as on a planned segment within ~2 deg on every joint
CAMERA = 'azure'


def load_progress(path):
    if not os.path.exists(path):
        return {'convention': 'measured_joints at the image; written by capture_views.py', 'poses': []}
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    data.setdefault('poses', [])
    return data


def save_progress(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    os.replace(tmp, path)


def segment_distance(q, a, b):
    """Max-norm joint distance from q to the joint-space segment a-b, and where on it (0..1)."""
    s = np.linspace(0.0, 1.0, 201)
    points = a[None, :] + s[:, None] * (b - a)[None, :]
    d = np.abs(points - q[None, :]).max(axis=1)
    i = int(np.argmin(d))
    return float(d[i]), float(s[i])


def rejoin(q, path, first_missing):
    """Planned path indices to drive through to reach path[first_missing + 1] from q.

    path[0] is READY, path[i + 1] is pose i. Returns (route, message) or (None, message)."""
    target = first_missing + 1
    best = None
    for j in range(target):
        d, s = segment_distance(q, path[j], path[j + 1])
        if best is None or d < best[0]:
            best = (d, j, s)
    d, j, s = best
    if d <= ON_PATH_RAD:
        where = 'READY' if j == 0 and s < 0.01 else f'segment {"READY" if j == 0 else f"pose {j - 1}"} -> pose {j}'
        return list(range(j + 1, target + 1)), f'arm on the planned path ({where}, {np.degrees(d):.1f} deg off)'
    return None, (f'the arm is {np.degrees(d):.1f} deg (largest joint) away from the planned path; '
                  f'nearest: {"READY" if j == 0 else f"pose {j - 1}"} -> pose {j}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('poses')
    p.add_argument('out')
    p.add_argument('--config', default=None, help='calibration_hardware.yaml')
    p.add_argument('--ns', default='/azure_camera')
    p.add_argument('--speed-scale', type=float, default=0.3)
    p.add_argument('--timeout-scale', type=float, default=3.0)
    a = p.parse_args(rospy.myargv(sys.argv)[1:])

    with open(a.poses) as handle:
        poses = [np.asarray(e['joints'], float) for e in yaml.safe_load(handle)['poses']]
    os.makedirs(os.path.join(a.out, 'images'), exist_ok=True)
    progress_file = os.path.join(a.out, 'poses.yaml')
    progress = load_progress(progress_file)
    done = {int(e['index']) for e in progress['poses']}
    missing = [i for i in range(len(poses)) if i not in done]
    print(f'{len(poses)} planned views, {len(done)} captured, {len(missing)} to go')
    if not missing:
        return 0

    config = load_config(a.config)
    joint_names = config['robot']['joint_names']
    capture = config['capture']
    rospy.init_node('capture_views', anonymous=True, disable_signals=True)
    driver = ArmDriver(joint_names, config['motion'], a.timeout_scale)
    stream = CameraStream(CAMERA, f'{a.ns}/rgb/image_raw', f'{a.ns}/rgb/camera_info')
    if not driver.wait_for_controller():
        print(f"error: no {config['motion']['controller_action']}; is robot.launch running?", file=sys.stderr)
        return 1
    if not driver.wait_for_state():
        print('error: no joint states', file=sys.stderr)
        return 1
    problem = driver.robot_mode_problem()
    if problem:
        print(f'error: {problem}', file=sys.stderr)
        return 3

    path = [np.asarray(PANDA_READY, float)] + poses
    route, message = rejoin(np.asarray(driver.positions, float), path, missing[0])
    print(message)
    if route is None:
        answer = input('Move straight to READY now? This path is NOT checked for collisions: look at the '
                       'arm first. y = move, anything else = stop (then guide the arm near READY or the '
                       'path and rerun) > ').strip().lower()
        if answer != 'y':
            return 1
        route = list(range(0, missing[0] + 2))      # READY, then the planned path up to the first missing view

    timestamp_gate = capture.get('require_post_settle_timestamp', False)
    info_written = os.path.exists(os.path.join(a.out, 'camera_info.json'))
    for step in route[:-1]:                  # back along the planned path, nothing captured
        label = 'READY' if step == 0 else f'pose {step - 1} (already captured)'
        print(f'--> {label}')
        ok, _, msg = driver.move_to(path[step], a.speed_scale)
        if not ok:
            print(f'error: moving to {label}: {msg}', file=sys.stderr)
            return 3
    # Capture every missing view in order; each move is a planned segment.
    for index in missing:
        print(f'--> pose {index} ({len(progress["poses"]) + 1}/{len(poses)})')
        ok, settle_error, msg = driver.move_to(poses[index], a.speed_scale)
        if not ok:
            print(f'error: pose {index}: {msg}', file=sys.stderr)
            problem = driver.robot_mode_problem()
            if problem:
                print(f'robot: {problem}', file=sys.stderr)
            return 3
        settled = driver.last_settled_state
        measured = np.asarray(driver.positions, float)
        frame = stream.grab_fresh_frame_record(
            capture['discard_frames'], capture['frame_timeout'] * a.timeout_scale,
            settled['header_stamp_s'] if (timestamp_gate and settled) else None,
            settled['received_monotonic_s'] if (timestamp_gate and settled) else None)
        if frame is None:
            print(f'error: pose {index}: no fresh camera frame (is azure_4k.launch running?)', file=sys.stderr)
            return 3
        bracket = driver.wait_for_joint_after(frame['received_monotonic_s'], capture['frame_timeout'])
        if bracket is None:
            print(f'error: pose {index}: no joint state after the image', file=sys.stderr)
            return 3
        image = os.path.join('images', f'pose_{index:03d}_{CAMERA}.png')
        if not cv2.imwrite(os.path.join(a.out, image), frame['image']):
            print(f'error: cannot write {image}', file=sys.stderr)
            return 1
        if not info_written and stream.info is not None:
            info = stream.info
            with open(os.path.join(a.out, 'camera_info.json'), 'w') as handle:
                json.dump({'width': info.width, 'height': info.height,
                           'distortion_model': info.distortion_model, 'd': list(info.D),
                           'k': list(info.K), 'r': list(info.R), 'p': list(info.P),
                           'frame_id': info.header.frame_id}, handle, indent=2)
            info_written = True
        progress['poses'].append({
            'index': index, 'joint_names': joint_names,
            'target_joints': [round(float(v), 6) for v in poses[index]],
            'measured_joints': [round(float(v), 6) for v in measured],
            'max_joint_error': round(float(np.abs(measured - poses[index]).max()), 6),
            'settle_error': round(float(settle_error), 6), 'image': image,
            'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
        save_progress(progress_file, progress)
        print(f'    saved ({len(progress["poses"])} views)')
    print(f'done: {len(progress["poses"])} views captured')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nstopped; the views captured so far are saved', file=sys.stderr)
        sys.exit(130)
