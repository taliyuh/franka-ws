"""Pivot calibration of the tool tip held in the Franka Hand.

Touch one fixed point -- ideally a small conical dimple, so the tip cannot
slide -- with the tool tip from many orientations, guiding the arm by hand.
Every recorded pose i puts the tip on the same base-frame point:

    R_i @ tip + t_i = pivot

(R_i, t_i) is the robot's measured panda_hand_tcp pose (O_T_EE, Desk end
effector = Franka Hand). Stacking [R_i, -I] [tip; pivot] = -t_i gives a linear
least-squares problem for the tip in TCP axes -- the frame of
probe.pen_tip_offset_tcp_m and of the peg tips `pipeline check/probe` print --
and for the pivot in the base frame. Only the tip position is observable:
turning about the tip says nothing about the direction of the tool's axis.

Run in the container while robot.launch is running:

    rosrun panda_calibration pivot_calibration [--out FILE]
    rosrun panda_calibration pivot_calibration --resolve FILE [--drop 3,7]

It stops position_joint_trajectory_controller so the arm can be guided by hand
and starts it again at the end.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

import numpy as np
import rospy
import yaml

from .context import build_chain, default_config_path, load_config
from .kinematics import rotation_angle_between

MOTION_CONTROLLER = 'position_joint_trajectory_controller'
WORKSPACE = '/franka_ws'
SAMPLE_SECONDS = 0.5
STILL_RAD = 0.0005      # max joint movement while one pose is sampled (~0.3 mm at the tip)
MIN_SPREAD_DEG = 30.0   # below this the tip is poorly determined along the tool axis


def _fit(rotations: np.ndarray, translations: np.ndarray):
    n = len(rotations)
    a = np.hstack([rotations.reshape(3 * n, 3), np.tile(-np.eye(3), (n, 1))])
    b = -translations.reshape(3 * n)
    return a, b, np.linalg.lstsq(a, b, rcond=None)[0]


def solve_pivot(rotations, translations) -> dict:
    """Tip in TCP axes and pivot in the base frame from N >= 4 measured TCP poses."""
    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    n = len(rotations)
    if n < 4:
        raise ValueError(f'need at least 4 poses, have {n}')
    a, b, x = _fit(rotations, translations)
    if np.linalg.matrix_rank(a, tol=1e-6) < 6:
        raise ValueError('the poses are too similar: tilt the tool in different directions')
    error = (a @ x - b).reshape(n, 3)
    residuals = np.linalg.norm(error, axis=1)
    covariance = float(np.sum(error ** 2)) / max(3 * n - 6, 1) * np.linalg.inv(a.T @ a)
    # Leave-one-out: how far each pose is from the fit of all the others, an
    # honest estimate of the error on a pose the fit has not seen.
    loo = None
    if n >= 5:
        loo = np.empty(n)
        for i in range(n):
            keep = np.arange(n) != i
            _, _, xi = _fit(rotations[keep], translations[keep])
            loo[i] = np.linalg.norm(rotations[i] @ xi[:3] + translations[i] - xi[3:])
    spread = max(rotation_angle_between(rotations[i], rotations[j])
                 for i in range(n) for j in range(i + 1, n))
    return {
        'tip': x[:3], 'pivot': x[3:], 'residuals': residuals,
        'rms': float(np.sqrt(np.mean(residuals ** 2))),
        'tip_std': np.sqrt(np.diag(covariance)[:3]),
        'loo': loo, 'spread_deg': float(np.degrees(spread)),
    }


def flange_tcp(chain) -> np.ndarray:
    """panda_link8 -> panda_hand_tcp (fixed: the Franka Hand mount and TCP offset)."""
    frames = chain.link_frames(np.zeros(len(chain.limits[0])))
    return np.linalg.inv(frames['panda_link8']) @ frames['panda_hand_tcp']


# ---------------------------------------------------------------- recording
def capture_pose(topic: str, seconds: float):
    """Average the robot state over *seconds*: (q, R, t, joint spread)."""
    from franka_msgs.msg import FrankaState

    samples = []
    subscriber = rospy.Subscriber(topic, FrankaState, samples.append, queue_size=1000)
    deadline = time.monotonic() + 2.0
    while not samples and time.monotonic() < deadline:
        time.sleep(0.01)
    if samples:
        first = len(samples)
        time.sleep(seconds)
    subscriber.unregister()
    if not samples or len(samples) - first < 10:
        raise RuntimeError(f'too few robot states on {topic}; is robot.launch running?')
    q = np.array([m.q for m in samples])
    poses = np.array([np.array(m.O_T_EE).reshape(4, 4).T for m in samples])
    u, _, vt = np.linalg.svd(poses[:, :3, :3].mean(axis=0))
    return q.mean(axis=0), u @ vt, poses[:, :3, 3].mean(axis=0), float(np.max(q.max(0) - q.min(0)))


def controller_running(name: str):
    """True/False, or None if no controller manager answers."""
    from controller_manager_msgs.srv import ListControllers

    try:
        rospy.wait_for_service('/controller_manager/list_controllers', timeout=5.0)
    except rospy.ROSException:
        return None
    states = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)().controller
    return any(c.name == name and c.state == 'running' for c in states)


def switch_controller(start=(), stop=()) -> bool:
    from controller_manager_msgs.srv import SwitchController

    proxy = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
    return proxy(start_controllers=list(start), stop_controllers=list(stop),
                 strictness=2, start_asap=False, timeout=0.0).ok


def record(topic: str, chain) -> list:
    print('\nGuide the tool tip into the point (dimple), release the guiding buttons so the arm\n'
          'holds still, then type at the prompt:\n'
          '  Enter      record this pose\n'
          '  u + Enter  drop the last pose\n'
          '  q + Enter  finish and solve\n'
          'Between poses tilt the tool 30-60 deg in different directions and turn it about its axis.\n')
    samples = []
    while True:
        command = input(f'[{len(samples)} poses] > ').strip().lower()
        if command == 'q':
            return samples
        if command == 'u':
            if samples:
                samples.pop()
            continue
        if command:
            print('  Enter, u or q')
            continue
        q, rotation, translation, moved = capture_pose(topic, SAMPLE_SECONDS)
        if moved > STILL_RAD:
            print(f'  the arm moved {moved * 1000:.2f} mrad while sampling; hold it still and try again')
            continue
        if not samples:
            fk = chain.fk(q)
            offset = np.linalg.norm(fk[:3, 3] - translation)
            if offset > 0.002 or rotation_angle_between(fk[:3, :3], rotation) > 0.01:
                print(f'  warning: O_T_EE is {offset * 1000:.1f} mm from panda_hand_tcp; '
                      'set the end effector in Desk to the Franka Hand')
        samples.append({'q': q, 'R': rotation, 't': translation})
        if len(samples) >= 4:
            try:
                fit = solve_pivot([s['R'] for s in samples], [s['t'] for s in samples])
                print(f'  tip {np.round(fit["tip"] * 1000, 2)} mm, residual rms {fit["rms"] * 1000:.3f} mm, '
                      f'spread {fit["spread_deg"]:.0f} deg')
            except ValueError as error:
                print(f'  {error}')


# ------------------------------------------------------------------ output
def report(fit: dict, tcp_to_flange: np.ndarray) -> list:
    tip_flange = tcp_to_flange[:3, :3] @ fit['tip'] + tcp_to_flange[:3, 3]
    worst = int(np.argmax(fit['residuals']))
    lines = [
        f'tip in panda_hand_tcp axes:   {np.round(fit["tip"] * 1000, 3)} mm '
        f'(1-sigma {np.round(fit["tip_std"] * 1000, 3)} mm)',
        f'tip in panda_link8 (flange):  {np.round(tip_flange * 1000, 3)} mm',
        f'pivot point in the base:      {np.round(fit["pivot"] * 1000, 3)} mm',
        f'{len(fit["residuals"])} poses, residual rms {fit["rms"] * 1000:.3f} mm, '
        f'max {fit["residuals"][worst] * 1000:.3f} mm (pose {worst}), '
        + (f'leave-one-out rms {np.sqrt(np.mean(fit["loo"] ** 2)) * 1000:.3f} mm, '
           if fit['loo'] is not None else '')
        + f'orientation spread {fit["spread_deg"]:.0f} deg',
    ]
    if fit['spread_deg'] < MIN_SPREAD_DEG:
        lines.append(f'warning: orientations span only {fit["spread_deg"]:.0f} deg; '
                     'tilt more for a tip you can trust')
    if fit['residuals'][worst] > 3 * fit['rms'] and len(fit['residuals']) >= 6:
        lines.append(f'pose {worst} fits far worse than the rest (did the tip slip?); '
                     f'drop it with --resolve FILE --drop {worst}')
    return lines


def write_result(path: str, fit: dict, samples: list, tcp_to_flange: np.ndarray,
                 dropped: list, shape: str) -> None:
    tip_flange = tcp_to_flange[:3, :3] @ fit['tip'] + tcp_to_flange[:3, 3]
    data = {
        'frame': 'panda_hand_tcp',
        'shape': shape,
        'tip_in_tcp_m': [float(v) for v in fit['tip']],
        'tip_in_flange_m': [float(v) for v in tip_flange],
        'tip_std_mm': [round(float(v) * 1000, 4) for v in fit['tip_std']],
        'pivot_in_base_m': [float(v) for v in fit['pivot']],
        'poses': len(samples),
        'dropped': dropped,
        'residual_rms_mm': round(fit['rms'] * 1000, 4),
        'residual_max_mm': round(float(fit['residuals'].max()) * 1000, 4),
        'leave_one_out_rms_mm': (None if fit['loo'] is None
                                 else round(float(np.sqrt(np.mean(fit['loo'] ** 2))) * 1000, 4)),
        'orientation_spread_deg': round(fit['spread_deg'], 2),
        'samples': [{'q': [float(v) for v in s['q']],
                     'O_T_EE': [[float(v) for v in row] for row in np.vstack(
                         [np.hstack([s['R'], s['t'][:, None]]), [0, 0, 0, 1]])],
                     'residual_mm': round(float(r) * 1000, 4)}
                    for s, r in zip(samples, fit['residuals'])],
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('# Pivot calibration (rosrun panda_calibration pivot_calibration).\n'
                     '# tip_in_tcp_m: TCP -> tool tip in panda_hand_tcp axes, the frame of\n'
                     '# probe.pen_tip_offset_tcp_m and of the "tip [...] mm from the TCP" that\n'
                     '# `pipeline check/probe` print for the held peg. Use it there with\n'
                     '# --tip-from THIS_FILE (applies to the `shape` peg only).\n')
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=None)
    # The container runs as root: hand the result to the workspace owner.
    owner = os.stat(WORKSPACE) if os.path.isdir(WORKSPACE) else None
    if owner is not None and os.geteuid() == 0:
        for target in (os.path.dirname(path), path):
            if target.startswith(WORKSPACE):
                os.chown(target, owner.st_uid, owner.st_gid)


def load_samples(path: str) -> tuple:
    """(poses, peg shape) saved in an earlier result."""
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    samples = []
    for entry in data['samples']:
        pose = np.asarray(entry['O_T_EE'], dtype=float)
        samples.append({'q': np.asarray(entry['q'], dtype=float),
                        'R': pose[:3, :3], 't': pose[:3, 3]})
    return samples, data.get('shape', 'round')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', help='result file (default: /franka_ws/tool_calibration/'
                                      'pivot_<date>_<time>.yaml)')
    parser.add_argument('--config', help='robot configuration (default: calibration_hardware.yaml)')
    parser.add_argument('--shape', help='peg shape of the held tool, as in pipeline/config/grasps.yaml '
                                        '(default: round; with --resolve: the one in FILE)')
    parser.add_argument('--resolve', metavar='FILE',
                        help='solve again from the poses saved in an earlier result')
    parser.add_argument('--drop', default='', metavar='I,J',
                        help='with --resolve: leave these pose numbers out')
    args = parser.parse_args(argv if argv is not None else rospy.myargv(sys.argv)[1:])
    config = load_config(args.config or default_config_path('calibration_hardware.yaml'))

    if args.resolve:
        samples, shape = load_samples(args.resolve)
        shape = args.shape or shape
        dropped = sorted({int(v) for v in args.drop.split(',') if v.strip()})
        samples = [s for i, s in enumerate(samples) if i not in dropped]
        chain = build_chain(config)
        out = args.out or os.path.splitext(args.resolve)[0] + '_resolved.yaml'
    else:
        shape = args.shape or 'round'
        rospy.init_node('pivot_calibration', anonymous=True, disable_signals=True)
        chain = build_chain(config, rospy.get_param('/robot_description', None))
        was_running = controller_running(MOTION_CONTROLLER)
        if was_running is None:
            print('error: no controller manager; start panda_calibration robot.launch first',
                  file=sys.stderr)
            return 1
        if was_running:
            input(f'{MOTION_CONTROLLER} will be stopped so you can guide the arm by hand '
                  '(it holds its pose). Press Enter to continue, Ctrl-C to abort... ')
            if not switch_controller(stop=[MOTION_CONTROLLER]):
                print(f'error: could not stop {MOTION_CONTROLLER}', file=sys.stderr)
                return 1
        try:
            samples = record(config['motion']['measured_tcp_pose_topic'], chain)
        finally:
            if was_running:
                input('Release the guiding buttons and let go of the arm, then press Enter to '
                      f'restart {MOTION_CONTROLLER}... ')
                if switch_controller(start=[MOTION_CONTROLLER]):
                    print(f'{MOTION_CONTROLLER} running again')
                else:
                    print(f'warning: could not restart {MOTION_CONTROLLER}; run\n'
                          '  rosservice call /controller_manager/switch_controller '
                          f'"{{start_controllers: [{MOTION_CONTROLLER}], stop_controllers: [], '
                          'strictness: 2}"', file=sys.stderr)
        dropped = []
        stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        out = args.out or os.path.join(WORKSPACE, 'tool_calibration', f'pivot_{stamp}.yaml')

    try:
        fit = solve_pivot([s['R'] for s in samples], [s['t'] for s in samples])
    except ValueError as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    tcp_to_flange = flange_tcp(chain)
    print('\n' + '\n'.join(report(fit, tcp_to_flange)))
    write_result(out, fit, samples, tcp_to_flange, dropped, shape)
    print(f'written to {out} ({shape} peg); use it with '
          f'`pipeline check/probe --hole {shape}_... --tip-from {out}`')
    return 0
