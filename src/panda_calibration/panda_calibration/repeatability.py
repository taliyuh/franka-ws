"""How well the arm returns to commanded joint angles, with no compensation.

You guide the arm by hand to a point. This tool records the joint angles there,
sends the arm home, commands those joint angles again with one ordinary
trajectory goal, waits, and records where the arm stopped. That repeats a few
times per point, for several points.

Nothing is corrected: no Cartesian correction loop, no calibrated joint offsets,
no IK. Errors are measured - commanded, for each joint and at the tool tip
(panda_hand_tcp plus the tip offset of the held peg). Both come from the robot's
own joint readings, so this shows how the arm settles against its commands, not
where the tip is in the real world.

Results go to --out: run.yaml (everything), measurements.csv (one row per
repeat) and repeatability.png. They are written after every repeat, so an
interrupted run keeps what it measured.

    pipeline repeatability                                  (on the host: peg geometry from the config)
    rosrun panda_calibration measure_repeatability --plot DIR   (draw the figure again)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import time

import numpy as np
import rospy
import yaml

from . import gripper
from .context import build_chain, default_config_path, load_config
from .kinematics import PANDA_READY, rotation_angle_between
from .pivot_calibration import (MOTION_CONTROLLER, SAMPLE_SECONDS, STILL_RAD, WORKSPACE,
                                capture_pose, controller_running, switch_controller)
from .trajectory_client import ArmDriver

RUN_FILE = 'run.yaml'
CSV_FILE = 'measurements.csv'
PLOT_FILE = 'repeatability.png'
AXES = ('x', 'y', 'z')
JOINTS = range(1, 8)

SURFACE, INK, INK_SOFT, GRID, MARK ='#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6'


def _floats(values) -> list:
    return [float(v) for v in values]


def _matrix(values) -> list:
    return [_floats(row) for row in values]


def tip_position(rotation, translation, tip_offset) -> np.ndarray:
    """Tool tip in the base frame from the TCP pose and the TCP -> tip vector in TCP axes."""
    return np.asarray(rotation) @ np.asarray(tip_offset, dtype=float) + np.asarray(translation, dtype=float)


def make_point(label: str, distance_cm: float, q, rotation, translation, tip_offset) -> dict:
    """A taught point: its joint angles are what every repeat commands."""
    tip = tip_position(rotation, translation, tip_offset)
    return {'label': label, 'nominal_distance_cm': float(distance_cm),
            'q_commanded': _floats(q), 'R_commanded': _matrix(rotation),
            'tcp_commanded_m': _floats(translation), 'tip_commanded_m': _floats(tip),
            'tip_distance_m': float(np.hypot(tip[0], tip[1])), 'repeats': []}


def add_repeat(point: dict, q, rotation, translation, joint_spread: float, tip_offset) -> dict:
    """Append one measurement to *point*; every error is measured - commanded."""
    tip = tip_position(rotation, translation, tip_offset)
    repeat = {
        'repeat': len(point['repeats']) + 1,
        'q_measured': _floats(q), 'R_measured': _matrix(rotation),
        'tcp_measured_m': _floats(translation), 'tip_measured_m': _floats(tip),
        'joint_error_mrad': _floats((np.asarray(q) - point['q_commanded']) * 1000),
        'tcp_offset_mm': _floats((np.asarray(translation) - point['tcp_commanded_m']) * 1000),
        'tip_offset_mm': _floats((tip - point['tip_commanded_m']) * 1000),
        'rotation_error_mrad': rotation_angle_between(np.asarray(point['R_commanded']),
                                                      np.asarray(rotation)) * 1000,
        'joint_spread_mrad': float(joint_spread) * 1000,
    }
    point['repeats'].append(repeat)
    return repeat


def csv_rows(data: dict) -> list:
    """One header row, then one row per repeat."""
    triples = ('tip_commanded_m', 'tip_measured_m', 'tip_offset_mm',
               'tcp_commanded_m', 'tcp_measured_m', 'tcp_offset_mm')
    joints = ('q_commanded', 'q_measured', 'joint_error_mrad')
    header = (['point', 'nominal_distance_cm', 'repeat']
              + [f'{name}_{axis}' for name in triples for axis in AXES]
              + [f'{name}_{j}' for name in joints for j in JOINTS]
              + ['rotation_error_mrad', 'joint_spread_mrad'])
    rows = [header]
    for point in data['points']:
        for repeat in point['repeats']:
            source = {**point, **repeat}
            rows.append([point['label'], point['nominal_distance_cm'], repeat['repeat']]
                        + [v for name in triples for v in source[name]]
                        + [v for name in joints for v in source[name]]
                        + [repeat['rotation_error_mrad'], repeat['joint_spread_mrad']])
    return rows


def save(data: dict, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, RUN_FILE), 'w', encoding='utf-8') as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=None)
    with open(os.path.join(out, CSV_FILE), 'w', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerows(csv_rows(data))


def measured_points(data: dict) -> list:
    return [point for point in data['points'] if point['repeats']]


def summary_lines(data: dict) -> list:
    lines = []
    for point in measured_points(data):
        tip = np.array([r['tip_offset_mm'] for r in point['repeats']])
        joints = np.array([r['joint_error_mrad'] for r in point['repeats']])
        worst = int(np.argmax(np.abs(joints.mean(axis=0))))
        lines.append(f"{point['label']} ({point['tip_distance_m'] * 100:.1f} cm, "
                     f"{len(tip)} repeats): tip offset x,y,z mean {np.round(tip.mean(axis=0), 2)} mm, "
                     f"std {np.round(tip.std(axis=0), 2)} mm; "
                     f"largest joint error j{worst + 1} {joints.mean(axis=0)[worst]:+.2f} mrad")
    return lines


# ---------------------------------------------------------------- figure
def _strip(axis, groups, names, label_all: bool) -> None:
    """A column of dots per name (one dot per repeat), the mean as a short tick."""
    means = [float(np.mean(values)) for values in groups]
    largest = int(np.argmax(np.abs(means)))
    axis.axhline(0.0, color=INK_SOFT, linewidth=0.8, zorder=1)
    for index, values in enumerate(groups):
        offsets = np.linspace(-0.15, 0.15, len(values)) if len(values) > 1 else np.zeros(1)
        axis.plot(index + offsets, values, 'o', markersize=6.5, markerfacecolor=MARK,
                  markeredgecolor=SURFACE, markeredgewidth=1.0, linestyle='none', zorder=3)
        axis.plot([index - 0.3, index + 0.3], [means[index]] * 2, color=INK, linewidth=1.6,
                  solid_capstyle='round', zorder=4)
        if label_all or index == largest:
            axis.annotate(f'{means[index]:+.2f}', (index + 0.33, means[index]), color=INK,
                          fontsize=8.5, va='center')
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names)
    axis.set_xlim(-0.6, len(names) - 0.2)
    axis.margins(y=0.14)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK_SOFT, length=0, labelsize=9)
    for side in ('top', 'right', 'bottom', 'left'):
        axis.spines[side].set_visible(False)


def plot_results(data: dict, path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    points = measured_points(data)
    figure, axes = plt.subplots(2, len(points), sharey='row', squeeze=False, facecolor=SURFACE,
                                figsize=(4.5 * len(points) + 1.0, 7.6), dpi=150)
    for column, point in enumerate(points):
        tip = np.array([r['tip_offset_mm'] for r in point['repeats']])
        joints = np.array([r['joint_error_mrad'] for r in point['repeats']])
        commanded = np.asarray(point['tip_commanded_m']) * 1000
        for axis in axes[:, column]:
            axis.set_facecolor(SURFACE)
        _strip(axes[0, column], list(tip.T), [a.upper() for a in AXES], label_all=True)
        _strip(axes[1, column], list(joints.T), [f'J{j}' for j in JOINTS], label_all=False)
        axes[0, column].set_title(
            f"{point['label']}: {point['tip_distance_m'] * 100:.1f} cm from the base\n"
            f'commanded tip x {commanded[0]:.1f}, y {commanded[1]:.1f}, z {commanded[2]:.1f} mm',
            loc='left', fontsize=10, color=INK)
        axes[1, column].set_title('Joint error', loc='left', fontsize=10, color=INK)
    axes[0, 0].set_ylabel('Tip offset, measured − commanded [mm]', color=INK_SOFT, fontsize=9.5)
    axes[1, 0].set_ylabel('Joint error, measured − commanded [mrad]', color=INK_SOFT, fontsize=9.5)
    tool = data['tool']
    figure.text(0.012, 0.975, 'Return to commanded joint angles, no compensation', fontsize=13,
                fontweight='bold', color=INK, va='top')
    figure.text(0.012, 0.94, f"Dot = one repeat, black tick = mean. Tip = panda_hand_tcp + {tool['shape']} peg "
                             f"({tool['tip_source']}). Base-frame axes, from the robot's joint readings.",
                fontsize=9.5, color=INK_SOFT, va='top')
    figure.subplots_adjust(top=0.83, bottom=0.05, left=0.09 if len(points) > 1 else 0.17, right=0.985,
                           hspace=0.3, wspace=0.08)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


# --------------------------------------------------------------- procedure
def prepare_peg(data: dict, cad_width: float) -> bool:
    """Clamp the peg and record the finger width it ended at; False to stop the test."""
    print('\nPut the peg in the V-block fingers.')
    while True:
        command = input('Enter = close on the peg, o = open the fingers, s = already held, '
                        'q = stop > ').strip().lower()
        if command == 'q':
            return False
        if command == 'o':
            gripper.open_hand()
            continue
        if command not in ('', 's'):
            print('  Enter, o, s or q')
            continue
        if command == '':
            width = gripper.clamp(cad_width)
            if width is None:
                continue
        else:
            width = gripper.finger_width()
        data['tool']['finger_width_m'] = width
        if width is not None:
            print(f'  peg held at {width * 1000:.2f} mm (CAD {cad_width * 1000:.2f} mm)')
        return True


def teach(topic: str, label: str, distance_cm: float, tip_offset):
    """The pose the operator guided the arm to, or None to stop the test."""
    print(f'\n=== {label}: guide the peg tip to about {distance_cm:g} cm from the robot base ===\n'
          'Hold the guiding buttons on the wrist, move the arm, then release the buttons.')
    while True:
        command = input('Enter = record this point, q = stop the test > ').strip().lower()
        if command == 'q':
            return None
        if command:
            print('  Enter or q')
            continue
        q, rotation, translation, moved = capture_pose(topic, SAMPLE_SECONDS)
        if moved > STILL_RAD:
            print(f'  the arm moved {moved * 1000:.2f} mrad while sampling; let it stop and try again')
            continue
        tip = tip_position(rotation, translation, tip_offset)
        print(f'  tip at {np.round(tip * 1000, 1)} mm, {np.hypot(tip[0], tip[1]) * 100:.1f} cm from the base axis')
        if input('Enter = use this point, r = record again > ').strip().lower() != 'r':
            return q, rotation, translation


def teach_with_controller_stopped(topic: str, label: str, distance_cm: float, tip_offset):
    if not switch_controller(stop=[MOTION_CONTROLLER]):
        raise RuntimeError(f'could not stop {MOTION_CONTROLLER}')
    try:
        return teach(topic, label, distance_cm, tip_offset)
    finally:
        input('Release the guiding buttons and let go of the arm, then press Enter to hold it again... ')
        if not switch_controller(start=[MOTION_CONTROLLER]):
            raise RuntimeError(f'could not restart {MOTION_CONTROLLER}; start it with rosservice '
                               '/controller_manager/switch_controller')
        time.sleep(0.5)


def measure_point(args, data: dict, driver: ArmDriver, topic: str, index: int, distance: float) -> bool:
    """Teach one point (a resumed run may have it) and run its repeats; False when the operator stopped."""
    label = f'P{index}'
    point = data['points'][index - 1] if index <= len(data['points']) else None
    if point is not None and len(point['repeats']) >= args.repeats:
        print(f"{label}: already measured ({len(point['repeats'])} repeats)")
        return True
    if point is not None:
        print(f"\n{label}: saved point, {point['tip_distance_m'] * 100:.1f} cm from the base axis, "
              f"{len(point['repeats'])} of {args.repeats} repeats done")
        if input('Enter = go back to this point, t = teach it again (drops its repeats) > '
                 ).strip().lower() == 't':
            point = None
    if point is None:
        pose = teach_with_controller_stopped(topic, label, distance, args.tip_offset)
        if pose is None:
            return False
        point = make_point(label, distance, *pose, args.tip_offset)
        data['points'][index - 1:index] = [point]
        save(data, args.out)
    q = point['q_commanded']
    print('The arm moves straight in joint space and nothing checks for collisions. '
          'Keep the e-stop at hand.')
    if input(f"Enter = run {args.repeats - len(point['repeats'])} repeats (home, then back to {label}), "
             'q = stop > ').strip().lower() == 'q':
        return False
    for number in range(len(point['repeats']) + 1, args.repeats + 1):
        for name, target in (('home', PANDA_READY), (label, q)):
            print(f'  repeat {number}/{args.repeats}: moving to {name}')
            ok, _, message = driver.move_to(target, args.speed_scale)
            if not ok:
                raise RuntimeError(f'the move to {name} failed: {message}')
        time.sleep(args.dwell)
        repeat = add_repeat(point, *capture_pose(topic, SAMPLE_SECONDS), args.tip_offset)
        save(data, args.out)
        print(f"  repeat {number}: tip offset {np.round(repeat['tip_offset_mm'], 2)} mm, "
              f"joint error {np.round(repeat['joint_error_mrad'], 2)} mrad")
    return True


def hand_over(out: str) -> None:
    """The container runs as root: give the results to the workspace owner."""
    if os.geteuid() != 0 or not out.startswith(WORKSPACE) or not os.path.isdir(out):
        return
    owner = os.stat(WORKSPACE)
    for target in [os.path.dirname(out), out] + [os.path.join(out, name) for name in os.listdir(out)]:
        os.chown(target, owner.st_uid, owner.st_gid)


def finish(data: dict, out: str) -> None:
    if measured_points(data):
        plot_results(data, os.path.join(out, PLOT_FILE))
        print('\n' + '\n'.join(summary_lines(data)))
        print(f'figure: {os.path.join(out, PLOT_FILE)}')
    hand_over(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', help='result folder (default: /franka_ws/repeatability/<date>_<time>)')
    parser.add_argument('--tip-offset', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='TCP -> peg tip in panda_hand_tcp axes, m')
    parser.add_argument('--tip-source', default='CAD', help='where the tip offset comes from (for the figure)')
    parser.add_argument('--shape', default='round', help='peg shape (for the figure)')
    parser.add_argument('--grasp-width', type=float, metavar='M',
                        help='CAD finger width on the peg, m; adds the step that closes the hand on it '
                             '(the hand clamps at the real width, up to 1 mm above this one)')
    parser.add_argument('--distances', nargs='+', type=float, default=[25.0, 50.0, 75.0], metavar='CM',
                        help='about how far from the base each point is, one per point (default 25 50 75)')
    parser.add_argument('--repeats', type=int, default=3, help='repeats per point (default 3)')
    parser.add_argument('--speed-scale', type=float, default=0.3,
                        help='fraction of motion.max_joint_speed (default 0.3)')
    parser.add_argument('--dwell', type=float, default=1.0,
                        help='seconds to wait after the arm has settled, before measuring (default 1)')
    parser.add_argument('--config', help='robot configuration (default: calibration_hardware.yaml)')
    parser.add_argument('--plot', metavar='DIR', help='only draw the figure of an earlier run')
    parser.add_argument('--resume', metavar='DIR',
                        help="finish an interrupted run's unfinished points, with its tool, speed and dwell")
    parser.add_argument('--no-grasp', action='store_true', help='with --resume: skip the step that closes the hand')
    args = parser.parse_args(argv if argv is not None else rospy.myargv(sys.argv)[1:])

    if args.plot:
        with open(os.path.join(args.plot, RUN_FILE), 'r', encoding='utf-8') as handle:
            finish(yaml.safe_load(handle), args.plot)
        return 0
    data = None
    if args.resume:
        with open(os.path.join(args.resume, RUN_FILE), 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle)
        args.out, args.tip_offset = args.resume, data['tool']['tip_offset_tcp_m']
        args.speed_scale, args.dwell = data['speed_scale'], data['dwell_s']
        args.grasp_width = None if args.no_grasp else data['tool'].get('grasp_width_m')
        saved = [point['nominal_distance_cm'] for point in data['points']]
        args.distances = saved + args.distances[len(saved):]
    if args.tip_offset is None:
        parser.error('--tip-offset is required (`pipeline repeatability` passes it from the config)')
    if args.repeats < 1:
        parser.error('--repeats must be at least 1')

    rospy.init_node('measure_repeatability', anonymous=True, disable_signals=True)
    config = load_config(args.config or default_config_path('calibration_hardware.yaml'))
    topic = config['motion']['measured_tcp_pose_topic']
    driver = ArmDriver(config['robot']['joint_names'], config['motion'])
    args.out = args.out or os.path.join(
        WORKSPACE, 'repeatability', datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    data = data or {'tool': {'shape': args.shape, 'tip_offset_tcp_m': _floats(args.tip_offset),
                             'tip_source': args.tip_source, 'grasp_width_m': args.grasp_width},
                    'home_q': _floats(PANDA_READY), 'speed_scale': args.speed_scale, 'dwell_s': args.dwell,
                    'points': []}
    status = 0
    try:
        if controller_running(MOTION_CONTROLLER) is not True:
            raise RuntimeError(f'{MOTION_CONTROLLER} is not running; start panda_calibration robot.launch')
        if not driver.wait_for_controller(timeout=10.0) or not driver.wait_for_state(timeout=10.0):
            raise RuntimeError('no trajectory action or joint states; start panda_calibration robot.launch')
        problem = driver.check_measured_tcp_frame(build_chain(config, rospy.get_param('/robot_description', None)))
        if problem:
            raise RuntimeError(problem)
        if args.grasp_width is None or prepare_peg(data, args.grasp_width):
            for index, distance in enumerate(args.distances, 1):
                if not measure_point(args, data, driver, topic, index, distance):
                    break
    except KeyboardInterrupt:
        driver.client.cancel_all_goals()
        print('\ninterrupted; what was measured so far is saved', file=sys.stderr)
        status = 130
    except RuntimeError as error:
        print(f'error: {error}', file=sys.stderr)
        status = 1
    finally:
        driver.shutdown()
        save(data, args.out)
        finish(data, args.out)
    print(f'results in {args.out}')
    return status
