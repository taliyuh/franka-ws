"""Target vs measured pose at every probe move, for `pipeline accuracy`.

`probe_board --record FILE` keeps one hole's record with this module. For every
move (the approach above the hole, each descent step, the retract) and every
attempt of the probe's correction loop it stores the commanded TCP pose and
joints and where the arm settled. A move's first attempt is not corrected, so its
error is how the arm settles on its own; the last attempt is the one the probe
accepted (within 0.2 mm). At the bottom the arm holds for photos; that pose is
stored with the operator's note.

"Measured" is the robot's own reading, O_T_EE: the forward kinematics of the
measured joints through the nominal model. It shows how the arm settles against
its commands, not where the tip is in the real world. The photos show that.

A run folder (written by `pipeline accuracy`) holds plan.yaml, with the holes in
order and their peg and clearance, and one <hole>.yaml per probed hole.

    rosrun panda_calibration plot_probe_accuracy DIR    (figure, table and summary of a run)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys

import numpy as np
import yaml

from .kinematics import make_transform, rotation_angle_between
from .repeatability import (AXES, GRID, INK, INK_SOFT, JOINTS, SURFACE, _floats, _matrix,
                            tip_position)

PLAN_FILE = 'plan.yaml'
CSV_FILE = 'measurements.csv'
PLOT_FILE = 'probe_accuracy.png'
FIRST, FINAL = '#eb6834', '#2a78d6'   # categorical slots 2 and 1 of the reference palette
OPERATOR_STOP = 'stopped by the operator'
TARGET_REACHED = 'reached the target'


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


def sample(desired, q_command, q_measured, measured, tip_offset, board_rotation) -> dict:
    """The desired and the measured TCP pose and joints; every error is measured - desired."""
    desired, measured = np.asarray(desired, dtype=float), np.asarray(measured, dtype=float)
    q_command = np.asarray(q_command, dtype=float)
    q_measured = np.full(len(q_command), np.nan) if q_measured is None else np.asarray(q_measured, dtype=float)
    desired_tip = tip_position(desired[:3, :3], desired[:3, 3], tip_offset)
    measured_tip = tip_position(measured[:3, :3], measured[:3, 3], tip_offset)
    tip_error = measured_tip - desired_tip
    return {
        'tcp_desired_m': _floats(desired[:3, 3]), 'R_desired': _matrix(desired[:3, :3]),
        'tip_desired_m': _floats(desired_tip),
        'q_command_rad': _floats(q_command), 'q_measured_rad': _floats(q_measured),
        'tcp_measured_m': _floats(measured[:3, 3]), 'R_measured': _matrix(measured[:3, :3]),
        'tip_measured_m': _floats(measured_tip),
        'tcp_error_mm': _floats((measured[:3, 3] - desired[:3, 3]) * 1000),
        'tip_error_mm': _floats(tip_error * 1000),
        'tip_error_board_mm': _floats(np.asarray(board_rotation).T @ tip_error * 1000),
        'rotation_error_mrad': rotation_angle_between(desired[:3, :3], measured[:3, :3]) * 1000,
        'joint_error_mrad': _floats((q_measured - q_command) * 1000),
    }


def describe(phase: str, entry: dict) -> str:
    """One line for the terminal: the tip error of one attempt and its largest joint error."""
    error = np.asarray(entry['tip_error_mm'])
    joints = np.asarray(entry['joint_error_mrad'])
    worst = int(np.nanargmax(np.abs(joints))) if np.isfinite(joints).any() else 0
    return (f"{phase}, try {entry['attempt']}: tip error x {error[0]:+.2f}, y {error[1]:+.2f}, "
            f"z {error[2]:+.2f} mm ({np.linalg.norm(error):.2f} mm); largest joint error "
            f"J{worst + 1} {joints[worst]:+.2f} mrad")


class Recorder:
    """One hole's record, saved to *path* after every change so an interrupted probe keeps it."""

    def __init__(self, path: str, base_frame: str, board_transform, target_tip, target_tcp, tip_offset):
        self.path = path
        self.tip_offset = np.asarray(tip_offset, dtype=float)
        self.board_rotation = np.asarray(board_transform, dtype=float)[:3, :3]
        self.data = {
            'hole': os.path.splitext(os.path.basename(path))[0],
            'started': _now(), 'status': 'running', 'message': '',
            'base_frame': base_frame,
            'tip_offset_tcp_m': _floats(self.tip_offset),
            'T_base_board': _matrix(board_transform),
            'target_tip_m': _floats(np.asarray(target_tip, dtype=float)[:3]),
            'target_tcp': _matrix(target_tcp),
            'moves': [], 'hold': None}
        self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        temporary = self.path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(self.data, handle, sort_keys=False, default_flow_style=None)
        os.replace(temporary, self.path)

    def start_move(self, phase: str, depth_mm: float | None = None) -> dict:
        move = {'phase': phase, 'depth_mm': None if depth_mm is None else float(depth_mm), 'attempts': []}
        self.data['moves'].append(move)
        return move

    def attempt(self, move: dict, number: int, desired, command, q_command, q_measured, measured,
                ok: bool, message: str) -> dict:
        entry = {'attempt': int(number), 'time': _now(), 'ok': bool(ok), 'message': str(message or ''),
                 'tcp_command_m': _floats(np.asarray(command, dtype=float)[:3, 3]),
                 **sample(desired, q_command, q_measured, measured, self.tip_offset, self.board_rotation)}
        move['attempts'].append(entry)
        self.save()
        return entry

    def last_attempt(self) -> dict | None:
        for move in reversed(self.data['moves']):
            if move['attempts']:
                return move['attempts'][-1]
        return None

    def hold(self, reason: str, descended_mm: float, q_measured, measured) -> dict:
        """The pose the arm holds for the photos, against the last command it was given."""
        last = self.last_attempt()
        if last is None:
            raise RuntimeError('no move was recorded before the hold')
        desired = make_transform(np.asarray(last['R_desired']), last['tcp_desired_m'])
        self.data['hold'] = {'time': _now(), 'reason': reason, 'descended_mm': float(descended_mm),
                             'note': '',
                             **sample(desired, last['q_command_rad'], q_measured, measured,
                                      self.tip_offset, self.board_rotation)}
        self.save()
        return self.data['hold']

    def note(self, text: str) -> None:
        self.data['hold']['note'] = text.strip()
        self.save()

    def finish(self, status: str, message: str = '') -> None:
        self.data.update(status=status, message=message, finished=_now())
        self.save()


# ------------------------------------------------------------------- reading
def _load(path: str):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def load_run(folder: str) -> tuple[dict, list]:
    """The plan and [(plan entry, record)] of the probed holes, in plan order."""
    plan_path = os.path.join(folder, PLAN_FILE)
    if os.path.exists(plan_path):
        plan = _load(plan_path)
    else:
        names = sorted(name[:-len('.yaml')] for name in os.listdir(folder) if name.endswith('.yaml'))
        plan = {'holes': [{'name': name} for name in names]}
    records = []
    for entry in plan['holes']:
        path = os.path.join(folder, f"{entry['name']}.yaml")
        if os.path.exists(path):
            records.append((entry, _load(path)))
    return plan, records


def approach_pair(record: dict) -> tuple:
    """First and last attempt of the approach above the hole."""
    for move in record['moves']:
        if move['phase'] == 'approach' and move['attempts']:
            return move['attempts'][0], move['attempts'][-1]
    return None, None


def bottom_pair(record: dict) -> tuple:
    """First attempt of the last move before the hold, and the hold itself."""
    hold = record.get('hold')
    if hold is None:
        return None, None
    moves = [move for move in record['moves'] if move['phase'] != 'retract' and move['attempts']]
    return (moves[-1]['attempts'][0] if moves else None), hold


def _norm(entry: dict | None) -> str:
    return '-' if entry is None else f"{np.linalg.norm(entry['tip_error_mm']):.2f}"


def summary_lines(records: list) -> list:
    lines = []
    for _, record in records:
        first, final = approach_pair(record)
        bottom, hold = bottom_pair(record)
        line = f"{record['hole']}: above the hole, tip error first try {_norm(first)} mm, " \
               f"after the correction {_norm(final)} mm"
        if hold is not None:
            joints = np.asarray(hold['joint_error_mrad'], dtype=float)
            worst = int(np.nanargmax(np.abs(joints))) if np.isfinite(joints).any() else 0
            line += (f"; at the bottom ({hold['reason']}, {hold['descended_mm']:.1f} mm down) "
                     f"first try {_norm(bottom)} mm, after the correction {_norm(hold)} mm, "
                     f"largest joint error J{worst + 1} {joints[worst]:+.2f} mrad")
            if hold.get('note'):
                line += f"; note: {hold['note']}"
        if record['status'] != 'done':
            line += f"  [{record['status']}{': ' + record['message'] if record.get('message') else ''}]"
        lines.append(line)
    return lines


# --------------------------------------------------------------------- table
TRIPLES = ('tip_desired_m', 'tip_measured_m', 'tip_error_mm', 'tip_error_board_mm',
           'tcp_desired_m', 'tcp_measured_m', 'tcp_error_mm')
JOINT_FIELDS = ('q_command_rad', 'q_measured_rad', 'joint_error_mrad')


def _row(hole, phase, depth, attempt, ok, entry, note, message) -> list:
    return ([hole, phase, '' if depth is None else depth, attempt, ok]
            + [v for name in TRIPLES for v in entry[name]]
            + [v for name in JOINT_FIELDS for v in entry[name]]
            + [entry['rotation_error_mrad'], entry['time'], note, message])


def csv_rows(records: list) -> list:
    """A header, then one row per attempt and one per hold, in the order they happened."""
    header = (['hole', 'phase', 'depth_mm', 'attempt', 'ok']
              + [f'{name}_{axis}' for name in TRIPLES for axis in AXES]
              + [f'{name}_{j}' for name in JOINT_FIELDS for j in JOINTS]
              + ['rotation_error_mrad', 'time', 'note', 'message'])
    rows = [header]
    for _, record in records:
        moves = record['moves']
        for move in [m for m in moves if m['phase'] != 'retract']:
            rows += [_row(record['hole'], move['phase'], move['depth_mm'], a['attempt'], a['ok'], a, '',
                          a['message']) for a in move['attempts']]
        hold = record.get('hold')
        if hold is not None:
            rows.append(_row(record['hole'], 'hold', hold['descended_mm'], '', '', hold,
                             hold.get('note', ''), hold['reason']))
        for move in [m for m in moves if m['phase'] == 'retract']:
            rows += [_row(record['hole'], move['phase'], move['depth_mm'], a['attempt'], a['ok'], a, '',
                          a['message']) for a in move['attempts']]
    return rows


def write_csv(records: list, path: str) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerows(csv_rows(records))


# -------------------------------------------------------------------- figure
def _values(entry: dict | None, key: str):
    return None if entry is None else np.asarray(entry[key], dtype=float)


def _dots(axis, names, first, final, label: str) -> None:
    """Per name: the first try (left) and the value after the correction (right)."""
    x = np.arange(len(names))
    axis.axhline(0.0, color=INK_SOFT, linewidth=0.8, zorder=1)
    if first is not None:
        axis.plot(x - 0.14, first, 'D', markersize=6.5, markerfacecolor=FIRST, markeredgecolor=SURFACE,
                  markeredgewidth=1.0, linestyle='none', zorder=3)
        finite = np.isfinite(first)
        labelled = (np.flatnonzero(finite) if label == 'all'
                    else [int(np.nanargmax(np.abs(first)))] if finite.any() else [])
        for index in labelled:
            axis.annotate(f'{first[index]:+.2f}', (index - 0.25, first[index]), color=INK, fontsize=8.5,
                          ha='right', va='center')
    if final is not None:
        axis.plot(x + 0.14, final, 'o', markersize=7.5, markerfacecolor=FINAL, markeredgecolor=SURFACE,
                  markeredgewidth=1.0, linestyle='none', zorder=4)
    axis.set_xticks(x)
    axis.set_xticklabels(names)
    axis.set_xlim(-0.75, len(names) - 0.35)
    axis.margins(y=0.16)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK_SOFT, length=0, labelsize=9)
    for side in ('top', 'right', 'bottom', 'left'):
        axis.spines[side].set_visible(False)


def _empty(axis, text: str) -> None:
    axis.text(0.5, 0.5, text, transform=axis.transAxes, ha='center', va='center', color=INK_SOFT,
              fontsize=9)
    axis.set_xticks([])
    axis.tick_params(colors=INK_SOFT, length=0, labelsize=9)
    for side in ('top', 'right', 'bottom', 'left'):
        axis.spines[side].set_visible(False)


def _short(text: str, length: int = 46) -> str:
    text = ' '.join(str(text).split())
    return text if len(text) <= length else text[:length - 1] + '…'


def _column_title(entry: dict, record: dict) -> str:
    clearance = entry.get('clearance_mm')
    lines = [record['hole'] + (f' · clearance {clearance:g} mm' if clearance is not None else '')]
    tip = np.asarray(record['target_tip_m']) * 1000
    lines.append(f'target tip x {tip[0]:.1f}, y {tip[1]:.1f}, z {tip[2]:.1f} mm')
    hold = record.get('hold')
    if hold is not None:
        lines.append(f"{hold['reason']}, {hold['descended_mm']:.1f} mm down")
        if hold.get('note'):
            lines.append(_short(f"note: {hold['note']}"))
    if record['status'] != 'done':
        lines.append(_short(f"{record['status']}: {record.get('message') or 'no message'}"))
    if entry.get('correction_mm') is not None:
        shift = entry['correction_mm']
        lines.append(f'joint-offset correction moved it {shift[0]:+.2f}, {shift[1]:+.2f}, {shift[2]:+.2f} mm')
    return '\n'.join(lines)


def plot_run(records: list, path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    count = len(records)
    figure, axes = plt.subplots(3, count, sharey='row', squeeze=False, facecolor=SURFACE,
                                figsize=(4.4 * count + 1.6, 11.0), dpi=150)
    tip_axes, joint_names = [a.upper() for a in AXES], [f'J{j}' for j in JOINTS]
    for column, (entry, record) in enumerate(records):
        for axis in axes[:, column]:
            axis.set_facecolor(SURFACE)
        first, final = approach_pair(record)
        if final is None:
            _empty(axes[0, column], 'no approach recorded')
        else:
            _dots(axes[0, column], tip_axes, _values(first, 'tip_error_mm'), _values(final, 'tip_error_mm'), 'all')
        first, hold = bottom_pair(record)
        if hold is None:
            _empty(axes[1, column], 'never held at the bottom')
            _empty(axes[2, column], 'never held at the bottom')
        else:
            _dots(axes[1, column], tip_axes, _values(first, 'tip_error_mm'), _values(hold, 'tip_error_mm'), 'all')
            _dots(axes[2, column], joint_names, _values(first, 'joint_error_mrad'),
                  _values(hold, 'joint_error_mrad'), 'largest')
        axes[0, column].set_title(_column_title(entry, record), loc='left', fontsize=9, color=INK)
    for row, text in enumerate(('Above the hole\ntip, measured − target [mm]',
                                'At the bottom (photo pose)\ntip, measured − target [mm]',
                                'At the bottom (photo pose)\njoints, measured − commanded [mrad]')):
        axes[row, 0].set_ylabel(text, color=INK_SOFT, fontsize=9.5)
    figure.text(0.012, 0.985, 'Probe accuracy: target vs measured at each hole', fontsize=13,
                fontweight='bold', color=INK, va='top')
    figure.text(0.012, 0.962, "Measured = the robot's own joint readings through the nominal model "
                              '(O_T_EE), base-frame axes. It cannot see calibration or peg errors: the photos '
                              'show where the tip really is.', fontsize=9.5, color=INK_SOFT, va='top')
    handles = [Line2D([], [], marker='D', linestyle='none', markersize=7, markerfacecolor=FIRST,
                      markeredgecolor=SURFACE, label='First try: where the arm settled on its own'),
               Line2D([], [], marker='o', linestyle='none', markersize=8, markerfacecolor=FINAL,
                      markeredgecolor=SURFACE,
                      label="After the probe's correction (it corrects until it is within 0.2 mm)")]
    legend = figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.008, 0.948), ncol=2,
                           frameon=False, fontsize=9.5, handletextpad=0.4, columnspacing=2.0)
    for text in legend.get_texts():
        text.set_color(INK)
    figure.subplots_adjust(top=0.815, bottom=0.04, left=0.07 if count > 1 else 0.2, right=0.99,
                           hspace=0.32, wspace=0.1)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', help='run folder (sessions/<name>/accuracy/<date>_<time>)')
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    _, records = load_run(args.run)
    if not records:
        print(f'error: no probed holes in {args.run}', file=sys.stderr)
        return 1
    write_csv(records, os.path.join(args.run, CSV_FILE))
    plot_run(records, os.path.join(args.run, PLOT_FILE))
    print('\n'.join(summary_lines(records)))
    print(f'figure: {os.path.join(args.run, PLOT_FILE)}\ntable:  {os.path.join(args.run, CSV_FILE)}')
    return 0
