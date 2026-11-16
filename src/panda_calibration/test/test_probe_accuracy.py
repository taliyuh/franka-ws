import json
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

from panda_calibration import board_manipulation, probe_accuracy
from panda_calibration.board_manipulation import PROBE_ACTION, PROBE_ACTION_TYPE, BoardManipulation
from panda_calibration.kinematics import make_transform, rotation_from_axis_angle
from panda_calibration.probe_accuracy import (CSV_FILE, OPERATOR_STOP, PLAN_FILE, PLOT_FILE, TARGET_REACHED,
                                              Recorder, approach_pair, bottom_pair, csv_rows, sample)

TIP = np.array([0.0, -0.016491, 0.039])
Q = np.array([0.1, -0.5, 0.2, -2.0, 0.1, 1.8, 0.7])
DOWN = np.diag([1.0, -1.0, -1.0])
POSE = make_transform(DOWN, [0.4, 0.05, 0.2])
BOARD = make_transform(rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2), [0.45, 0.0, 0.0])


def shifted(pose, millimetres):
    moved = pose.copy()
    moved[:3, 3] += np.asarray(millimetres, dtype=float) / 1000
    return moved


def recorder(folder, hole='round_c3mm') -> Recorder:
    return Recorder(os.path.join(folder, f'{hole}.yaml'), 'panda_link0', BOARD,
                    np.r_[POSE[:3, 3] + DOWN @ TIP, 1.0], POSE, TIP)


def full_record(folder, hole='round_c3mm') -> Recorder:
    """Approach (two tries), one descent step, the hold and the retract."""
    rec = recorder(folder, hole)
    move = rec.start_move('approach')
    rec.attempt(move, 1, POSE, POSE, Q, Q + 0.002, shifted(POSE, [1.5, 0, -1]), False, 'settled Cartesian error')
    rec.attempt(move, 2, POSE, shifted(POSE, [-1.5, 0, 1]), Q, Q + 0.002, shifted(POSE, [0.1, 0, 0]), True, '')
    move = rec.start_move('descent', 3.0)
    lower = shifted(POSE, [0, 0, -3])
    rec.attempt(move, 1, lower, lower, Q, Q + 0.001, shifted(lower, [0.3, 0, 0]), True, '')
    rec.hold(TARGET_REACHED, 3.0, Q + 0.001, shifted(lower, [0.05, 0, 0]))
    rec.note('  touches the edge ')
    move = rec.start_move('retract')
    rec.attempt(move, 1, POSE, POSE, Q, Q, POSE, True, '')
    rec.finish('done')
    return rec


class SampleTest(unittest.TestCase):
    def test_errors_are_measured_minus_desired(self):
        q_measured = Q + np.array([0, 0.002, 0, -0.003, 0, 0, 0])
        entry = sample(POSE, Q, q_measured, shifted(POSE, [-1.5, 0.0, -1.0]), TIP, BOARD[:3, :3])
        np.testing.assert_allclose(entry['tcp_error_mm'], [-1.5, 0, -1], atol=1e-9)
        np.testing.assert_allclose(entry['tip_error_mm'], [-1.5, 0, -1], atol=1e-9)
        np.testing.assert_allclose(entry['joint_error_mrad'], [0, 2, 0, -3, 0, 0, 0], atol=1e-9)
        # The board is turned 90 deg about z: base -x is board +y.
        np.testing.assert_allclose(entry['tip_error_board_mm'], [0, 1.5, -1], atol=1e-9)
        self.assertAlmostEqual(entry['rotation_error_mrad'], 0.0)

    def test_tip_error_includes_the_lever_of_a_rotation(self):
        turned = POSE.copy()
        turned[:3, :3] = rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.01) @ DOWN
        entry = sample(POSE, Q, Q, turned, TIP, np.eye(3))
        self.assertAlmostEqual(entry['rotation_error_mrad'], 10.0, places=6)
        # The tip is 16.49 mm off the rotation axis, so 10 mrad moves it 0.165 mm.
        self.assertAlmostEqual(float(np.linalg.norm(entry['tip_error_mm'])), 0.16491, places=3)
        np.testing.assert_allclose(entry['tcp_error_mm'], [0, 0, 0], atol=1e-9)


class RecorderTest(unittest.TestCase):
    def test_every_attempt_and_the_hold_are_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            rec = full_record(folder)
            with open(rec.path, encoding='utf-8') as handle:
                saved = yaml.safe_load(handle)
        self.assertEqual(saved, rec.data)
        self.assertEqual([m['phase'] for m in saved['moves']], ['approach', 'descent', 'retract'])
        first, final = approach_pair(saved)
        np.testing.assert_allclose(first['tip_error_mm'], [1.5, 0, -1], atol=1e-9)
        np.testing.assert_allclose(final['tip_error_mm'], [0.1, 0, 0], atol=1e-9)
        np.testing.assert_allclose(final['tcp_command_m'], shifted(POSE, [-1.5, 0, 1])[:3, 3])
        bottom, hold = bottom_pair(saved)
        np.testing.assert_allclose(bottom['tip_error_mm'], [0.3, 0, 0], atol=1e-9)
        # The hold is compared with the last command: the descent step's.
        np.testing.assert_allclose(hold['tip_error_mm'], [0.05, 0, 0], atol=1e-9)
        np.testing.assert_allclose(hold['joint_error_mrad'], [1] * 7, atol=1e-9)
        self.assertEqual(hold['note'], 'touches the edge')
        self.assertEqual(saved['status'], 'done')

    def test_table_has_a_row_per_attempt_and_hold_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            rec = full_record(folder)
        rows = csv_rows([({'name': 'round_c3mm'}, rec.data)])
        self.assertEqual([row[1] for row in rows[1:]], ['approach', 'approach', 'descent', 'hold', 'retract'])
        self.assertEqual({len(row) for row in rows}, {len(rows[0])})
        self.assertIn('tip_error_board_mm_y', rows[0])
        self.assertIn('joint_error_mrad_7', rows[0])

    def test_figure_table_and_summary_of_a_run_with_an_aborted_hole(self):
        with tempfile.TemporaryDirectory() as folder:
            full_record(folder, 'round_c0p03mm')
            aborted = recorder(folder, 'round_c3mm')
            move = aborted.start_move('approach')
            aborted.attempt(move, 1, POSE, POSE, Q, Q, shifted(POSE, [4, 0, 0]), False, 'settled Cartesian error')
            aborted.finish('aborted', 'Cartesian position residual is 4.000 mm; correction window is 3.000 mm')
            plan = {'holes': [{'name': 'round_c0p03mm', 'shape': 'round', 'clearance_mm': 0.03,
                               'correction_mm': [1.52, -2.96, -0.34]},
                              {'name': 'round_c3mm', 'shape': 'round', 'clearance_mm': 3},
                              {'name': 'rectangle_c3mm', 'shape': 'rectangle', 'clearance_mm': 3}]}
            with open(os.path.join(folder, PLAN_FILE), 'w', encoding='utf-8') as handle:
                yaml.safe_dump(plan, handle)
            with mock.patch('builtins.print') as printed:
                self.assertEqual(probe_accuracy.main([folder]), 0)
            self.assertGreater(os.path.getsize(os.path.join(folder, PLOT_FILE)), 10000)
            with open(os.path.join(folder, CSV_FILE), encoding='utf-8') as handle:
                self.assertEqual(len(handle.readlines()), 1 + 5 + 1)
        summary = printed.call_args_list[0][0][0].splitlines()
        self.assertEqual(len(summary), 2, 'the hole never probed has no line')
        self.assertIn('note: touches the edge', summary[0])
        self.assertIn('[aborted: Cartesian position residual', summary[1])


class FakeDriver:
    """Settles at the given poses in turn; accepts within 0.2 mm like ArmDriver.move_to."""

    def __init__(self, settled):
        self.settled = list(settled)
        self.positions = Q + 0.0025
        self.last_measured_tcp_pose = None

    def move_to(self, joints, speed_scale, cartesian_target=None, kinematic_chain=None, cartesian_frame=None):
        self.last_measured_tcp_pose = self.settled.pop(0)
        error = np.linalg.norm(self.last_measured_tcp_pose[:3, 3] - cartesian_target[:3, 3])
        if error > 0.0002:
            return False, 0.0, f'settled Cartesian error is {error * 1000:.3f} mm'
        return True, 0.0, ''


class CorrectionLoopTest(unittest.TestCase):
    def test_each_attempt_of_the_correction_loop_is_recorded(self):
        node = BoardManipulation.__new__(BoardManipulation)
        node.config = {'motion': {'max_cartesian_position_error_m': 0.0002,
                                  'max_cartesian_rotation_error_rad': 0.0008,
                                  'cartesian_correction_attempts': 3,
                                  'cartesian_correction_window_m': 0.003,
                                  'cartesian_correction_window_rad': 0.008}}
        node.args = SimpleNamespace(dry_run=False)
        node.kinematic_chain = None
        node.driver = FakeDriver([shifted(POSE, [-1.5, 0, -1]), POSE])
        node.solve_ik = mock.Mock(return_value=list(Q))
        with tempfile.TemporaryDirectory() as folder:
            node.recorder = recorder(folder)
            node.move_cartesian_pose(POSE, 'panda_link0', 0.5, record={'phase': 'approach'})
        attempts = node.recorder.data['moves'][0]['attempts']
        self.assertEqual(len(attempts), 2)
        np.testing.assert_allclose(attempts[0]['tip_error_mm'], [-1.5, 0, -1], atol=1e-9)
        np.testing.assert_allclose(attempts[0]['joint_error_mrad'], [2.5] * 7, atol=1e-9)
        self.assertFalse(attempts[0]['ok'])
        # The second command is shifted against the first try's error.
        np.testing.assert_allclose(attempts[1]['tcp_command_m'], shifted(POSE, [1.5, 0, 1])[:3, 3])
        np.testing.assert_allclose(attempts[1]['tip_error_mm'], [0, 0, 0], atol=1e-9)
        self.assertTrue(attempts[1]['ok'])


class FakeNode:
    """A perfect arm: it settles exactly on every command and records like BoardManipulation."""

    def __init__(self, during_descent_step=None):
        self.recorder = None
        self.pose = POSE.copy()
        self.driver = SimpleNamespace(positions=Q)
        self.phases = []
        self.during_descent_step = during_descent_step

    def wait_for_robot(self):
        return True

    def measured_tcp(self, base_frame):
        return self.pose.copy()

    def shutdown(self):
        pass

    def move_cartesian_pose(self, pose, frame_id, speed_scale, pose_validator=None, record=None):
        if pose_validator is not None:
            pose_validator(pose)
        self.pose = pose.copy()
        self.phases.append(record['phase'])
        move = self.recorder.start_move(**record)
        self.recorder.attempt(move, 1, pose, pose, Q, Q, pose, True, '')
        if record['phase'] == 'descent' and self.during_descent_step:
            self.during_descent_step()


class RecordedProbeTest(unittest.TestCase):
    """probe_board main() with --record and the arm replaced: the hold, the note, the retract."""

    def run_main(self, node, reply):
        with tempfile.TemporaryDirectory() as folder:
            board = os.path.join(folder, 'board.json')
            with open(board, 'w', encoding='utf-8') as handle:
                json.dump({'base_frame': 'panda_link0', 'T_base_board': make_transform(np.eye(3), [0.45, 0, 0]).tolist(),
                           'board_surface_z_m': 0.025}, handle)
            tool = os.path.join(folder, 'tool.yaml')
            with open(tool, 'w', encoding='utf-8') as handle:
                yaml.safe_dump({'probe': {'staging_z_m': None, 'approach_height_m': 0.02,
                                          'approach_speed_scale': 0.5, 'descent_speed_scale': 0.6,
                                          'descent_segment_m': 0.003, 'max_descent_m': 0.15,
                                          'tcp_orientation_board_rpy': [np.pi, 0.0, 0.0],
                                          'pen_tip_offset_tcp_m': [0.0, 0.0, 0.1]}}, handle)
            record = os.path.join(folder, 'round_c3mm.yaml')
            config = {'motion': {'action_type': PROBE_ACTION_TYPE, 'controller_action': PROBE_ACTION},
                      'robot': {'joint_names': []}}
            with mock.patch.object(board_manipulation, 'BoardManipulation', return_value=node), \
                    mock.patch.object(board_manipulation, 'load_config', return_value=config), \
                    mock.patch.object(board_manipulation, 'default_config_path', return_value='unused'), \
                    mock.patch.object(board_manipulation.rospy, 'init_node'), \
                    mock.patch.object(board_manipulation.rospy, 'signal_shutdown'), \
                    mock.patch('builtins.print'), \
                    mock.patch('builtins.input', side_effect=reply):
                status = board_manipulation.main(['--board-pose', board, '--offset', '0', '0', '0',
                                                  '--manipulation-config', tool, '--record', record])
            with open(record, encoding='utf-8') as handle:
                return status, yaml.safe_load(handle)

    def test_reaching_the_target_holds_for_photos_and_keeps_the_note(self):
        held = threading.Event()
        original = Recorder.hold

        def hold(rec, *args):
            entry = original(rec, *args)
            held.set()
            return entry

        def reply(prompt=''):
            if prompt.startswith('Descending'):
                held.wait(10)   # nobody stops the descent; the line typed at the hold is the note
                return 'touches the +x edge'
            return ''

        node = FakeNode()
        with mock.patch.object(Recorder, 'hold', hold):
            status, saved = self.run_main(node, reply)
        self.assertEqual(status, 0)
        self.assertEqual(node.phases, ['approach'] + ['descent'] * 7 + ['retract'])
        self.assertEqual(saved['status'], 'done')
        self.assertEqual(saved['hold']['reason'], TARGET_REACHED)
        self.assertAlmostEqual(saved['hold']['descended_mm'], 20.0)
        self.assertEqual(saved['hold']['note'], 'touches the +x edge')
        np.testing.assert_allclose(saved['hold']['tip_measured_m'], [0.45, 0.0, 0.025], atol=1e-9)

    def test_an_operator_stop_holds_there_and_asks_for_the_note(self):
        stepping, answered = threading.Event(), threading.Event()

        def reply(prompt=''):
            if prompt.startswith('Descending'):
                stepping.wait(10)   # Enter while the first step moves: stop after it
                answered.set()
                return ''
            return 'in' if prompt.startswith('>') else ''

        def during_descent_step():
            stepping.set()
            answered.wait(10)
            time.sleep(0.05)   # let the reader thread set the stop

        node = FakeNode(during_descent_step)
        status, saved = self.run_main(node, reply)
        self.assertEqual(status, 0)
        self.assertEqual(node.phases, ['approach', 'descent', 'retract'])
        self.assertEqual(saved['hold']['reason'], OPERATOR_STOP)
        self.assertAlmostEqual(saved['hold']['descended_mm'], 3.0)
        self.assertEqual(saved['hold']['note'], 'in')
        self.assertEqual(saved['status'], 'done')


if __name__ == '__main__':
    unittest.main()
