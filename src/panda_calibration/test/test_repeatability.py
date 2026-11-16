import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from panda_calibration import repeatability
from panda_calibration.kinematics import PANDA_READY, rotation_from_axis_angle
from panda_calibration.repeatability import (CSV_FILE, PLOT_FILE, RUN_FILE, add_repeat, csv_rows,
                                             make_point, plot_results, save, summary_lines)

TIP = np.array([0.0, -0.01649, 0.039])
Q = np.array([0.1, -0.5, 0.2, -2.0, 0.1, 1.8, 0.7])
DOWN = np.diag([1.0, -1.0, -1.0])
TCP = np.array([0.4, 0.05, 0.2])


def run_data(points=1, repeats=3) -> dict:
    data = {'tool': {'shape': 'round', 'tip_source': 'CAD'}, 'points': []}
    for p in range(points):
        point = make_point(f'P{p + 1}', 25.0, Q, DOWN, TCP, TIP)
        for r in range(repeats):
            add_repeat(point, Q + 1e-3, DOWN, TCP + [0.001 * r, 0, 0], 1e-4, TIP)
        data['points'].append(point)
    return data


class RepeatabilityTest(unittest.TestCase):
    def test_errors_are_measured_minus_commanded(self):
        point = make_point('P1', 25.0, Q, DOWN, TCP, TIP)
        measured_q = Q + np.array([0, 0.002, 0, -0.003, 0, 0, 0])
        repeat = add_repeat(point, measured_q, DOWN, TCP + [-0.0015, 0.0, -0.001], 2e-4, TIP)
        np.testing.assert_allclose(repeat['joint_error_mrad'], [0, 2, 0, -3, 0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(repeat['tcp_offset_mm'], [-1.5, 0.0, -1.0], atol=1e-9)
        np.testing.assert_allclose(repeat['tip_offset_mm'], [-1.5, 0.0, -1.0], atol=1e-9)
        self.assertAlmostEqual(repeat['rotation_error_mrad'], 0.0)
        self.assertAlmostEqual(repeat['joint_spread_mrad'], 0.2)

    def test_tip_offset_includes_the_lever_of_a_rotation(self):
        point = make_point('P1', 25.0, Q, DOWN, TCP, TIP)
        turned = rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.01) @ DOWN
        repeat = add_repeat(point, Q, turned, TCP, 0.0, TIP)
        self.assertAlmostEqual(repeat['rotation_error_mrad'], 10.0, places=6)
        # The tip is 16.49 mm off the rotation axis, so 10 mrad moves it 0.165 mm.
        self.assertAlmostEqual(float(np.linalg.norm(repeat['tip_offset_mm'])), 0.1649, places=3)
        np.testing.assert_allclose(repeat['tcp_offset_mm'], [0, 0, 0], atol=1e-9)

    def test_csv_has_one_row_per_repeat(self):
        rows = csv_rows(run_data(points=2, repeats=3))
        self.assertEqual(len(rows), 1 + 6)
        self.assertEqual({len(row) for row in rows}, {len(rows[0])})
        self.assertIn('tip_offset_mm_x', rows[0])
        self.assertIn('joint_error_mrad_7', rows[0])

    def test_save_round_trip_and_figure(self):
        data = run_data(points=3)
        data['points'].append(make_point('P4', 75.0, Q, DOWN, TCP, TIP))   # taught, never measured
        with tempfile.TemporaryDirectory() as folder:
            save(data, folder)
            with open(os.path.join(folder, RUN_FILE), encoding='utf-8') as handle:
                self.assertEqual(yaml.safe_load(handle), data)
            self.assertTrue(os.path.exists(os.path.join(folder, CSV_FILE)))
            path = os.path.join(folder, PLOT_FILE)
            plot_results(data, path)
            self.assertGreater(os.path.getsize(path), 10000)
        self.assertEqual(len(summary_lines(data)), 3)


class ProcedureTest(unittest.TestCase):
    """main() with the robot replaced: what is commanded, and in which order."""

    def resume(self, driver, answers=None):
        """Resume a saved run whose P1 is finished and P2 has one of two repeats; returns the saved run."""
        other = Q + 0.1
        data = run_data(points=1, repeats=2)
        data.update(speed_scale=0.3, dwell_s=0.0)
        data['tool'].update(tip_offset_tcp_m=TIP.tolist(), grasp_width_m=None)
        second = make_point('P2', 50.0, other, DOWN, TCP, TIP)
        add_repeat(second, other, DOWN, TCP, 0.0, TIP)
        data['points'].append(second)
        with tempfile.TemporaryDirectory() as folder:
            save(data, folder)
            self.assertEqual(self.run_main(driver, folder, answers=answers,
                                           argv=['--resume', folder, '--repeats', '2', '--distances', '25', '50']), 0)
            with open(os.path.join(folder, RUN_FILE), encoding='utf-8') as handle:
                return other, yaml.safe_load(handle)

    def test_resume_only_finishes_the_unfinished_point(self):
        driver = self.driver()
        other, saved = self.resume(driver)
        self.assertEqual([len(p['repeats']) for p in saved['points']], [2, 2])
        calls = driver.move_to.call_args_list
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0][0][0], PANDA_READY)
        np.testing.assert_array_equal(calls[1][0][0], other)
        self.assertEqual(calls[0][0][1], 0.3)

    def test_resume_can_teach_the_unfinished_point_again(self):
        driver = self.driver()
        # t, record, use it, release the arm, run the repeats
        _, saved = self.resume(driver, answers=['t', '', '', '', ''])
        self.assertEqual(len(saved['points']), 2)
        self.assertEqual(len(saved['points'][1]['repeats']), 2, 'the old repeat is dropped')
        np.testing.assert_allclose(saved['points'][1]['q_commanded'], Q)
        self.assertEqual(driver.move_to.call_count, 4)

    def run_main(self, driver, folder, repeats='2', extra=(), argv=None, answers=None):
        with mock.patch.multiple(repeatability, ArmDriver=mock.Mock(return_value=driver),
                                 load_config=mock.Mock(return_value={
                                     'motion': {'measured_tcp_pose_topic': 'state'},
                                     'robot': {'joint_names': []}}),
                                 build_chain=mock.DEFAULT, default_config_path=mock.DEFAULT,
                                 controller_running=mock.Mock(return_value=True),
                                 switch_controller=mock.Mock(return_value=True),
                                 capture_pose=mock.Mock(return_value=(Q, DOWN, TCP, 1e-5))), \
                mock.patch.object(repeatability.rospy, 'init_node'), \
                mock.patch.object(repeatability.rospy, 'get_param'), \
                mock.patch('builtins.input', **({'side_effect': answers} if answers else {'return_value': ''})):
            return repeatability.main(argv or ['--out', folder, '--tip-offset', '-0.000000', '-0.016491', '0.039',
                                               '--distances', '25', '--repeats', repeats, '--dwell', '0',
                                               *extra])

    def grasp_run(self, widths):
        """Run with the grasp step; returns (the grasp mock, the saved tool section)."""
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(repeatability.gripper, 'grasp', return_value=True) as grasp, \
                mock.patch.object(repeatability.gripper, 'finger_width', side_effect=widths):
            self.assertEqual(self.run_main(self.driver(), folder, repeats='1',
                                           extra=('--grasp-width', '0.005019')), 0)
            with open(os.path.join(folder, RUN_FILE), encoding='utf-8') as handle:
                return grasp, yaml.safe_load(handle)['tool']

    def test_the_hand_clamps_below_the_cad_width_and_records_the_real_one(self):
        grasp, tool = self.grasp_run([0.0021])
        self.assertEqual(grasp.call_count, 1)
        self.assertLess(grasp.call_args[0][0], 0.001, 'the target must sit below any peg')
        self.assertAlmostEqual(grasp.call_args[1]['epsilon_outer'] + grasp.call_args[0][0], 0.006019)
        self.assertEqual(tool['finger_width_m'], 0.0021)

    def test_an_empty_hand_is_not_accepted(self):
        grasp, tool = self.grasp_run([0.0, 0.0021])
        self.assertEqual(grasp.call_count, 2)
        self.assertEqual(tool['finger_width_m'], 0.0021)

    def driver(self):
        driver = mock.Mock()
        driver.move_to.return_value = (True, 0.0, '')
        driver.check_measured_tcp_frame.return_value = None
        return driver

    def test_each_repeat_is_home_then_the_taught_joints_and_nothing_else(self):
        driver = self.driver()
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(self.run_main(driver, folder), 0)
            self.assertTrue(os.path.exists(os.path.join(folder, PLOT_FILE)))
        calls = driver.move_to.call_args_list
        self.assertEqual(len(calls), 4)
        for call, target in zip(calls, (PANDA_READY, Q, PANDA_READY, Q)):
            np.testing.assert_array_equal(call[0][0], target)
            self.assertEqual(call[1], {}, 'no Cartesian target: nothing may correct the move')

    def test_a_failed_move_stops_the_run_and_keeps_the_files(self):
        driver = self.driver()
        driver.move_to.side_effect = [(True, 0.0, ''), (False, None, 'reflex')]
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(self.run_main(driver, folder), 1)
            with open(os.path.join(folder, RUN_FILE), encoding='utf-8') as handle:
                saved = yaml.safe_load(handle)
        self.assertEqual(len(saved['points']), 1)
        self.assertEqual(saved['points'][0]['repeats'], [])


if __name__ == '__main__':
    unittest.main()
