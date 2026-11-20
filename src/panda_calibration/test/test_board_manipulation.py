import json
import math
import tempfile
import unittest

import numpy as np

from panda_calibration.board_pose import (
    board_target_pose,
    load_board_pose,
    signed_board_clearance,
    tip_position_in_base,
    validate_board_offset,
)
from panda_calibration.board_manipulation import (
    BoardManipulation,
    compensate_pose_for_measured_error,
    pen_tip_report,
    posture_change,
)
from panda_calibration.kinematics import (
    conservative_joint_tolerances,
    make_transform,
    matrix_to_quaternion,
    pose_error,
    rotation_from_axis_angle,
)


class BoardManipulationGeometryTest(unittest.TestCase):
    def test_load_board_pose_and_make_approach(self):
        artifact = {
            'base_frame': 'panda_link0',
            'T_base_board': np.eye(4).tolist(),
            'board_surface_z_m': 0.025,
        }
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as handle:
            json.dump(artifact, handle)
            handle.flush()
            base_frame, transform, surface_z = load_board_pose(handle.name)

        target, approach = board_target_pose(
            transform, surface_z, np.array([0.05, -0.02, 0.0]), 0.15,
            [np.pi, 0.0, 0.0], np.zeros(3))
        self.assertEqual(base_frame, 'panda_link0')
        np.testing.assert_allclose(target[:3, 3], [0.05, -0.02, 0.025])
        np.testing.assert_allclose(approach[:3, 3], [0.05, -0.02, 0.175])
        np.testing.assert_allclose(target[:3, 2], [0.0, 0.0, -1.0], atol=1e-8)

    def test_tip_offset_moves_tcp_above_the_requested_board_point(self):
        tip_offset = np.array([0.0, 0.0, 0.125])
        target, approach = board_target_pose(
            np.eye(4), 0.025, np.zeros(3), 0.15,
            [np.pi, 0.0, 0.0], tip_offset)

        np.testing.assert_allclose(
            tip_position_in_base(target, tip_offset),
            [0.0, 0.0, 0.025])
        self.assertAlmostEqual(
            signed_board_clearance(
                approach, tip_offset, np.array([0.0, 0.0, 0.025]),
                np.array([0.0, 0.0, 1.0])),
            0.15)

    def test_rejects_non_rigid_transform(self):
        artifact = {
            'base_frame': 'panda_link0',
            'T_base_board': [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            'board_surface_z_m': 0.025,
        }
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as handle:
            json.dump(artifact, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, 'orthonormal'):
                load_board_pose(handle.name)

    def test_rejects_an_offset_beyond_the_board_rim(self):
        with self.assertRaisesRegex(ValueError, 'outside the board footprint'):
            validate_board_offset(np.array([0.161, 0.0, 0.0]))

    def test_pose_error_reports_translation_and_geodesic_rotation(self):
        target = np.eye(4)
        actual = make_transform(
            rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.004),
            [0.0005, 0.0, 0.0],
        )

        translation_error, rotation_error = pose_error(target, actual)

        self.assertAlmostEqual(translation_error, 0.0005)
        self.assertAlmostEqual(rotation_error, 0.004)

    def test_pose_error_at_cartesian_gate_boundary_is_not_exceeded(self):
        position_limit = 0.0001
        rotation_limit = math.atan2(0.0001, 0.125)
        target = np.eye(4)
        actual = make_transform(
            rotation_from_axis_angle(np.array([1.0, 0.0, 0.0]), rotation_limit),
            [position_limit, 0.0, 0.0],
        )

        translation_error, rotation_error = pose_error(target, actual)

        self.assertLessEqual(translation_error, position_limit)
        self.assertLessEqual(rotation_error, rotation_limit)

    def test_repeated_corrections_converge_on_a_constant_sag(self):
        desired = make_transform(
            rotation_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi), [0.4, 0.1, 0.3])
        sag = make_transform(
            rotation_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.004), [0.0, 0.0, -0.0012])
        command = desired.copy()
        for _ in range(3):
            actual = sag @ command  # the arm settles short by the same sag every time
            command = compensate_pose_for_measured_error(command, desired, actual)

        translation_error, rotation_error = pose_error(desired, sag @ command)
        self.assertLess(translation_error, 1e-9)
        self.assertLess(rotation_error, 1e-9)

    def test_compensated_rotation_is_orthonormal_for_a_skewed_measurement(self):
        # The Panda's O_T_EE is orthonormal only to ~3e-5; MoveIt's IK fails on
        # the non-unit quaternion a skewed command rotation gives.
        desired = make_transform(
            rotation_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi), [0.4, 0.1, 0.3])
        actual = desired.copy()
        actual[:3, :3] = (rotation_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.004)
                          @ desired[:3, :3] * (1.0 - 1e-5))

        rotation = compensate_pose_for_measured_error(desired, desired, actual)[:3, :3]

        self.assertLess(np.linalg.norm(rotation.T @ rotation - np.eye(3)), 1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(matrix_to_quaternion(rotation))), 1.0, places=12)

    def test_joint_tolerance_is_conservative_for_jacobian_bound(self):
        jacobian = np.vstack((np.eye(3, 3), np.eye(3, 3)))
        tolerance = conservative_joint_tolerances(jacobian, 0.0001, 0.0002)
        self.assertEqual(tolerance.shape, (3,))
        self.assertAlmostEqual(float(tolerance[0]), 0.0001 / 3.0)

    def test_pen_tip_report_in_board_axes(self):
        # A board turned -92 deg and tilted 0.5 deg, and the round peg's off-axis tip.
        board = make_transform(
            rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(-92.0))
            @ rotation_from_axis_angle(np.array([1.0, 0.3, 0.0]), math.radians(0.5)),
            [0.5, 0.1, 0.0])
        offset = np.array([0.0, -0.01649, 0.039])
        point = np.array([-0.127838, -0.03865, 0.0])
        tcp, _ = board_target_pose(board, 0.025, point, 0.02, [math.pi, 0.0, -math.pi / 2], offset)
        target = board @ np.r_[point[:2], 0.025, 1.0]

        on_target = pen_tip_report(tcp, offset, board, target)
        self.assertIn('x +0.00, y +0.00 mm (0.00 mm beside), 0.00 mm above it', on_target)

        # 3 mm along the board normal, and 1 mm along board -x.
        moved = tcp.copy()
        moved[:3, 3] += board[:3, :3] @ np.array([-0.001, 0.0, 0.003])
        report = pen_tip_report(moved, offset, board, target)
        self.assertIn('x -1.00, y +0.00 mm (1.00 mm beside), 3.00 mm above it', report)
        below = tcp.copy()
        below[:3, 3] -= board[:3, 2] * 0.0005
        self.assertIn('0.50 mm below it', pen_tip_report(below, offset, board, target))


class _FakeChain:
    """Stands in for kinematics.Chain: returns the seed, shifted a little."""

    def ik(self, pose, seed, **_):
        return np.asarray(seed, dtype=float) + 0.01, True

    def ik_with_restarts(self, pose, seed=None, **_):
        return np.asarray(seed, dtype=float) + 0.01, True


class _Driver:
    positions = np.zeros(7)


def _node(moveit_answers, seed=None, limit=0.5):
    """A BoardManipulation whose MoveIt IK replays *moveit_answers* (no ROS)."""
    node = BoardManipulation.__new__(BoardManipulation)
    node.driver = _Driver()
    node.config = {'motion': {}}
    node.kinematic_chain = node.local_chain = _FakeChain()
    node.use_moveit_ik = True
    node.posture = None if seed is None else np.asarray(seed, dtype=float)
    node.max_posture_change = limit
    node.seeds = []
    answers = iter(moveit_answers)

    def moveit(pose, frame_id, ik_seed):
        node.seeds.append(np.asarray(ik_seed, dtype=float))
        answer = next(answers)
        return list(ik_seed) if answer is None else list(answer)   # None: echo the seed
    node._moveit_ik = moveit
    return node


PLANNED = np.array([-0.35, 0.24, 0.38, -2.65, -0.36, 2.86, -2.38])
FLIPPED = np.array([2.897, -0.16, -2.83, -2.67, -0.19, 2.82, -2.51])   # logged 2026-11-05


class PlannedPostureTest(unittest.TestCase):
    def test_posture_change_is_the_largest_joint_difference(self):
        self.assertAlmostEqual(posture_change(PLANNED, FLIPPED), 2.897 + 0.35)

    def test_ik_starts_at_the_planned_joints_not_the_current_ones(self):
        node = _node([None], seed=PLANNED)
        joints = node.solve_ik(np.eye(4), 'panda_link0')
        np.testing.assert_allclose(node.seeds[0], PLANNED)
        np.testing.assert_allclose(joints, PLANNED)

    def test_a_flipped_moveit_solution_is_replaced_by_the_nearby_one(self):
        node = _node([FLIPPED, None], seed=PLANNED)
        joints = node.solve_ik(np.eye(4), 'panda_link0')
        np.testing.assert_allclose(joints, PLANNED + 0.01)      # local IK, confirmed by MoveIt
        np.testing.assert_allclose(node.seeds[1], PLANNED + 0.01)

    def test_refuses_when_moveit_rejects_the_nearby_solution_too(self):
        node = _node([FLIPPED, FLIPPED], seed=PLANNED)
        with self.assertRaisesRegex(RuntimeError, 'near the planned posture'):
            node.solve_ik(np.eye(4), 'panda_link0')

    def test_later_solves_follow_the_previous_solution(self):
        node = _node([None, FLIPPED, FLIPPED], seed=PLANNED)
        node.solve_ik(np.eye(4), 'panda_link0')
        with self.assertRaises(RuntimeError):                    # no jump mid-descent
            node.solve_ik(np.eye(4), 'panda_link0')

    def test_without_a_seed_the_first_solve_is_free(self):
        node = _node([FLIPPED])
        joints = node.solve_ik(np.eye(4), 'panda_link0')
        np.testing.assert_allclose(node.seeds[0], np.zeros(7))  # current joints, as before
        np.testing.assert_allclose(joints, FLIPPED)


if __name__ == '__main__':
    unittest.main()
