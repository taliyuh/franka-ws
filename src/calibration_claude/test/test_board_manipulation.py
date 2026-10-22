import json
import math
import tempfile
import unittest

import numpy as np

from calibration_claude.board_pose import (
    board_target_pose,
    load_board_pose,
    signed_board_clearance,
    tip_position_in_base,
    validate_board_offset,
)
from calibration_claude.kinematics import (
    conservative_joint_tolerances,
    make_transform,
    pose_error,
    rotation_from_axis_angle,
)


class BoardManipulationGeometryTest(unittest.TestCase):
    def test_load_board_pose_and_make_approach(self):
        artifact = {
            'base_frame': 'fr3_link0',
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
        self.assertEqual(base_frame, 'fr3_link0')
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
            'base_frame': 'fr3_link0',
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

    def test_joint_tolerance_is_conservative_for_jacobian_bound(self):
        jacobian = np.vstack((np.eye(3, 3), np.eye(3, 3)))
        tolerance = conservative_joint_tolerances(jacobian, 0.0001, 0.0002)
        self.assertEqual(tolerance.shape, (3,))
        self.assertAlmostEqual(float(tolerance[0]), 0.0001 / 3.0)


if __name__ == '__main__':
    unittest.main()
