import unittest

import cv2
import numpy as np

try:
    from calibration_claude.cameras import tag_corners
    from calibration_claude.kinematics import make_transform, rotation_from_axis_angle
    from calibration_claude.solve_handeye import (
        _leave_one_pose_out_validation,
        invert,
        joint_handeye_fit,
    )
except ImportError as error:  # pragma: no cover - depends on the ROS image runtime
    RUNTIME_ERROR = str(error)
    RUNTIME_AVAILABLE = False
else:
    RUNTIME_ERROR = ''
    RUNTIME_AVAILABLE = True


@unittest.skipUnless(RUNTIME_AVAILABLE, f'joint solver unavailable: {RUNTIME_ERROR}')
class JointHandeyeFitTest(unittest.TestCase):
    def test_shared_mount_fit_is_deterministic_and_recovers_both_cameras(self):
        tag_mount = make_transform(
            rotation_from_axis_angle(np.array([0.2, -0.4, 0.7]), 0.3),
            [0.12, -0.01, -0.05])
        camera_transforms = {
            'alpha': make_transform(
                rotation_from_axis_angle(np.array([0.1, 0.4, 0.2]), 0.7),
                [0.8, -0.2, 0.9]),
            'beta': make_transform(
                rotation_from_axis_angle(np.array([-0.3, 0.2, 0.5]), 1.1),
                [-0.4, 0.6, 1.2]),
        }
        samples = {camera: [] for camera in camera_transforms}
        camera_matrix = np.array([[700.0, 0.0, 320.0],
                                  [0.0, 700.0, 240.0],
                                  [0.0, 0.0, 1.0]])
        distortion = np.zeros((5, 1))
        for index in range(6):
            tcp = make_transform(
                rotation_from_axis_angle(
                    np.array([0.2 + 0.1 * index, 0.6, -0.3]),
                    0.2 + 0.11 * index),
                [0.3 + 0.02 * index, -0.25 + 0.03 * index,
                 0.55 + 0.01 * index])
            for camera, transform in camera_transforms.items():
                camera_tag = invert(transform) @ tcp @ tag_mount
                rotation_vector, _ = cv2.Rodrigues(camera_tag[:3, :3])
                corners_px, _ = cv2.projectPoints(
                    tag_corners(0.106), rotation_vector,
                    camera_tag[:3, 3].reshape(3, 1), camera_matrix, distortion)
                samples[camera].append({
                    'index': index,
                    'base_tcp': tcp,
                    'camera_tag': camera_tag,
                    'reprojection_rms_px': 0.0,
                    'corners_px': corners_px.reshape(4, 2),
                    'camera_matrix': camera_matrix,
                    'distortion': distortion,
                })

        first = joint_handeye_fit(samples, 0.106, tag_mount)
        second = joint_handeye_fit(samples, 0.106, tag_mount)

        for camera, expected in camera_transforms.items():
            np.testing.assert_allclose(
                first['camera_transforms'][camera], expected, atol=1e-8)
            np.testing.assert_allclose(
                second['camera_transforms'][camera], first['camera_transforms'][camera],
                atol=1e-12)
        np.testing.assert_allclose(first['tag_mount'], tag_mount, atol=1e-8)
        self.assertEqual(first['quality']['observation_model'],
                         'direct_distorted_image_corners')
        self.assertLess(first['quality']['direct_corner_rms_px'], 1e-5)
        self.assertEqual(first['quality']['samples_rejected'], 0)

        validation = _leave_one_pose_out_validation(
            samples, 0.106, tag_mount, first,
            attempted_indices={camera: list(range(6)) for camera in samples},
            undetected_indices={camera: [] for camera in samples},
        )
        self.assertEqual(validation['status'], 'completed')
        self.assertEqual(validation['samples_evaluated'], 12)
        self.assertEqual(len(validation['failures']), 0)


if __name__ == '__main__':
    unittest.main()
