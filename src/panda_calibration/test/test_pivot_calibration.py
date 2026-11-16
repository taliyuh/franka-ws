import math
import unittest

import numpy as np

from panda_calibration.kinematics import rotation_from_axis_angle
from panda_calibration.pivot_calibration import solve_pivot

TIP = np.array([0.0, 0.0165, 0.039])      # round peg: 16.5 mm off axis, 39 mm below the TCP
PIVOT = np.array([0.45, 0.12, 0.025])
DOWN = rotation_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi)   # TCP z pointing down


def poses(count: int, tilt_deg: float, seed: int = 0):
    """TCP poses that all put TIP on PIVOT, tilted up to *tilt_deg* in random directions."""
    rng = np.random.default_rng(seed)
    rotations, translations = [], []
    for _ in range(count):
        heading = rng.uniform(0.0, 2.0 * math.pi)
        tilt = math.radians(tilt_deg) * rng.uniform(0.3, 1.0)
        axis = np.array([math.cos(heading), math.sin(heading), 0.0])
        spin = rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), rng.uniform(-math.pi, math.pi))
        rotation = rotation_from_axis_angle(axis, tilt) @ DOWN @ spin
        rotations.append(rotation)
        translations.append(PIVOT - rotation @ TIP)
    return np.array(rotations), np.array(translations)


class PivotCalibrationTest(unittest.TestCase):
    def test_recovers_tip_and_pivot_exactly_without_noise(self):
        rotations, translations = poses(8, 45.0)
        fit = solve_pivot(rotations, translations)
        np.testing.assert_allclose(fit['tip'], TIP, atol=1e-9)
        np.testing.assert_allclose(fit['pivot'], PIVOT, atol=1e-9)
        self.assertLess(fit['rms'], 1e-9)

    def test_noisy_poses_stay_within_the_reported_uncertainty(self):
        rotations, translations = poses(20, 45.0, seed=1)
        noise = np.random.default_rng(2).normal(0.0, 0.0002, translations.shape)  # 0.2 mm
        fit = solve_pivot(rotations, translations + noise)
        self.assertLess(np.max(np.abs(fit['tip'] - TIP)), 5 * np.max(fit['tip_std']))
        self.assertLess(np.max(np.abs(fit['tip'] - TIP)), 0.0005)
        self.assertEqual(len(fit['loo']), 20)

    def test_turning_only_about_one_axis_is_rejected(self):
        spins = [DOWN @ rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), a)
                 for a in np.linspace(0.0, 2.0, 6)]
        with self.assertRaisesRegex(ValueError, 'too similar'):
            solve_pivot(spins, [PIVOT - r @ TIP for r in spins])

    def test_needs_four_poses(self):
        rotations, translations = poses(3, 45.0)
        with self.assertRaisesRegex(ValueError, 'at least 4'):
            solve_pivot(rotations, translations)


if __name__ == '__main__':
    unittest.main()
