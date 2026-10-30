"""ROS-independent board-pose artifact validation and target geometry."""

from __future__ import annotations

import json

import numpy as np

from .kinematics import make_transform, rpy_to_matrix

BOARD_HALF_WIDTH_M = 0.160
BOARD_HALF_HEIGHT_M = 0.125


def tip_position_in_base(tcp_pose: np.ndarray,
                         tip_offset_tcp: np.ndarray) -> np.ndarray:
    """Return the physical pen-tip position for a TCP pose."""
    return tcp_pose[:3, 3] + tcp_pose[:3, :3] @ tip_offset_tcp


def signed_board_clearance(tcp_pose: np.ndarray,
                           tip_offset_tcp: np.ndarray,
                           board_point: np.ndarray,
                           board_normal: np.ndarray) -> float:
    """Return pen-tip clearance from the board plane along its +normal."""
    tip_position = tip_position_in_base(tcp_pose, tip_offset_tcp)
    return float(np.dot(board_normal, tip_position - board_point))


def load_board_pose(path: str) -> tuple[str, np.ndarray, float]:
    """Load and validate the canonical robot-base to board transform artifact."""
    with open(path, 'r', encoding='utf-8') as handle:
        artifact = json.load(handle)

    required = ('base_frame', 'T_base_board', 'board_surface_z_m')
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ValueError(f'board pose is missing required keys: {missing}')
    if artifact.get('units') not in (None, 'm'):
        raise ValueError('board pose units must be metres')
    base_frame = artifact['base_frame']
    if not isinstance(base_frame, str) or not base_frame:
        raise ValueError('board pose base_frame must be a non-empty string')
    surface_z = float(artifact['board_surface_z_m'])
    if not np.isfinite(surface_z):
        raise ValueError('board_surface_z_m must be finite')

    transform = np.asarray(artifact['T_base_board'], dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError('T_base_board must be a finite 4x4 matrix')
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError('T_base_board must have homogeneous final row [0, 0, 0, 1]')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError('T_base_board rotation is not orthonormal')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError('T_base_board rotation must have determinant +1')
    return base_frame, transform, surface_z


def validate_board_offset(offset: np.ndarray) -> None:
    """Reject a requested point outside the 320 by 250 mm board footprint."""
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('offset must contain exactly three finite values')
    if abs(offset[0]) > BOARD_HALF_WIDTH_M or abs(offset[1]) > BOARD_HALF_HEIGHT_M:
        raise ValueError(
            'offset lies outside the board footprint: '
            f'x must be within +/-{BOARD_HALF_WIDTH_M:.3f} m and '
            f'y within +/-{BOARD_HALF_HEIGHT_M:.3f} m'
        )


def board_target_pose(transform: np.ndarray, surface_z: float, offset: np.ndarray,
                      approach_height: float, tcp_orientation_board_rpy: list[float],
                      pen_tip_offset_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return TCP target and approach transforms in the robot base frame.

    ``offset`` is expressed in board axes. The pen tip is placed at the target;
    the TCP translation is adjusted by the configured TCP-to-tip vector.
    """
    validate_board_offset(offset)
    rotation = transform[:3, :3] @ rpy_to_matrix(*tcp_orientation_board_rpy)
    point_board = np.array([offset[0], offset[1], surface_z + offset[2], 1.0])
    tip_base = transform @ point_board
    tcp_base = tip_base[:3] - rotation @ pen_tip_offset_tcp
    target = make_transform(rotation, tcp_base)
    approach = target.copy()
    approach[:3, 3] += transform[:3, 2] * approach_height
    return target, approach
