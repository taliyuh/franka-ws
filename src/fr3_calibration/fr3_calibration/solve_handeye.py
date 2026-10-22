"""Solve a fixed-camera AprilTag calibration dataset.

The dataset records ``T_base_tcp`` while the AprilTag is rigidly mounted to
the TCP.  OpenCV's hand-eye API is formulated for an eye-in-hand camera, so we
use the inverse measured ``T_base_tag`` as ``gripper2base``.  With
``T_camera_tag`` from PnP, its returned camera-to-gripper transform is then
``T_base_camera`` (the eye-to-hand substitution described in the README).
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

try:  # Supports both ``python -m`` and direct execution during diagnosis.
    from . import tag as tag_module
    from .cameras import tag_corners
    from .context import build_chain, load_config
    from .kinematics import Chain, axis_angle, make_transform, transform_from_xyz_rpy
except ImportError:  # pragma: no cover - direct script invocation only
    from fr3_calibration import tag as tag_module
    from fr3_calibration.cameras import tag_corners
    from fr3_calibration.context import build_chain, load_config
    from fr3_calibration.kinematics import Chain, axis_angle, make_transform, transform_from_xyz_rpy


MIN_SAMPLES = 4
MAD_TO_GAUSSIAN_SIGMA = 1.482602218505602
JOINT_OUTLIER_SIGMA = 3.0


def invert(transform: np.ndarray) -> np.ndarray:
    """Return the inverse of a rigid homogeneous transform."""
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the angle between two rotation matrices in degrees."""
    cosine = (np.trace(first.T @ second) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def matrix_from_record(record: dict[str, Any], label: str) -> np.ndarray:
    """Read and validate a 4x4 rigid transform saved by capture_node."""
    matrix = np.asarray(record.get('matrix'), dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'{label} must contain a finite 4x4 matrix')
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f'{label} has an invalid homogeneous bottom row')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError(f'{label} rotation is not orthonormal')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise ValueError(f'{label} rotation determinant is not +1')
    return matrix


def solve_camera_tag_pose(
    corners_px: Any, camera_info: dict[str, Any], tag_size_m: float,
) -> tuple[np.ndarray, float]:
    """Estimate ``T_camera_tag`` with IPPE_SQUARE and return RMS pixel error."""
    image_points = np.asarray(corners_px, dtype=np.float64)
    if image_points.shape != (4, 2) or not np.isfinite(image_points).all():
        raise ValueError('detected tag corners must be four finite [x, y] pairs')
    camera_matrix = np.asarray(camera_info.get('K'), dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera_info.get('D', []), dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(camera_matrix).all() or camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError('camera K must be a valid finite pinhole matrix')
    success, rotation_vector, translation = cv2.solvePnP(
        tag_corners(tag_size_m).astype(np.float64), image_points, camera_matrix,
        distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success or translation[2, 0] <= 0:
        raise ValueError('IPPE_SQUARE did not return a tag pose in front of the camera')
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    projected, _ = cv2.projectPoints(
        tag_corners(tag_size_m).astype(np.float64), rotation_vector, translation,
        camera_matrix, distortion)
    rms = float(np.sqrt(np.mean((projected.reshape(4, 2) - image_points) ** 2)))
    return transform, rms


def solve_handeye(
    base_tag_poses: list[np.ndarray], camera_tag_poses: list[np.ndarray], method: int,
) -> np.ndarray:
    """Solve ``T_base_camera`` using OpenCV's eye-to-hand substitution."""
    if len(base_tag_poses) < MIN_SAMPLES:
        raise ValueError(f'at least {MIN_SAMPLES} detections are required')
    tag_base_poses = [invert(pose) for pose in base_tag_poses]
    rotation, translation = cv2.calibrateHandEye(
        [pose[:3, :3] for pose in tag_base_poses],
        [pose[:3, 3].reshape(3, 1) for pose in tag_base_poses],
        [pose[:3, :3] for pose in camera_tag_poses],
        [pose[:3, 3].reshape(3, 1) for pose in camera_tag_poses],
        method=method)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation).reshape(3)
    return transform


def _inlier_mask(base_tag: list[np.ndarray], camera_tag: list[np.ndarray], estimate: np.ndarray) -> np.ndarray:
    """Reject views inconsistent with a provisional fixed camera estimate."""
    per_view = [base @ invert(camera) for base, camera in zip(base_tag, camera_tag)]
    translation_errors = np.array([np.linalg.norm(view[:3, 3] - estimate[:3, 3]) for view in per_view])
    rotation_errors = np.array([rotation_error_deg(view[:3, :3], estimate[:3, :3]) for view in per_view])
    # Median scaling resists a few bad corner detections while retaining normal noise.
    translation_limit = max(0.010, 3.0 * float(np.median(translation_errors)))
    rotation_limit = max(1.0, 3.0 * float(np.median(rotation_errors)))
    return (translation_errors <= translation_limit) & (rotation_errors <= rotation_limit)


def solve_robust(base_tag: list[np.ndarray], camera_tag: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PARK estimate, ANDREFF estimate, and a robust inlier mask."""
    park = solve_handeye(base_tag, camera_tag, cv2.CALIB_HAND_EYE_PARK)
    mask = _inlier_mask(base_tag, camera_tag, park)
    if int(mask.sum()) < MIN_SAMPLES:
        raise ValueError('outlier filtering left fewer than four usable detections')
    selected_base = [pose for pose, keep in zip(base_tag, mask) if keep]
    selected_camera = [pose for pose, keep in zip(camera_tag, mask) if keep]
    park = solve_handeye(selected_base, selected_camera, cv2.CALIB_HAND_EYE_PARK)
    andreff = solve_handeye(selected_base, selected_camera, cv2.CALIB_HAND_EYE_ANDREFF)
    return park, andreff, mask


def load_camera_samples(dataset: Path, camera_name: str) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray], list[float], list[int]]:
    """Load matched measured robot/tag poses for one camera from a dataset."""
    def read_yaml(name: str) -> dict[str, Any]:
        with (dataset / name).open(encoding='utf-8') as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f'{name} must contain a mapping')
        return loaded

    meta, poses_file, detections_file = (read_yaml(name) for name in ('meta.yaml', 'poses.yaml', 'detections.yaml'))
    camera_info = meta.get('cameras', {}).get(camera_name)
    if not camera_info:
        raise ValueError(f'camera {camera_name!r} is absent from meta.yaml')
    tag = meta.get('tag', {})
    try:
        tag_size = float(tag['size'])
        tcp_tag = transform_from_xyz_rpy(tag['mount_xyz'], tag['mount_rpy'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('meta.yaml tag must contain size, mount_xyz, and mount_rpy') from error
    pose_by_index = {entry['index']: entry for entry in poses_file.get('poses', [])}
    base_tag, camera_tag, reprojection, indices = [], [], [], []
    for detection_pose in detections_file.get('poses', []):
        index = detection_pose.get('index')
        detection = detection_pose.get('cameras', {}).get(camera_name, {})
        if not detection.get('detected'):
            continue
        if index not in pose_by_index:
            continue
        try:
            base_tcp = matrix_from_record(pose_by_index[index]['T_base_tcp'], f'pose {index} T_base_tcp')
            camera_tag_pose, rms = solve_camera_tag_pose(detection['corners_px'], camera_info, tag_size)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'pose {index}, camera {camera_name}: {error}') from error
        base_tag.append(base_tcp @ tcp_tag)
        camera_tag.append(camera_tag_pose)
        reprojection.append(rms)
        indices.append(index)
    return meta, base_tag, camera_tag, reprojection, indices


def load_taught_camera_samples(dataset: Path, camera_name: str,
                               config_path: str | None) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray], list[float], list[int]]:
    """Load a ``run_taught_capture`` dataset by detecting its saved tag images.

    ``poses.txt`` contains Cartesian TCP targets, not the generated joint-pose
    format. Taught capture records measured joints and live intrinsics after
    every target, which are sufficient to reconstruct the same hand-eye inputs.
    """
    with (dataset / 'poses.yaml').open(encoding='utf-8') as handle:
        taught = yaml.safe_load(handle)
    if not isinstance(taught, dict):
        raise ValueError('poses.yaml must contain a mapping')

    config = load_config(config_path)
    chain = build_chain(config)
    tag = config['tag']
    tag_size = float(tag['size'])
    tcp_tag = transform_from_xyz_rpy(tag['mount_xyz'], tag['mount_rpy'])
    base_tag, camera_tag, reprojection, indices = [], [], [], []
    camera_info = None

    for pose in taught.get('poses', []):
        camera = pose.get('cameras', {}).get(camera_name)
        image_path = pose.get('images', {}).get(camera_name)
        if camera is None or not image_path:
            continue
        image = cv2.imread(str(dataset / image_path))
        if image is None:
            raise ValueError(f'could not read captured image {dataset / image_path}')
        detection = tag_module.detect(image, int(tag['id']), str(tag['family']))
        if not detection.detected or detection.corners is None:
            continue
        try:
            measured_joints = np.asarray(pose['measured_joints'], dtype=float)
            if measured_joints.shape != (len(chain.joint_names),):
                raise ValueError(
                    f'expected {len(chain.joint_names)} measured joints, got '
                    f'{measured_joints.shape}'
                )
            camera_tag_pose, rms = solve_camera_tag_pose(
                detection.corners, camera, tag_size)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'taught pose {pose.get("index", "?")}: {error}') from error
        base_tag.append(chain.fk(measured_joints) @ tcp_tag)
        camera_tag.append(camera_tag_pose)
        reprojection.append(rms)
        indices.append(int(pose['index']))
        camera_info = camera

    if camera_info is None:
        raise ValueError(f'{camera_name!r} has no detected calibration tag in taught-capture images')
    return {
        'base_link': config['robot']['base_link'],
        'cameras': {camera_name: camera_info},
    }, base_tag, camera_tag, reprojection, indices


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    """Read a YAML mapping with an error that names the offending file."""
    try:
        with path.open(encoding='utf-8') as handle:
            loaded = yaml.safe_load(handle)
    except OSError as error:
        raise ValueError(f'could not read {path}: {error}') from error
    if not isinstance(loaded, dict):
        raise ValueError(f'{label} must contain a mapping')
    return loaded


def _joint_sample(index: int, base_tcp: np.ndarray, camera_tag: np.ndarray,
                  reprojection_rms_px: float, corners_px: Any,
                  camera_info: dict[str, Any]) -> dict[str, Any]:
    """Create one normalized observation for the joint hand-eye fit."""
    camera_matrix = np.asarray(camera_info.get('K'), dtype=float).reshape(3, 3)
    distortion = np.asarray(camera_info.get('D', []), dtype=float).reshape(-1, 1)
    corners = np.asarray(corners_px, dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError('detected tag corners must be four finite [x, y] pairs')
    if not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
        raise ValueError('camera intrinsics must be finite')
    return {
        'index': int(index),
        'base_tcp': np.asarray(base_tcp, dtype=float),
        'camera_tag': np.asarray(camera_tag, dtype=float),
        'reprojection_rms_px': float(reprojection_rms_px),
        'corners_px': corners,
        'camera_matrix': camera_matrix,
        'distortion': distortion,
    }


def _load_joint_samples_from_capture(dataset: Path) -> dict[str, Any]:
    """Load robot poses, PnP tag poses, and camera metadata from a capture."""
    meta = _read_yaml_mapping(dataset / 'meta.yaml', 'meta.yaml')
    poses_file = _read_yaml_mapping(dataset / 'poses.yaml', 'poses.yaml')
    detections_file = _read_yaml_mapping(dataset / 'detections.yaml', 'detections.yaml')
    tag = meta.get('tag', {})
    try:
        tag_size = float(tag['size'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('meta.yaml tag must contain a numeric size') from error

    camera_info = meta.get('cameras')
    if not isinstance(camera_info, dict) or not camera_info:
        raise ValueError('meta.yaml must contain at least one camera')
    pose_by_index = {entry['index']: entry for entry in poses_file.get('poses', [])}
    samples = {camera: [] for camera in sorted(camera_info)}
    attempted_indices = {camera: [] for camera in sorted(camera_info)}
    undetected_indices = {camera: [] for camera in sorted(camera_info)}
    for detection_pose in sorted(detections_file.get('poses', []),
                                 key=lambda entry: int(entry['index'])):
        index = detection_pose.get('index')
        pose = pose_by_index.get(index)
        if pose is None:
            continue
        try:
            base_tcp = matrix_from_record(pose['T_base_tcp'],
                                          f'pose {index} T_base_tcp')
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'pose {index}: {error}') from error
        for camera in sorted(samples):
            detection = detection_pose.get('cameras', {}).get(camera, {})
            if camera in detection_pose.get('cameras', {}):
                attempted_indices[camera].append(index)
            if not detection.get('detected'):
                if camera in detection_pose.get('cameras', {}):
                    undetected_indices[camera].append(index)
                continue
            try:
                camera_tag, rms = solve_camera_tag_pose(
                    detection['corners_px'], camera_info[camera], tag_size)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'pose {index}, camera {camera}: {error}') from error
            samples[camera].append(_joint_sample(
                index, base_tcp, camera_tag, rms, detection['corners_px'],
                camera_info[camera]))

    return {
        'base_frame': meta.get('base_link', 'fr3_link0'),
        'tag_size': tag_size,
        'tag_mount': None,
        'camera_info': camera_info,
        'samples': samples,
        'attempted_indices': attempted_indices,
        'undetected_indices': undetected_indices,
        'dataset_kind': 'generated_calibration_capture',
    }


def _chain_for_joint_capture(config: dict[str, Any], urdf_path: str | None) -> Chain:
    """Build the exact runtime chain, with an explicit offline URDF escape hatch."""
    if urdf_path is None:
        return build_chain(config)
    xml = Path(urdf_path).read_text(encoding='utf-8')
    links = {element.get('name') for element in ET.fromstring(xml).findall('link')}
    robot = config['robot']
    base_link = robot.get('kinematic_base_link', robot['base_link'])
    if base_link not in links:
        base_link = robot['base_link']
    return Chain.from_urdf(xml, base_link, robot['tip_link'])


def _load_joint_samples_from_taught(dataset: Path, config_path: str | None,
                                    urdf_path: str | None) -> dict[str, Any]:
    """Load a taught capture while retaining the measured TCP pose inputs."""
    if config_path is None:
        raise ValueError('--config is required for a taught-capture dataset')
    config = load_config(config_path)
    chain = _chain_for_joint_capture(config, urdf_path)
    taught = _read_yaml_mapping(dataset / 'poses.yaml', 'poses.yaml')
    tag = config['tag']
    tag_size = float(tag['size'])
    camera_info: dict[str, Any] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    attempted_indices: dict[str, list[int]] = {}
    undetected_indices: dict[str, list[int]] = {}
    for pose in sorted(taught.get('poses', []), key=lambda entry: int(entry['index'])):
        try:
            measured_joints = np.asarray(pose['measured_joints'], dtype=float)
            if measured_joints.shape != (len(chain.joint_names),):
                raise ValueError(
                    f'expected {len(chain.joint_names)} measured joints, got '
                    f'{measured_joints.shape}')
            if 'T_base_tcp' in pose:
                base_tcp = matrix_from_record(
                    pose['T_base_tcp'],
                    f'pose {pose.get("index", "?")} T_base_tcp')
            else:
                base_tcp = chain.fk(measured_joints)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'taught pose {pose.get("index", "?")}: {error}') from error

        for camera in sorted(pose.get('cameras', {})):
            camera_entry = pose['cameras'][camera]
            image_path = pose.get('images', {}).get(camera)
            if not image_path:
                continue
            attempted_indices.setdefault(camera, []).append(int(pose['index']))
            image = cv2.imread(str(dataset / image_path))
            if image is None:
                raise ValueError(f'could not read captured image {dataset / image_path}')
            detection = tag_module.detect(
                image, int(tag['id']), str(tag['family']))
            if not detection.detected or detection.corners is None:
                undetected_indices.setdefault(camera, []).append(int(pose['index']))
                continue
            try:
                camera_tag, rms = solve_camera_tag_pose(
                    detection.corners, camera_entry, tag_size)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'taught pose {pose.get("index", "?")}, '
                                 f'camera {camera}: {error}') from error
            camera_info.setdefault(camera, camera_entry)
            samples.setdefault(camera, []).append(_joint_sample(
                pose['index'], base_tcp, camera_tag, rms, detection.corners,
                camera_entry))

    if not samples:
        raise ValueError('taught capture contains no detected camera samples')
    return {
        'base_frame': config['robot']['base_link'],
        'tag_size': tag_size,
        'tag_mount': transform_from_xyz_rpy(tag['mount_xyz'], tag['mount_rpy']),
        'camera_info': camera_info,
        'samples': {camera: samples[camera] for camera in sorted(samples)},
        'attempted_indices': {
            camera: attempted_indices.get(camera, []) for camera in sorted(attempted_indices)
        },
        'undetected_indices': {
            camera: undetected_indices.get(camera, [])
            for camera in sorted(undetected_indices)
        },
        'dataset_kind': 'taught_capture',
    }


def load_joint_camera_samples(dataset: Path, config_path: str | None = None,
                              urdf_path: str | None = None) -> dict[str, Any]:
    """Load all camera streams for one deterministic joint calibration."""
    if (dataset / 'meta.yaml').exists() and (dataset / 'detections.yaml').exists():
        loaded = _load_joint_samples_from_capture(dataset)
        meta = _read_yaml_mapping(dataset / 'meta.yaml', 'meta.yaml')
        tag = meta.get('tag', {})
        if {'mount_xyz', 'mount_rpy'} <= set(tag):
            loaded['tag_mount'] = transform_from_xyz_rpy(
                tag['mount_xyz'], tag['mount_rpy'])
        return loaded
    return _load_joint_samples_from_taught(dataset, config_path, urdf_path)


def _pose_vector(transform: np.ndarray) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(np.asarray(transform[:3, :3], dtype=float))
    return np.r_[rotation_vector.reshape(3), np.asarray(transform[:3, 3], dtype=float)]


def _pose_matrix(vector: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.asarray(vector[:3], dtype=float))[0]
    transform[:3, 3] = np.asarray(vector[3:6], dtype=float)
    return transform


def _average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    """Average rotations by SVD and translations by componentwise median."""
    if not transforms:
        raise ValueError('cannot average an empty transform list')
    rotation_sum = sum(transform[:3, :3] for transform in transforms)
    left, _, right = np.linalg.svd(rotation_sum)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return make_transform(rotation,
                          np.median([transform[:3, 3] for transform in transforms], axis=0))


def _joint_corner_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform tag-frame points into the destination frame."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def _has_direct_corner_observation(sample: dict[str, Any]) -> bool:
    """Return whether a sample contains the raw image-corner observation."""
    return all(key in sample for key in ('corners_px', 'camera_matrix', 'distortion'))


def _project_tag_corners(camera_tag: np.ndarray, sample: dict[str, Any],
                         points: np.ndarray) -> np.ndarray:
    """Project tag-frame points into one sample's distorted image."""
    rotation_vector, _ = cv2.Rodrigues(camera_tag[:3, :3])
    projected, _ = cv2.projectPoints(
        points.astype(np.float64), rotation_vector,
        camera_tag[:3, 3].reshape(3, 1), sample['camera_matrix'],
        sample['distortion'])
    return projected.reshape(-1, 2)


def _direct_corner_residual(sample: dict[str, Any], base_camera: np.ndarray,
                            tag_mount: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return predicted-minus-observed image-corner residuals in pixels."""
    base_tag = sample['base_tcp'] @ tag_mount
    camera_tag = invert(base_camera) @ base_tag
    projected = _project_tag_corners(camera_tag, sample, points)
    return projected - sample['corners_px']


def _joint_camera_seed(samples: list[dict[str, Any]], tag_mount: np.ndarray,
                       tag_size: float, minimum_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic camera seed and data-derived inlier mask.

    The legacy solver uses fixed metric thresholds for its outlier mask.  That
    is inappropriate here because the scale of the residual changes with the
    camera geometry and robot motion.  This seed instead clips only samples
    beyond three robust standard deviations from the per-camera residual
    median, with the scale estimated from the samples' MAD.  The residual is
    measured in image pixels when raw corners are available and otherwise in
    3-D tag-corner metres for the compatibility path.  The final joint
    optimizer still uses a robust loss, so this mask is only for initialization
    and diagnostics.
    """
    points = tag_corners(tag_size)
    base_tag = [sample['base_tcp'] @ tag_mount for sample in samples]
    camera_tag = [sample['camera_tag'] for sample in samples]
    mask = np.ones(len(samples), dtype=bool)
    estimate = None
    for _ in range(len(samples)):
        selected_base = [pose for pose, keep in zip(base_tag, mask) if keep]
        selected_camera = [pose for pose, keep in zip(camera_tag, mask) if keep]
        try:
            estimate = solve_handeye(
                selected_base, selected_camera, cv2.CALIB_HAND_EYE_PARK)
        except (cv2.error, ValueError, TypeError):
            fallback = _average_transforms([
                robot @ invert(observed)
                for robot, observed in zip(selected_base, selected_camera)])
            return fallback, mask

        errors = []
        use_direct = all(_has_direct_corner_observation(sample) for sample in samples)
        for sample, robot, observed in zip(samples, base_tag, camera_tag):
            if use_direct:
                residual = _direct_corner_residual(
                    sample, estimate, tag_mount, points)
                errors.append(float(np.sqrt(np.mean(residual ** 2))))
                continue
            robot_points = _joint_corner_points(robot, points)
            camera_points = _joint_corner_points(estimate @ observed, points)
            errors.append(float(np.sqrt(np.mean(
                np.linalg.norm(robot_points - camera_points, axis=1) ** 2))))
        errors = np.asarray(errors)
        inlier_errors = errors[mask]
        median = float(np.median(inlier_errors))
        mad = float(np.median(np.abs(inlier_errors - median)))
        robust_sigma = MAD_TO_GAUSSIAN_SIGMA * mad
        cutoff = median + JOINT_OUTLIER_SIGMA * robust_sigma
        if robust_sigma <= np.finfo(float).eps:
            new_mask = errors <= median
        else:
            new_mask = errors <= cutoff
        if int(new_mask.sum()) < minimum_samples or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    if estimate is None:  # guarded by the minimum sample check in the caller
        raise ValueError('could not initialize a joint camera calibration')
    return estimate, mask


def joint_handeye_fit(samples_by_camera: dict[str, list[dict[str, Any]]],
                      tag_size: float, tag_mount: np.ndarray,
                      minimum_samples: int = MIN_SAMPLES) -> dict[str, Any]:
    """Fit all camera extrinsics and one shared TCP-to-tag transform together.

    When samples contain raw image corners and camera intrinsics, the residual
    is the direct distorted pixel-corner error.  The saved PnP pose is then
    used only to initialize the optimizer.  Samples without raw corners use
    the older 3-D PnP residual as a compatibility path for existing tests and
    datasets.
    """
    camera_names = sorted(samples_by_camera)
    if not camera_names:
        raise ValueError('at least one camera is required')
    for camera in camera_names:
        if len(samples_by_camera[camera]) < minimum_samples:
            raise ValueError(f'{camera} has {len(samples_by_camera[camera])} valid '
                             f'samples; need at least {minimum_samples}')
    tag_mount = np.asarray(tag_mount, dtype=float)
    if tag_mount.shape != (4, 4):
        raise ValueError('tag_mount must be a 4x4 transform')
    if not np.isfinite(tag_mount).all():
        raise ValueError('tag_mount must contain finite values')
    if not np.allclose(tag_mount[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError('tag_mount has an invalid homogeneous bottom row')
    if not (np.allclose(tag_mount[:3, :3].T @ tag_mount[:3, :3], np.eye(3), atol=2e-4)
            and np.isclose(np.linalg.det(tag_mount[:3, :3]), 1.0, atol=2e-4)):
        raise ValueError('tag_mount rotation is not a proper rotation')
    if not (np.isfinite(tag_size) and tag_size > 0.0):
        raise ValueError('tag_size must be finite and positive')

    points = tag_corners(tag_size)
    all_samples = {
        camera: sorted(samples_by_camera[camera], key=lambda sample: sample['index'])
        for camera in camera_names
    }
    direct_modes = {
        _has_direct_corner_observation(sample)
        for samples in all_samples.values() for sample in samples
    }
    if len(direct_modes) != 1:
        raise ValueError('joint calibration samples must consistently contain '
                         'raw corners and camera intrinsics')
    use_direct_corners = direct_modes.pop()
    fit_samples = {}
    inlier_indices = {}
    camera_initial = {}
    for camera in camera_names:
        samples = all_samples[camera]
        camera_initial[camera], mask = _joint_camera_seed(
            samples, tag_mount, tag_size, minimum_samples)
        fit_samples[camera] = [sample for sample, keep in zip(samples, mask) if keep]
        inlier_indices[camera] = {sample['index'] for sample, keep in zip(samples, mask)
                                  if keep}
        if len(fit_samples[camera]) < minimum_samples:
            raise ValueError(f'{camera} has only {len(fit_samples[camera])} usable '
                             'samples after deterministic outlier filtering')

    initial_vector = np.concatenate(
        [_pose_vector(camera_initial[camera]) for camera in camera_names]
        + [_pose_vector(tag_mount)])
    camera_parameter_count = 6 * len(camera_names)

    def unpack(vector: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        cameras = {
            camera: _pose_matrix(vector[6 * index:6 * (index + 1)])
            for index, camera in enumerate(camera_names)
        }
        mount = _pose_matrix(vector[camera_parameter_count:camera_parameter_count + 6])
        return cameras, mount

    def unscaled_residual(vector: np.ndarray) -> np.ndarray:
        cameras, mount = unpack(vector)
        blocks = []
        for camera in camera_names:
            camera_blocks = []
            for sample in fit_samples[camera]:
                if use_direct_corners:
                    camera_blocks.append(_direct_corner_residual(
                        sample, cameras[camera], mount, points).ravel())
                    continue
                robot_points = _joint_corner_points(sample['base_tcp'] @ mount, points)
                camera_points = _joint_corner_points(
                    cameras[camera] @ sample['camera_tag'], points)
                camera_blocks.append((robot_points - camera_points).ravel())
            # Equalize camera influence when one detector has fewer valid views.
            blocks.append(np.concatenate(camera_blocks) /
                          math.sqrt(len(fit_samples[camera])))
        return np.concatenate(blocks)

    initial_residual = unscaled_residual(initial_vector)
    residual_scale = float(np.median(np.abs(initial_residual)))
    if not np.isfinite(residual_scale) or residual_scale <= np.finfo(float).eps:
        residual_scale = 1.0

    def residual(vector: np.ndarray) -> np.ndarray:
        return unscaled_residual(vector) / residual_scale

    solution = least_squares(
        residual, initial_vector, loss='soft_l1', f_scale=1.0,
        x_scale='jac', max_nfev=1000)
    if not solution.success or not np.isfinite(solution.x).all():
        raise ValueError(f'joint hand-eye optimizer failed: {solution.message}')

    camera_transforms, fitted_mount = unpack(solution.x)
    per_camera = {}
    used_corner_errors = []
    all_corner_errors = []
    used_pixel_errors = []
    all_pixel_errors = []
    for camera in camera_names:
        sample_errors = []
        for sample in all_samples[camera]:
            used = sample['index'] in inlier_indices[camera]
            if use_direct_corners:
                pixel_error = _direct_corner_residual(
                    sample, camera_transforms[camera], fitted_mount, points)
                pixel_distances = np.linalg.norm(pixel_error, axis=1)
                all_pixel_errors.extend(pixel_error.ravel())
                if used:
                    used_pixel_errors.extend(pixel_error.ravel())
                sample_errors.append({
                    'index': sample['index'],
                    'used': used,
                    'direct_corner_rms_px': float(np.sqrt(np.mean(pixel_distances ** 2))),
                    'direct_corner_max_px': float(np.max(pixel_distances)),
                    'pnp_reprojection_rms_px': sample['reprojection_rms_px'],
                })
                continue
            robot_points = _joint_corner_points(sample['base_tcp'] @ fitted_mount, points)
            camera_points = _joint_corner_points(
                camera_transforms[camera] @ sample['camera_tag'], points)
            corner_error = robot_points - camera_points
            distances = np.linalg.norm(corner_error, axis=1)
            sample_errors.append({
                'index': sample['index'],
                'used': used,
                'corner_rms_mm': float(np.sqrt(np.mean(distances ** 2)) * 1000.0),
                'corner_max_mm': float(np.max(distances) * 1000.0),
                'reprojection_rms_px': sample['reprojection_rms_px'],
            })
            all_corner_errors.extend(corner_error.ravel())
            if used:
                used_corner_errors.extend(corner_error.ravel())
        values = np.asarray([row.get(
            'direct_corner_rms_px' if use_direct_corners else 'corner_rms_mm',
            0.0) for row in sample_errors])
        used_values = np.asarray([row[
            'direct_corner_rms_px' if use_direct_corners else 'corner_rms_mm']
            for row in sample_errors if row['used']])
        pnp_values = [row['pnp_reprojection_rms_px'] if use_direct_corners
                      else row['reprojection_rms_px'] for row in sample_errors]
        per_camera_quality = {
            'samples_total': len(sample_errors),
            'samples_used': int(len(used_values)),
            'samples_rejected': int(len(sample_errors) - len(used_values)),
            'mean_reprojection_rms_px': float(np.mean(pnp_values)),
            'max_reprojection_rms_px': float(np.max(pnp_values)),
            'samples': sample_errors,
        }
        if use_direct_corners:
            per_camera_quality.update({
                'direct_corner_rms_px': float(np.sqrt(np.mean(used_values ** 2))),
                'all_sample_direct_corner_rms_px': float(np.sqrt(np.mean(values ** 2))),
                'direct_corner_median_px': float(np.median(values)),
                'direct_corner_p90_px': float(np.percentile(values, 90)),
                'direct_corner_max_rms_px': float(np.max(values)),
                'direct_corner_max_px': float(np.max([
                    row['direct_corner_max_px'] for row in sample_errors])),
                'pnp_mean_reprojection_rms_px': float(np.mean(pnp_values)),
                'pnp_max_reprojection_rms_px': float(np.max(pnp_values)),
            })
        else:
            per_camera_quality.update({
                'joint_corner_rms_mm': float(np.sqrt(np.mean(used_values ** 2))),
                'all_sample_corner_rms_mm': float(np.sqrt(np.mean(values ** 2))),
                'joint_corner_median_mm': float(np.median(values)),
                'joint_corner_p90_mm': float(np.percentile(values, 90)),
                'joint_corner_max_mm': float(np.max(values)),
            })
        per_camera[camera] = per_camera_quality

    mount_translation_delta = float(np.linalg.norm(
        fitted_mount[:3, 3] - tag_mount[:3, 3]) * 1000.0)
    mount_rotation_delta = float(np.degrees(np.linalg.norm(axis_angle(
        fitted_mount[:3, :3] @ tag_mount[:3, :3].T))))
    quality = {
        'samples_total': int(sum(len(samples) for samples in all_samples.values())),
        'samples_used': int(sum(len(samples) for samples in fit_samples.values())),
        'samples_rejected': int(sum(len(all_samples[camera]) -
                                    len(fit_samples[camera]) for camera in camera_names)),
        'per_camera': per_camera,
        'observation_model': ('direct_distorted_image_corners'
                              if use_direct_corners else 'pnp_tag_pose_corners'),
        'initial_residual_units': 'px' if use_direct_corners else 'm',
        'initial_median_residual': float(residual_scale),
        'outlier_filter': (
            'per-camera median plus three robust standard deviations; '
            'standard deviation estimated as 1.482602218505602 times MAD'),
        'tag_mount_delta_mm': mount_translation_delta,
        'tag_mount_delta_deg': mount_rotation_delta,
        'optimizer': {
            'success': bool(solution.success),
            'message': solution.message,
            'nfev': int(solution.nfev),
            'optimality': float(solution.optimality),
            'cost': float(solution.cost),
        },
    }
    if use_direct_corners:
        quality.update({
            'direct_corner_rms_px': float(
                np.sqrt(np.mean(np.asarray(used_pixel_errors) ** 2))),
            'all_sample_direct_corner_rms_px': float(
                np.sqrt(np.mean(np.asarray(all_pixel_errors) ** 2))),
            'direct_corner_median_coordinate_px': float(
                np.median(np.abs(all_pixel_errors))),
        })
    else:
        quality.update({
            'joint_corner_rms_mm': float(
                np.sqrt(np.mean(np.asarray(used_corner_errors) ** 2)) * 1000.0),
            'all_sample_joint_corner_rms_mm': float(
                np.sqrt(np.mean(np.asarray(all_corner_errors) ** 2)) * 1000.0),
            'joint_corner_median_coordinate_mm': float(
                np.median(np.abs(all_corner_errors)) * 1000.0),
        })
    return {
        'camera_transforms': camera_transforms,
        'tag_mount': fitted_mount,
        'quality': quality,
        'camera_names': camera_names,
    }


def _direct_sample_quality(sample: dict[str, Any], camera_transform: np.ndarray,
                           tag_mount: np.ndarray, points: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Measure one raw-corner sample without changing or filtering it."""
    residual = _direct_corner_residual(sample, camera_transform, tag_mount, points)
    distances = np.linalg.norm(residual, axis=1)
    return ({
        'index': sample['index'],
        'direct_corner_rms_px': float(np.sqrt(np.mean(distances ** 2))),
        'direct_corner_max_px': float(np.max(distances)),
    }, residual)


def _evaluate_direct_samples(samples_by_camera: dict[str, list[dict[str, Any]]],
                             camera_transforms: dict[str, np.ndarray],
                             tag_mount: np.ndarray, tag_size: float) -> dict[str, Any]:
    """Evaluate every supplied raw-corner sample, including outliers."""
    points = tag_corners(tag_size)
    per_camera = {}
    all_errors = []
    for camera in sorted(samples_by_camera):
        rows = []
        values = []
        camera_errors = []
        for sample in sorted(samples_by_camera[camera], key=lambda item: item['index']):
            row, residual = _direct_sample_quality(
                sample, camera_transforms[camera], tag_mount, points)
            rows.append(row)
            values.append(row['direct_corner_rms_px'])
            camera_errors.extend(residual.ravel())
            all_errors.extend(residual.ravel())
        if not rows:
            per_camera[camera] = {
                'samples_evaluated': 0,
                'direct_corner_rms_px': None,
                'direct_corner_median_px': None,
                'direct_corner_p90_px': None,
                'direct_corner_max_rms_px': None,
                'direct_corner_max_px': None,
                'samples': [],
            }
            continue
        values = np.asarray(values, dtype=float)
        camera_errors = np.asarray(camera_errors, dtype=float)
        per_camera[camera] = {
            'samples_evaluated': len(rows),
            'direct_corner_rms_px': float(np.sqrt(np.mean(camera_errors ** 2))),
            'direct_corner_median_px': float(np.median(values)),
            'direct_corner_p90_px': float(np.percentile(values, 90)),
            'direct_corner_max_rms_px': float(np.max(values)),
            'direct_corner_max_px': float(max(row['direct_corner_max_px'] for row in rows)),
            'samples': rows,
        }
    all_errors = np.asarray(all_errors, dtype=float)
    return {
        'samples_evaluated': int(sum(item['samples_evaluated']
                                    for item in per_camera.values())),
        'per_camera': per_camera,
        'direct_corner_rms_px': float(np.sqrt(np.mean(all_errors ** 2))),
        'direct_corner_median_coordinate_px': float(np.median(np.abs(all_errors))),
    }


def _leave_one_pose_out_validation(
        samples_by_camera: dict[str, list[dict[str, Any]]], tag_size: float,
        initial_tag_mount: np.ndarray, fitted: dict[str, Any],
        attempted_indices: dict[str, list[int]] | None = None,
        undetected_indices: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """Validate each recorded pose with that pose excluded from the fit."""
    camera_names = fitted['camera_names']
    if fitted['quality'].get('observation_model') != 'direct_distorted_image_corners':
        raise ValueError('leave-one-pose-out validation requires raw image corners')

    valid_indices = {
        sample['index'] for camera in camera_names
        for sample in samples_by_camera[camera]
    }
    attempted_indices = attempted_indices or {}
    pose_indices = sorted(valid_indices | {
        index for camera in camera_names
        for index in attempted_indices.get(camera, [])
    })
    undetected_indices = undetected_indices or {}
    folds = []
    evaluated_samples = {camera: [] for camera in camera_names}
    failures = []
    for index in pose_indices:
        held_out = {
            camera: [sample for sample in samples_by_camera[camera]
                     if sample['index'] == index]
            for camera in camera_names
        }
        training = {
            camera: [sample for sample in samples_by_camera[camera]
                     if sample['index'] != index]
            for camera in camera_names
        }
        missing_training = {
            camera: len(samples) for camera, samples in training.items()
            if len(samples) < MIN_SAMPLES
        }
        if missing_training:
            reason = f'not enough training samples after holding out pose {index}'
            fold = {'index': index, 'status': 'unavailable',
                    'reason': reason, 'training_samples': {
                        camera: len(training[camera]) for camera in camera_names}}
            failures.append(fold)
            folds.append(fold)
            continue
        if not any(held_out.values()):
            fold = {'index': index, 'status': 'unavailable',
                    'reason': 'no camera detected the tag at this pose',
                    'training_samples': {
                        camera: len(training[camera]) for camera in camera_names}}
            failures.append(fold)
            folds.append(fold)
            continue
        try:
            fold_fit = joint_handeye_fit(
                training, tag_size, initial_tag_mount, minimum_samples=MIN_SAMPLES)
            evaluation = _evaluate_direct_samples(
                held_out, fold_fit['camera_transforms'], fold_fit['tag_mount'], tag_size)
        except (cv2.error, ValueError, TypeError) as error:
            fold = {'index': index, 'status': 'unavailable', 'reason': str(error),
                    'training_samples': {
                        camera: len(training[camera]) for camera in camera_names}}
            failures.append(fold)
            folds.append(fold)
            continue

        for camera in camera_names:
            evaluated_samples[camera].extend(evaluation['per_camera'][camera]['samples'])
        folds.append({
            'index': index,
            'status': 'completed',
            'training_samples': {
                camera: len(training[camera]) for camera in camera_names},
            'held_out': evaluation,
        })

    fold_rows = {
        camera: sorted(rows, key=lambda row: row['index'])
        for camera, rows in evaluated_samples.items()
    }
    all_fold_values = [row['direct_corner_rms_px']
                       for rows in fold_rows.values() for row in rows]
    per_camera = {}
    for camera in camera_names:
        rows = fold_rows[camera]
        values = np.asarray([row['direct_corner_rms_px'] for row in rows], dtype=float)
        per_camera[camera] = {
            'samples_evaluated': len(rows),
            'direct_corner_rms_px': float(np.sqrt(np.mean(values ** 2))) if len(values) else None,
            'direct_corner_median_px': float(np.median(values)) if len(values) else None,
            'direct_corner_p90_px': float(np.percentile(values, 90)) if len(values) else None,
            'direct_corner_max_rms_px': float(np.max(values)) if len(values) else None,
            'samples': rows,
        }
    status = 'completed' if not failures and not any(undetected_indices.get(camera)
                                                     for camera in camera_names) else 'incomplete'
    return {
        'status': status,
        'method': 'leave_one_pose_out',
        'observation_model': 'direct_distorted_image_corners',
        'fit_pose_exclusion': 'all cameras at the held-out pose index are excluded',
        'undetected_indices': {
            camera: sorted(set(undetected_indices.get(camera, [])))
            for camera in camera_names
        },
        'folds': folds,
        'samples_evaluated': int(sum(len(rows) for rows in fold_rows.values())),
        'per_camera': per_camera,
        'direct_corner_rms_px': float(np.sqrt(np.mean(np.asarray(all_fold_values) ** 2)))
        if all_fold_values else None,
        'failures': failures,
    }


def joint_calibration_document(dataset: Path, camera_frames: dict[str, str] | None = None,
                               config_path: str | None = None,
                               urdf_path: str | None = None,
                               cameras: list[str] | None = None) -> dict[str, Any]:
    """Create a versioned artifact from one joint multi-camera calibration fit."""
    loaded = load_joint_camera_samples(dataset, config_path, urdf_path)
    available = loaded['samples']
    selected = sorted(cameras or available)
    missing = [camera for camera in selected if camera not in available]
    if missing:
        raise ValueError(f'camera(s) {missing} are absent from the dataset')
    if loaded['tag_mount'] is None:
        raise ValueError('dataset does not provide the TCP-to-tag mount transform')
    selected_samples = {camera: available[camera] for camera in selected}
    fit = joint_handeye_fit(selected_samples, loaded['tag_size'], loaded['tag_mount'])
    validation = _leave_one_pose_out_validation(
        selected_samples, loaded['tag_size'], loaded['tag_mount'], fit,
        {camera: loaded.get('attempted_indices', {}).get(camera, [])
         for camera in selected},
        {camera: loaded.get('undetected_indices', {}).get(camera, [])
         for camera in selected})
    camera_info = loaded['camera_info']
    camera_frames = camera_frames or {}
    unknown_frames = set(camera_frames) - set(selected)
    if unknown_frames:
        raise ValueError(f'camera frame supplied for unknown camera(s): '
                         f'{sorted(unknown_frames)}')
    frames = {
        camera: camera_frames.get(camera,
                                  camera_info[camera].get('frame_id', f'{camera}_camera'))
        for camera in selected
    }
    return {
        'schema_version': 2,
        'calibration_type': 'joint_robot_camera_handeye',
        'base_frame': loaded['base_frame'],
        'units': 'm',
        'camera_frames': frames,
        'T_base_camera': {
            camera: fit['camera_transforms'][camera].tolist() for camera in selected
        },
        'T_tcp_tag': fit['tag_mount'].tolist(),
        'tag_size_m': loaded['tag_size'],
        'quality': fit['quality'],
        'validation': validation,
        'provenance': {
            'dataset': str(dataset.resolve()),
            'dataset_kind': loaded['dataset_kind'],
            'camera_names': selected,
            'sample_indices': {
                camera: [sample['index'] for sample in sorted(
                    selected_samples[camera], key=lambda item: item['index'])]
                for camera in selected
            },
            'pnp_method': 'SOLVEPNP_IPPE_SQUARE (initialization and diagnostic only)',
            'fit_method': 'joint_se3_direct_distorted_image_corner_fit',
            'outlier_filter': fit['quality']['outlier_filter'],
            'residual': ('robot-predicted versus camera-observed tag-corner '
                         'positions in the base frame'),
            'camera_order': selected,
            'convention': ('T_base_camera maps camera-frame points into base_frame; '
                           'T_tcp_tag maps tag-frame points into tcp frame'),
        },
    }


def calibration_document(dataset: Path, camera_name: str, camera_frame: str | None = None,
                         config_path: str | None = None) -> dict[str, Any]:
    """Solve one dataset camera and return the portable robot-to-camera artifact."""
    if (dataset / 'meta.yaml').exists() and (dataset / 'detections.yaml').exists():
        meta, base_tag, camera_tag, reprojection, indices = load_camera_samples(dataset, camera_name)
        dataset_kind = 'generated_calibration_capture'
    else:
        meta, base_tag, camera_tag, reprojection, indices = load_taught_camera_samples(
            dataset, camera_name, config_path)
        dataset_kind = 'taught_capture'
    if len(base_tag) < MIN_SAMPLES:
        raise ValueError(f'{camera_name} has {len(base_tag)} valid detections; need at least {MIN_SAMPLES}')
    park, andreff, inliers = solve_robust(base_tag, camera_tag)
    selected_reprojection = [value for value, keep in zip(reprojection, inliers) if keep]
    return {
        'schema_version': 1,
        'base_frame': meta.get('base_link', 'fr3_link0'),
        'camera_frame': camera_frame or meta['cameras'][camera_name].get('frame_id', f'{camera_name}_camera'),
        'units': 'm',
        'T_base_camera': park.tolist(),
        'quality': {
            'samples_total': len(base_tag),
            'samples_used': int(inliers.sum()),
            'samples_rejected': int((~inliers).sum()),
            'mean_reprojection_rms_px': float(np.mean(selected_reprojection)),
            'max_reprojection_rms_px': float(np.max(selected_reprojection)),
            'park_andreff_translation_disagreement_m': float(np.linalg.norm(park[:3, 3] - andreff[:3, 3])),
            'park_andreff_rotation_disagreement_deg': rotation_error_deg(park[:3, :3], andreff[:3, :3]),
        },
        'provenance': {
            'dataset': str(dataset), 'camera_name': camera_name,
            'sample_indices': [index for index, keep in zip(indices, inliers) if keep],
            'dataset_kind': dataset_kind,
            'pnp_method': 'SOLVEPNP_IPPE_SQUARE', 'handeye_method': 'PARK',
            'cross_check_method': 'ANDREFF',
            'convention': 'T_base_camera maps camera-frame points into base_frame',
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Solve fixed-camera eye-to-hand calibration from a capture dataset.')
    parser.add_argument('dataset', type=Path, help='capture directory')
    parser.add_argument('--camera', required=True, help='camera name in the capture dataset')
    parser.add_argument('--camera-frame', help='output frame name (defaults to the capture frame ID)')
    parser.add_argument('--config', help='calibration.yaml used for a taught-capture dataset')
    parser.add_argument('--output', type=Path, help='write JSON here; otherwise print it only')
    args = parser.parse_args(argv)
    try:
        document = calibration_document(args.dataset, args.camera, args.camera_frame, args.config)
    except (OSError, ValueError, cv2.error) as error:
        parser.error(str(error))
    rendered = json.dumps(document, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
        print(f'wrote {args.output}')
    else:
        print(rendered, end='')
    return 0


def _parse_camera_frame(value: str) -> tuple[str, str]:
    """Parse the explicit ``CAMERA=FRAME`` output-frame option."""
    try:
        camera, frame = value.split('=', 1)
    except ValueError as error:
        raise ValueError('--camera-frame must use CAMERA=FRAME') from error
    if not camera or not frame:
        raise ValueError('--camera-frame must use non-empty CAMERA and FRAME')
    return camera, frame


def main_joint(argv: list[str] | None = None) -> int:
    """CLI entry point for a joint multi-camera calibration artifact."""
    parser = argparse.ArgumentParser(
        description='Fit all camera base transforms and the shared TCP-to-tag mount together.')
    parser.add_argument('dataset', type=Path, help='capture directory')
    parser.add_argument('--camera', action='append',
                        help='camera to include; repeat for a subset (default: all)')
    parser.add_argument('--camera-frame', action='append', default=[],
                        metavar='CAMERA=FRAME',
                        help='output frame alias, for example mega=mega_camera')
    parser.add_argument('--config', help='calibration.yaml for a taught-capture dataset')
    parser.add_argument('--urdf', help='offline URDF used instead of invoking xacro')
    parser.add_argument('--output', type=Path, required=True,
                        help='joint calibration JSON to write')
    args = parser.parse_args(argv)
    try:
        frames = {}
        for value in args.camera_frame:
            camera, frame = _parse_camera_frame(value)
            if camera in frames:
                raise ValueError(f'duplicate --camera-frame for {camera!r}')
            frames[camera] = frame
        document = joint_calibration_document(
            args.dataset, frames, args.config, args.urdf, args.camera)
    except (OSError, ValueError, cv2.error) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {args.output}')
    direct_corners = document['quality']['observation_model'] == 'direct_distorted_image_corners'
    for camera in document['provenance']['camera_names']:
        metrics = document['quality']['per_camera'][camera]
        if direct_corners:
            metric = metrics['direct_corner_rms_px']
            unit = 'px'
        else:
            metric = metrics['joint_corner_rms_mm']
            unit = 'mm'
        print(f'{camera}: {metrics["samples_used"]} samples, '
              f'joint corner RMS {metric:.3f} {unit}')
    validation = document['validation']
    print(f'leave-one-pose-out validation: {validation["status"]}, '
          f'{validation["samples_evaluated"]} held-out samples')
    if validation['undetected_indices']:
        for camera, indices in validation['undetected_indices'].items():
            if indices:
                print(f'  {camera}: tag not detected at poses {indices}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
