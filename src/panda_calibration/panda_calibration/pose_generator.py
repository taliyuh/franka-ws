"""Generate FR3 joint configurations suitable for eye-to-hand calibration.

The hard part is not reachability, it is that AX=XB is only well conditioned
when the poses differ by large rotations about *different* axes.  So the
generator samples tag poses that both cameras can actually resolve, converts
them to joint space, and then selects a subset that maximises rotational spread
while guaranteeing a minimum number of views per camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import yaml

from .cameras import VisibilityThresholds
from .kinematics import (
    Chain,
    FR3_READY,
    check_configuration,
    invert,
    make_transform,
    matrix_to_quaternion,
    rotation_angle_between,
    rotation_from_axis_angle,
    transform_from_xyz_rpy,
)


@dataclass
class CalibrationPose:
    index: int
    joints: np.ndarray
    tcp_pose: np.ndarray          # base -> fr3_hand_tcp
    tag_pose: np.ndarray          # base -> tag
    views: dict = field(default_factory=dict)   # camera name -> TagView

    @property
    def visible_cameras(self) -> list:
        return sorted(name for name, view in self.views.items() if view.visible)

    def as_dict(self) -> dict:
        return {
            'index': int(self.index),
            'joints': [round(float(v), 6) for v in self.joints],
            'tcp_xyz': [round(float(v), 6) for v in self.tcp_pose[:3, 3]],
            'tcp_quat_xyzw': [round(float(v), 6)
                              for v in matrix_to_quaternion(self.tcp_pose[:3, :3])],
            'tag_xyz': [round(float(v), 6) for v in self.tag_pose[:3, 3]],
            'visible_cameras': self.visible_cameras,
            'views': {
                name: {
                    'visible': bool(view.visible),
                    'apparent_px': round(float(view.apparent_px), 1),
                    'incidence_deg': round(float(view.incidence_deg), 1),
                    'distance_m': round(float(view.distance), 3),
                    'reasons': list(view.reasons),
                }
                for name, view in sorted(self.views.items())
            },
        }


def _look_at_rotation(normal: np.ndarray, roll: float,
                      up_hint=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation whose +Z is *normal*, rolled by *roll* about that normal."""
    z = np.asarray(normal, dtype=float)
    z = z / np.linalg.norm(z)
    up = np.asarray(up_hint, dtype=float)
    if abs(float(np.dot(up, z))) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    base = np.column_stack([x, y, z])
    return base @ rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), roll)


class PoseGenerator:
    def __init__(self, chain: Chain, cameras: dict, config: dict):
        self.chain = chain
        self.cameras = cameras
        self.config = config
        self.tag_size = float(config['tag']['size'])
        self.tcp_to_tag = transform_from_xyz_rpy(config['tag']['mount_xyz'],
                                                 config['tag']['mount_rpy'])
        self.tag_to_tcp = invert(self.tcp_to_tag)
        self.thresholds = VisibilityThresholds(**config['visibility'])
        self.occluders = config.get('occluders') or {}
        self.settings = config['pose_generation']
        self.feasibility = self.settings.get('feasibility', {})

    # -- candidate construction -------------------------------------------- #

    def _sample_tag_pose(self, rng: np.random.Generator, roll: float) -> np.ndarray:
        centre = np.asarray(self.settings['centre'], dtype=float)
        extents = np.asarray(self.settings['extents'], dtype=float)
        position = centre + rng.uniform(-extents, extents)

        weight = float(rng.choice(self.settings['aim_weights']))
        names = sorted(self.cameras)
        aim_point = ((1.0 - weight) * self.cameras[names[0]].pose[:3, 3]
                     + weight * self.cameras[names[-1]].pose[:3, 3])
        normal = aim_point - position
        normal = normal / np.linalg.norm(normal)

        max_tilt = math.radians(float(self.settings['max_tilt_deg']))
        tilt_axis = np.cross(normal, rng.normal(size=3))
        if np.linalg.norm(tilt_axis) > 1e-6:
            normal = rotation_from_axis_angle(
                tilt_axis, rng.uniform(-max_tilt, max_tilt)) @ normal

        return make_transform(_look_at_rotation(normal, roll), position)

    def _capsules(self, frames: dict) -> list:
        """The robot's own links as ``(start, end, radius)`` capsules.

        Built from the chain itself rather than a hand-written body model: a
        capsule between each pair of consecutive link origins, radius from
        ``occluders.link_radius`` (falling back to ``occluders.radius``).  Crude,
        but it is the difference between predicting a view the detector can
        actually use and predicting one the forearm is lying across.
        """
        if not self.occluders:
            return []
        default = float(self.occluders.get('radius', 0.065))
        per_link = self.occluders.get('link_radius', {})
        links = [self.chain.base_link] + [j.child for j in self.chain.joints]
        capsules = []
        for parent, child in zip(links, links[1:]):
            start, end = frames[parent][:3, 3], frames[child][:3, 3]
            if float(np.linalg.norm(end - start)) < 1e-6:
                continue
            radius = max(float(per_link.get(parent, default)),
                         float(per_link.get(child, default)))
            capsules.append((start, end, radius))
        return capsules

    def _score_views(self, tag_pose: np.ndarray, occluders=None) -> dict:
        return {name: camera.view_tag(tag_pose, self.tag_size, self.thresholds, occluders)
                for name, camera in self.cameras.items()}

    def candidates(self, rng: np.random.Generator, seed_joints=None, limit=None):
        """Yield feasible, at-least-one-camera-visible :class:`CalibrationPose`.

        Stops after *limit* successes; IK dominates the runtime and a pool an
        order of magnitude larger than the requested pose count is already
        plenty for the diversity selection to work with.
        """
        roll_step = math.radians(float(self.settings['roll_step_deg']))
        roll_count = max(1, int(round(2 * math.pi / roll_step)))
        seed = np.array(FR3_READY if seed_joints is None else seed_joints, dtype=float)
        produced = 0

        for attempt in range(int(self.settings['max_candidates'])):
            tag_pose = self._sample_tag_pose(rng, roll_step * (attempt % roll_count))

            views = self._score_views(tag_pose)
            if not any(view.visible for view in views.values()):
                continue

            target = tag_pose @ self.tag_to_tcp
            joints, converged = self.chain.ik_with_restarts(target, seed=seed, rng=rng)
            if not converged:
                continue
            if check_configuration(self.chain, joints, **self.feasibility):
                continue

            # The IK solution is only accurate to ~1 mm / 0.5 deg, so re-score
            # visibility from the pose the robot will actually hold.
            frames = self.chain.link_frames(joints)
            tcp_pose = frames[self.chain.tip_link]
            actual_tag_pose = tcp_pose @ self.tcp_to_tag
            views = self._score_views(actual_tag_pose, self._capsules(frames))
            if not any(view.visible for view in views.values()):
                continue

            seed = joints
            yield CalibrationPose(index=produced, joints=joints, tcp_pose=tcp_pose,
                                  tag_pose=actual_tag_pose, views=views)
            produced += 1
            if limit is not None and produced >= limit:
                return

    # -- selection --------------------------------------------------------- #

    def select(self, candidates: list, count: int) -> list:
        """Greedily pick *count* poses with maximal rotational spread.

        Each step takes the candidate whose minimum rotation distance to the
        already-selected set is largest, with a bonus for being seen by every
        camera; cameras still short of ``min_poses_per_camera`` get priority so
        no camera ends up under-constrained.
        """
        if not candidates:
            return []
        min_separation = math.radians(float(self.settings['min_rotation_separation_deg']))
        min_per_camera = min(int(self.settings['min_poses_per_camera']), count)
        camera_names = sorted(self.cameras)

        both = [p for p in candidates if len(p.visible_cameras) == len(camera_names)]
        selected = [max(both or candidates, key=lambda p: len(p.visible_cameras))]
        remaining = [p for p in candidates if p is not selected[0]]

        while len(selected) < count and remaining:
            counts = {name: sum(1 for p in selected if name in p.visible_cameras)
                      for name in camera_names}
            starved = {name for name, n in counts.items() if n < min_per_camera}

            def score(pose):
                spread = min(rotation_angle_between(pose.tcp_pose[:3, :3],
                                                    other.tcp_pose[:3, :3])
                             for other in selected)
                visible = set(pose.visible_cameras)
                bonus = 0.0
                if starved:
                    bonus += 2.0 * len(visible & starved)
                if len(visible) == len(camera_names):
                    bonus += 0.5
                return spread + bonus

            best = max(remaining, key=score)
            spread = min(rotation_angle_between(best.tcp_pose[:3, :3],
                                                other.tcp_pose[:3, :3])
                         for other in selected)
            if spread < min_separation and not starved:
                break
            selected.append(best)
            remaining.remove(best)

        return selected

    @staticmethod
    def order(poses: list, start_joints=None) -> list:
        """Nearest-neighbour ordering in joint space, to keep the moves small."""
        remaining = list(poses)
        ordered = []
        current = np.array(FR3_READY if start_joints is None else start_joints, dtype=float)
        while remaining:
            nearest = min(remaining,
                          key=lambda p: float(np.max(np.abs(p.joints - current))))
            ordered.append(nearest)
            remaining.remove(nearest)
            current = nearest.joints
        for index, pose in enumerate(ordered):
            pose.index = index
        return ordered

    # -- entry point ------------------------------------------------------- #

    def generate(self, count: int | None = None, seed: int | None = None) -> list:
        count = int(count if count is not None else self.settings['count'])
        rng = np.random.default_rng(
            int(seed if seed is not None else self.settings['seed']))
        limit = int(self.settings.get('pool_size', 0)) or max(120, 12 * count)
        pool = list(self.candidates(rng, limit=limit))
        return self.order(self.select(pool, count))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_poses(path: str, poses: list, config: dict, cameras: dict) -> None:
    payload = {
        'tag': {
            'id': config['tag']['id'],
            'family': config['tag']['family'],
            'size': config['tag']['size'],
            'mount_xyz': config['tag']['mount_xyz'],
            'mount_rpy': config['tag']['mount_rpy'],
        },
        'joint_names': config['robot']['joint_names'],
        'cameras': {
            name: {
                'optical_xyz': [round(float(v), 6) for v in camera.pose[:3, 3]],
                'optical_quat_xyzw': [round(float(v), 6)
                                      for v in matrix_to_quaternion(camera.pose[:3, :3])],
            }
            for name, camera in sorted(cameras.items())
        },
        'poses': [pose.as_dict() for pose in poses],
    }
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=None)


def load_joint_poses(path: str) -> list:
    """Read joint vectors from generated or taught-capture pose manifests.

    Generated manifests call the field ``joints``; ``run_taught_capture``
    records the measured vector as ``measured_joints``.  Both represent the
    robot pose that should be replayed by ``run_calibration_capture``.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    poses = []
    for entry in data['poses']:
        values = entry.get('joints')
        if values is None:
            values = entry.get('measured_joints')
        if values is None:
            raise KeyError(f'pose {entry.get("index", "?")} has no joints or measured_joints')
        poses.append(np.asarray(values, dtype=float))
    return poses
