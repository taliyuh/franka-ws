"""Pinhole camera model and AprilTag visibility scoring.

The visibility test is what makes the generated poses usable: at the distances
in this cell a 106 mm tag is only 30-60 px across, so "the tag is inside the
image" is nowhere near a strong enough condition.  We also require a minimum
apparent size, a bounded incidence angle and a sane range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import yaml

from .kinematics import invert, transform_from_xyz_rpy

# Gazebo camera *bodies* look down +X with +Z up.  The optical frame that
# ``camera_info`` and OpenCV use looks down +Z with +X right and +Y down.
# Columns are the optical axes expressed in the body frame.
BODY_TO_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def tag_corners(tag_size: float) -> np.ndarray:
    """The 4 corners of the tag's *black square*, in the tag frame.

    Order matches OpenCV's marker convention (the same order
    ``cv2.aruco.detectMarkers`` returns), with +Z out of the tag face.
    """
    half = tag_size / 2.0
    return np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ])


def segment_distance(p1: np.ndarray, q1: np.ndarray,
                     p2: np.ndarray, q2: np.ndarray) -> float:
    """Shortest distance between segments p1-q1 and p2-q2 (Ericson, RTCD 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, min(1.0, max(0.0, f / e))
    else:
        c = float(d1 @ r)
        if e <= eps:
            t, s = 0.0, min(1.0, max(0.0, -c / a))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / denom)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    return float(np.linalg.norm(r + s * d1 - t * d2))


@dataclass
class VisibilityThresholds:
    border_px: float = 30.0
    min_apparent_px: float = 35.0
    max_incidence_deg: float = 65.0
    min_range: float = 0.40
    max_range: float = 1.60


@dataclass
class TagView:
    """Result of projecting a tag pose into one camera."""

    camera: str
    visible: bool
    reasons: list = field(default_factory=list)
    corners_px: np.ndarray | None = None
    apparent_px: float = 0.0
    incidence_deg: float = 180.0
    distance: float = 0.0

    def summary(self) -> str:
        if self.visible:
            return (f'{self.apparent_px:5.1f}px {self.incidence_deg:4.1f}deg '
                    f'{self.distance:4.2f}m')
        return ','.join(self.reasons) or 'not visible'


@dataclass
class PinholeCamera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    pose: np.ndarray  # world -> optical frame

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_dict(cls, name: str, spec: dict) -> 'PinholeCamera':
        """Build from a config entry.

        The pose may be given either as the Gazebo model (body) pose --
        ``body_xyz``/``body_rpy``, which is what the world SDF contains -- or
        directly as the optical frame pose (``optical_xyz``/``optical_rpy``),
        which is what a real measured extrinsic normally is.
        """
        if 'optical_xyz' in spec:
            pose = transform_from_xyz_rpy(spec['optical_xyz'], spec['optical_rpy'])
        else:
            body = transform_from_xyz_rpy(spec['body_xyz'], spec['body_rpy'])
            pose = body.copy()
            pose[:3, :3] = body[:3, :3] @ BODY_TO_OPTICAL
        return cls(
            name=name,
            width=int(spec['width']),
            height=int(spec['height']),
            fx=float(spec['fx']),
            fy=float(spec['fy']),
            cx=float(spec['cx']),
            cy=float(spec['cy']),
            pose=pose,
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    def apply_camera_info(self, msg) -> None:
        """Overwrite intrinsics from a live ``sensor_msgs/CameraInfo``."""
        k = list(msg.k)
        self.fx, self.fy, self.cx, self.cy = k[0], k[4], k[2], k[5]
        self.width, self.height = int(msg.width), int(msg.height)

    # -- projection -------------------------------------------------------- #

    def to_optical(self, points_world: np.ndarray) -> np.ndarray:
        points_world = np.atleast_2d(points_world)
        world_to_optical = invert(self.pose)
        return (world_to_optical[:3, :3] @ points_world.T).T + world_to_optical[:3, 3]

    def project(self, points_world: np.ndarray):
        """Returns ``(pixels Nx2, depths N)``.  Depths <= 0 are behind the camera."""
        cam = self.to_optical(points_world)
        depth = cam[:, 2]
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        pixels = np.stack([self.fx * cam[:, 0] / safe + self.cx,
                           self.fy * cam[:, 1] / safe + self.cy], axis=1)
        return pixels, depth

    # -- visibility -------------------------------------------------------- #

    def view_tag(self, tag_pose: np.ndarray, tag_size: float,
                 thresholds: VisibilityThresholds | None = None,
                 occluders=None) -> TagView:
        """Project the tag into this camera and judge whether it is usable.

        *occluders* is a list of ``(start, end, radius)`` capsules approximating
        the robot's own links.  Without it the geometry happily reports "69 px,
        comfortably in frame, 23 deg off-normal" for views where the forearm or
        the gripper is lying across the black square and no detector will ever
        find it -- which is exactly what Gazebo showed before this was added.
        """
        thresholds = thresholds or VisibilityThresholds()
        corners_world = (tag_pose[:3, :3] @ tag_corners(tag_size).T).T + tag_pose[:3, 3]
        pixels, depths = self.project(corners_world)

        centre = tag_pose[:3, 3]
        to_camera = self.pose[:3, 3] - centre
        distance = float(np.linalg.norm(to_camera))
        normal = tag_pose[:3, 2]
        incidence = math.degrees(math.acos(
            max(-1.0, min(1.0, float(np.dot(normal, to_camera / max(distance, 1e-9)))))))

        edges = [float(np.linalg.norm(pixels[(i + 1) % 4] - pixels[i])) for i in range(4)]
        apparent = float(np.mean(edges))

        reasons = []
        if np.any(depths <= 0.0):
            reasons.append('behind camera')
        else:
            border = thresholds.border_px
            if (np.any(pixels[:, 0] < border) or np.any(pixels[:, 0] > self.width - border)
                    or np.any(pixels[:, 1] < border)
                    or np.any(pixels[:, 1] > self.height - border)):
                reasons.append('outside frame')
            if apparent < thresholds.min_apparent_px:
                reasons.append(f'too small ({apparent:.0f}px)')
        if incidence > thresholds.max_incidence_deg:
            reasons.append(f'incidence {incidence:.0f}deg')
        eye = self.pose[:3, 3]
        for start, end, radius in (occluders or []):
            if any(segment_distance(eye, corner, start, end) < radius
                   for corner in corners_world):
                reasons.append('occluded by the robot')
                break
        if distance < thresholds.min_range or distance > thresholds.max_range:
            reasons.append(f'range {distance:.2f}m')

        return TagView(
            camera=self.name,
            visible=not reasons,
            reasons=reasons,
            corners_px=pixels,
            apparent_px=apparent,
            incidence_deg=incidence,
            distance=distance,
        )


def load_cameras(path: str) -> dict:
    """Load ``config/cameras_*.yaml`` into ``{name: PinholeCamera}``."""
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    return {name: PinholeCamera.from_dict(name, spec)
            for name, spec in data['cameras'].items()}
