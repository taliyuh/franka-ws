"""Minimal URDF forward/inverse kinematics for the FR3 arm.

Deliberately dependency-free (numpy only): the container has no ``kdl_parser``,
no ``moveit_py`` and no ``urdf_parser_py``, and hard-coding DH parameters would
drift away from whatever ``franka_description`` actually ships.  So the chain is
read straight out of the URDF that Gazebo/``robot_state_publisher`` are using.
"""

from __future__ import annotations

import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

FR3_JOINT_NAMES = [f'fr3_joint{i}' for i in range(1, 8)]

# Franka's "ready" pose, also the URDF's initial_value for each arm joint.
FR3_READY = np.array([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4])


# --------------------------------------------------------------------------- #
# Small SE(3) helpers.  Everything is a plain 4x4 numpy array.
# --------------------------------------------------------------------------- #

def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF/ROS fixed-axis RPY (== Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`rpy_to_matrix` (returns roll, pitch, yaw)."""
    pitch = math.atan2(-rot[2, 0], math.hypot(rot[0, 0], rot[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:  # gimbal lock
        return math.atan2(-rot[1, 2], rot[1, 1]), pitch, 0.0
    return math.atan2(rot[2, 1], rot[2, 2]), pitch, math.atan2(rot[1, 0], rot[0, 0])


def make_transform(rot: np.ndarray, pos) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = np.asarray(pos, dtype=float)
    return out


def quaternion_to_matrix(xyzw) -> np.ndarray:
    """Convert an ``(x, y, z, w)`` quaternion to a rotation matrix."""
    x, y, z, w = np.asarray(xyzw, dtype=float)
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0.0:
        raise ValueError('quaternion must not be zero')
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_error(target: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return translation (m) and geodesic rotation (rad) error for two poses."""
    translation_error = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    rotation_error = rotation_angle_between(target[:3, :3], actual[:3, :3])
    return translation_error, rotation_error


def conservative_joint_tolerances(jacobian: np.ndarray, position_tolerance: float,
                                  rotation_tolerance: float) -> np.ndarray:
    """Return equal joint bounds that conservatively bound Cartesian error.

    If every joint error is within the returned bound, the triangle inequality
    applied to the linear and angular Jacobian rows bounds the corresponding
    first-order Cartesian errors. The bound is deliberately conservative and is
    intended for a motion action's final joint goal tolerance; it is not a
    substitute for measuring the achieved TCP pose.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != 6 or jacobian.shape[1] == 0:
        raise ValueError('jacobian must have shape (6, number_of_joints)')
    if not (np.isfinite(jacobian).all() and position_tolerance > 0.0
            and rotation_tolerance > 0.0):
        raise ValueError('jacobian and Cartesian tolerances must be finite and positive')

    linear_sensitivity = float(np.linalg.norm(jacobian[:3], axis=0).sum())
    angular_sensitivity = float(np.linalg.norm(jacobian[3:], axis=0).sum())
    if linear_sensitivity <= 0.0 or angular_sensitivity <= 0.0:
        raise ValueError('jacobian must contain linear and angular sensitivity')
    bounds = [position_tolerance / linear_sensitivity]
    bounds.append(rotation_tolerance / angular_sensitivity)
    return np.full(jacobian.shape[1], min(bounds))


def transform_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    return make_transform(rpy_to_matrix(*rpy), xyz)


def invert(transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ transform[:3, 3]
    return out


def axis_angle(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle)."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rot) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    if abs(angle - math.pi) < 1e-6:
        # Near-pi: read the axis off (R + I) instead, which stays well conditioned.
        col = np.argmax(np.diag(rot))
        axis = (rot + np.eye(3))[:, col]
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    vec = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return vec * (angle / (2.0 * math.sin(angle)))


def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def rotation_angle_between(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Geodesic angle (rad) between two rotations."""
    return float(np.linalg.norm(axis_angle(rot_a @ rot_b.T)))


def matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    """Returns (x, y, z, w)."""
    trace = np.trace(rot)
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return np.array([(rot[2, 1] - rot[1, 2]) / scale,
                         (rot[0, 2] - rot[2, 0]) / scale,
                         (rot[1, 0] - rot[0, 1]) / scale,
                         0.25 * scale])
    idx = int(np.argmax(np.diag(rot)))
    if idx == 0:
        scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        return np.array([0.25 * scale,
                         (rot[0, 1] + rot[1, 0]) / scale,
                         (rot[0, 2] + rot[2, 0]) / scale,
                         (rot[2, 1] - rot[1, 2]) / scale])
    if idx == 1:
        scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        return np.array([(rot[0, 1] + rot[1, 0]) / scale,
                         0.25 * scale,
                         (rot[1, 2] + rot[2, 1]) / scale,
                         (rot[0, 2] - rot[2, 0]) / scale])
    scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
    return np.array([(rot[0, 2] + rot[2, 0]) / scale,
                     (rot[1, 2] + rot[2, 1]) / scale,
                     0.25 * scale,
                     (rot[1, 0] - rot[0, 1]) / scale])


# --------------------------------------------------------------------------- #
# URDF chain
# --------------------------------------------------------------------------- #

@dataclass
class ChainJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float = 0.0
    upper: float = 0.0
    velocity: float = 0.0

    @property
    def actuated(self) -> bool:
        return self.joint_type in ('revolute', 'continuous', 'prismatic')


@dataclass
class Chain:
    """A serial base -> tip chain extracted from a URDF."""

    base_link: str
    tip_link: str
    joints: list = field(default_factory=list)

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_urdf(cls, urdf_xml: str, base_link: str, tip_link: str) -> 'Chain':
        root = ET.fromstring(urdf_xml)
        by_child = {}
        for elem in root.findall('joint'):
            origin = elem.find('origin')
            xyz = [float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None
                                      else '0 0 0').split()]
            rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None
                                      else '0 0 0').split()]
            axis_elem = elem.find('axis')
            axis = ([float(v) for v in axis_elem.get('xyz').split()]
                    if axis_elem is not None else [0.0, 0.0, 1.0])
            limit = elem.find('limit')
            joint = ChainJoint(
                name=elem.get('name'),
                joint_type=elem.get('type'),
                parent=elem.find('parent').get('link'),
                child=elem.find('child').get('link'),
                origin=transform_from_xyz_rpy(xyz, rpy),
                axis=np.asarray(axis, dtype=float),
                lower=float(limit.get('lower', 0.0)) if limit is not None else 0.0,
                upper=float(limit.get('upper', 0.0)) if limit is not None else 0.0,
                velocity=float(limit.get('velocity', 0.0)) if limit is not None else 0.0,
            )
            by_child[joint.child] = joint

        walked = []
        link = tip_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(
                    f'link "{link}" has no parent joint; cannot reach "{base_link}" '
                    f'from "{tip_link}"')
            joint = by_child[link]
            walked.append(joint)
            link = joint.parent
        walked.reverse()
        return cls(base_link=base_link, tip_link=tip_link, joints=walked)

    @classmethod
    def from_xacro(cls, xacro_path: str, base_link: str, tip_link: str,
                   mappings: dict | None = None) -> 'Chain':
        """Run ``xacro`` on *xacro_path* and build the chain from the result."""
        args = ['xacro', xacro_path]
        for key, value in (mappings or {}).items():
            args.append(f'{key}:={value}')
        urdf = subprocess.run(args, check=True, capture_output=True, text=True).stdout
        return cls.from_urdf(urdf, base_link, tip_link)

    # -- queries ----------------------------------------------------------- #

    @property
    def joint_names(self) -> list:
        return [j.name for j in self.joints if j.actuated]

    @property
    def limits(self) -> tuple:
        actuated = [j for j in self.joints if j.actuated]
        return (np.array([j.lower for j in actuated]),
                np.array([j.upper for j in actuated]))

    @property
    def max_velocities(self) -> np.ndarray:
        return np.array([j.velocity for j in self.joints if j.actuated])

    def link_frames(self, q) -> dict:
        """base -> link transform for every link along the chain."""
        q = np.asarray(q, dtype=float)
        frames = {self.base_link: np.eye(4)}
        current = np.eye(4)
        index = 0
        for joint in self.joints:
            current = current @ joint.origin
            if joint.actuated:
                if joint.joint_type == 'prismatic':
                    current = current @ make_transform(np.eye(3), joint.axis * q[index])
                else:
                    current = current @ make_transform(
                        rotation_from_axis_angle(joint.axis, float(q[index])), np.zeros(3))
                index += 1
            frames[joint.child] = current.copy()
        return frames

    def fk(self, q) -> np.ndarray:
        return self.link_frames(q)[self.tip_link]

    def fk_and_jacobian(self, q) -> tuple:
        """Tip pose and geometric Jacobian (6 x n) in one pass.

        IK runs this in its inner loop, so it deliberately avoids the dict
        allocation that :meth:`link_frames` does.
        """
        q = np.asarray(q, dtype=float)
        current = np.eye(4)
        axes = []
        origins = []
        types = []
        index = 0
        for joint in self.joints:
            current = current @ joint.origin
            if joint.actuated:
                axes.append(current[:3, :3] @ joint.axis)
                origins.append(current[:3, 3].copy())
                types.append(joint.joint_type)
                if joint.joint_type == 'prismatic':
                    current = current @ make_transform(np.eye(3), joint.axis * q[index])
                else:
                    current = current @ make_transform(
                        rotation_from_axis_angle(joint.axis, float(q[index])), np.zeros(3))
                index += 1

        tip_pos = current[:3, 3]
        jac = np.empty((6, len(axes)))
        for i, (axis_world, origin_world) in enumerate(zip(axes, origins)):
            if types[i] == 'prismatic':
                jac[:3, i] = axis_world
                jac[3:, i] = 0.0
            else:
                jac[:3, i] = np.cross(axis_world, tip_pos - origin_world)
                jac[3:, i] = axis_world
        return current, jac

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian (6 x n) of the tip, expressed in the base frame."""
        return self.fk_and_jacobian(q)[1]

    # -- inverse kinematics ------------------------------------------------ #

    def ik(self, target: np.ndarray, seed=None, *, max_iterations: int = 150,
           position_tolerance: float = 1e-4, rotation_tolerance: float = 8.0e-4,
           damping: float = 0.03, nullspace_target=None, nullspace_gain: float = 0.02,
           limit_margin: float = 0.05):
        """Levenberg-Marquardt IK with joint-limit clamping and a line search.

        Returns ``(q, converged)``. Defaults target 0.1 mm and 0.046 deg
        convergence for the calibrated board task.
        The nullspace term biases the redundant DOF towards ``nullspace_target``
        (``FR3_READY`` by default) so solutions stay in a sane elbow posture.
        """
        lower, upper = self.limits
        lower = lower + limit_margin
        upper = upper - limit_margin
        q = np.clip(np.array(FR3_READY if seed is None else seed, dtype=float), lower, upper)
        bias = np.array(FR3_READY if nullspace_target is None else nullspace_target, dtype=float)
        identity = np.eye(len(q))
        eye6 = np.eye(6)

        def task_error(config):
            pose, jac = self.fk_and_jacobian(config)
            pos_err = target[:3, 3] - pose[:3, 3]
            rot_err = axis_angle(target[:3, :3] @ pose[:3, :3].T)
            return pos_err, rot_err, jac

        pos_err, rot_err, jac = task_error(q)
        cost = np.linalg.norm(pos_err) + 0.1 * np.linalg.norm(rot_err)
        lam = damping
        stalled = 0

        for _ in range(max_iterations):
            if np.linalg.norm(pos_err) < position_tolerance \
                    and np.linalg.norm(rot_err) < rotation_tolerance:
                return q, True

            # Cap the task error so a far-away target cannot request a huge,
            # wildly non-linear joint step.
            error = np.concatenate([
                pos_err * min(1.0, 0.08 / max(np.linalg.norm(pos_err), 1e-9)),
                rot_err * min(1.0, 0.35 / max(np.linalg.norm(rot_err), 1e-9)),
            ])
            pinv = jac.T @ np.linalg.solve(jac @ jac.T + (lam ** 2) * eye6, eye6)
            delta = pinv @ error + (identity - pinv @ jac) @ (nullspace_gain * (bias - q))
            norm = np.linalg.norm(delta)
            if norm > 0.5:
                delta *= 0.5 / norm

            improved = False
            for _ in range(4):  # backtracking line search
                trial = np.clip(q + delta, lower, upper)
                t_pos, t_rot, t_jac = task_error(trial)
                trial_cost = np.linalg.norm(t_pos) + 0.1 * np.linalg.norm(t_rot)
                if trial_cost < cost:
                    if cost - trial_cost < 1e-7:
                        stalled += 1
                    else:
                        stalled = 0
                    q, pos_err, rot_err, jac, cost = trial, t_pos, t_rot, t_jac, trial_cost
                    lam = max(damping, lam * 0.7)
                    improved = True
                    break
                delta *= 0.5
            if not improved:
                lam *= 2.5
                stalled += 1
                if lam > 5.0 or stalled > 12:
                    break
            elif stalled > 12:
                break

        return q, bool(np.linalg.norm(pos_err) < position_tolerance
                       and np.linalg.norm(rot_err) < rotation_tolerance)

    def ik_with_restarts(self, target: np.ndarray, seed=None, *, attempts: int = 6,
                         rng=None, **kwargs):
        """:meth:`ik` with random restarts.  Returns ``(q, converged)``."""
        rng = rng if rng is not None else np.random.default_rng(0)
        lower, upper = self.limits
        seeds = [FR3_READY if seed is None else np.asarray(seed, dtype=float)]
        if seed is not None:
            seeds.append(FR3_READY)
        best = None
        for attempt in range(attempts):
            candidate = (seeds[attempt] if attempt < len(seeds)
                         else rng.uniform(lower + 0.2, upper - 0.2))
            q, converged = self.ik(target, candidate, **kwargs)
            if converged:
                return q, True
            if best is None:
                best = q
        return best, False


# --------------------------------------------------------------------------- #
# Cheap feasibility checks (no collision model available in this workspace)
# --------------------------------------------------------------------------- #

def check_configuration(chain: Chain, q, *, limit_margin: float = 0.1,
                        min_link_height: float = 0.05,
                        base_radius: float = 0.20,
                        base_height: float = 0.35) -> list:
    """Return a list of human-readable reasons why *q* is unusable (empty == ok).

    There is no collision geometry available here (no MoveIt/SRDF on the Gazebo
    side), so this is a deliberately conservative geometric screen: joint limits
    with margin, nothing dipping into the floor, and the wrist staying out of the
    cylinder occupied by the robot's own base.
    """
    reasons = []
    q = np.asarray(q, dtype=float)
    lower, upper = chain.limits
    if np.any(q < lower + limit_margin) or np.any(q > upper - limit_margin):
        bad = np.where((q < lower + limit_margin) | (q > upper - limit_margin))[0]
        reasons.append('joint limit margin: ' + ','.join(chain.joint_names[i] for i in bad))

    frames = chain.link_frames(q)
    for link, transform in frames.items():
        if link in (chain.base_link, 'fr3_link0'):
            continue
        if transform[2, 3] < min_link_height:
            reasons.append(f'{link} below z={min_link_height:.2f} m')

    tip = frames[chain.tip_link][:3, 3]
    if math.hypot(tip[0], tip[1]) < base_radius and tip[2] < base_height:
        reasons.append('tip inside base cylinder')
    return reasons
