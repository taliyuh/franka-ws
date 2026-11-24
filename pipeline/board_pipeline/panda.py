"""Franka Panda kinematics (DH), pose vectors and a small IK, independent of ROS.

The Panda and the FR3 share these DH parameters (Franka's published table);
only the joint limits differ. The model reproduces the URDF FK of
franka_description's panda.urdf.xacro (hand:=true) to machine precision
(checked against the Panda URDF in the container), and of the FR3 datasets it
was taken from (calibration_diagnostics/solve_eye_in_hand.py).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
D = [0.333, 0, 0.316, 0, 0.384, 0, 0]
ALPHA = [0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2]
# franka_description/robots/panda/joint_limits.yaml (Panda datasheet limits).
LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
READY = np.array([0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4])
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def dh(a, d, alpha, theta):
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def fk(q, tip):
    """Panda FK from panda_link0 to panda_link6 or panda_hand_tcp (Franka Hand)."""
    t = np.eye(4)
    for i in range(6):
        t = t @ dh(A[i], D[i], ALPHA[i], q[i])
    if tip == "panda_link6":
        return t
    t = t @ dh(A[6], D[6], ALPHA[6], q[6]) @ dh(0, 0.107, 0, 0) @ dh(0, 0, 0, -math.pi / 4)
    t[:, 3] = t @ np.array([0, 0, 0.1034, 1])
    return t


def frame_origins(q):
    """Base-frame positions of link origins 1..6, flange and hand TCP."""
    t, out = np.eye(4), []
    for i in range(7):
        t = t @ dh(A[i], D[i], ALPHA[i], q[i])
        out.append(t[:3, 3].copy())
    out.append((t @ dh(0, 0.107, 0, 0))[:3, 3])
    out.append(fk(q, "panda_hand_tcp")[:3, 3])
    return np.array(out)


def to_vec(t):
    return np.r_[cv2.Rodrigues(t[:3, :3])[0].ravel(), t[:3, 3]]


def from_vec(v):
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(np.asarray(v[:3], float))[0]
    t[:3, 3] = v[3:]
    return t


def arm_angle(q) -> float | None:
    """Elbow arm angle psi (rad, 0 = elbow straight up = J3 near 0), matching rl_manip arm_angle.h.
    Returns None if the shoulder-wrist line is within 18 deg of vertical or elbow within 5 cm of it."""
    o = frame_origins(q)
    s, e, w = o[1], o[3], o[5]
    sw = w - s
    n_sw = float(np.linalg.norm(sw))
    if n_sw < 1e-6:
        return None
    u = sw / n_sw
    if abs(u[2]) > 0.95:
        return None
    v = np.array([0.0, 0.0, 1.0]) - u[2] * u
    v /= float(np.linalg.norm(v))
    el = (e - s) - float(np.dot(e - s, u)) * u
    if float(np.linalg.norm(el)) < 0.05:
        return None
    return math.atan2(float(np.dot(u, np.cross(v, el))), float(np.dot(v, el)))


def ik(target, seed, tip="panda_hand_tcp", arm_angle_rad: float | None = None):
    """Joint angles reaching `target` (4x4) from `seed`; returns (q, max 6D pose residual).
    With arm_angle_rad (e.g. 0.0), uses the 7th DoF to also hold the elbow arm angle there."""
    def err(q):
        t = fk(q, tip)
        base = np.r_[(t[:3, 3] - target[:3, 3]) * 10,
                     Rotation.from_matrix(target[:3, :3].T @ t[:3, :3]).as_rotvec(),
                     1e-5 * (q - seed)]
        if arm_angle_rad is None:
            return base
        psi = arm_angle(q)
        psi_err = 0.0 if psi is None else math.remainder(psi - arm_angle_rad, 2 * math.pi)
        return np.r_[base, 0.05 * psi_err]
    solution = least_squares(err, np.clip(seed, LOWER, UPPER), bounds=(LOWER, UPPER))
    return solution.x, float(np.abs(err(solution.x)[:6]).max())


def ik_link6(target, seed):
    """IK for panda_link6 with joint 7 held at the seed (it does not move link6)."""
    def err(x):
        t = fk(np.r_[x, seed[6]], "panda_link6")
        rot = Rotation.from_matrix(target[:3, :3].T @ t[:3, :3]).as_rotvec()
        return np.r_[(t[:3, 3] - target[:3, 3]) * 10, rot]
    solution = least_squares(err, seed[:6], bounds=(LOWER[:6], UPPER[:6]))
    return np.r_[solution.x, seed[6]], float(np.max(np.abs(err(solution.x))))
