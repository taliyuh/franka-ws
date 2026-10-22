"""FR3 kinematics (DH), pose vectors and a small IK, independent of ROS.

The DH model reproduces the URDF FK of the stored datasets exactly (checked in
calibration_diagnostics/solve_eye_in_hand.py, where this was taken from).
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
LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
READY = np.array([0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4])
JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]


def dh(a, d, alpha, theta):
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def fk(q, tip):
    """FR3 FK from base to fr3_link6 or fr3_hand_tcp (franka_hand)."""
    t = np.eye(4)
    for i in range(6):
        t = t @ dh(A[i], D[i], ALPHA[i], q[i])
    if tip == "fr3_link6":
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
    out.append(fk(q, "fr3_hand_tcp")[:3, 3])
    return np.array(out)


def to_vec(t):
    return np.r_[cv2.Rodrigues(t[:3, :3])[0].ravel(), t[:3, 3]]


def from_vec(v):
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(np.asarray(v[:3], float))[0]
    t[:3, 3] = v[3:]
    return t


def ik(target, seed, tip="fr3_hand_tcp"):
    """Joint angles reaching `target` (4x4) from `seed`; returns (q, max residual)."""
    def err(q):
        t = fk(q, tip)
        return np.r_[(t[:3, 3] - target[:3, 3]) * 10,
                     Rotation.from_matrix(target[:3, :3].T @ t[:3, :3]).as_rotvec(),
                     1e-5 * (q - seed)]
    solution = least_squares(err, seed, bounds=(LOWER, UPPER))
    return solution.x, float(np.abs(err(solution.x)[:6]).max())


def ik_link6(target, seed):
    """IK for fr3_link6 with joint 7 held at the seed (it does not move link6)."""
    def err(x):
        t = fk(np.r_[x, seed[6]], "fr3_link6")
        rot = Rotation.from_matrix(target[:3, :3].T @ t[:3, :3]).as_rotvec()
        return np.r_[(t[:3, 3] - target[:3, 3]) * 10, rot]
    solution = least_squares(err, seed[:6], bounds=(LOWER[:6], UPPER[:6]))
    return np.r_[solution.x, seed[6]], float(np.max(np.abs(err(solution.x))))
