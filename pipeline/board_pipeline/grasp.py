"""The peg held in the printed V-block fingers and the TCP pose that puts it on a hole.

The fingers (franka_fr3_fingers v2) hold every peg off the TCP axis, at a
shape-specific offset and yaw (config/grasps.yaml). For a hole, the held peg
is the one of the hole's shape. Its frame is placed on the hole frame, turned
about the peg axis by a multiple of the peg's symmetry period (60 deg for the
hexagon, 120 for the triangle, 180 for the rectangle, any angle for the round
peg, none for the L), and the TCP follows:

    T_board_tcp = T_board_hole @ Rz(turn) @ T_peg_tcp

All the turns put the same peg into the same hole; they differ only in where
the wrist ends up. `choose_turn` solves the IK for each and keeps the one that
turns joint 7 least from where the arm stands, among those that leave joint 7
at least WRIST_MARGIN_RAD inside its limits.

probe_board takes a tool as an orientation in the board frame plus a TCP ->
tip vector in TCP axes; `write_manipulation_config` writes both for one hole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from board_pipeline import panda
from board_pipeline.config import PipelineError, load_yaml

DEFAULT_GRASPS = "pipeline/config/grasps.yaml"
FLIP = np.diag([1.0, -1.0, -1.0])  # Rx(180 deg): the TCP looks down the peg
SURFACE_Z = 0.025          # board top surface, board frame (holes.yaml surface_z_m)
# Joint 7 must stay this far inside its +-2.8973 rad limits at the hole: the
# guarded descent and the Cartesian corrections move it a little, and at
# -2.85 rad the robot stopped with joint_motion_generator_position_limits_violation.
WRIST_MARGIN_RAD = 0.35
ROUND_STEP_DEG = 10.0      # the round peg may turn by any angle; candidates this far apart


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Tool:
    shape: str
    rotation: np.ndarray    # panda_hand_tcp orientation in the board frame
    tip_offset: np.ndarray  # TCP -> peg tip (peg frame origin), TCP axes, m
    point: np.ndarray       # board x, y of the peg tip target (centred in the hole), m
    peg_yaw: float          # peg frame yaw in the board frame, rad
    grasp_width_mm: float
    turn: float = 0.0       # turn about the peg axis from the hole frame, rad
    period: float = 2 * math.pi  # symmetry period of the peg, rad (0 = any turn)
    joints: np.ndarray | None = None  # predicted arm joints at the hole (choose_turn)

    def tcp_pose(self, board: np.ndarray, z: float) -> np.ndarray:
        """Base-frame TCP pose with the peg tip at board height z."""
        tip = board @ np.r_[self.point, z, 1.0]
        target = np.eye(4)
        target[:3, :3] = board[:3, :3] @ self.rotation
        target[:3, 3] = tip[:3] - target[:3, :3] @ self.tip_offset
        return target


def hole_tool(cfg, hole: dict, name: str, turn: float | None = None) -> Tool:
    """The held peg on this hole, turned by `turn` (rad, a multiple of its period).

    Without `turn` the peg is turned the multiple of its period that brings the
    TCP yaw closest to the configured tcp_orientation_board_rpy; that is only a
    starting point: with a board pose, `choose_turn` picks the turn by joint 7.
    """
    # Sessions created before the fingers were swapped have no `grasps` entry.
    source = cfg.data.get("grasps", DEFAULT_GRASPS)
    grasps = load_yaml(cfg.path(source))
    try:
        shape = grasps["shapes"][hole["shape"]]
        centre = grasps["holes"][name]
    except KeyError as error:
        raise PipelineError(f"{source} has no grasp for {name} ({hole['shape']}); "
                            "regenerate it with pipeline/tools/extract_grasps.py") from error
    tcp = shape["tcp"]
    r_peg_tcp = _rz(math.radians(tcp["yaw_deg"]+180.0)) @ FLIP
    t_peg_tcp = np.array([tcp["x_m"], tcp["y_m"], tcp["z_m"]])
    hole_yaw = math.radians(hole["yaw_deg"])
    period = math.radians(shape["yaw_period_deg"])

    if turn is None:
        # Only the yaw of the configured orientation is used: the peg must stay
        # vertical to go in.
        probe = load_yaml(cfg.path(cfg["robot"]["manipulation_config"]))["probe"]
        reference = Rotation.from_euler("xyz", probe["tcp_orientation_board_rpy"]).as_matrix()
        natural = _rz(hole_yaw) @ r_peg_tcp
        turn = math.remainder(math.atan2(reference[1, 0], reference[0, 0])
                              - math.atan2(natural[1, 0], natural[0, 0]), 2 * math.pi)
        if period > 0:
            turn = period * round(turn / period)
    elif period > 0 and abs(math.remainder(turn, period)) > 1e-9:
        raise PipelineError(f"a {hole['shape']} peg turned by {math.degrees(turn):.2f} deg does "
                            f"not fit its hole (symmetry period {shape['yaw_period_deg']} deg)")
    peg_yaw = hole_yaw + turn

    point = np.array([hole["x_m"], hole["y_m"]]) \
        + _rz(hole_yaw)[:2, :2] @ np.array([centre["x_m"], centre["y_m"]])
    return Tool(shape=hole["shape"], rotation=_rz(peg_yaw) @ r_peg_tcp,
                tip_offset=-r_peg_tcp.T @ t_peg_tcp, point=point,
                peg_yaw=peg_yaw, grasp_width_mm=float(shape["grasp_width_mm"]),
                turn=turn, period=period)


def symmetric_turns(period: float) -> list[float]:
    """The turns about the peg axis that fit the hole, in [-pi, pi]."""
    step = period if period > 0 else math.radians(ROUND_STEP_DEG)
    count = max(1, round(2 * math.pi / step))
    return sorted({round(math.remainder(k * step, 2 * math.pi), 12) for k in range(count)})


def wrist_margin(q7: float) -> float:
    """Distance of joint 7 from its nearer limit, rad."""
    return float(min(q7 - panda.LOWER[6], panda.UPPER[6] - q7))


def _solve(target: np.ndarray, seeds: list[np.ndarray], near: np.ndarray) -> np.ndarray | None:
    """IK from each seed in turn; the converged solution nearest `near`, None if none."""
    good = [q for q, residual in (panda.ik(target, np.clip(seed, panda.LOWER, panda.UPPER))
                                  for seed in seeds) if residual < 1e-4]
    return min(good, key=lambda q: float(np.abs(q - near).max())) if good else None


def _predicted_joints(tools: list[Tool], board: np.ndarray, start: np.ndarray) -> list:
    """Arm joints at the hole for each turn, as probe_board's IK (seeded at `start`) finds them.

    Joint 7 turns about the TCP z axis, so turning the peg by `turn` about the
    board normal moves joint 7 by about turn * (TCP z . board z) and leaves the
    other joints nearly where they were. The turn nearest 0 is solved from a
    spread of seeds; every other turn starts from that solution with joint 7
    pre-turned. A turn that would take joint 7 past its limit is unreachable
    without reconfiguring the whole arm and gets None; the spread is retried
    only when the pre-turned seed does not converge.
    """
    spread = [np.r_[base[:6], base[6] + offset] for base in (start, panda.READY)
              for offset in (0.0, -math.pi / 2, math.pi / 2, -math.pi, math.pi)]
    order = sorted(range(len(tools)), key=lambda i: abs(tools[i].turn))
    first = tools[order[0]]
    anchor = _solve(first.tcp_pose(board, SURFACE_Z), spread, start)
    joints: list = [None] * len(tools)
    joints[order[0]] = anchor
    for i in order[1:]:
        tool = tools[i]
        target = tool.tcp_pose(board, SURFACE_Z)
        if anchor is None:
            joints[i] = _solve(target, spread, start)
            continue
        seed = anchor.copy()
        seed[6] += (tool.turn - first.turn) * tool.rotation[2, 2]
        if not panda.LOWER[6] <= seed[6] <= panda.UPPER[6]:
            continue
        q = _solve(target, [seed], seed)
        joints[i] = q if q is not None else _solve(target, spread, seed)
    return joints


def choose_turn(cfg, hole: dict, name: str, board: np.ndarray,
                start: np.ndarray | None = None) -> tuple[Tool, list[tuple[Tool, np.ndarray | None]]]:
    """The symmetric turn of the held peg that turns the wrist least.

    `board` is T_base_board, `start` the arm joints the motion starts from
    (None: unknown, then joint 7 is kept as close to mid-range as possible).
    Every turn that fits the hole is solved with IK; of those that keep
    joint 7 at least WRIST_MARGIN_RAD inside its limits, the one with the
    smallest joint 7 travel from `start` wins. Returns the chosen tool and
    every candidate with its predicted joints (None: no IK solution).
    """
    base = start if start is not None else panda.READY
    reference = float(base[6]) if start is not None else 0.0
    period = hole_tool(cfg, hole, name).period
    tools = [hole_tool(cfg, hole, name, turn=turn) for turn in symmetric_turns(period)]
    candidates = list(zip(tools, _predicted_joints(tools, board, base)))
    solved = [(tool, q) for tool, q in candidates if q is not None]
    if not solved:
        raise PipelineError(f"no IK solution over {name} for any turn of the {hole['shape']} peg")
    safe = [(tool, q) for tool, q in solved if wrist_margin(q[6]) >= WRIST_MARGIN_RAD]
    if safe:
        tool, q = min(safe, key=lambda c: abs(c[1][6] - reference))
    else:
        tool, q = max(solved, key=lambda c: wrist_margin(c[1][6]))
        print(f"warning: no turn of the {hole['shape']} peg keeps joint 7 "
              f"{WRIST_MARGIN_RAD:.2f} rad inside its limits at {name}; the best leaves "
              f"{wrist_margin(q[6]):.2f} rad. Turn the board towards the robot's mid-range wrist.")
    return replace(tool, joints=q), candidates


def write_manipulation_config(cfg, tool: Tool, out: Path) -> Path:
    """The robot manipulation config with the probe tool replaced by this peg."""
    data = load_yaml(cfg.path(cfg["robot"]["manipulation_config"]))
    data["probe"]["tcp_orientation_board_rpy"] = \
        [float(v) for v in Rotation.from_matrix(tool.rotation).as_euler("xyz")]
    data["probe"]["pen_tip_offset_tcp_m"] = [float(v) for v in tool.tip_offset]
    out.parent.mkdir(exist_ok=True)
    out.write_text(f"# {tool.shape} peg in the V-block fingers; written by `pipeline check/probe`\n"
                   + yaml.safe_dump(data, sort_keys=False))
    return out
