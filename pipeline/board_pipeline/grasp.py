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

The tip comes from the CAD grasp unless `pipeline check/probe --tip-from FILE`
passes a pivot_calibration result measured on the same peg shape;
`apply_pivot_tip` swaps it in after comparing it with the CAD tip. Without
--tip-from, a measured tip kept in the session for the shape
(<session>/tips/<shape>.yaml, same format) is used; `--tip-from cad` forces
the CAD tip (`resolve_tip`).
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
# 0.25 rad (~14.3 deg): keeps l_shape_c1mm (0.30 rad margin) at arm angle 0 while
# still triggering the elbow fallback on l_shape_c3mm (0.11 rad margin).
WRIST_MARGIN_RAD = 0.25
ROUND_STEP_DEG = 10.0      # the round peg may turn by any angle; candidates this far apart
# Elbow fallback (choose_turn): when no peg turn keeps joint 7 WRIST_MARGIN_RAD inside its limits with the elbow
# arm angle at 0 (the L: one turn only), the elbow is turned, smallest first, in these steps up to the maximum.
# Turning the elbow by psi moves joint 7 by about 1.3 psi at the same hand pose (2026-11-25 sweep over the L holes),
# but the robot's kinematic error moves the reported tip about 0.1 mm per degree of arm angle
# (rl_manip robot_session/POSTURE_FINDINGS.md): the smallest turn that works is taken.
ELBOW_STEP_DEG = 5.0
ELBOW_MAX_DEG = 45.0
ELBOW_TOLERANCE_RAD = 0.05  # a solution this close to the asked arm angle is at it (as _solve)


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
    arm_angle: float = 0.0  # elbow arm angle psi at the hole, rad (choose_turn's elbow fallback; 0 otherwise)

    def tcp_pose(self, board: np.ndarray, z: float) -> np.ndarray:
        """Base-frame TCP pose with the peg tip at board height z."""
        tip = board @ np.r_[self.point, z, 1.0]
        target = np.eye(4)
        target[:3, :3] = board[:3, :3] @ self.rotation
        target[:3, 3] = tip[:3] - target[:3, :3] @ self.tip_offset
        return target


def _grasps(cfg) -> tuple[str, dict]:
    # Sessions created before the fingers were swapped have no `grasps` entry.
    source = cfg.data.get("grasps", DEFAULT_GRASPS)
    return source, load_yaml(cfg.path(source))


def grasp_width_mm(cfg, shape: str) -> float:
    """Finger width with the peg of this shape seated in the V-block, mm."""
    source, grasps = _grasps(cfg)
    shapes = grasps["shapes"]
    if shape not in shapes:
        raise PipelineError(f"no peg shape {shape!r} in {source}; the shapes are {', '.join(shapes)}")
    return float(shapes[shape]["grasp_width_mm"])


def mirror_seat(tcp: dict, centre: dict) -> tuple[dict, dict]:
    """The seat of the mirror image of the CAD peg (2026-11-23: the real rectangle and L pegs are the CAD's mirror
    images). Mirrored through the peg frame's xz plane (y -> -y; the frame's x, the longest flat's normal, and the
    centroid stay) with the fingers mirrored through the hand's yz plane (they are symmetric about it): T_peg_tcp' =
    M_peg T_peg_tcp M_hand, i.e. x kept, y negated, yaw -> 180 deg - yaw; the L's centred origin y negated too."""
    return ({**tcp, "y_m": -float(tcp["y_m"]), "yaw_deg": 180.0 - float(tcp["yaw_deg"])},
            {**centre, "y_m": -float(centre["y_m"])})


def hole_tool(cfg, hole: dict, name: str, turn: float | None = None,
              extra_turn_deg: float = 0.0, mirror: bool = False) -> Tool:
    """The held peg on this hole, turned by `turn` (rad, a multiple of its period).

    Without `turn` the peg is turned the multiple of its period that brings the
    TCP yaw closest to the configured tcp_orientation_board_rpy; that is only a
    starting point: with a board pose, `choose_turn` picks the turn by joint 7.
    `extra_turn_deg` turns the hand further about the peg axis (`--turn`, also a
    multiple of the period).
    """
    source, grasps = _grasps(cfg)
    try:
        shape = grasps["shapes"][hole["shape"]]
        centre = grasps["holes"][name]
    except KeyError as error:
        raise PipelineError(f"{source} has no grasp for {name} ({hole['shape']}); "
                            "regenerate it with pipeline/tools/extract_grasps.py") from error
    tcp = shape["tcp"]
    if mirror:
        tcp, centre = mirror_seat(tcp, centre)
    r_peg_tcp = _rz(math.radians(tcp["yaw_deg"])) @ FLIP
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
    if extra_turn_deg:
        extra = math.radians(extra_turn_deg)
        if period > 0 and abs(math.remainder(extra, period)) > 1e-6:
            raise PipelineError(f"the {hole['shape']} peg only turns by multiples of "
                                f"{shape['yaw_period_deg']:g} deg, not {extra_turn_deg:g}")
        turn = math.remainder(turn + extra, 2 * math.pi)
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


def at_arm_angle(q: np.ndarray, arm_angle_rad: float) -> bool:
    """Whether q has the elbow at arm_angle_rad (an undefined arm angle counts only for 0, as peg_search's)."""
    psi = panda.arm_angle(q)
    if psi is None:
        return abs(arm_angle_rad) < ELBOW_TOLERANCE_RAD
    return abs(math.remainder(psi - arm_angle_rad, 2 * math.pi)) < ELBOW_TOLERANCE_RAD


def _solve(target: np.ndarray, seeds: list[np.ndarray], near: np.ndarray,
           arm_angle_rad: float | None = 0.0, strict: bool = False) -> np.ndarray | None:
    """IK from each seed in turn; prefers solutions at `arm_angle_rad` (default 0: elbow up, matching peg_search's
    posture rule) with >= 0.15 rad joint margin on J1-6, falling back to unconstrained IK if none exist (strict: no
    fallback, None)."""
    if arm_angle_rad is not None:
        psi_seeds = [np.r_[s[:2], 0.0, s[3:]] for s in seeds]
        good_psi = []
        for seed in psi_seeds:
            q, residual = panda.ik(target, np.clip(seed, panda.LOWER, panda.UPPER), arm_angle_rad=arm_angle_rad)
            psi = panda.arm_angle(q)
            margin16 = float(np.min(np.minimum(q[:6] - panda.LOWER[:6], panda.UPPER[:6] - q[:6])))
            if residual < 1e-4 and psi is not None and abs(math.remainder(psi - arm_angle_rad, 2 * math.pi)) < 0.05 \
                    and margin16 >= 0.15:
                good_psi.append(q)
        if good_psi:
            return min(good_psi, key=lambda q: float(np.abs(q - near).max()))
        if strict:
            return None
    good = [q for q, residual in (panda.ik(target, np.clip(seed, panda.LOWER, panda.UPPER))
                                  for seed in seeds) if residual < 1e-4]
    return min(good, key=lambda q: float(np.abs(q - near).max())) if good else None


def _predicted_joints(tools: list[Tool], board: np.ndarray, start: np.ndarray,
                      arm_angle_rad: float = 0.0, strict: bool = False) -> list:
    """Arm joints at the hole for each turn, nearest `start`; probe_board's IK is seeded with them.
    arm_angle_rad, strict: the elbow wanted and whether a solution elsewhere is refused (_solve).

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
    elbow = {"arm_angle_rad": arm_angle_rad, "strict": strict}
    anchor = _solve(first.tcp_pose(board, SURFACE_Z), spread, start, **elbow)
    joints: list = [None] * len(tools)
    joints[order[0]] = anchor
    for i in order[1:]:
        tool = tools[i]
        target = tool.tcp_pose(board, SURFACE_Z)
        if anchor is None:
            joints[i] = _solve(target, spread, start, **elbow)
            continue
        seed = anchor.copy()
        seed[6] += (tool.turn - first.turn) * tool.rotation[2, 2]
        if not panda.LOWER[6] <= seed[6] <= panda.UPPER[6]:
            continue
        q = _solve(target, [seed], seed, **elbow)
        joints[i] = q if q is not None else _solve(target, spread, seed, **elbow)
    return joints


def elbow_steps(step_deg: float = ELBOW_STEP_DEG, max_deg: float = ELBOW_MAX_DEG) -> list[float]:
    """The elbow fallback's arm angles, rad, smallest first, each size both ways: 5, -5, 10, -10, ... deg."""
    count = int(math.floor(max_deg / step_deg + 1e-9))
    return [math.radians(sign * k * step_deg) for k in range(1, count + 1) for sign in (1, -1)]


def choose_turn(cfg, hole: dict, name: str, board: np.ndarray,
                start: np.ndarray | None = None, mirror: bool = False) -> tuple[Tool, list[tuple[Tool, np.ndarray | None]]]:
    """The symmetric turn of the held peg that turns the wrist least.

    `board` is T_base_board, `start` the arm joints the motion starts from
    (None: unknown, then joint 7 is kept as close to mid-range as possible).
    Every turn that fits the hole is solved with IK; of those that keep
    joint 7 at least WRIST_MARGIN_RAD inside its limits, the one with the
    smallest joint 7 travel from `start` wins. Returns the chosen tool and
    every candidate with its predicted joints (None: no IK solution).

    All of that is with the elbow arm angle at 0. Only when no turn is safe
    there (the L, which has one turn, on some board yaws) is the elbow turned:
    the arm angles of `elbow_steps`, smallest first, until one makes a turn
    safe; the tool then carries that arm_angle (peg_search holds it, the
    joint-offset correction is evaluated there) and the candidates are that
    arm angle's. When none does, the elbow stays at 0 with the warning.
    """
    base = start if start is not None else panda.READY
    reference = float(base[6]) if start is not None else 0.0
    period = hole_tool(cfg, hole, name, mirror=mirror).period
    tools = [hole_tool(cfg, hole, name, turn=turn, mirror=mirror) for turn in symmetric_turns(period)]
    candidates = list(zip(tools, _predicted_joints(tools, board, base)))
    solved = [(tool, q) for tool, q in candidates if q is not None]
    if not solved:
        raise PipelineError(f"no IK solution over {name} for any turn of the {hole['shape']} peg")
    safe = [(tool, q) for tool, q in solved if wrist_margin(q[6]) >= WRIST_MARGIN_RAD and at_arm_angle(q, 0.0)]
    if safe:
        tool, q = min(safe, key=lambda c: abs(c[1][6] - reference))
        return replace(tool, joints=q), candidates
    best0 = max(solved, key=lambda c: wrist_margin(c[1][6]))
    for psi in elbow_steps():
        bent = [replace(t, arm_angle=psi) for t in tools]
        found = list(zip(bent, _predicted_joints(bent, board, base, arm_angle_rad=psi, strict=True)))
        safe = [(t, q) for t, q in found if q is not None and wrist_margin(q[6]) >= WRIST_MARGIN_RAD]
        if safe:
            tool, q = min(safe, key=lambda c: abs(c[1][6] - reference))
            print(f"elbow fallback at {name}: with the elbow at 0 no turn of the {hole['shape']} peg keeps joint 7 "
                  f"{WRIST_MARGIN_RAD:.2f} rad inside its limits (the best leaves {wrist_margin(best0[1][6]):.2f} "
                  f"rad); arm angle {math.degrees(psi):+.0f} deg leaves {wrist_margin(q[6]):.2f} rad. The reported "
                  f"tip there is off by about 0.1 mm per degree of arm angle (POSTURE_FINDINGS.md)")
            return replace(tool, joints=q), found
    tool, q = best0
    print(f"warning: no turn of the {hole['shape']} peg keeps joint 7 "
          f"{WRIST_MARGIN_RAD:.2f} rad inside its limits at {name}, with the elbow at 0 or turned up to "
          f"+-{ELBOW_MAX_DEG:g} deg; the best leaves {wrist_margin(q[6]):.2f} rad. Turn the board towards the "
          "robot's mid-range wrist.")
    return replace(tool, joints=q), candidates


def turn_further(cfg, hole: dict, name: str, tool: Tool, board: np.ndarray, extra_turn_deg: float) -> Tool:
    """`tool` (from choose_turn) turned `extra_turn_deg` further about the peg axis (`--turn`).

    Its joints are predicted from the chosen ones with joint 7 pre-turned; a turn
    that leaves joint 7 near its limit is only warned about: it was asked for.
    """
    turned = replace(hole_tool(cfg, hole, name, turn=tool.turn, extra_turn_deg=extra_turn_deg),
                     arm_angle=tool.arm_angle)
    joints = None
    if tool.joints is not None:
        seed = tool.joints.copy()
        seed[6] += math.remainder(turned.turn - tool.turn, 2 * math.pi) * turned.rotation[2, 2]
        joints = _solve(turned.tcp_pose(board, SURFACE_Z), [seed], seed, arm_angle_rad=tool.arm_angle)
    if joints is None:
        print(f"warning: --turn {extra_turn_deg:g}: no IK solution near the chosen wrist; "
              "joint 7 may have to swing a long way")
    else:
        print(f"--turn {extra_turn_deg:g}: j7 {joints[6]:+.2f} rad ({wrist_margin(joints[6]):.2f} rad "
              "from its limit)")
        if wrist_margin(joints[6]) < WRIST_MARGIN_RAD:
            print(f"warning: that is less than {WRIST_MARGIN_RAD:.2f} rad; the move may stop on "
                  "joint_motion_generator_position_limits_violation")
    return replace(turned, joints=joints)


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


# A measured tip further than this from the CAD one, in distance from the TCP
# axis or along it, is a wrong file or peg rather than a CAD error.
PIVOT_MAX_DEVIATION_M = 0.005


@dataclass(frozen=True)
class PivotTip:
    shape: str
    tip: np.ndarray         # TCP -> tool tip, TCP axes, m (pivot_calibration's tip_in_tcp_m)
    path: Path


def load_pivot_tip(path: Path) -> PivotTip:
    data = load_yaml(path)
    if data.get("frame") != "panda_hand_tcp" or "tip_in_tcp_m" not in data:
        raise PipelineError(f"{path} is not a pivot_calibration result")
    # Results from before the shape was recorded were all measured with the round pointer.
    return PivotTip(shape=data.get("shape", "round"),
                    tip=np.asarray(data["tip_in_tcp_m"], dtype=float), path=Path(path))


def apply_pivot_tip(tool: Tool, pivot: PivotTip) -> tuple[Tool, str]:
    """The tool with the measured tip when it was measured on this peg shape, and where the tip came from."""
    if pivot.shape != tool.shape:
        return tool, f"CAD; {pivot.path.name} is for the {pivot.shape} peg"
    cad, measured = tool.tip_offset, pivot.tip
    # Compare in a way a turn about the TCP axis does not change: the grasp yaw
    # is itself a CAD value the measurement may disagree with.
    radius_cad, radius = np.linalg.norm(cad[:2]), np.linalg.norm(measured[:2])
    if abs(radius - radius_cad) > PIVOT_MAX_DEVIATION_M or abs(measured[2] - cad[2]) > PIVOT_MAX_DEVIATION_M:
        raise PipelineError(
            f"{pivot.path.name}: tip {np.round(measured * 1e3, 2)} mm is {radius * 1e3:.1f} mm from the "
            f"TCP axis and {measured[2] * 1e3:.1f} mm along it; the CAD {tool.shape} peg has "
            f"{radius_cad * 1e3:.1f} and {cad[2] * 1e3:.1f} mm. Wrong file, or not the {tool.shape} peg?")
    source = (f"measured, {pivot.path.name}; CAD {np.round(cad * 1e3, 2)} mm, "
              f"{np.linalg.norm(measured - cad) * 1e3:.2f} mm apart")
    if radius > 0.001 and radius_cad > 0.001:
        turn = math.degrees(abs(math.remainder(
            math.atan2(measured[1], measured[0]) - math.atan2(cad[1], cad[0]), 2 * math.pi)))
        if turn > 20.0:
            source += (f"; the measured tip is turned {turn:.0f} deg about the TCP axis from the CAD "
                       "one, so the grasp yaw in grasps.yaml / grasp.py looks off")
    return replace(tool, tip_offset=measured), source


def session_tip_path(session_path: Path, shape: str) -> Path:
    """Where a session keeps the measured tip of a peg shape (used without --tip-from)."""
    return Path(session_path) / "tips" / f"{shape}.yaml"


def resolve_tip(tool: Tool, tip_from: str | None, session_path: Path | None, host_path=None) -> tuple[Tool, str]:
    """The tool's tip: --tip-from FILE (host_path maps a container path), `--tip-from cad` the CAD tip, else the
    session's measured tip of this shape (session_tip_path) when there is one, else CAD."""
    if tip_from and tip_from.lower() == "cad":
        return tool, "CAD (--tip-from cad)"
    if tip_from:
        return apply_pivot_tip(tool, load_pivot_tip(host_path(tip_from) if host_path else Path(tip_from)))
    if session_path is not None:
        path = session_tip_path(session_path, tool.shape)
        if path.exists():
            tool, source = apply_pivot_tip(tool, load_pivot_tip(path))
            return tool, f"{source}, the session's tips/{path.name}"
    return tool, "CAD"


# ----------------------------------------------------------------- neighbor clearance (v3 fingers)
# Lower grip section (Z = 34..54 mm, the part that descends below peg tops) in hand/TCP axes, mm:
#   Finger A (V-block, +Y): x in [-16.8, +16.8], y in [q, q + 30.0]
#   Finger B (pusher,  -Y): x in [-10.0, +10.0], y in [-(q + 30.0), -q + 10.0]   (q = width / 2)
FINGER_A_HALF_X_MM = 16.8
FINGER_A_DEPTH_MM = 30.0
FINGER_B_HALF_X_MM = 10.0
FINGER_B_DEPTH_MM = 30.0
FINGER_B_TONGUE_MM = 10.0

# Neighbor peg 2D footprints in each hole's own frame (mm; x = outward normal of longest flat).
# Exact convex polygons for rectangle (16x12) and hexagon (17 AF); circumcircles for the rest.
_RECT_POLY_MM = np.array([[-6.0, -8.0], [6.0, -8.0], [6.0, 8.0], [-6.0, 8.0]], dtype=float)
_HEX_R_MM = 17.0 / math.sqrt(3.0)
_HEX_POLY_MM = np.array(
    [[_HEX_R_MM * math.cos(math.radians(30.0 + 60.0 * k)),
      _HEX_R_MM * math.sin(math.radians(30.0 + 60.0 * k))] for k in range(6)],
    dtype=float,
)
_PEG_RADIUS_MM = {"round": 6.0, "hexagon": _HEX_R_MM, "triangle": 12.0, "rectangle": 10.0, "l_shape": 14.0}
_PEG_POLYS_MM = {"rectangle": _RECT_POLY_MM, "hexagon": _HEX_POLY_MM}


def _rect_point_dist(px: float, py: float, xmin: float, xmax: float, ymin: float, ymax: float) -> float:
    dx = max(xmin - px, 0.0, px - xmax)
    dy = max(ymin - py, 0.0, py - ymax)
    if dx == 0.0 and dy == 0.0:
        return -min(px - xmin, xmax - px, py - ymin, ymax - py)
    return math.hypot(dx, dy)


def _point_in_convex_ccw(p: np.ndarray, poly: np.ndarray) -> bool:
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-9:
            return False
    return True


def _seg_point_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom <= 1e-12 else max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    return float(np.linalg.norm(p - (a + t * ab)))


def _seg_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    d1, d2 = a2 - a1, b2 - b1
    denom = cross(d1, d2)
    if abs(denom) < 1e-12:
        return False
    t = cross(b1 - a1, d2) / denom
    u = cross(b1 - a1, d1) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def _convex_poly_dist(p_poly: np.ndarray, q_poly: np.ndarray) -> float:
    """Distance between two CCW convex 2D polygons (0.0 if overlapping)."""
    for p in p_poly:
        if _point_in_convex_ccw(p, q_poly):
            return 0.0
    for q in q_poly:
        if _point_in_convex_ccw(q, p_poly):
            return 0.0
    for i in range(len(p_poly)):
        a1, a2 = p_poly[i], p_poly[(i + 1) % len(p_poly)]
        for j in range(len(q_poly)):
            if _seg_intersect(a1, a2, q_poly[j], q_poly[(j + 1) % len(q_poly)]):
                return 0.0
    best = math.inf
    for p in p_poly:
        for j in range(len(q_poly)):
            best = min(best, _seg_point_dist(p, q_poly[j], q_poly[(j + 1) % len(q_poly)]))
    for q in q_poly:
        for i in range(len(p_poly)):
            best = min(best, _seg_point_dist(q, p_poly[i], p_poly[(i + 1) % len(p_poly)]))
    return best


def neighbor_clearance_mm(tool: Tool, holes: dict, hole_name: str, width_mm: float) -> tuple[float, str | None]:
    """Minimum 2D clearance (mm) between the v3 fingers opened to `width_mm` over `hole_name` and every other-shape
    hole on the board (assuming a peg may sit in any other shape's hole; same-shape holes share one physical peg)."""
    target_shape = getattr(tool, "shape", None) or (holes.get(hole_name) or {}).get("shape")
    p_tcp_b = (np.asarray(tool.point, float)[:2] - (np.asarray(tool.rotation, float) @ np.asarray(tool.tip_offset, float))[:2]) * 1e3
    ex_h = np.asarray(tool.rotation, float)[:2, 0]
    ey_h = np.asarray(tool.rotation, float)[:2, 1]
    q = float(width_mm) / 2.0
    finger_a = np.array([[-FINGER_A_HALF_X_MM, q], [FINGER_A_HALF_X_MM, q],
                         [FINGER_A_HALF_X_MM, q + FINGER_A_DEPTH_MM], [-FINGER_A_HALF_X_MM, q + FINGER_A_DEPTH_MM]])
    finger_b = np.array([[-FINGER_B_HALF_X_MM, -(q + FINGER_B_DEPTH_MM)], [FINGER_B_HALF_X_MM, -(q + FINGER_B_DEPTH_MM)],
                         [FINGER_B_HALF_X_MM, -q + FINGER_B_TONGUE_MM], [-FINGER_B_HALF_X_MM, -q + FINGER_B_TONGUE_MM]])

    worst_d, worst_name = math.inf, None
    for other_name, h2 in holes.items():
        if other_name == hole_name or h2.get("shape") == target_shape:
            continue
        p2_b = np.array([float(h2["x_m"]), float(h2["y_m"])]) * 1e3
        shape2 = h2.get("shape", "")
        poly2 = _PEG_POLYS_MM.get(shape2)
        if poly2 is not None:
            r2 = _rz(math.radians(float(h2.get("yaw_deg", 0.0))))[:2, :2]
            corners_b = p2_b + (r2 @ poly2.T).T
            dp = corners_b - p_tcp_b
            corners_h = np.column_stack([dp @ ex_h, dp @ ey_h])
            d = min(_convex_poly_dist(finger_a, corners_h), _convex_poly_dist(finger_b, corners_h))
        else:
            dp = p2_b - p_tcp_b
            xh, yh = float(np.dot(dp, ex_h)), float(np.dot(dp, ey_h))
            rad = _PEG_RADIUS_MM.get(shape2, 12.0)
            d_a = _rect_point_dist(xh, yh, -FINGER_A_HALF_X_MM, FINGER_A_HALF_X_MM, q, q + FINGER_A_DEPTH_MM) - rad
            d_b = _rect_point_dist(xh, yh, -FINGER_B_HALF_X_MM, FINGER_B_HALF_X_MM, -(q + FINGER_B_DEPTH_MM), -q + FINGER_B_TONGUE_MM) - rad
            d = min(d_a, d_b)
        if d < worst_d:
            worst_d, worst_name = d, other_name
    return float(worst_d), worst_name


def safe_open_width_mm(tool, holes: dict | None, hole_name: str | None, requested_mm: float,
                       min_clearance_mm: float | None = 2.5,
                       min_width_mm: float | None = None) -> tuple[float, str | None, float | None]:
    """Cap `requested_mm` so both v3 fingers keep at least `min_clearance_mm` from every other-shape hole on the board.
    Returns `(width_mm, capped_by_hole_or_None, clearance_mm_or_None)`. Never caps below `min_width_mm`."""
    req = float(requested_mm)
    if (min_clearance_mm is None or float(min_clearance_mm) <= 0.0 or not holes or not hole_name
            or tool is None or not all(hasattr(tool, a) for a in ("point", "rotation", "tip_offset"))):
        return req, None, None
    target_clear = float(min_clearance_mm)
    clear_req, nearest = neighbor_clearance_mm(tool, holes, hole_name, req)
    if nearest is None or clear_req >= target_clear:
        return req, None, clear_req
    lo = req if min_width_mm is None else min(req, float(min_width_mm))
    clear_lo, nearest_lo = neighbor_clearance_mm(tool, holes, hole_name, lo)
    if clear_lo < target_clear:
        return lo, nearest_lo, clear_lo
    hi = req
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if neighbor_clearance_mm(tool, holes, hole_name, mid)[0] >= target_clear:
            lo = mid
        else:
            hi = mid
    clear_cap, nearest_cap = neighbor_clearance_mm(tool, holes, hole_name, lo)
    return round(lo, 2), nearest_cap, clear_cap
