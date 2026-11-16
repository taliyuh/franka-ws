"""The peg held in the printed V-block fingers and the TCP pose that puts it on a hole.

The fingers (franka_fr3_fingers v2) hold every peg off the TCP axis, at a
shape-specific offset and yaw (config/grasps.yaml). For a hole, the held peg
is the one of the hole's shape. Its frame is placed on the hole frame (turned
by a multiple of the peg's symmetry period, chosen so the hand stays closest
to the configured tcp_orientation_board_rpy), and the TCP follows:

    T_board_tcp = T_board_hole @ Rz(turn) @ T_peg_tcp

probe_board takes a tool as an orientation in the board frame plus a TCP ->
tip vector in TCP axes; `write_manipulation_config` writes both for one hole.

The tip comes from the CAD grasp unless `pipeline check/probe --tip-from FILE`
passes a pivot_calibration result measured on the same peg shape;
`apply_pivot_tip` swaps it in after comparing it with the CAD tip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from board_pipeline.config import PipelineError, load_yaml

DEFAULT_GRASPS = "pipeline/config/grasps.yaml"
FLIP = np.diag([1.0, -1.0, -1.0])  # Rx(180 deg): the TCP looks down the peg


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


def hole_tool(cfg, hole: dict, name: str, extra_turn_deg: float = 0.0) -> Tool:
    """The held peg on this hole; extra_turn_deg turns the hand further about the peg axis
    (a multiple of the peg's symmetry period; `pipeline diag 3` / `--turn`)."""
    source, grasps = _grasps(cfg)
    try:
        shape = grasps["shapes"][hole["shape"]]
        centre = grasps["holes"][name]
    except KeyError as error:
        raise PipelineError(f"{source} has no grasp for {name} ({hole['shape']}); "
                            "regenerate it with pipeline/tools/extract_grasps.py") from error
    tcp = shape["tcp"]
    r_peg_tcp = _rz(math.radians(tcp["yaw_deg"]+180.0)) @ FLIP
    t_peg_tcp = np.array([tcp["x_m"], tcp["y_m"], tcp["z_m"]])

    # Turn the peg about its axis by the multiple of its symmetry period (any
    # angle for the round peg) that brings the TCP x axis closest to the
    # configured one. Only the yaw of the configured orientation is used: the
    # peg must stay vertical to go in.
    probe = load_yaml(cfg.path(cfg["robot"]["manipulation_config"]))["probe"]
    reference = Rotation.from_euler("xyz", probe["tcp_orientation_board_rpy"]).as_matrix()
    hole_yaw = math.radians(hole["yaw_deg"])
    natural = _rz(hole_yaw) @ r_peg_tcp
    turn = math.remainder(math.atan2(reference[1, 0], reference[0, 0])
                          - math.atan2(natural[1, 0], natural[0, 0]), 2 * math.pi)
    period = math.radians(shape["yaw_period_deg"])
    if period > 0:
        turn = period * round(turn / period)
    if extra_turn_deg:
        extra = math.radians(extra_turn_deg)
        if period > 0 and abs(math.remainder(extra, period)) > 1e-6:
            raise PipelineError(f"the {hole['shape']} peg only turns by multiples of "
                                f"{shape['yaw_period_deg']:g} deg, not {extra_turn_deg:g}")
        turn += extra
    peg_yaw = hole_yaw + turn

    point = np.array([hole["x_m"], hole["y_m"]]) \
        + _rz(hole_yaw)[:2, :2] @ np.array([centre["x_m"], centre["y_m"]])
    return Tool(shape=hole["shape"], rotation=_rz(peg_yaw) @ r_peg_tcp,
                tip_offset=-r_peg_tcp.T @ t_peg_tcp, point=point,
                peg_yaw=peg_yaw, grasp_width_mm=float(shape["grasp_width_mm"]))


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
