"""The peg held in the printed V-block fingers and the TCP pose that puts it on a hole.

The fingers (franka_fr3_fingers v2) hold every peg off the TCP axis, at a
shape-specific offset and yaw (config/grasps.yaml). For a hole, the held peg
is the one of the hole's shape. Its frame is placed on the hole frame (turned
by a multiple of the peg's symmetry period, chosen so the hand stays closest
to the configured tcp_orientation_board_rpy), and the TCP follows:

    T_board_tcp = T_board_hole @ Rz(turn) @ T_peg_tcp

probe_board takes a tool as an orientation in the board frame plus a TCP ->
tip vector in TCP axes; `write_manipulation_config` writes both for one hole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    rotation: np.ndarray    # fr3_hand_tcp orientation in the board frame
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


def hole_tool(cfg, hole: dict, name: str) -> Tool:
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
    r_peg_tcp = _rz(math.radians(tcp["yaw_deg"])) @ FLIP
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
