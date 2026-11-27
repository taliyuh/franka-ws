"""Tests for dynamic neighbor-clearance capping of extract open_width and insert release_width."""

from __future__ import annotations

import math
from pathlib import Path

from board_pipeline import grasp, insert
from board_pipeline.config import Config, load_yaml

WORKSPACE = Path(__file__).resolve().parents[2]


def _cfg_and_holes():
    cfg = Config.template()
    holes = load_yaml(cfg.path(cfg["holes"]))["holes"]
    return cfg, holes


def test_l_shape_c0p2mm_caps_open_and_release_width():
    cfg, holes = _cfg_and_holes()
    tool = grasp.hole_tool(cfg, holes["l_shape_c0p2mm"], "l_shape_c0p2mm")
    min_open = tool.grasp_width_mm + 3.0  # 2 * min_open_margin_mm (1.5)

    # Extract open_width (31.4 mm requested) is capped for rectangle_c0p2mm
    req_open = tool.grasp_width_mm + 9.0
    w_open, by, clear = grasp.safe_open_width_mm(
        tool, holes, "l_shape_c0p2mm", req_open, min_clearance_mm=2.5, min_width_mm=min_open)
    assert by == "rectangle_c0p2mm"
    assert min_open <= w_open < req_open
    assert clear is not None and clear >= 2.49

    # Insert release_width (30.0 mm requested) is also capped
    s = insert.settings(cfg)
    w_rel = insert.release_width_mm(s, tool, holes=holes, hole_name="l_shape_c0p2mm")
    assert w_rel < float(s["release_width_mm"])
    assert w_rel >= tool.grasp_width_mm + 2.0


def test_other_l_shape_holes_uncapped():
    cfg, holes = _cfg_and_holes()
    for name in ("l_shape_c3mm", "l_shape_c1mm", "l_shape_c0p03mm"):
        tool = grasp.hole_tool(cfg, holes[name], name)
        req_open = tool.grasp_width_mm + 9.0
        w_open, by, clear = grasp.safe_open_width_mm(
            tool, holes, name, req_open, min_clearance_mm=2.5, min_width_mm=tool.grasp_width_mm + 3.0)
        assert by is None
        assert w_open == req_open
        assert clear is not None and clear > 10.0


def test_round_bad_turn_capped_good_turn_uncapped():
    cfg, holes = _cfg_and_holes()
    bad_tool = grasp.hole_tool(cfg, holes["round_c0p2mm"], "round_c0p2mm", turn=math.radians(270.0))
    w_bad, by, _ = grasp.safe_open_width_mm(
        bad_tool, holes, "round_c0p2mm", bad_tool.grasp_width_mm + 9.0,
        min_clearance_mm=2.5, min_width_mm=bad_tool.grasp_width_mm + 3.0)
    assert by == "hexagon_c0p2mm"
    assert w_bad < bad_tool.grasp_width_mm + 9.0

    good_tool = grasp.hole_tool(cfg, holes["round_c0p2mm"], "round_c0p2mm", turn=0.0)
    w_good, by_good, _ = grasp.safe_open_width_mm(
        good_tool, holes, "round_c0p2mm", good_tool.grasp_width_mm + 9.0,
        min_clearance_mm=2.5, min_width_mm=good_tool.grasp_width_mm + 3.0)
    assert by_good is None
    assert w_good == good_tool.grasp_width_mm + 9.0


def test_release_width_backward_compatible_without_holes():
    cfg, holes = _cfg_and_holes()
    s = insert.settings(cfg)
    tool = grasp.hole_tool(cfg, holes["l_shape_c0p2mm"], "l_shape_c0p2mm")
    assert insert.release_width_mm(s, tool) == min(float(s["release_width_mm"]),
                                                   tool.grasp_width_mm + float(s["release_margin_mm"]))
