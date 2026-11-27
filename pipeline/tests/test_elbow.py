"""grasp.choose_turn's elbow fallback and what it changes downstream (no robot, no ROS).

The planning tests use the board pose of sessions/2026-11-19_eye_in_hand_2 and are skipped without it; they need
OpenCV (grasp -> panda), so run them with the pipeline's interpreter.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_pipeline import boardheight, holefit, insert  # noqa: E402
from board_pipeline.config import Config, load_yaml  # noqa: E402

PIPELINE = Path(__file__).resolve().parents[1]
HOLES = load_yaml(PIPELINE / "config" / "holes.yaml")["holes"]
BOARD = PIPELINE.parent / "sessions" / "2026-11-19_eye_in_hand_2" / "board_pose.json"


def test_posture_args_only_when_the_elbow_is_turned():
    assert insert.posture_args(SimpleNamespace(arm_angle=0.0)) == []
    assert insert.posture_args(SimpleNamespace()) == []
    assert insert.posture_args(SimpleNamespace(arm_angle=math.radians(-10.0))) == ["--arm-angle-deg", "-10.000"]


def test_elbow_tagged_entries_are_not_fitted(tmp_path):
    found = {"xy": [0.5, 0.0], "sigma_m": 1e-4, "source": "test"}
    holefit.record_found(tmp_path, "hexagon_c3mm", [0.5, 0.0], found, "sha", "run_a")
    holefit.record_found(tmp_path, "hexagon_c1mm", [0.5, 0.05], found, "sha", "run_b", arm_angle_deg=-10.0)
    entries = holefit.load_found(tmp_path)
    assert "arm_angle_deg" not in entries[0] and entries[1]["arm_angle_deg"] == -10.0
    s = Config.template()["insert"]
    kept = insert.correction_entries(entries, {**s, "correction_exclude_shapes": []}, HOLES)
    assert [e["hole"] for e in kept] == ["hexagon_c3mm"]

    boardheight.record(tmp_path, "r0", "round", "sha", [(0.5, 0.0, 0.010)], [0.009])
    boardheight.record(tmp_path, "r1", "round", "sha", [(0.5, 0.05, 0.012)], [0.009], arm_angle_deg=10.0)
    model = boardheight.fit(boardheight.load(tmp_path), "sha")
    assert model["c"] == pytest.approx(0.001, abs=2e-4)  # the turned-elbow touch (+3 mm) left out


def test_elbow_steps_smallest_first():
    pytest.importorskip("cv2")
    from board_pipeline import grasp
    steps = [round(math.degrees(v)) for v in grasp.elbow_steps(5.0, 15.0)]
    assert steps == [5, -5, 10, -10, 15, -15]


@pytest.fixture(scope="module")
def board():
    if not BOARD.exists():
        pytest.skip(f"no {BOARD}")
    return np.asarray(json.loads(BOARD.read_text())["T_base_board"])


def test_unreachable_l_hole_turns_the_elbow(board):
    pytest.importorskip("cv2")
    from board_pipeline import grasp, panda
    tool, _ = grasp.choose_turn(Config.template(), HOLES["l_shape_c3mm"], "l_shape_c3mm", board)
    assert tool.arm_angle != 0.0
    assert abs(math.degrees(tool.arm_angle)) <= 20.0  # the smallest turn that works, not the whole range
    assert grasp.wrist_margin(tool.joints[6]) >= grasp.WRIST_MARGIN_RAD
    assert grasp.at_arm_angle(tool.joints, tool.arm_angle)
    assert np.allclose(panda.fk(tool.joints, "panda_hand_tcp"), tool.tcp_pose(board, grasp.SURFACE_Z), atol=1e-4)


def test_reachable_hole_keeps_the_elbow_at_zero(board):
    pytest.importorskip("cv2")
    from board_pipeline import grasp
    for name in ("l_shape_c0p2mm", "l_shape_c1mm", "hexagon_c3mm"):
        tool, _ = grasp.choose_turn(Config.template(), HOLES[name], name, board)
        assert tool.arm_angle == 0.0, name
        assert grasp.at_arm_angle(tool.joints, 0.0), name


def test_same_hole_elbow_correction_matches_hole_angle_and_tip(tmp_path):
    tip_v3 = [0.003459, 0.012303, 0.039]
    tip_cad = [0.004491, 0.012479, 0.039]
    vision = [0.3934, -0.1348]
    # Legacy entry without tip_in_tcp_m (must be ignored when tip_offset is provided)
    holefit.record_found(tmp_path, "l_shape_c3mm", vision, {"xy": [0.3922, -0.1371], "sigma_m": 0.001, "source": "s"},
                         "sha", "r0", arm_angle_deg=-5.0)
    # Entry with different tip (must be ignored)
    holefit.record_found(tmp_path, "l_shape_c3mm", vision, {"xy": [0.3922, -0.1371], "sigma_m": 0.001, "source": "s"},
                         "sha", "r1", arm_angle_deg=-5.0, tip_offset=tip_cad)
    # Entry with matching tip and arm angle
    holefit.record_found(tmp_path, "l_shape_c3mm", vision, {"xy": [0.3930, -0.1364], "sigma_m": 0.0005, "source": "s"},
                         "sha", "r2", arm_angle_deg=-5.0, tip_offset=tip_v3)
    # Entry at different arm angle (must be ignored)
    holefit.record_found(tmp_path, "l_shape_c3mm", vision, {"xy": [0.3950, -0.1340], "sigma_m": 0.0005, "source": "s"},
                         "sha", "r3", arm_angle_deg=0.0, tip_offset=tip_v3)
    entries = holefit.load_found(tmp_path)
    corr = holefit.same_hole_elbow_correction(entries, "l_shape_c3mm", "sha", -5.0, tip_offset=tip_v3)
    assert corr is not None and corr["holes"] == 1 and corr["elbow_same_hole"] is True
    shifted = holefit.correct_xy_full(vision, corr)
    assert shifted == pytest.approx([0.3930, -0.1364], abs=1e-4)
    M = holefit.full_matrix(corr)
    assert (M @ np.array([vision[0], vision[1], 0.0, 1.0]))[:2] == pytest.approx(shifted, abs=1e-9)
    assert holefit.same_hole_elbow_correction(entries, "l_shape_c3mm", "sha", -10.0, tip_offset=tip_v3) is None


def test_session_l_shape_tip_resolves():
    pytest.importorskip("cv2")
    from board_pipeline import grasp
    session_path = BOARD.parent
    tip_file = grasp.session_tip_path(session_path, "l_shape")
    if not tip_file.exists():
        pytest.skip(f"no {tip_file}")
    base_tool = grasp.hole_tool(Config.template(), HOLES["l_shape_c3mm"], "l_shape_c3mm")
    resolved, source = grasp.resolve_tip(base_tool, None, session_path)
    assert "tips/l_shape.yaml" in source
    assert resolved.tip_offset == pytest.approx([0.003459, 0.012303, 0.039], abs=1e-6)
