"""Offline tests: no plunge into the tight holes (insert.no_plunge_max_clearance_mm), the board touches only."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_pipeline import campaign, insert  # noqa: E402
from board_pipeline.config import Config, load_yaml  # noqa: E402

PIPELINE = Path(__file__).resolve().parents[1]
S = Config.template()["insert"]
TABLE = load_yaml(PIPELINE / "config" / "tilt_leads.yaml")["holes"]


def _tool():  # as tests/test_insert.py's
    return SimpleNamespace(point=np.array([0.0, 0.0]), tip_offset=np.array([0.0, -0.017, 0.039]))


def _dry_run(tmp_path, monkeypatch, hole):
    session_dir = tmp_path / "session"
    (session_dir / "targets").mkdir(parents=True, exist_ok=True)
    board = np.eye(4)
    board[:3, 3] = [0.45, -0.0166, -0.019]
    target = session_dir / "targets" / f"{hole}.json"
    target.write_text(json.dumps({"T_base_board": board.tolist(), "board_surface_z_m": 0.025,
                                  "source": {"board_pose_sha256": "sha-n"}}))

    def probe_target(session, holes, name, tip_from, height_mm, turn, joint_correction=True):
        return _tool(), "CAD", target, ["--board-pose", "/franka_ws/x.json", "--offset", 0, 0, 0.006]

    seen = {}
    monkeypatch.setattr(insert, "peg_search", lambda cfg, s, out, argv: seen.update(argv=list(map(str, argv))) or 0)
    monkeypatch.setattr(insert.ros, "controller_states", lambda cfg: {})
    monkeypatch.setattr(insert.ros, "container_path", lambda cfg, p: f"/c/{Path(p).name}")
    session = SimpleNamespace(cfg=Config(dict(Config.template().data)), path=session_dir, name="t")
    args = SimpleNamespace(hole=hole, attempts=None, height=None, tip_from=None, turn=0.0, no_joint_correction=True,
                           slide_dir=None, pattern=None, method="plunge", here=False, dry_run=True,
                           no_board_correction=True, no_centre=False, search=None, touches="0")
    insert.run(session, args, probe_target)
    return seen["argv"]


@pytest.mark.parametrize("hole,points", [("rectangle_c0p2mm", "0"), ("rectangle_c0p03mm", "0"),
                                         ("rectangle_c1mm", "1"), ("rectangle_c3mm", "1")])
def test_tight_holes_get_no_plunge_point(tmp_path, monkeypatch, hole, points):
    argv = _dry_run(tmp_path, monkeypatch, hole)
    last = len(argv) - 1 - argv[::-1].index("--peck-max-points")   # the last one wins (peck_options has 60)
    assert argv[last + 1] == points
    if points == "0":
        assert argv.count("--peck-touch") >= 1          # peg_search needs a touch with 0 points (and the plane)


def test_touch_only_run_is_logged_as_touch_and_not_counted(tmp_path, monkeypatch, capsys):
    touched = {"ok": True, "stage": "done", "inserted": False, "board_measured": None,
               "plane": {"point": [0.452, -0.0166, 0.0065], "slope": [0.0, 0.0]},
               "touch_points": [[0.43, -0.0166, 0.0065], [0.43, -0.0386, 0.0065], [0.408, -0.0386, 0.0065]],
               "first": {"xy": [0.452, -0.0166], "tip_z": None, "result": "none"}}
    attempt = {"ok": True, "stage": "done", "inserted": True, "outcomes": ["inserted"]}

    def fake(cfg, s, out, argv):
        name = "peck.json" if argv[1] == "peck" else Path(argv[argv.index("--result") + 1]).name
        (out / name).write_text(json.dumps(touched if argv[1] == "peck" else attempt))
        return 0

    monkeypatch.setattr(insert, "peg_search", fake)
    monkeypatch.setattr(insert, "robot_program", lambda *a, **k: 0)
    summary = {"attempts": [], "inserted": False}
    drop = ["ip", "peck", "/insert/peck.csv", "--peck-max-points", "0", "--result", "/insert/peck.json"]
    ok = insert._plunge_then_tilt(None, S, tmp_path, drop, ["--result-placeholder"], summary, 1, 6.0,
                                  np.array([0.452, -0.0166, 0.0065]), np.array([0.43, -0.0166, 0.0065]),
                                  TABLE["triangle_c0p2mm"]["leads"][0], 0.002, plunge=False)
    out = capsys.readouterr().out
    assert ok
    assert summary["attempts"][0]["method"] == "touch"
    assert "plunge: not in" not in out and "board touches (no plunge" in out
    assert campaign._attempts(summary) == 1                       # the tilt search only
    assert campaign._method_succeeded(summary, True) == "tilt"


def test_config_covers_0p2_and_0p03():
    assert float(S["no_plunge_max_clearance_mm"]) >= 0.2
    assert float(S["no_plunge_max_clearance_mm"]) < 1.0
