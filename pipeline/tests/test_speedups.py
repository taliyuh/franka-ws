"""Offline tests for pipeline speedups: persistent docker exec container, peg_search move,
free-air speed args, zero-handover campaign flow, and trimmed contact config defaults."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_pipeline import campaign, extract, insert, ros  # noqa: E402
from board_pipeline.config import Config, load_yaml  # noqa: E402

PIPELINE = Path(__file__).resolve().parents[1]
HOLES = load_yaml(PIPELINE / "config" / "holes.yaml")["holes"]


def test_pipeline_yaml_speedup_defaults():
    cfg = Config.template()
    s = cfg["insert"]
    e = cfg["extract"]
    assert "moveit:=false" in s["robot_launch"]
    assert "load_gripper:=false" in s["robot_launch"]
    assert s["travel"] == "libfranka"
    assert s["persistent_container"] is True
    assert s["free_move_speed_m_s"] == pytest.approx(0.12)
    assert s["free_joint_speed_rad_s"] == pytest.approx(0.7)
    assert s["board_touches"] == 0
    assert s["peck_insert_depth_m"] == pytest.approx(0.010)
    assert s["centre_after_insert"] is True  # item 4 intentionally kept
    assert e["hover_mm"] == pytest.approx(6.0)
    assert e["speed_mm_s"] == pytest.approx(5.0)
    assert e["pull_speed_mm_s"] == pytest.approx(6.0)
    assert e["pull_stiffness"] == [150, 150, 4000, 70, 70, 2]
    assert e["pull_max_lag_mm"] == pytest.approx(5.5)
    assert e["wiggle_deg"] == pytest.approx(0.15)
    assert e["wiggle_hz"] == pytest.approx(2.0)
    assert "--max-tip-speed" in e["options"]


def test_rewrite_run_paths_and_container_run_dir(tmp_path):
    cfg = Config.template()
    ws_run = cfg.workspace / "sessions" / "s1" / "insert" / "r1"
    mapped = insert._container_run_dir(cfg, ws_run)
    assert mapped == f"{cfg['docker']['workspace']}/sessions/s1/insert/r1"

    tmp_run = Path("/tmp/insert_dry_xyz")
    assert insert._container_run_dir(cfg, tmp_run) == "/tmp/insert_dry_xyz"

    rewritten = insert._rewrite_run_paths(
        ["172.16.0.2", "lift", "/insert/lift.csv", "--result", "/insert/lift.json", "--yes"],
        mapped,
    )
    assert rewritten == [
        "172.16.0.2", "lift", f"{mapped}/lift.csv", "--result", f"{mapped}/lift.json", "--yes",
    ]


def test_robot_program_uses_docker_exec_when_persistent(monkeypatch, tmp_path):
    cfg = Config.template()
    s = dict(cfg["insert"], persistent_container=True)
    run_dir = cfg.workspace / "sessions" / "fake_run"
    seen = []

    monkeypatch.setattr(insert, "robot_session_dir", lambda c, st: tmp_path)
    monkeypatch.setattr(insert, "ensure_robot_container", lambda c, st: "peg_search_robot")

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = insert.robot_program(cfg, s, run_dir, "peg_search",
                              ["172.16.0.2", "lift", "/insert/a.csv", "--yes"])
    assert rc == 0
    assert seen[0][:4] == ["docker", "exec", "-i", "peg_search_robot"]
    assert seen[0][4] == "./build/peg_search"
    assert seen[0][7] == f"{cfg['docker']['workspace']}/sessions/fake_run/a.csv"


def test_extract_ik_seed_and_move_to_hole(monkeypatch, tmp_path):
    q = [0.1, 0.2, 0.0, -2.5, 0.0, 2.8, -0.5]
    argv = ["--board-pose", "/x.json", "--ik-seed", *(str(v) for v in q), "--stay"]
    assert insert.extract_ik_seed(argv) == pytest.approx(q)
    assert insert.extract_ik_seed(["--board-pose", "/x.json"]) is None

    cfg = Config.template()
    s = cfg["insert"]
    captured = []
    monkeypatch.setattr(insert, "robot_program",
                        lambda c, st, rd, prog, a: (captured.append((prog, a)), 0)[1])
    ok = insert.move_to_hole(cfg, s, tmp_path, q, "approach", travel_z=0.06, target_z=0.012)
    assert ok is True
    prog, a = captured[0]
    assert prog == "peg_search"
    assert a[1] == "move"
    assert "--target-q" in a
    assert "--travel-z" in a and "--target-z" in a
    assert "--free-move-speed" in a and "--free-joint-speed" in a
    assert "--yes" in a


def test_campaign_zero_handover_in_libfranka_travel(tmp_path, monkeypatch):
    """With travel=libfranka, campaign must not restart robot.launch between holes (only in finally)."""
    start_calls = []
    extract_restart_flags = []

    def fake_insert(session, args, probe_target, restart_launch=True, raise_on_fail=True):
        d = tmp_path / f"ins_{args.hole}"
        d.mkdir(exist_ok=True)
        release = {"hole": args.hole, "tip_pose": [float(v) for v in np.eye(4).reshape(-1)],
                   "board_z_m": 0.0, "tip_offset_tcp_m": [0, 0, 0.039]}
        return {"inserted": True, "stopped": None, "out": d,
                "summary": {"attempts": [{}], "release": release}}

    def fake_extract(session, args, probe_target, record=None, here=False, restart_launch=True):
        extract_restart_flags.append(restart_launch)
        d = tmp_path / f"ext_{args.hole}"
        d.mkdir(exist_ok=True)
        return {"ok": True, "peg": "hand", "out": d,
                "result": {"held": True, "contact_above_board": 0.01, "seated_shift_xy": [0.0, 0.0],
                           "seated_turn_deg": 0.0, "width_off": 0.0}, "error": None}

    monkeypatch.setattr(insert, "run", fake_insert)
    monkeypatch.setattr(extract, "run", fake_extract)
    monkeypatch.setattr(insert, "start_launch", lambda cfg, s: start_calls.append("start"))
    monkeypatch.setattr(insert, "launch_running", lambda cfg, s: False)

    session = SimpleNamespace(path=tmp_path / "session", cfg=Config.template())
    c = Config.template()["campaign"]
    campaign.save_state(session, campaign.new_state(HOLES, c, 1, ["round"]))
    campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)

    assert all(flag is False for flag in extract_restart_flags)
    # Only the single `finally:` start_launch at the very end of the campaign
    assert start_calls == ["start"]


def test_launch_kept_stopped_suppresses_ros_and_pgrep_and_restores(monkeypatch):
    cfg = Config.template()
    s = cfg["insert"]
    calls = []

    monkeypatch.setattr(insert, "_container", lambda c, script: (calls.append(("pgrep", script)), SimpleNamespace(returncode=1))[1])
    monkeypatch.setattr(ros, "output", lambda c, argv, timeout=None: (calls.append(("ros", argv)), "")[1])
    monkeypatch.setattr(insert, "stop_launch", lambda c, st: calls.append(("stop",)))

    assert insert.launch_running(cfg, s) is False
    assert ros.current_joints(cfg) is None
    assert [c[0] for c in calls] == ["pgrep", "ros"]
    calls.clear()

    with insert.launch_kept_stopped(cfg, s):
        assert len(calls) == 1 and calls[0][0] == "pgrep"
        calls.clear()
        assert insert.launch_running(cfg, s) is False
        assert ros.current_joints(cfg) is None
        assert calls == []

    assert insert.launch_running(cfg, s) is False
    assert ros.current_joints(cfg) is None
    assert [c[0] for c in calls] == ["pgrep", "ros"]

    calls.clear()
    with pytest.raises(RuntimeError):
        with insert.launch_kept_stopped(cfg, s):
            assert insert.launch_running(cfg, s) is False
            raise RuntimeError("boom")
    assert ros._SKIP_CURRENT_JOINTS is False
    assert insert._ASSUME_LAUNCH_STOPPED is False
