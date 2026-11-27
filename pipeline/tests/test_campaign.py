"""Offline tests of extract's geometry (the mirrored seat, the equivalent grasp) and the campaign's peg tracking
(insert and extract replaced by stand-ins: no robot)."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_pipeline import campaign, extract, grasp, holefit, insert, ros  # noqa: E402
from board_pipeline.config import Config, load_yaml  # noqa: E402

PIPELINE = Path(__file__).resolve().parents[1]
HOLES = load_yaml(PIPELINE / "config" / "holes.yaml")["holes"]


def _t_peg_tcp(tcp: dict) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = grasp._rz(math.radians(tcp["yaw_deg"])) @ grasp.FLIP
    t[:3, 3] = [tcp["x_m"], tcp["y_m"], tcp["z_m"]]
    return t


@pytest.mark.parametrize("shape", ["rectangle", "l_shape", "hexagon"])
def test_mirror_seat_is_the_mirrored_transform(shape):
    grasps = load_yaml(PIPELINE / "config" / "grasps.yaml")
    tcp = grasps["shapes"][shape]["tcp"]
    mirrored, centre = grasp.mirror_seat(tcp, {"x_m": 0.001, "y_m": 0.002})
    m_peg, m_hand = np.diag([1.0, -1.0, 1.0, 1.0]), np.diag([-1.0, 1.0, 1.0, 1.0])
    assert np.allclose(_t_peg_tcp(mirrored), m_peg @ _t_peg_tcp(tcp) @ m_hand, atol=1e-12)
    assert centre == {"x_m": 0.001, "y_m": -0.002}
    assert np.linalg.det(_t_peg_tcp(mirrored)[:3, :3]) == pytest.approx(1.0)


def test_mirrored_hole_tool_keeps_the_tip_on_the_hole():
    cfg = Config.template()
    for name in ("rectangle_c1mm", "l_shape_c0p03mm"):
        plain = grasp.hole_tool(cfg, HOLES[name], name, turn=0.0)
        mirrored = grasp.hole_tool(cfg, HOLES[name], name, turn=0.0, mirror=True)
        assert plain.tip_offset[2] == pytest.approx(mirrored.tip_offset[2])
        assert not np.allclose(plain.rotation, mirrored.rotation)
        if name.startswith("rectangle"):
            assert np.allclose(plain.point, mirrored.point)


def test_equivalent_rotation_picks_the_nearest_symmetric_turn():
    flip = grasp.FLIP
    r_peg = grasp._rz(0.3) @ flip
    r_ref = grasp._rz(0.3 + math.radians(118.0)) @ flip
    out = extract.equivalent_rotation(r_peg, r_ref, math.radians(120.0))
    # the peg axis is kept, the hand turned by -120 about it (body z points down: base yaw +120)
    assert np.allclose(out[:, 2], r_peg[:, 2])
    yaw = math.atan2(out[1, 0], out[0, 0])
    assert math.degrees(math.remainder(yaw - math.atan2(r_ref[1, 0], r_ref[0, 0]), 2 * math.pi)) == pytest.approx(2.0, abs=1e-6)
    assert np.allclose(extract.equivalent_rotation(r_peg, r_ref, 2 * math.pi), r_peg)
    assert np.allclose(extract.equivalent_rotation(r_peg, r_ref, 0.0), r_ref)


def test_plan_covers_every_hole_in_order():
    c = Config.template()["campaign"]
    state = campaign.new_state(HOLES, c, 2, list(c["shapes"]))
    plan = campaign._plan(state)
    assert len(plan) == 2 * 5 * 3
    assert [h for _, _, h in plan[:6]] == [
        "round_c3mm", "hexagon_c3mm", "triangle_c3mm", "rectangle_c3mm", "l_shape_c3mm", "round_c1mm"
    ]
    for _ in range(3 * 5):
        campaign._advance(state)
    assert (state["round"], state["shape_index"], state["step"]) == (2, 0, 0)


class FakeRobot:
    """insert.run / extract.run stand-ins: the hole names in `fail` do not take the peg (`stuck`: no extract)."""

    def __init__(self, tmp: Path, fail=(), stuck=(), tilt_win=(), record_found=False):
        self.tmp = tmp
        self.fail = set(fail)
        self.stuck = set(stuck)
        self.tilt_win = set(tilt_win)
        self.record_found = record_found
        self.calls = []
        self.insert_args = []
        self.found_counts_at_insert = []
        self.n = 0

    def _dir(self, kind):
        self.n += 1
        d = self.tmp / f"{kind}_{self.n}"
        d.mkdir()
        return d

    def insert(self, session, args, probe_target, restart_launch=True, raise_on_fail=True):
        self.calls.append(("insert", args.hole))
        self.insert_args.append(args)
        self.found_counts_at_insert.append((args.hole, len(holefit.load_found(session.path))))
        ok = args.hole not in self.fail
        self.fail.discard(args.hole)  # fails once
        if not ok:
            attempts = [
                {"attempt": 0, "method": "plunge", "ok": True, "inserted": False, "stage": "done"},
                {"attempt": 1, "ok": True, "inserted": False, "stage": "done", "outcomes": ["miss"]},
            ]
        elif args.hole in self.tilt_win:
            attempts = [
                {"attempt": 0, "method": "plunge", "ok": True, "inserted": False, "stage": "done"},
                {"attempt": 1, "ok": True, "inserted": True, "stage": "done", "outcomes": ["inserted"]},
            ]
        else:
            attempts = [{"attempt": 0, "method": "plunge", "ok": True, "inserted": True, "stage": "done"}]
        if ok and self.record_found and not getattr(args, "no_record", False):
            holefit.record_found(session.path, args.hole, [0.5, 0.0],
                                 {"xy": [0.501, 0.0], "sigma_m": 1e-4, "source": "wall touches (6)"},
                                 "sha", f"run_{self.n}")
        release = {"hole": args.hole, "tip_pose": [float(v) for v in np.eye(4).reshape(-1)], "board_z_m": 0.0,
                   "tip_offset_tcp_m": [0, 0, 0.039]} if ok else None
        return {"inserted": ok, "stopped": None, "out": self._dir("insert"),
                "summary": {"attempts": attempts, "release": release}}

    def extract(self, session, args, probe_target, record=None, here=False, restart_launch=True):
        self.calls.append(("extract", args.hole, here))
        if args.hole in self.stuck:
            return {"ok": False, "peg": "hole", "out": self._dir("extract"), "result": None, "error": "stuck"}
        return {"ok": True, "peg": "hand", "out": self._dir("extract"),
                "result": {"held": True, "contact_above_board": 0.03, "seated_shift_xy": [0.0, 0.0],
                           "seated_turn_deg": 0.0, "width_off": 0.0}, "error": None}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def make(fail=(), stuck=(), tilt_win=(), record_found=False, shapes=("round", "hexagon"), rounds=1):
        robot = FakeRobot(tmp_path, fail, stuck, tilt_win=tilt_win, record_found=record_found)
        monkeypatch.setattr(insert, "run", robot.insert)
        monkeypatch.setattr(extract, "run", robot.extract)
        monkeypatch.setattr(insert, "start_launch", lambda cfg, s: None)
        monkeypatch.setattr(insert, "stop_launch", lambda cfg, s: None)
        monkeypatch.setattr(insert, "launch_running", lambda cfg, s: False)
        monkeypatch.setattr(campaign, "rehome_gripper",
                            lambda session, state, finished_round: robot.calls.append(("rehome", finished_round)))
        session = SimpleNamespace(path=tmp_path / "session", cfg=Config.template())
        c = Config.template()["campaign"]
        campaign.save_state(session, campaign.new_state(HOLES, c, rounds, list(shapes)))
        return session, robot
    return make


def test_campaign_round_moves_every_peg_through_its_holes(setup):
    session, robot = setup()
    campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)
    state = campaign.load_state(session)
    assert state["round"] == 2
    assert state["pegs"]["round"]["at"] == "round_c0p2mm" and state["pegs"]["hexagon"]["at"] == "hexagon_c0p2mm"
    assert robot.calls[:6] == [
        ("extract", "round_c0p2mm", False), ("insert", "round_c3mm"),
        ("extract", "hexagon_c0p2mm", False), ("insert", "hexagon_c3mm"),
        ("extract", "round_c3mm", False), ("insert", "round_c1mm"),
    ]
    rows = (session.path / "campaign" / "results.csv").read_text().splitlines()
    assert len(rows) == 1 + 2 * 3 * 2  # header, an extract and an insert per hole


def test_a_failed_insert_parks_the_peg_and_the_next_hole_takes_it_from_there(setup):
    session, robot = setup(fail={"round_c1mm"}, shapes=("round",))
    campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)
    i = robot.calls.index(("insert", "round_c1mm"))
    assert robot.calls[i + 1] == ("insert", "round_c3mm")          # parked
    assert robot.calls[i + 2] == ("extract", "round_c3mm", True)   # taken from there
    assert robot.calls[i + 3] == ("insert", "round_c0p2mm")
    assert campaign.load_state(session)["pegs"]["round"]["at"] == "round_c0p2mm"
    park_idx = [idx for idx, a in enumerate(robot.insert_args) if a.no_centre]
    assert len(park_idx) == 1
    assert robot.insert_args[park_idx[0]].hole == "round_c3mm"
    assert robot.insert_args[park_idx[0]].no_record is True
    assert all(not robot.insert_args[k].no_record for k in range(len(robot.insert_args)) if k != park_idx[0])


def test_a_stuck_peg_stops_the_campaign_by_default(setup):
    session, robot = setup(stuck={"round_c0p2mm"})
    with pytest.raises(Exception, match="extracting the round peg failed"):
        campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)
    assert campaign.load_state(session)["pegs"]["round"]["at"] == "round_c0p2mm"
    assert not any(c[0] == "insert" for c in robot.calls)


def test_a_stuck_peg_blocks_its_shape_and_the_others_go_on(setup):
    session, robot = setup(stuck={"round_c0p2mm"})
    session.cfg.data["campaign"]["on_extract_failure"] = "skip_shape"
    campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)
    state = campaign.load_state(session)
    assert "round" in state["blocked"] and state["pegs"]["round"]["at"] == "round_c0p2mm"
    assert ("insert", "hexagon_c0p2mm") in robot.calls
    assert not any(c[0] == "insert" and c[1].startswith("round") for c in robot.calls)


def test_a_run_stopped_mid_action_asks_where_the_peg_is(setup):
    session, _ = setup()
    state = campaign.load_state(session)
    state["pending"] = {"action": "insert", "shape": "round", "hole": "round_c3mm"}
    campaign.save_state(session, state)
    with pytest.raises(Exception, match="campaign peg round"):
        campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)


def test_found_holes_isolated_per_round_and_start_archives_files(setup):
    session, robot = setup(record_found=True, shapes=("round", "hexagon"), rounds=2)
    holefit.record_found(session.path, "round_c3mm", [0.4, 0.0],
                         {"xy": [0.402, 0.001], "sigma_m": 1e-4, "source": "wall touches (6)"},
                         "sha", "pre_campaign")
    heights_file = session.path / "insert" / "board_heights.yaml"
    heights_file.write_text("touches:\n- run: old\n")
    (session.path / "campaign" / "results.csv").write_text("stale\n")
    (session.path / "campaign" / "attempts.csv").write_text("stale\n")

    campaign.cmd_start(session, SimpleNamespace(fresh=True, rounds=2, shapes=["round", "hexagon"]))
    assert not holefit.found_path(session.path).exists()
    assert (session.path / "campaign" / campaign.EXTRACT_FOUND_FILE).exists()
    assert list((session.path / "campaign").glob("found_holes_before_start_*.yaml"))
    assert list((session.path / "campaign").glob("results_*.csv"))
    assert list((session.path / "campaign").glob("attempts_*.csv"))
    assert not (session.path / "campaign" / "results.csv").exists()
    assert heights_file.exists()

    campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)
    # 2 rounds * 3 clearances * 2 shapes = 12 inserts: round 1 grows 0..5, round 2 resets to 0..5
    assert [cnt for _, cnt in robot.found_counts_at_insert] == list(range(6)) + list(range(6))
    assert (session.path / "campaign" / "found_holes_round_1.yaml").exists()
    assert (session.path / "campaign" / "found_holes_round_2.yaml").exists()
    assert not holefit.found_path(session.path).exists()
    assert heights_file.read_text() == "touches:\n- run: old\n"


def test_extract_vision_board_falls_back_to_campaign_snapshot(tmp_path):
    session = SimpleNamespace(path=tmp_path / "session", cfg=Config.template())
    board = np.eye(4)
    board[:3, 3] = [0.5, 0.0, 0.01]
    snap_dir = session.path / "campaign"
    snap_dir.mkdir(parents=True)
    (snap_dir / campaign.EXTRACT_FOUND_FILE).write_text(yaml.safe_dump({
        "holes": [{
            "hole": "round_c3mm",
            "vision_xy": [0.5, 0.0],
            "found_xy": [0.502, -0.001],
            "sigma_m": 1e-4,
            "source": "wall touches (6)",
            "board_sha256": "sha1",
            "run": "r0",
        }]
    }))
    data = {
        "T_base_board": board.tolist(),
        "board_surface_z_m": 0.025,
        "source": {"board_pose_sha256": "sha1"},
    }
    corrected = extract._vision_board(session, data, None)
    assert corrected[0, 3] > board[0, 3] + 0.001
    assert corrected[1, 3] < board[1, 3] - 0.0005


def test_attempts_csv_and_status_metrics(setup, capsys):
    session, _ = setup(tilt_win={"round_c1mm"}, fail={"hexagon_c0p2mm"}, shapes=("round", "hexagon"), rounds=1)
    campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)

    results = list(csv.DictReader((session.path / "campaign" / "results.csv").open()))
    inserts = [r for r in results if r["role"] == "insert"]
    by_hole = {r["hole"]: r for r in inserts}
    assert by_hole["round_c3mm"]["method_succeeded"] == "plunge"
    assert by_hole["round_c3mm"]["attempts"] == "1"
    assert by_hole["round_c3mm"]["retries"] == "0"
    assert by_hole["round_c1mm"]["method_succeeded"] == "tilt"
    assert by_hole["round_c1mm"]["attempts"] == "2"
    assert by_hole["round_c1mm"]["retries"] == "1"
    assert by_hole["hexagon_c0p2mm"]["ok"] == "0"
    assert by_hole["hexagon_c0p2mm"]["method_succeeded"] == ""
    assert by_hole["hexagon_c0p2mm"]["retries"] == "1"

    attempts = list(csv.DictReader((session.path / "campaign" / "attempts.csv").open()))
    r1_attempts = [a for a in attempts if a["hole"] == "round_c1mm"]
    assert [(a["attempt_index"], a["method"], a["inserted"]) for a in r1_attempts] == [
        ("0", "plunge", "0"),
        ("1", "tilt", "1"),
    ]

    capsys.readouterr()
    campaign.cmd_status(session, SimpleNamespace())
    out = capsys.readouterr().out
    assert "retries/ins" in out
    assert "by clearance" in out
    assert "insert overall: 5/6 (83%), plunge 4, tilt 1" in out
    assert "avg retries/insertion 0.33" in out
    assert "parks: 1/1 into the c3mm holes" in out


def test_no_board_correction_holes_setting(setup):
    session, robot = setup(shapes=("round", "l_shape"), rounds=1)
    campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)
    by_hole = {a.hole: a.no_board_correction for a in robot.insert_args}
    assert by_hole["l_shape_c3mm"] is True
    for h, flag in by_hole.items():
        if h != "l_shape_c3mm":
            assert flag is False, f"expected no_board_correction=False for {h}"


def test_the_gripper_is_rehomed_after_each_completed_board(setup):
    session, robot = setup(shapes=("round",), rounds=2)
    campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)
    rehomes = [c for c in robot.calls if c[0] == "rehome"]
    assert rehomes == [("rehome", 1), ("rehome", 2)]
    # right after each round's last insert
    i = robot.calls.index(("rehome", 1))
    assert robot.calls[i - 1] == ("insert", "round_c0p2mm")


def test_rehome_lifts_then_homes_and_skips_with_a_peg_in_the_hand(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(insert, "lift", lambda cfg, s, out, d, name, **k: seen.append(("lift", d, name)) or True)
    monkeypatch.setattr(insert, "robot_program", lambda cfg, s, out, prog, argv: seen.append((prog, argv[1])) or 0)
    monkeypatch.setattr(insert, "gripper_state", lambda cfg, s: {"width": 0.0792, "grasped": False})
    session = SimpleNamespace(path=tmp_path / "session", cfg=Config.template())
    state = campaign.new_state(HOLES, Config.template()["campaign"], 2, ["round"])
    campaign.rehome_gripper(session, state, 1)
    assert seen == [("lift", 0.06, "rehome_lift"), ("gripper", "home")]
    seen.clear()
    state["pegs"]["round"]["at"] = "hand"
    campaign.rehome_gripper(session, state, 1)
    assert seen == []                                      # homing would open the hand


def test_a_failed_homing_stops_the_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(insert, "lift", lambda *a, **k: True)
    monkeypatch.setattr(insert, "robot_program", lambda *a, **k: 1)
    session = SimpleNamespace(path=tmp_path / "session", cfg=Config.template())
    state = campaign.new_state(HOLES, Config.template()["campaign"], 2, ["round"])
    with pytest.raises(Exception, match="homing after round 1 failed"):
        campaign.rehome_gripper(session, state, 1)


def test_insert_args_pass_corner_in():
    args = campaign._insert_args(SimpleNamespace(corner_in="on"), "rectangle_c0p2mm")
    assert args.corner_in == "on"
    assert campaign._insert_args(SimpleNamespace(), "rectangle_c0p2mm").corner_in is None
