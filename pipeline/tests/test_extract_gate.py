"""Offline tests of extract's grasp gate and put-back (robot 2026-11-25 logs as the cases; no robot)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_pipeline import campaign, extract, insert  # noqa: E402
from board_pipeline.config import Config  # noqa: E402

# The cases below are v2-finger extracts: the gate with the v2 fingers' normal width offsets (the config holds v3's).
GATE = {**Config.template()["extract"]["grasp_gate"],
        "width_off_mm": {"round": -0.80, "hexagon": -0.77, "triangle": -0.64, "rectangle": -1.25, "l_shape": -1.44}}


def _res(contact_mm, width_off_mm, seated_w=None, width=None, stalls=0):
    r = {"contact_above_board": contact_mm / 1e3, "width_off": width_off_mm / 1e3, "pull_stalls": stalls}
    if seated_w is not None:
        r.update(grasp_width_seated=seated_w / 1e3, width=width / 1e3)
    return r


# (shape, contact mm, width off mm) of real extracts, 2026-11-25
GOOD = [("hexagon", -22.80, -0.77), ("hexagon", -20.91, -0.77), ("round", -21.97, -0.81),
        ("triangle", -21.95, -0.64), ("rectangle", -22.29, -1.25), ("l_shape", -21.06, -1.34),
        ("l_shape", -22.01, -1.44)]
BAD = [("hexagon", -3.73, -2.95),     # 21:31 finger on the hexagon top, held by its top 4 mm (knocked over later)
       ("l_shape", -22.5, -2.44)]     # 19:28 the L slipped ~16 mm in the fingers during a stalled pull


@pytest.mark.parametrize("shape,contact,width_off", GOOD)
def test_good_extracts_pass_the_gate(shape, contact, width_off):
    assert extract.grasp_problem(_res(contact, width_off), shape, GATE) is None


@pytest.mark.parametrize("shape,contact,width_off", BAD)
def test_the_nights_bad_grasps_are_caught(shape, contact, width_off):
    assert extract.grasp_problem(_res(contact, width_off), shape, GATE)


def test_high_contact_alone_is_caught():
    assert "peg top" in extract.grasp_problem(_res(-3.7, -0.77), "hexagon", GATE)


def test_slip_between_grasp_and_pull_end_is_caught():
    # width off still within tolerance, but it changed 0.4 mm after the grasp
    p = extract.grasp_problem(_res(-22.0, -1.44, seated_w=11.0, width=10.6, stalls=10), "l_shape", GATE)
    assert p and "slipped" in p
    assert extract.grasp_problem(_res(-22.0, -1.44, seated_w=11.0, width=10.9), "l_shape", GATE) is None


def test_no_gate_and_unknown_shape_are_lenient():
    assert extract.grasp_problem(_res(-3.0, -5.0), "hexagon", None) is None
    assert extract.grasp_problem(_res(-22.0, -5.0), "star", GATE) is None


class FakeArm:
    """insert.lift / open_hand stand-ins: the tip goes where asked unless something is under it (stop_z: the tip
    stops there, the program fails like the force-change stop)."""

    def __init__(self, start_z, stop_z=None, open_ok=True):
        self.z, self.stop_z, self.open_ok = start_z, stop_z, open_ok
        self.calls = []

    def lift(self, cfg, s, out, distance, name, tool=None, dq=None, result=False, to_z=None, to_xy=None,
             upright=False):
        self.calls.append(("lift", name, to_z, to_xy, upright, s.get("lift_max_force_change")))
        assert to_z is not None and to_z - self.z >= -0.0500001, "a leg of more than 50 mm down"
        ok = True
        if self.stop_z is not None and to_z < self.stop_z <= self.z:
            self.z, ok = self.stop_z, False
        else:
            self.z = to_z
        (out / f"{name}.csv").write_text(f"time,p_x,p_y,p_z\n0,0,0,{self.z}\n")
        return ok

    def open_hand(self, cfg, s, out, width_mm=None):
        self.calls.append(("open", width_mm))
        return self.open_ok


def _put_back(tmp_path, monkeypatch, arm, seat_z=0.0036):
    monkeypatch.setattr(insert, "lift", arm.lift)
    monkeypatch.setattr(insert, "open_hand", arm.open_hand)
    result = {"seated_tip": [0.3899, 0.0517, seat_z], "end_tip": [0.3935, 0.0523, 0.0603]}
    s = {"lift_max_force_change": 50.0}
    return extract.put_back(None, s, tmp_path, result, None, None, GATE, 1, 20.5, 0.015, 0.0603)


def test_put_back_goes_over_the_seat_upright_then_down_in_short_legs(tmp_path, monkeypatch):
    arm = FakeArm(0.0603)
    assert _put_back(tmp_path, monkeypatch, arm) == "hole"
    over = arm.calls[0]
    assert over[1] == "put_back_1_over" and over[3] == [0.3899, 0.0517] and over[4] is True
    downs = [c for c in arm.calls if c[0] == "lift" and "_over" not in c[1] and "clear" not in c[1]]
    assert all(c[5] == GATE["put_back_force_change_n"] for c in downs)   # gentle stop on the way in
    assert abs(downs[-1][2] - (0.0036 + 0.001)) < 1e-9                 # 1 mm over the seat
    assert ("open", 20.5) in arm.calls
    assert arm.calls[-1][1] == "put_back_1_clear" and arm.calls[-1][2] >= 0.015


def test_put_back_stopped_high_keeps_the_peg_in_the_hand(tmp_path, monkeypatch):
    arm = FakeArm(0.0603, stop_z=0.0036 + 0.015)   # the peg on the rim, 15 mm over the seat
    assert _put_back(tmp_path, monkeypatch, arm) == "hand"
    assert not any(c[0] == "open" for c in arm.calls)
    assert arm.calls[-1][2] == pytest.approx(0.0603)                  # back up to the travel height


def test_put_back_stopped_a_little_short_is_in_the_hole(tmp_path, monkeypatch):
    arm = FakeArm(0.0603, stop_z=0.0036 + 0.004)    # slipped 4 mm: bottoms early, still deep in the pocket
    assert _put_back(tmp_path, monkeypatch, arm) == "hole"


def test_put_back_open_failure_is_unknown(tmp_path, monkeypatch):
    assert _put_back(tmp_path, monkeypatch, FakeArm(0.0603, open_ok=False)) == "unknown"


def test_campaign_keeps_an_untrusted_held_peg_in_the_hand_and_stops(tmp_path, monkeypatch):
    def fake_extract(session, args, probe_target, record=None, here=False, restart_launch=True):
        d = tmp_path / "x"
        d.mkdir(exist_ok=True)
        return {"ok": False, "peg": "hand", "out": d, "result": None, "error": "grasp not trusted: test"}

    monkeypatch.setattr(extract, "run", fake_extract)
    monkeypatch.setattr(insert, "start_launch", lambda cfg, s: None)
    monkeypatch.setattr(insert, "stop_launch", lambda cfg, s: None)
    monkeypatch.setattr(insert, "launch_running", lambda cfg, s: False)
    from board_pipeline.config import load_yaml
    holes = load_yaml(Path(__file__).resolve().parents[1] / "config" / "holes.yaml")["holes"]
    session = SimpleNamespace(path=tmp_path / "session", cfg=Config.template())
    campaign.save_state(session, campaign.new_state(holes, Config.template()["campaign"], 1, ["round"]))
    with pytest.raises(Exception, match="it is in the hand"):
        campaign.cmd_run(session, SimpleNamespace(dry_run=False, max_inserts=None), None)
    assert campaign.load_state(session)["pegs"]["round"]["at"] == "hand"


def test_upright_rotation_strips_tilt_and_keeps_yaw_modulo_period():
    from board_pipeline import grasp
    r_ref = grasp._rz(0.4) @ grasp.FLIP
    tilt = np.array([[math.cos(math.radians(2.5)), 0.0, math.sin(math.radians(2.5))],
                     [0.0, 1.0, 0.0],
                     [-math.sin(math.radians(2.5)), 0.0, math.cos(math.radians(2.5))]])
    r_peg = tilt @ grasp._rz(0.4 + math.radians(122.0)) @ grasp.FLIP
    out = extract.upright_rotation(r_peg, r_ref, math.radians(120.0))
    assert np.allclose(out[:, 2], r_ref[:, 2], atol=1e-12)
    yaw = math.atan2(out[1, 0], out[0, 0])
    assert math.degrees(math.remainder(yaw - math.atan2(r_ref[1, 0], r_ref[0, 0]), 2 * math.pi)) == pytest.approx(2.0, abs=0.2)
    assert np.allclose(extract.upright_rotation(r_peg, r_ref, 0.0), r_ref, atol=1e-12)


def test_extract_clamps_hover_z_above_board_and_levels_pose(tmp_path, monkeypatch):
    from board_pipeline import grasp
    cfg = Config.template()
    session_dir = tmp_path / "session"
    (session_dir / "targets").mkdir(parents=True)
    board = np.eye(4)
    board[:3, 3] = [0.39, 0.05, -0.0175]  # board_z = 0.0075
    target = session_dir / "targets" / "hexagon_c3mm.json"
    target.write_text(json.dumps({"T_base_board": board.tolist(), "board_surface_z_m": 0.025,
                                  "source": {"board_pose_sha256": "sha"}}))
    tool = SimpleNamespace(point=np.array([0.0, 0.0]), rotation=grasp.FLIP,
                           tip_offset=np.array([0.0, 0.0152, 0.039]), grasp_width_mm=12.519, period=math.radians(60.0))

    def probe_target(session, holes, name, tip_from, height_mm, turn, joint_correction=True, mirror=False):
        return tool, "CAD", target, ["--board-pose", "/x.json", "--ik-seed", 0, 0, 0, -2, 0, 2, 0]

    tilt = np.array([[math.cos(math.radians(2.8)), 0.0, math.sin(math.radians(2.8))],
                     [0.0, 1.0, 0.0],
                     [-math.sin(math.radians(2.8)), 0.0, math.cos(math.radians(2.8))]])
    tip_pose = np.eye(4)
    tip_pose[:3, :3] = tilt @ grasp.FLIP
    tip_pose[:3, 3] = [0.39, 0.05, 0.0075 - 0.0107]  # 10.7 mm deep
    record = {"hole": "hexagon_c3mm", "shape": "hexagon", "tip_pose": [float(v) for v in tip_pose.reshape(-1)],
              "board_z_m": 0.0075, "vision_board_z_m": 0.0075, "tip_offset_tcp_m": [0.0, 0.0152, 0.039],
              "period_rad": math.radians(60.0), "run": "r1"}

    seen = {}

    def fake_peg_search(c, st, out_dir, argv):
        seen["argv"] = list(map(str, argv))
        res_name = Path(seen["argv"][seen["argv"].index("--result") + 1]).name
        (out_dir / res_name).write_text(json.dumps({
            "ok": True, "stage": "done", "held": True, "contact_above_board": -0.022,
            "seated_shift_xy": [0.0, 0.0], "seated_turn_deg": 0.0, "width": 0.01175,
            "width_off": -0.00021, "pull_peak_lag_m": 0.002,  # width_off: the v3 hexagon normal (config)
        }))
        return 0

    monkeypatch.setattr(insert, "robot_session_dir", lambda c, st: tmp_path)
    monkeypatch.setattr(insert, "launch_running", lambda c, st: False)
    monkeypatch.setattr(insert, "move_to_hole", lambda *a, **k: True)
    monkeypatch.setattr(insert, "peg_search", fake_peg_search)

    session = SimpleNamespace(name="test", path=session_dir, cfg=cfg)
    args = SimpleNamespace(hole="hexagon_c3mm", no_joint_correction=True, dry_run=False)
    outcome = extract.run(session, args, probe_target, record=record, here=False, restart_launch=False)
    assert outcome["ok"] is True
    hover_z = float(seen["argv"][seen["argv"].index("--extract-hover-z") + 1])
    assert hover_z == pytest.approx(0.0075 + 0.002, abs=1e-6)
    gp = np.array([float(v) for v in seen["argv"][seen["argv"].index("--grasp-pose") + 1:
                                                  seen["argv"].index("--grasp-pose") + 17]]).reshape(4, 4)
    assert np.allclose(gp[:3, 2], grasp.FLIP[:, 2], atol=1e-9)


def test_identical_pull_failure_stops_retries_early(tmp_path, monkeypatch, capsys):
    cfg = Config.template()
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session = SimpleNamespace(name="test", path=session_dir, cfg=cfg)
    args = SimpleNamespace(hole="triangle_c3mm", no_joint_correction=True, dry_run=False)
    record = {"hole": "triangle_c3mm", "shape": "triangle", "tip_pose": [float(v) for v in np.eye(4).reshape(-1)],
              "board_z_m": 0.0075, "vision_board_z_m": 0.0075,
              "tip_offset_tcp_m": [0.0, 0.017, 0.039], "run": "r1"}

    calls = []

    def fake_peg_search(c, st, out_dir, argv):
        argv = list(map(str, argv))
        calls.append("extract")
        res_name = Path(argv[argv.index("--result") + 1]).name
        (out_dir / res_name).write_text(json.dumps({
            "ok": False, "stage": "pull", "error": "6.0 mm off the pull line",
            "contact_above_board": -0.0138, "end_tip": [0.39, -0.015, 0.005],
        }))
        return 1

    lifts = []
    monkeypatch.setattr(insert, "robot_session_dir", lambda c, st: tmp_path)
    monkeypatch.setattr(insert, "launch_running", lambda c, st: False)
    monkeypatch.setattr(insert, "peg_search", fake_peg_search)
    monkeypatch.setattr(insert, "open_hand", lambda *a, **k: True)
    monkeypatch.setattr(insert, "robot_program", lambda c, st, out_dir, prog, argv: 0)
    monkeypatch.setattr(insert, "lift", lambda c, st, out_dir, d, name, **k: lifts.append((name, k.get("to_z"))) or True)

    outcome = extract.run(session, args, None, record=record, here=True, restart_launch=False)
    out = capsys.readouterr().out
    assert outcome["ok"] is False
    assert outcome["attempts"] == 2  # stopped after 2 identical failures instead of 4
    assert "same failure repeated" in out
    assert lifts[-1][0] == "lift_recovery" and lifts[-1][1] == pytest.approx(0.0075 + 0.040)
