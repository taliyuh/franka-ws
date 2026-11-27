"""Offline tests of insert.move_to_hole's free-air force-trip retry (robot 2026-11-25 results as the cases)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_pipeline import insert  # noqa: E402

TRIP_2038 = {"ok": False, "stage": "run",
             "error": "joint move (transit): the external force changed by 54.261023 N, above --posture-max-force-change",
             "end_tip": [0.4531324409, -0.1381433252, 0.06091017186]}
TRIP_2104 = {"ok": False, "stage": "run",
             "error": "joint move (descend to target): the external force changed by 50.237722 N, above "
                      "--posture-max-force-change",
             "end_tip": [0.3930972972, 0.0518998258, 0.05218548236]}
HELD_BACK = {"ok": False, "stage": "run", "error": "joint move (transit): the tip 3.1 mm off the commanded one, above "
             "--posture-max-error (held back: the peg touched something?)", "end_tip": [0.45, -0.13, 0.061]}
S = {"robot_ip": "x", "free_air_trip_retries": 1, "free_air_trip_margin_mm": 5.0, "free_air_trip_wait_s": 0.0}


def test_the_nights_trips_are_free_air_and_touches_are_not():
    assert insert.free_air_trip(TRIP_2038, 0.050)
    assert insert.free_air_trip(TRIP_2104, 0.047)
    assert not insert.free_air_trip(HELD_BACK, 0.050)
    assert not insert.free_air_trip(TRIP_2104, 0.055)        # below the planned heights: maybe a touch
    assert not insert.free_air_trip({"ok": True}, 0.05)
    assert not insert.free_air_trip(TRIP_2038, None)


def test_gripper_state_parsing_and_held_check():
    st = insert.parse_gripper_state("width 11.62 mm (max 79.8 mm), grasped, temperature 31 C")
    assert st["grasped"] is True and abs(st["width"] - 0.01162) < 1e-12
    assert insert.parse_gripper_state("width 30 mm (max 79.8 mm), not grasped, temperature 31 C")["grasped"] is False
    assert insert.parse_gripper_state("garbage") is None
    assert insert.peg_still_held(st, 12.5, S)
    assert not insert.peg_still_held({"width": 0.0040, "grasped": True}, 12.5, S)   # closed past the peg: dropped
    assert not insert.peg_still_held({"width": 0.0116, "grasped": False}, 12.5, S)
    assert not insert.peg_still_held(None, 12.5, S)


class FakePrograms:
    def __init__(self, run_dir: Path, results: list):
        self.run_dir, self.results, self.names = run_dir, list(results), []

    def __call__(self, cfg, s, run_dir, program, argv):
        name = Path(argv[argv.index("--result") + 1]).stem
        self.names.append(name)
        res = self.results.pop(0)
        (run_dir / f"{name}.json").write_text(json.dumps(res))
        return 0 if res.get("ok") else 1


def _move(tmp_path, monkeypatch, results, holding=False, gripper=None, travel_z=0.0605, target_z=0.0525):
    prog = FakePrograms(tmp_path, results)
    monkeypatch.setattr(insert, "robot_program", prog)
    monkeypatch.setattr(insert, "gripper_state", lambda cfg, s: gripper)
    monkeypatch.setattr(insert, "_tool_and_load_args", lambda s, tool, dq: [])
    ok = insert.move_to_hole(None, S, tmp_path, [0.0] * 7, "approach", travel_z=travel_z, target_z=target_z,
                             tool=SimpleNamespace(grasp_width_mm=12.5), holding=holding)
    return ok, prog.names


def test_a_free_air_trip_is_moved_again_once(tmp_path, monkeypatch):
    ok, names = _move(tmp_path, monkeypatch, [TRIP_2104, {"ok": True}])
    assert ok and names == ["approach", "approach_retry1"]
    assert (tmp_path / "approach.json").exists()      # the trip's record kept


def test_only_one_retry(tmp_path, monkeypatch):
    ok, names = _move(tmp_path, monkeypatch, [TRIP_2104, TRIP_2104, {"ok": True}])
    assert not ok and names == ["approach", "approach_retry1"]


def test_a_touch_is_not_retried(tmp_path, monkeypatch):
    ok, names = _move(tmp_path, monkeypatch, [HELD_BACK, {"ok": True}])
    assert not ok and names == ["approach"]


def test_with_a_peg_the_retry_needs_it_still_in_the_hand(tmp_path, monkeypatch):
    held = {"width": 0.0116, "grasped": True}
    ok, names = _move(tmp_path, monkeypatch, [TRIP_2038, {"ok": True}], holding=True, gripper=held)
    assert ok and names == ["approach", "approach_retry1"]
    ok, names = _move(tmp_path, monkeypatch, [TRIP_2038, {"ok": True}], holding=True,
                      gripper={"width": 0.002, "grasped": False})
    assert not ok and names == ["approach"]
    ok, names = _move(tmp_path, monkeypatch, [TRIP_2038, {"ok": True}], holding=True, gripper=None)
    assert not ok and names == ["approach"]
