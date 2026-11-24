"""Offline tests of board_pipeline.insert's search geometry and belief (no robot, no ROS).

    cd <workspace>/pipeline && python3 -m pytest tests -q

The robot data tests replay results from sessions/2026-11-19_eye_in_hand_2/insert (2026-11-21) and are skipped if
that session is not here. PEG_SEARCH_BIN=<build dir>/peg_search also runs peg_search tilt-slide --dry-run with the
arguments insert builds for every hole (in the peg-search-libfranka-0.9.2 image).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_pipeline import insert  # noqa: E402
from board_pipeline.config import Config, PipelineError, load_yaml  # noqa: E402

PIPELINE = Path(__file__).resolve().parents[1]
TABLE = load_yaml(PIPELINE / "config" / "tilt_leads.yaml")["holes"]
HOLES = load_yaml(PIPELINE / "config" / "holes.yaml")["holes"]
S = Config.template()["insert"]
RUNS = PIPELINE.parent / "sessions" / "2026-11-19_eye_in_hand_2" / "insert"


def lead_of(hole: str, k: int = 0) -> dict:
    return TABLE[hole]["leads"][k]


def run_json(run: str, name: str) -> dict:
    path = RUNS / run / name
    if not path.exists():
        pytest.skip(f"{path} not here")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- B: lines
def test_flat_leads_keep_the_1mm_cap():
    plan = insert.line_plan(lead_of("hexagon_c1mm"), S, 0.005)
    assert plan["spacing_m"] == pytest.approx(0.001)
    assert plan["lines"] == 11  # 0, +-1 ... +-5 mm (line 5 covers 4.5..5.5)


def test_corner_leads_use_the_band_spacing():
    for hole in ("triangle_c1mm", "rectangle_c1mm", "l_shape_c1mm"):
        plan = insert.line_plan(lead_of(hole), S, 0.005)
        assert plan["spacing_m"] == pytest.approx(lead_of(hole)["line_spacing_m"]) and plan["spacing_m"] > 0.004
        assert plan["lines"] == 3, hole


def test_corner_cap_applies_when_set():
    plan = insert.line_plan(lead_of("triangle_c1mm"), {**S, "max_corner_line_spacing_m": 0.002}, 0.005)
    assert plan["spacing_m"] == pytest.approx(0.002) and plan["lines"] == 5


def test_lines_cover_the_disc():
    """Line k covers half a spacing either side of k x spacing (from its centre); together out to the error."""
    for hole, entry in TABLE.items():
        for lead in entry["leads"]:
            if lead.get("dips") is False:
                continue
            for error in (0.0015, 0.003, 0.005):
                plan = insert.line_plan(lead, S, error)
                k = (plan["lines"] - 1) // 2
                assert (k + 0.5) * plan["spacing_m"] >= error - 1e-12, (hole, error)
                assert k == 0 or (k - 0.5) * plan["spacing_m"] < error, (hole, error)  # no ring more than needed


def test_line_centre_puts_line_0_on_the_band_middle():
    """The lines' centre across the slide (tip line from the hole) puts the lead's line on the dip band's middle."""
    for hole in ("l_shape_c1mm", "rectangle_c1mm", "triangle_c1mm", "hexagon_c1mm"):
        lead = lead_of(hole)
        centre = insert.line_plan(lead, S, 0.005)["centre_m"]
        band = lead["eff_band_m"]
        assert centre + lead["lead_across_m"] == pytest.approx(0.5 * (band[0] + band[1]))
    assert abs(insert.line_plan(lead_of("l_shape_c1mm"), S, 0.005)["centre_m"]) > 0.008  # the corner 8.3 mm across
    assert abs(insert.line_plan(lead_of("hexagon_c1mm"), S, 0.005)["centre_m"]) < 1e-4  # symmetric


def test_l_shape_line_0_now_covers_both_sides():
    """Before: the tip's line through the hole, the corner lead_across beside it: line 0 covered hole errors
    [lead_across - band hi, lead_across - band lo] (-2.3..+19 mm at 1 mm). Now centred: +- half the band."""
    lead = lead_of("l_shape_c1mm")
    lo, hi = lead["eff_band_m"]
    before = (lead["lead_across_m"] - hi, lead["lead_across_m"] - lo)
    centre = insert.line_plan(lead, S, 0.005)["centre_m"]
    after = (centre + lead["lead_across_m"] - hi, centre + lead["lead_across_m"] - lo)
    assert before[0] > -0.005  # missed part of the 5 mm disc
    assert after[0] < -0.005 and after[1] > 0.005  # line 0 alone covers it


def test_search_args_pass_the_plan():
    lead = lead_of("l_shape_c1mm")
    args = list(map(str, insert.search_args(lead, math.radians(90.0), S, "lines", 0.005)))
    plan = insert.line_plan(lead, S, 0.005)
    assert args[args.index("--line-centre") + 1] == f"{plan['centre_m']:.5f}"
    assert args[args.index("--lines") + 1] == str(plan["lines"])
    assert float(args[args.index("--line-spacing") + 1]) == pytest.approx(plan["spacing_m"], abs=1e-5)


def test_choose_lead_prefers_the_base_direction_among_equals():
    entry = TABLE["triangle_c1mm"]
    lead, slide = insert.choose_lead(entry, 0.0, 90.0, S, 0.005)
    gaps = [abs(math.remainder(math.radians(l["d_deg"]) - math.radians(90.0), 2 * math.pi)) for l in entry["leads"]]
    assert abs(math.remainder(slide - math.radians(90.0), 2 * math.pi)) == pytest.approx(min(gaps))


def test_choose_lead_keeps_the_simulations_group_on_a_tie():
    for hole in ("l_shape_c3mm", "l_shape_c1mm", "l_shape_c0p2mm", "l_shape_c0p03mm"):
        lead, _ = insert.choose_lead(TABLE[hole], 0.0, 90.0, S, 0.005)
        assert lead["group"] == 0, hole


def test_choose_lead_takes_fewer_lines():
    entry = {"leads": [{**lead_of("l_shape_c1mm", 0), "eff_band_m": [-0.0005, 0.0005], "line_spacing_m": 0.0007},
                       lead_of("l_shape_c1mm", 1)]}
    lead, _ = insert.choose_lead(entry, 0.0, 90.0, S, 0.005)
    assert lead["group"] == 1


def test_old_table_is_refused():
    old = {k: v for k, v in lead_of("triangle_c1mm").items() if k not in ("near_m", "band_m", "along_m")}
    with pytest.raises(PipelineError, match="regenerate"):
        insert.line_plan(old, S, 0.005)


# ---------------------------------------------------------------- A1: the dip belief
def test_dip_belief_on_the_robots_triangle_run():
    """Robot 2026-11-21 141115, triangle_c1mm, inserted on line 0: the hold (the tip on the hole axis just before
    it went in) is the hole. The old belief (dip lead + peg_radius along d) was 5.2 mm off it."""
    result = run_json("2026-11-21_141115", "attempt_1.json")
    plan = yaml.safe_load((RUNS / "2026-11-21_141115" / "plan.yaml").read_text())
    lead, truth = plan["lead"], np.asarray(result["hold_xy"])
    lead = next(l for l in TABLE["triangle_c1mm"]["leads"] if l["lead"] == lead["lead"])
    d = np.asarray(result["slide_dir"][:2]) / np.linalg.norm(result["slide_dir"][:2])
    old = np.asarray(result["dip_lead"][:2]) + lead["peg_radius_m"] * d
    belief = insert.next_belief({**result, "hold_xy": None}, lead, S, None, prior_xy=plan["hole_base_m"][:2],
                                prior_radius=0.005, max_radius=0.005)
    new = np.asarray(belief["xy"])
    print(f"old belief {np.linalg.norm(old - truth) * 1e3:.2f} mm from the hole, new {np.linalg.norm(new - truth) * 1e3:.2f}"
          f" mm (radius {belief['radius_m'] * 1e3:.2f} mm)")
    assert np.linalg.norm(old - truth) > S["dip_radius_m"]  # the bug: outside its own radius
    assert np.linalg.norm(new - truth) < belief["radius_m"]
    assert abs((new - truth) @ d) < 0.0015  # along the slide (the hole chamfer dips the lead early: ~1.2 mm)


def test_dip_belief_geometry():
    """Synthetic: a lead entering the dip region at (along_m[0], offset) from a hole at H."""
    lead = lead_of("l_shape_c1mm")
    d = np.array([0.0, 1.0])
    a = np.array([-1.0, 0.0])
    H = np.array([0.40, -0.02])
    offset = 0.5 * (lead["band_m"][0] + lead["band_m"][1]) + 0.001
    dip = H + lead["along_m"][0] * d + offset * a
    xy, half = insert.dip_hole(list(dip) + [0.0], list(d) + [0.0], lead, H, 0.05)
    assert abs((xy - H) @ d) < 1e-9
    assert abs((xy - H) @ a) <= half + 1e-9
    # the prior disc limits the across estimate
    xy2, half2 = insert.dip_hole(list(dip) + [0.0], list(d) + [0.0], lead, H, 0.002)
    assert half2 == pytest.approx(0.002) and abs((xy2 - H) @ a) < 1e-9


def test_dip_belief_radius_is_capped_by_the_hole_error():
    result = {"dip_lead": [0.4, -0.03, 0.006], "slide_dir": [0.0, 1.0, 0.0], "expected_hole": [0.4, -0.016]}
    belief = insert.next_belief(result, lead_of("triangle_c1mm"), S, None, max_radius=0.005)
    assert belief["radius_m"] <= 0.005


# ---------------------------------------------------------------- A3: a hold that jammed
def test_jammed_hold_gets_the_jam_radius():
    """Robot 2026-11-21 134814: the plunge stuck, the hold jammed 2.02 mm from the hole (141115's hold, the same
    hole): outside the 2 mm located radius, inside the jam radius."""
    jammed = run_json("2026-11-21_134814", "attempt_1.json")
    truth = np.asarray(run_json("2026-11-21_141115", "attempt_1.json")["hold_xy"])
    belief = insert.next_belief(jammed, lead_of("triangle_c1mm"), S, None)
    distance = np.linalg.norm(np.asarray(belief["xy"]) - truth)
    assert belief["source"].startswith("hold that jammed")
    assert S["located_radius_m"] < distance < belief["radius_m"] == pytest.approx(S["jam_radius_m"])


def test_clean_hold_keeps_the_located_radius():
    result = {"hold_xy": [0.45, -0.016], "outcomes": ["shallow"], "inserted": False}
    belief = insert.next_belief(result, lead_of("triangle_c1mm"), S, None)
    assert belief["radius_m"] == pytest.approx(S["located_radius_m"])


def test_a_hold_beats_a_later_dip():
    hold = insert.next_belief({"hold_xy": [0.45, -0.016], "outcomes": ["jammed"]}, lead_of("triangle_c1mm"), S, None)
    after = insert.next_belief({"dip_lead": [0.4, -0.03, 0.0], "slide_dir": [0, 1, 0]}, lead_of("triangle_c1mm"), S,
                               hold)
    assert after is hold


# ---------------------------------------------------------------- A4: the summary of the plunge flow
def test_touch_points_clear_all_holes():
    board = np.eye(4)
    for name, hole in HOLES.items():
        tool = SimpleNamespace(point=np.array([hole["x_m"], hole["y_m"]]))
        pts = insert.touch_points(HOLES, name, tool, board, 0.025, 0.022)
        assert len(pts) == 3
        # Non-collinear triangle
        u, v = pts[1][:2] - pts[0][:2], pts[2][:2] - pts[0][:2]
        area2 = abs(float(u[0] * v[1] - u[1] * v[0]))
        assert area2 == pytest.approx(0.022 ** 2, rel=1e-6)
        for pt in pts:
            nearest = min(math.hypot(pt[0] - h["x_m"], pt[1] - h["y_m"]) for h in HOLES.values())
            assert nearest >= 0.0219, (name, nearest)


def test_plunge_passes_measured_plane_and_avoids_zero_slope_freeze(tmp_path, monkeypatch):
    seen = []

    def run_case(peck_json):
        seen.clear()

        def fake(cfg, s, out, argv):
            argv = list(map(str, argv))
            seen.append(argv)
            if argv[1] == "peck":
                (out / "peck.json").write_text(json.dumps(peck_json))
            else:
                (out / Path(argv[argv.index("--result") + 1]).name).write_text(
                    json.dumps({"ok": True, "stage": "done", "inserted": False, "outcomes": ["missed"]}))
            return 0

        monkeypatch.setattr(insert, "peg_search", fake)
        monkeypatch.setattr(insert, "robot_program", lambda *a, **k: 0)
        summary = {"attempts": [], "inserted": False}
        insert._plunge_then_tilt(None, S, tmp_path, ["ip", "peck", "/insert/peck.csv", "--result", "/insert/peck.json"],
                                 ["--hole-xy", "0.452", "-0.0166"], summary, 1, 6.0,
                                 np.array([0.452, -0.0166, 0.0065]), np.array([0.43, -0.0166, 0.0065]),
                                 lead_of("triangle_c1mm"), 0.002)
        return seen[1], summary

    # With a 3-point measured plane: tilt-slide receives the measured slope
    argv_plane, summary = run_case({"ok": True, "stage": "done", "inserted": False, "board_measured": 0.0064,
                                    "plane": {"point": [0.452, -0.0166, 0.0064], "slope": [0.001, -0.006]},
                                    "first": {"xy": [0.452, -0.0166], "tip_z": 0.0063, "result": "board"}})
    i = argv_plane.index("--board-plane")
    assert [float(v) for v in argv_plane[i + 1:i + 6]] == pytest.approx([0.452, -0.0166, 0.0064, 0.001, -0.006])
    assert summary["plane"]["slope"] == pytest.approx([0.001, -0.006])

    # Without a measured plane (legacy 1-point): does NOT freeze [0, 0], uses --probe-points 0 (online BoardPlane)
    argv_no_plane, _ = run_case({"ok": True, "stage": "done", "inserted": False, "board_measured": 0.0064,
                                 "first": {"xy": [0.452, -0.0166], "tip_z": 0.0063, "result": "board"}})
    assert "--board-plane" not in argv_no_plane
    assert argv_no_plane[argv_no_plane.index("--probe-points") + 1] == "0"


def test_plunge_flow_keeps_its_plane_and_belief(tmp_path, monkeypatch):
    """_plunge_then_tilt's plane and belief reach summary.yaml (2026-11-21: run()'s finally overwrote them with null)."""
    plunge = {"ok": True, "stage": "done", "inserted": False, "board_measured": 0.0065,
              "first": {"xy": [0.452, -0.0166], "tip_z": 0.0065, "result": "board"}}
    attempt = {"ok": True, "stage": "done", "inserted": False, "outcomes": ["jammed"], "hold_xy": [0.4505, -0.0177]}

    def fake(cfg, s, out, argv):
        name = "peck.json" if argv[1] == "peck" else Path(argv[argv.index("--result") + 1]).name
        (out / name).write_text(json.dumps(plunge if argv[1] == "peck" else attempt))
        return 0

    monkeypatch.setattr(insert, "peg_search", fake)
    monkeypatch.setattr(insert, "robot_program", lambda *a, **k: 0)
    summary = {"attempts": [], "inserted": False}
    common = ["--result-placeholder"]
    drop = ["ip", "peck", "/insert/peck.csv", "--result", "/insert/peck.json"]
    ok = insert._plunge_then_tilt(None, S, tmp_path, drop, common, summary, 1, 6.0, np.array([0.452, -0.0166, 0.0065]),
                                  np.array([0.43, -0.0166, 0.0065]), lead_of("triangle_c1mm"), 0.005)
    assert not ok
    assert summary["plane"]["point"][2] == pytest.approx(0.0065)
    assert summary["belief"]["source"].startswith("hold that jammed")
    source = (PIPELINE / "board_pipeline" / "insert.py").read_text()
    assert 'if method != "plunge":' in source  # run()'s finally leaves them


# ---------------------------------------------------------------- peg_search accepts what insert builds
@pytest.mark.skipif(not os.environ.get("PEG_SEARCH_BIN"), reason="PEG_SEARCH_BIN not set")
def test_peg_search_dry_run_takes_every_holes_arguments(tmp_path):
    binary = Path(os.environ["PEG_SEARCH_BIN"]).resolve()
    hole_xy, board_z = (0.45, -0.0166), 0.0065
    failures = []
    for name in HOLES:
        _, entry = insert.hole_leads(type("C", (), {"workspace": PIPELINE.parent})(), HOLES, name)
        shape = HOLES[name]["shape"]
        error = insert.search_error(S, entry)
        lead, slide = insert.round_lead(entry, 90.0) if shape == "round" else insert.choose_lead(entry, 0.3, 90.0, S, error)
        lead = insert.sized_lead(lead, float(entry["error_m"]), error)
        argv = ["172.16.0.2", "tilt-slide", f"/out/dry_{name}.csv", "--approach", "0.0075", *map(str, S["peg_search_options"]),
                "--peg-axis-x", "0", "--peg-axis-y", "0", "--peg-tip-z", "0.15", "--load-mass", "0.68", "--load-com",
                "0.02", "-0.10", "-0.12", "--probe-radius", "0.025", "--hole-xy", *map(str, hole_xy),
                "--board-z", str(board_z), "--lift-at-end", "--yes",
                *map(str, insert.search_args(lead, slide, S, "lines", error)),
                "--board-plane", "0.43", "-0.0166", str(board_z), "0", "0", "--posture", "start",
                "--located-xy", "0.4505", "-0.0177", "--located-radius", f"{min(S['jam_radius_m'], error):.4f}", "--hole-error", f"{error:.4f}"]
        if lead.get("dip_region_m"):  # and the grid's plan (--line-offsets, --hole-xy, --lines) after them
            spacing = insert.line_plan(lead, S, error)["spacing_m"]
            g = insert.make_grid(S, np.array([*hole_xy, board_z]), error, lead, slide, entry["free_m"], spacing)
            _, extra = insert.grid_args(g, S, lead, error, tmp_path, 1)
            argv = argv[:argv.index("--located-xy")] + ["--hole-error", f"{error:.4f}"] + extra  # grid: no located xy
        argv.append("--dry-run")
        command = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}", "-v", f"{binary.parent}:/bin_dir:ro", "-v", f"{tmp_path}:/out",
                   "peg-search-libfranka-0.9.2", "/bin_dir/peg_search", *argv]
        done = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if done.returncode != 0:
            failures.append(f"{name}: {(done.stderr or done.stdout).strip().splitlines()[-1:]}")
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------- C.2: a 2 mm error disc
def test_template_error_is_2mm():
    assert insert.search_error(S, TABLE["triangle_c1mm"]) == pytest.approx(0.002)
    assert insert.search_error({**S, "error_m": None}, TABLE["triangle_c1mm"]) == pytest.approx(0.005)


def test_sized_lead_shrinks_run_up_and_slide():
    lead = lead_of("hexagon_c1mm")
    sized = insert.sized_lead(lead, 0.005, 0.002)
    assert sized["lead_run_up_m"] == pytest.approx(lead["lead_run_up_m"] - 0.003)
    assert sized["slide_length_m"] == pytest.approx(lead["slide_length_m"] - 0.006)
    assert sized["spiral_radius_m"] == pytest.approx(lead["spiral_radius_m"] - 0.003)
    # still the run-up the simulation sizes: the error + the near rim + 3 mm
    assert sized["lead_run_up_m"] == pytest.approx(0.002 + lead["near_m"] + 0.003, abs=1e-5)


def test_2mm_disc_lines():
    assert insert.line_plan(lead_of("hexagon_c1mm"), S, 0.002)["lines"] == 5  # flat, 1 mm: 0, +-1, +-2
    for hole in ("triangle_c1mm", "rectangle_c1mm", "l_shape_c1mm"):
        assert insert.line_plan(lead_of(hole), S, 0.002)["lines"] == 1, hole  # one 5 mm line covers +-2.5


def test_search_args_for_2mm():
    lead = insert.sized_lead(lead_of("triangle_c1mm"), 0.005, 0.002)
    args = list(map(str, insert.search_args(lead, 1.5, S, "lines", 0.002)))
    assert args[args.index("--hole-error") + 1] == "0.0020"
    assert float(args[args.index("--located-radius") + 1]) <= 0.002
    assert float(args[args.index("--lead-run-up") + 1]) == pytest.approx(lead["lead_run_up_m"], abs=1e-5)


def test_belief_radii_capped_at_the_disc():
    jam = insert.next_belief({"hold_xy": [0.45, -0.016], "outcomes": ["jammed"]}, lead_of("triangle_c1mm"), S, None,
                             max_radius=0.002)
    assert jam["radius_m"] == pytest.approx(0.002)


# ---------------------------------------------------------------- C.1: centring, the record, the correction
from types import SimpleNamespace  # noqa: E402

from board_pipeline import holefit  # noqa: E402
from tests.test_holefit import simulate_touches  # noqa: E402


def _tool():
    return SimpleNamespace(point=np.array([0.0, 0.0]), tip_offset=np.array([0.0, -0.017, 0.039]))


def test_centre_and_record_from_touches(tmp_path, monkeypatch):
    hole, yaw = "hexagon_c1mm", 0.4
    truth = np.array([0.4520, -0.0163])
    rng = np.random.default_rng(11)
    seat, touches = simulate_touches(hole, truth, yaw, rng)
    seen = {}

    def fake(cfg, s, out, argv):
        seen["argv"] = list(map(str, argv))
        (out / "centre.json").write_text(json.dumps({"ok": True, "stage": "done", "contacts": 6, "touches": touches}))
        return 0

    monkeypatch.setattr(insert, "peg_search", fake)
    session = SimpleNamespace(path=tmp_path)
    out = tmp_path / "insert" / "run1"
    out.mkdir(parents=True)
    summary = {"inserted_by": {"inserted": True, "inserted_xy": list(seat)}, "plane": None}
    entry = insert._centre_and_record(None, S, out, session, SimpleNamespace(hole=hole, no_centre=False), _tool(),
                                      TABLE[hole], HOLES, hole, np.array([0.4525, -0.0166, 0.0065]), yaw, "sha",
                                      summary, np.array([0.4525, -0.0166, 0.0065]))
    assert seen["argv"][1] == "centre" and "--centre-dirs-deg" in seen["argv"]
    assert "--posture-max-force-change" in seen["argv"]
    assert entry["source"].startswith("wall touches")
    assert np.linalg.norm(np.asarray(entry["found_xy"]) - truth) < 5e-5
    assert holefit.load_found(tmp_path)[0]["vision_xy"] == pytest.approx([0.4525, -0.0166])


def test_centre_skipped_for_tight_holes(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(insert, "peg_search", lambda *a, **k: called.append(a) or 0)
    session = SimpleNamespace(path=tmp_path)
    out = tmp_path / "insert" / "run_tight"
    out.mkdir(parents=True)
    for hole in ("hexagon_c0p03mm", "triangle_c0p2mm"):
        summary = {"inserted_by": {"inserted": True, "hold_xy": [0.452, -0.0163]}}
        entry = insert._centre_and_record(None, S, out, session, SimpleNamespace(hole=hole, no_centre=False),
                                          _tool(), TABLE[hole], HOLES, hole,
                                          np.array([0.4524, -0.0166, 0.0065]), 0.0, "sha", summary,
                                          np.array([0.4524, -0.0166, 0.0065]))
        assert entry["source"] == "seated tip (hold_xy)"
    assert not called


def test_lift_and_reset_pass_tool_and_load(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(insert, "robot_program", lambda cfg, s, run_dir, prog, argv: seen.append(list(map(str, argv))) or 0)
    dq = [0.0, -0.0024, 0.0045, -0.0036, 0.0072, -0.0083, 0.0]
    insert.lift(None, S, tmp_path, 0.15, "lift_test", tool=_tool(), dq=dq)
    assert len(seen) == 1
    argv = seen[0]
    assert "--peg-axis-y" in argv and "--peg-tip-z" in argv
    assert "--load-mass" in argv and "--load-com" in argv
    assert "--link6-load-mass" in argv and "--link6-load-com" in argv
    assert "--joint-offsets" in argv


def test_calibrated_handoff_passes_true_frame_and_joint_offsets(tmp_path, monkeypatch):
    """When calibration.json has joint_offsets_rad and target has T_base_board_true_frame:
    - probe_board keeps the nominal-FK corrected board pose (with board_correction applied)
    - peg_search gets --joint-offsets, --link6-load-mass/--link6-load-com, and aims in the TRUE calibrated frame."""
    session_dir = tmp_path / "session"
    (session_dir / "targets").mkdir(parents=True)
    dq = {f"panda_joint{i}": v for i, v in enumerate([0.0, -0.0024, 0.0045, -0.0036, 0.0072, -0.0083, 0.0], 1)}
    (session_dir / "calibration.json").write_text(json.dumps({"joint_offsets_rad": dq}))
    true_board = np.eye(4)
    true_board[:3, 3] = [0.450, -0.016, -0.019]
    nom_board = np.eye(4)
    nom_board[:3, 3] = [0.453, -0.019, -0.0185]  # +3, -3, +0.5 mm nominal shift
    target = session_dir / "targets" / "triangle_c1mm.json"
    target.write_text(json.dumps({
        "T_base_board": nom_board.tolist(), "board_surface_z_m": 0.025,
        "source": {"board_pose_sha256": "sha-c", "T_base_board_true_frame": true_board.tolist()}}))
    holefit.record_found(session_dir, "round_c3mm", [0.42, -0.10],
                         {"xy": [0.4194, -0.10], "sigma_m": 1e-4, "source": "t"}, "sha-c", "r0")

    def probe_target(session, holes, name, tip_from, height_mm, turn, joint_correction=True):
        return _tool(), "CAD", target, ["--board-pose", "/franka_ws/x.json", "--offset", 0, 0, 0.006]

    seen = {}

    def fake_peg_search(cfg, s, out, argv):
        seen["argv"] = list(map(str, argv))
        return 0

    monkeypatch.setattr(insert, "peg_search", fake_peg_search)
    monkeypatch.setattr(insert.ros, "controller_states", lambda cfg: {})
    monkeypatch.setattr(insert.ros, "container_path", lambda cfg, p: f"/c/{Path(p).name}")
    cfg = Config(dict(Config.template().data))
    session = SimpleNamespace(cfg=cfg, path=session_dir, name="t")
    args = SimpleNamespace(hole="triangle_c1mm", attempts=None, height=None, tip_from=None, turn=0.0,
                           no_joint_correction=False, slide_dir=None, pattern=None, method="plunge", here=False,
                           dry_run=True, no_board_correction=False, no_centre=False, search=None)
    insert.run(session, args, probe_target)
    argv = seen["argv"]
    assert "--joint-offsets" in argv
    assert "--link6-load-mass" in argv and "--link6-load-com" in argv
    assert argv.count("--peck-touch") == 3
    # peg_search aims in the TRUE calibrated frame (0.450 - 0.0006 board correction = 0.4494), NOT 0.453!
    hole_x = float(argv[argv.index("--hole-xy") + 1])
    hole_z = float(argv[argv.index("--board-z") + 1])
    assert hole_x == pytest.approx(0.450 - 0.0006, abs=1e-5)
    assert hole_z == pytest.approx(0.006, abs=1e-5)
    # probe_board's corrected file stays in the NOMINAL frame shifted by the same -0.6 mm
    corrected = json.loads((session_dir / "targets" / "triangle_c1mm_corrected.json").read_text())
    assert corrected["T_base_board"][0][3] == pytest.approx(0.453 - 0.0006, abs=1e-5)
    # With --no-joint-correction, peg_search omits --joint-offsets and uses the nominal board pose
    args.no_joint_correction = True
    insert.run(session, args, probe_target)
    assert "--joint-offsets" not in seen["argv"]
    assert float(seen["argv"][seen["argv"].index("--hole-xy") + 1]) == pytest.approx(0.453 - 0.0006, abs=1e-5)


def test_board_height_model_one_touch_and_predicted_board(tmp_path, monkeypatch, capsys):
    """Touches of two earlier runs spread over the board (this board pose): insert touches once and aims the peck at
    the model's board (vision's + 1.5 mm here), not vision's; with --touches 3 the three again."""
    from board_pipeline import boardheight
    session_dir = tmp_path / "session"
    (session_dir / "targets").mkdir(parents=True)
    board = np.eye(4)
    board[:3, 3] = [0.45, -0.0166, -0.019]  # vision's board top at z 0.006
    target = session_dir / "targets" / "triangle_c1mm.json"
    target.write_text(json.dumps({"T_base_board": board.tolist(), "board_surface_z_m": 0.025,
                                  "source": {"board_pose_sha256": "sha-h"}}))
    for run, (x, y) in (("r1", (0.45, -0.02)), ("r2", (0.50, -0.08))):
        pts = [(x, y, 0.0075), (x - 0.022, y, 0.0075), (x - 0.022, y - 0.022, 0.0075)]
        boardheight.record(session_dir, run, "triangle", "sha-h", pts, [0.006] * 3)

    def probe_target(session, holes, name, tip_from, height_mm, turn, joint_correction=True):
        return _tool(), "CAD", target, ["--board-pose", "/franka_ws/x.json", "--offset", 0, 0, 0.006]

    seen = {}
    monkeypatch.setattr(insert, "peg_search", lambda cfg, s, out, argv: seen.update(argv=list(map(str, argv))) or 0)
    monkeypatch.setattr(insert.ros, "controller_states", lambda cfg: {})
    monkeypatch.setattr(insert.ros, "container_path", lambda cfg, p: f"/c/{Path(p).name}")
    cfg = Config(dict(Config.template().data))
    session = SimpleNamespace(cfg=cfg, path=session_dir, name="t")
    args = SimpleNamespace(hole="triangle_c1mm", attempts=None, height=None, tip_from=None, turn=0.0,
                           no_joint_correction=True, slide_dir=None, pattern=None, method="plunge", here=False,
                           dry_run=True, no_board_correction=True, no_centre=False, search=None, touches=None)
    insert.run(session, args, probe_target)
    argv = seen["argv"]
    assert argv.count("--peck-touch") == 0
    assert float(argv[argv.index("--board-z") + 1]) == pytest.approx(0.0075, abs=2e-5)
    assert "no touch" in capsys.readouterr().out
    args.touches = "auto"
    insert.run(session, args, probe_target)
    assert seen["argv"].count("--peck-touch") == 1
    assert "one board touch" in capsys.readouterr().out
    args.touches = "3"
    insert.run(session, args, probe_target)
    assert seen["argv"].count("--peck-touch") == 3


def test_board_ref_travel_from_the_lead():
    lead = {"lead_run_up_m": 0.01616, "near_m": 0.01116}
    assert insert.board_ref_travel(lead, {"board_ref_travel_max_m": 0.002}, 0.002) == pytest.approx(0.002)
    assert insert.board_ref_travel(lead, {"board_ref_travel_max_m": 0.002, "board_ref_margin_m": 0.0025},
                                   0.002) == pytest.approx(0.0005)
    assert insert.board_ref_travel(lead, {"board_ref_travel_max_m": 0.0}, 0.002) == 0.0
    assert insert.board_ref_travel({"lead_run_up_m": 0.01}, {}, 0.002) == 0.0


def test_centre_failure_records_the_seat(tmp_path, monkeypatch):
    monkeypatch.setattr(insert, "peg_search", lambda cfg, s, out, argv: 1)
    session = SimpleNamespace(path=tmp_path)
    out = tmp_path / "insert" / "run2"
    out.mkdir(parents=True)
    summary = {"inserted_by": {"inserted": True, "hold_xy": [0.452, -0.0163], "end_tip": [0.4518, -0.0174, -0.009]}}
    entry = insert._centre_and_record(None, S, out, session, SimpleNamespace(hole="triangle_c1mm", no_centre=False),
                                      _tool(), TABLE["triangle_c1mm"], HOLES, "triangle_c1mm",
                                      np.array([0.4524, -0.0166, 0.0065]), 0.0, "sha", summary,
                                      np.array([0.4524, -0.0166, 0.0065]))
    assert entry["source"] == "seated tip (hold_xy)"
    assert entry["found_xy"] == pytest.approx([0.452, -0.0163])


def test_dry_run_applies_the_board_correction(tmp_path, monkeypatch, capsys):
    """A found hole of this board pose: insert aims at the corrected target (the probe's file and hole_base)."""
    session_dir = tmp_path / "session"
    (session_dir / "targets").mkdir(parents=True)
    board = np.eye(4)
    board[:3, 3] = [0.45, -0.0166, -0.0185]
    target = session_dir / "targets" / "triangle_c1mm.json"
    target.write_text(json.dumps({"T_base_board": board.tolist(), "board_surface_z_m": 0.025,
                                  "source": {"board_pose_sha256": "sha-b"}}))
    holefit.record_found(session_dir, "round_c3mm", [0.42, -0.10], {"xy": [0.4195, -0.1], "sigma_m": 1e-4,
                                                                     "source": "t"}, "sha-b", "r0")

    def probe_target(session, holes, name, tip_from, height_mm, turn, joint_correction=True):
        return _tool(), "CAD", target, ["--board-pose", "/franka_ws/x.json", "--offset", 0, 0, 0.006]

    seen = {}

    def fake_peg_search(cfg, s, out, argv):
        seen["argv"] = list(map(str, argv))
        return 0

    monkeypatch.setattr(insert, "peg_search", fake_peg_search)
    monkeypatch.setattr(insert.ros, "controller_states", lambda cfg: {})
    monkeypatch.setattr(insert.ros, "container_path", lambda cfg, p: f"/c/{Path(p).name}")
    cfg = Config(dict(Config.template().data))
    session = SimpleNamespace(cfg=cfg, path=session_dir, name="t")
    args = SimpleNamespace(hole="triangle_c1mm", attempts=None, height=None, tip_from=None, turn=0.0,
                           no_joint_correction=False, slide_dir=None, pattern=None, method="peck", here=False,
                           dry_run=True, no_board_correction=False, no_centre=False, search=None)
    insert.run(session, args, probe_target)
    corrected = json.loads((session_dir / "targets" / "triangle_c1mm_corrected.json").read_text())
    assert corrected["T_base_board"][0][3] == pytest.approx(0.45 - 0.0005, abs=1e-5)
    hole_xy = seen["argv"][seen["argv"].index("--hole-xy") + 1]
    assert float(hole_xy) == pytest.approx(0.45 - 0.0005, abs=1e-5)
    assert "board correction from 1 found hole" in capsys.readouterr().out
    args.no_board_correction = True
    insert.run(session, args, probe_target)
    assert float(seen["argv"][seen["argv"].index("--hole-xy") + 1]) == pytest.approx(0.45, abs=1e-6)


@pytest.mark.skipif(not os.environ.get("PEG_SEARCH_BIN"), reason="PEG_SEARCH_BIN not set")
def test_peg_search_centre_dry_run_takes_every_holes_arguments(tmp_path, monkeypatch):
    """The centre argv insert builds (captured from _centre_and_record), checked by peg_search --dry-run."""
    binary = Path(os.environ["PEG_SEARCH_BIN"]).resolve()
    failures = []
    for name in HOLES:
        table_name, entry = insert.hole_leads(type("C", (), {"workspace": PIPELINE.parent})(), HOLES, name)
        captured = {}

        def capture(cfg, s, out, argv):
            captured["argv"] = list(map(str, argv))
            return 1

        monkeypatch.setattr(insert, "peg_search", capture)
        out = tmp_path / name
        out.mkdir()
        insert._centre_and_record(None, {**S, "centre_min_clearance_mm": 0.0}, out, SimpleNamespace(path=tmp_path / "s"),
                                  SimpleNamespace(hole=name, no_centre=False), _tool(), entry, HOLES, table_name,
                                  np.array([0.45, -0.016, 0.0065]), 0.3,
                                  "sha", {"inserted_by": {"end_tip": [0.45, -0.016, -0.009]}},
                                  np.array([0.45, -0.016, 0.0065]))
        argv = [a.replace("/insert/", "/out/") for a in captured["argv"]]
        argv[2] = f"/out/centre_{name}.csv"
        command = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}", "-v", f"{binary.parent}:/bin_dir:ro",
                   "-v", f"{tmp_path}:/out", "peg-search-libfranka-0.9.2", "/bin_dir/peg_search", *argv, "--dry-run"]
        done = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if done.returncode != 0 or "dry run: options and trajectory ok" not in done.stdout:
            failures.append(f"{name}: {(done.stderr or done.stdout).strip().splitlines()[-1:]}")
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------- C.3: the grid in the plunge flow
def test_plunge_flow_with_the_grid(tmp_path, monkeypatch):
    """A fake robot: the plunge lands on the board; each tilt-search line slides its lead along the slide and only
    goes in when its strip holds the hidden hole. The attempts' line offsets come from the grid and the missed ones
    are not tried first again."""
    hole = "hexagon_c1mm"
    error = 0.002
    lead = insert.sized_lead(lead_of(hole), 0.005, error)
    slide = math.radians(90.0)
    spacing = insert.line_plan(lead, S, error)["spacing_m"]
    hole_base = np.array([0.452, -0.0166, 0.0065])
    truth = hole_base[:2] + np.array([-0.0018, 0.0004])  # 1.8 mm across (base -x = +a for d = +y): line 2 or so
    grid = insert.make_grid(S, hole_base, error, lead, slide, TABLE[hole]["free_m"], spacing)
    d = np.array([0.0, 1.0])
    a = np.array([-1.0, 0.0])
    tried = []

    def fake(cfg, s, out, argv):
        argv = list(map(str, argv))
        if argv[1] == "peck":
            (out / "peck.json").write_text(json.dumps({"ok": True, "stage": "done", "inserted": False,
                                                       "board_measured": 0.0065,
                                                       "first": {"xy": list(hole_base[:2]), "tip_z": 0.0066,
                                                                 "result": "board"}}))
            return 0
        k = int(Path(argv[argv.index("--result") + 1]).stem.split("_")[1])
        ref = np.array([float(argv[argv.index("--hole-xy") + 1 + i]) for i in range(2)])
        # the last --hole-xy given wins in peg_search
        idx = [i for i, v in enumerate(argv) if v == "--hole-xy"][-1]
        ref = np.array([float(argv[idx + 1]), float(argv[idx + 2])])
        offsets = [float(v) for v in argv[argv.index("--line-offsets") + 1].split(",")]
        centre = float(argv[argv.index("--line-centre") + 1])
        tried.append([float((ref - hole_base[:2]) @ a) + o for o in offsets])  # absolute across positions
        rows, outcomes, inserted = [], [], False
        for line, o in enumerate(offsets[:2]):  # the run aborts after 2 lines (a guard), as on the robot
            run_up = lead["lead_run_up_m"]
            tip0 = ref + (centre + o) * a - (run_up + lead["peg_radius_m"]) * d
            for s_ in np.linspace(0.0, lead["slide_length_m"], 300):
                tip = tip0 + s_ * d
                rows.append((16, line, tip[0], tip[1], 0.0066))
            if abs((truth - ref) @ a - o) <= 0.5 * spacing:
                outcomes.append("inserted")
                inserted = True
                break
            outcomes.append("missed")
        with open(out / f"attempt_{k}.csv", "w") as f:
            f.write("time,phase,segment,p_x,p_y,p_z\n")
            for i, r in enumerate(rows):
                f.write(f"{i * 0.001},{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
        (out / f"attempt_{k}.json").write_text(json.dumps({
            "ok": inserted, "stage": "done" if inserted else "run", "error": None if inserted else "guard",
            "outcomes": outcomes, "inserted": inserted, "slide_dir": list(d), "plunge": "none",
            "end_tip": [*truth, -0.009] if inserted else [*ref, 0.012]}))
        return 0 if inserted else 1

    monkeypatch.setattr(insert, "peg_search", fake)
    monkeypatch.setattr(insert, "robot_program", lambda *a, **k: 0)
    summary = {"attempts": [], "inserted": False}
    common = list(map(str, insert.search_args(lead, slide, S, "lines", error)))
    drop = ["ip", "peck", "/insert/peck.csv", "--result", "/insert/peck.json"]
    ok = insert._plunge_then_tilt(None, S, tmp_path, drop, common, summary, 5, 6.0, hole_base,
                                  np.array([0.43, -0.0166, 0.0065]), lead, error, grid)
    print("offsets per attempt (mm):", [[round(o * 1e3, 1) for o in t] for t in tried])
    assert ok, tried
    assert len(tried) <= 3
    # the lines that ran and missed (the first two of each failed attempt) are not the next attempt's first line
    for before, after in zip(tried, tried[1:]):
        assert all(abs(after[0] - m) > 0.5 * spacing for m in before[:2]), tried
    assert all(a_["grid_plan"] is not None for a_ in summary["attempts"][1:])
    assert (tmp_path / "grid_1.npz").exists()
