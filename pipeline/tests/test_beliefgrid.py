"""Offline tests of board_pipeline.beliefgrid (C.3): the evidence, the plan, a simulated search, the robot's logs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_pipeline import beliefgrid, insert  # noqa: E402
from board_pipeline.config import Config, load_yaml  # noqa: E402

PIPELINE = Path(__file__).resolve().parents[1]
TABLE = load_yaml(PIPELINE / "config" / "tilt_leads.yaml")["holes"]
S = Config.template()["insert"]
RUNS = PIPELINE.parent / "sessions" / "2026-11-19_eye_in_hand_2" / "insert"


def grid_for(hole: str, centre=(0.45, -0.0166), slide=math.radians(90.0), error=0.002):
    lead = TABLE[hole]["leads"][0]
    spacing = insert.line_plan(lead, S, error)["spacing_m"]
    return beliefgrid.HoleGrid(centre, error, lead, slide, TABLE[hole]["free_m"], spacing), lead, spacing


def lead_path(grid, ref, offset, lead, error):
    """The lead's path of a line at offset (tip line through ref + (line_centre + offset) a), along the slide."""
    centre = insert.line_plan(lead, S, error)["centre_m"]
    run_up = float(lead["lead_run_up_m"]) - (float(TABLE_ERROR) - error)
    tip0 = np.asarray(ref) + (centre + offset) * grid.a - (run_up + float(lead["peg_radius_m"])) * grid.d
    s = np.linspace(0.0, 2 * run_up + 0.004, 400)
    tip = tip0 + s[:, None] * grid.d
    return tip + float(lead["peg_radius_m"]) * grid.d + float(lead["lead_across_m"]) * grid.a


TABLE_ERROR = 0.005


def test_prior_plan_corner_one_line_flat_five():
    g, lead, _ = grid_for("triangle_c1mm")
    plan = g.plan()
    assert len(plan["offsets_m"]) == 1 and abs(plan["offsets_m"][0]) < 3e-4
    g, lead, _ = grid_for("hexagon_c1mm")
    plan = g.plan()
    assert 4 <= len(plan["offsets_m"]) <= 8
    assert abs(plan["offsets_m"][0]) < 6e-4  # the middle first


def test_missed_line_moves_the_plan_away():
    g, lead, spacing = grid_for("hexagon_c1mm")
    before = g.mass_within(g.centre, 0.0005)
    g.missed_path(lead_path(g, g.centre, 0.0, lead, 0.002), "0")
    assert g.mass_within(g.centre, 0.0003) < 0.5 * before * (0.0003 / 0.0005) ** 2 + 1e-9 or \
        g.mass_within(g.centre, 0.0003) < before
    plan = g.plan()
    assert all(abs(o) > 0.4 * spacing for o in plan["offsets_m"][:2])  # not the same strip again first


def test_hold_concentrates():
    g, lead, _ = grid_for("triangle_c1mm")
    xy = g.centre + np.array([0.0008, -0.0004])
    g.hold(xy, 0.0005)  # a clean hold (sigma a quarter of the 2 mm located radius) against the 1 mm prior
    assert g.mass_within(xy, 0.0015) > 0.85
    assert np.linalg.norm(g.mean() - xy) < 0.3 * np.linalg.norm(xy - g.centre)


def test_plunge_board_rules_out_the_centre():
    g, lead, _ = grid_for("round_c1mm")
    before = g.mass_within(g.centre, 0.0004)
    g.plunge(g.centre, "board")
    assert g.mass_within(g.centre, 0.0004) < 0.2 * before


def test_plunge_stuck_rules_in():
    g, lead, _ = grid_for("triangle_c1mm")
    xy = g.centre + np.array([-0.001, 0.0])
    g.plunge(xy, "stuck")
    assert g.mass_within(xy, 0.003) > 0.9


def test_dip_puts_the_hole_ahead_of_the_lead():
    g, lead, _ = grid_for("triangle_c1mm")
    truth = g.centre + np.array([0.0005, 0.0003])
    dip_lead = truth + float(lead["along_m"][0]) * g.d + 0.5 * (lead["band_m"][0] + lead["band_m"][1]) * g.a
    g.dip(dip_lead)
    assert np.abs((g.mean() - truth) @ g.d) < 0.0008


def simulate(hole: str, trials: int = 60, attempts: int = 5, p_sim: float = 0.6, p_abort: float = 0.3, seed: int = 3):
    """True holes drawn from the prior. A line whose strip holds the hole (and within the along reach) finds it with
    probability p_sim (a worse robot than the grid's p_detect 0.8); after any line that does not, the run aborts with
    probability p_abort (robot 2026-11-21: 6 runs ended on a guard: lift tolerance, no board contact, tip speed).
    grid: each attempt planned on the posterior; ring: today's search, 0, +s, -s, ... from line 0 every attempt.
    Returns (grid lines, ring lines) per trial, 99 when not found."""
    rng = np.random.default_rng(seed)
    error = 0.002
    out = {"grid": [], "ring": []}
    for trial in range(trials):
        g, lead, spacing = grid_for(hole)
        truth = g.centre + rng.normal(0.0, 0.5 * error, 2)
        if np.linalg.norm(truth - g.centre) > error:
            continue
        draws = rng.random((2, 200))  # the same luck for both searches
        n_lines = insert.line_plan(lead, S, error)["lines"]
        ring = [0.0] + [s * k * spacing for k in range(1, n_lines // 2 + 1) for s in (1, -1)]
        for kind in ("grid", "ring"):
            g, lead, spacing = grid_for(hole)
            lines, found, used = 0, False, 0
            for attempt in range(attempts):
                if kind == "grid":
                    plan = g.plan()
                    ref, offsets = np.asarray(plan["hole_xy"]), plan["offsets_m"]
                else:
                    ref, offsets = g.centre, ring
                for o in offsets:
                    lines += 1
                    hit = abs((truth - ref) @ g.a - o) <= 0.5 * spacing and abs((truth - ref) @ g.d) <= error
                    if hit and draws[0, used] < p_sim:
                        found = True
                        break
                    if kind == "grid":
                        g.missed_path(lead_path(g, ref, o, lead, error), f"{attempt}")
                    aborted = draws[1, used] < p_abort
                    used += 1
                    if aborted:
                        break
                if found:
                    break
            out[kind].append(lines if found else 99)
    return np.array(out["grid"]), np.array(out["ring"])


@pytest.mark.parametrize("hole", ["hexagon_c1mm", "triangle_c0p2mm"])
def test_simulated_search_finds_the_hole(hole):
    grid, ring = simulate(hole)
    print(f"{hole}: grid found {np.mean(grid < 99):.2f} (mean {grid[grid < 99].mean():.2f} lines), "
          f"ring found {np.mean(ring < 99):.2f} (mean {ring[ring < 99].mean():.2f} lines)")
    # Missed lines alone (no hold, dip or plunge): the grid gains only where runs abort part-way (hexagon 2026-11-21
    # sim: the same share found in ~14 % fewer lines); a one-line corner plan has nothing to gain. Not worse, within
    # one trial:
    assert np.mean(grid < 99) >= np.mean(ring < 99) - 1.5 / len(grid)
    assert grid[grid < 99].mean() <= ring[ring < 99].mean() + 0.2


def test_replay_the_jammed_triangle_run():
    """Robot 2026-11-21 134814 (triangle_c1mm): the plunge stuck in the hole mouth, the hold jammed 2 mm from the hole
    (the hole: 141115's inserting hold). The grid, from vision and that run, puts more mass near the hole."""
    run, truth_run = RUNS / "2026-11-21_134814", RUNS / "2026-11-21_141115"
    if not (run / "attempt_1.csv").exists() or not (truth_run / "attempt_1.json").exists():
        pytest.skip("the 2026-11-21 runs are not here")
    plan = yaml.safe_load((run / "plan.yaml").read_text())
    truth = np.asarray(json.loads((truth_run / "attempt_1.json").read_text())["hold_xy"])
    lead = next(l for l in TABLE["triangle_c1mm"]["leads"] if l["lead"] == plan["lead"]["lead"])
    spacing = insert.line_plan(lead, S, 0.002)["spacing_m"]
    g = beliefgrid.HoleGrid(plan["hole_base_m"][:2], 0.002, lead, math.radians(plan["slide_dir_base_deg"]),
                            TABLE["triangle_c1mm"]["free_m"], spacing)
    before = g.mass_within(truth, 0.001)
    peck = json.loads((run / "peck.json").read_text())
    g.update_from_peck({**peck, "first": {**peck["first"], "result": "stuck"}}, "plunge")
    result = json.loads((run / "attempt_1.json").read_text())
    g.update_from_run(result, run / "attempt_1.csv", "1", jam_radius=0.003)
    after = g.mass_within(truth, 0.001)
    print(f"mass within 1 mm of the hole: {before:.3f} -> {after:.3f}; events {g.events}")
    assert after > before
    assert any(e.startswith("hold") for e in g.events)


def test_aborted_slide_does_not_rule_out_hole_and_truncated_csv_is_read(tmp_path):
    g, lead, _ = grid_for("hexagon_c1mm")
    before = g.mass_within(g.centre, 0.0005)
    csv = tmp_path / "aborted.csv"
    lines = ["time,phase,segment,p_x,p_y,p_z"]
    for i, s in enumerate(np.linspace(-0.015, 0.015, 50)):
        lines.append(f"{i * 0.001},16,0,{g.centre[0]},{g.centre[1] + s},0.006")
    lines.append("0.050,16,0,0.45")  # truncated trailing line from a disconnect
    csv.write_text("\n".join(lines) + "\n")
    log = beliefgrid.read_log(csv)
    assert len(log["phase"]) == 50
    g.update_from_run({"outcomes": [], "slide_dir": [0.0, 1.0]}, csv, "abort")
    assert g.mass_within(g.centre, 0.0005) == pytest.approx(before)
    assert not g.events
