"""Offline tests of board_pipeline.holefit: wall-touch centring and the board correction (C.1)."""

from __future__ import annotations

import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_pipeline import holefit  # noqa: E402
from board_pipeline.config import load_yaml  # noqa: E402

PIPELINE = Path(__file__).resolve().parents[1]
TABLE = load_yaml(PIPELINE / "config" / "tilt_leads.yaml")["holes"]


def ray_hit(poly: np.ndarray, start: np.ndarray, u: np.ndarray) -> np.ndarray | None:
    """The first crossing of the ray start + s u (s > 0) with the polygon's outline."""
    best = None
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        m = np.array([[u[0], -e[0]], [u[1], -e[1]]])
        if abs(np.linalg.det(m)) < 1e-15:
            continue
        s, r = np.linalg.solve(m, a - start)
        if s > 1e-12 and -1e-12 <= r <= 1 + 1e-12 and (best is None or s < best):
            best = s
    return None if best is None else start + best * u


def simulate_touches(hole: str, centre: np.ndarray, yaw: float, rng, noise: float = 2.0e-5):
    free = holefit.to_base(TABLE[hole]["free_m"], yaw)
    # a start somewhere in the free region (the seated peg)
    for _ in range(1000):
        start_rel = free.mean(axis=0) + rng.uniform(-1, 1, 2) * 0.9 * holefit.reach(free)
        if holefit.inside(free, start_rel[None])[0]:
            break
    touches = []
    for deg in holefit.touch_directions(yaw):
        u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        hit = ray_hit(free, start_rel, u)
        tip = centre + hit + rng.normal(0.0, noise, 2)
        touches.append({"dir_deg": deg, "contact": True, "tip": [*tip, 0.0]})
    return centre + start_rel, touches


@pytest.mark.parametrize("hole", [h for h in TABLE if TABLE[h].get("free_m") and holefit.reach(TABLE[h]["free_m"]) > 1e-4])
def test_fit_centre_from_touches(hole):
    rng = np.random.default_rng(zlib.crc32(hole.encode()))
    centre = np.array([0.45, -0.016])
    yaw = rng.uniform(-math.pi, math.pi)
    seat, touches = simulate_touches(hole, centre, yaw, rng)
    found = holefit.found_from_touches({"ok": True, "touches": touches}, TABLE[hole]["free_m"], yaw, seat)
    assert found is not None, hole
    error = np.linalg.norm(np.asarray(found["xy"]) - centre)
    print(f"{hole}: centre {error * 1e6:.1f} um off (seat {np.linalg.norm(seat - centre) * 1e6:.0f} um)")
    assert error < 5.0e-5, f"{hole}: {error * 1e6:.1f} um"


def test_one_sided_touches_are_refused():
    rng = np.random.default_rng(1)
    seat, touches = simulate_touches("round_c3mm", np.array([0.4, 0.0]), 0.0, rng)
    one_side = [t for t in touches if -80.0 <= t["dir_deg"] <= 80.0]
    assert holefit.found_from_touches({"ok": True, "touches": one_side}, TABLE["round_c3mm"]["free_m"], 0.0, seat) is None


def test_touches_off_the_outline_are_refused():
    rng = np.random.default_rng(2)
    seat, touches = simulate_touches("hexagon_c3mm", np.array([0.4, 0.0]), 0.3, rng)
    touches[0]["tip"][0] += 0.0008  # a contact 0.8 mm beyond the wall (something else in the way)
    assert holefit.found_from_touches({"ok": True, "touches": touches}, TABLE["hexagon_c3mm"]["free_m"], 0.3, seat) is None


def test_offset_outline_fits_a_bigger_hole():
    """The 3 mm round hole's touches fitted with the 1 mm hole's region: the outline moves out 1 mm, the centre stays."""
    rng = np.random.default_rng(8)
    centre = np.array([0.42, -0.1])
    seat, touches = simulate_touches("round_c3mm", centre, 0.0, rng)
    fit = holefit.fit_centre([t["tip"] for t in touches], TABLE["round_c1mm"]["free_m"], 0.0, seat, max_delta=0.0012)
    assert fit["delta_m"] == pytest.approx(0.001, abs=5e-5)
    assert np.linalg.norm(np.asarray(fit["xy"]) - centre) < 5e-5
    found = holefit.found_from_touches({"ok": True, "touches": touches}, TABLE["round_c1mm"]["free_m"], 0.0, seat,
                                       max_delta=0.0012)
    assert found is not None and found["delta_m"] == pytest.approx(0.001, abs=5e-5)


def test_too_few_contacts():
    rng = np.random.default_rng(3)
    seat, touches = simulate_touches("round_c3mm", np.array([0.4, 0.0]), 0.0, rng)
    for t in touches[3:]:
        t["contact"] = False
    assert holefit.found_from_touches({"ok": True, "touches": touches}, TABLE["round_c3mm"]["free_m"], 0.0, seat) is None


def test_geometry_helpers():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert holefit.inside(square, np.array([[0.5, 0.5], [1.5, 0.5]])).tolist() == [True, False]
    assert holefit.boundary_distance(square, np.array([[0.5, 0.5]]))[0] == pytest.approx(0.5)
    assert holefit.signed_distance(square, np.array([[0.5, 0.2], [0.5, -0.3]])).tolist() == pytest.approx([-0.2, 0.3])
    assert holefit.diameter(square) == pytest.approx(math.sqrt(2))


# ---------------------------------------------------------------- the board correction
BOARD = "sha-a"
HOLES_XY = {"round_c3mm": (0.42, -0.10), "hexagon_c3mm": (0.48, -0.10), "triangle_c3mm": (0.54, -0.10),
            "rectangle_c1mm": (0.60, -0.04), "l_shape_c3mm": (0.66, -0.10), "triangle_c1mm": (0.54, -0.04)}


def synthetic(shift, turn_deg, sigma, rng, holes=HOLES_XY, centre=(0.54, -0.07)):
    R = holefit.rotation(math.radians(turn_deg))
    c = np.asarray(centre)
    out = []
    for name, xy in holes.items():
        v = np.asarray(xy)
        f = R @ (v - c) + c + np.asarray(shift) + rng.normal(0.0, sigma, 2)
        out.append({"hole": name, "vision_xy": list(v), "found_xy": list(f), "sigma_m": sigma, "board_sha256": BOARD,
                    "run": "r"})
    return out


def test_correction_recovers_shift_and_turn():
    rng = np.random.default_rng(4)
    entries = synthetic((-0.0005, 0.0002), 0.15, 5e-5, rng)
    corr = holefit.fit_correction(entries, BOARD)
    assert corr["shift_m"][0] == pytest.approx(-0.0005, abs=6e-5)
    assert corr["shift_m"][1] == pytest.approx(0.0002, abs=6e-5)
    assert corr["turn_deg"] == pytest.approx(0.15, abs=0.03)
    for e in entries:  # the corrected vision lands on the found holes
        assert np.linalg.norm(holefit.correct_xy(e["vision_xy"], corr) - np.asarray(e["found_xy"])) < 2e-4


def test_one_hole_gives_a_shift_not_a_turn():
    rng = np.random.default_rng(5)
    entries = synthetic((-0.0006, 0.0), 0.0, 5e-5, rng, holes={"round_c3mm": (0.42, -0.10)})
    corr = holefit.fit_correction(entries, BOARD)
    assert corr["holes"] == 1 and abs(corr["turn_deg"]) < 1e-6
    assert corr["shift_m"][0] == pytest.approx(-0.0006, abs=1e-4)


def test_loose_holes_barely_turn():
    """Seated tips (sigma 0.75 mm): the turn stays near its prior, the shift is the mean."""
    rng = np.random.default_rng(6)
    entries = synthetic((-0.0005, 0.0), 0.0, 7.5e-4, rng)
    corr = holefit.fit_correction(entries, BOARD)
    assert abs(corr["turn_deg"]) < 0.3
    assert corr["sigma_shift_m"] > 2e-4


def test_other_board_pose_ignored_and_outlier_dropped():
    rng = np.random.default_rng(7)
    entries = synthetic((-0.0005, 0.0), 0.0, 5e-5, rng)
    entries.append({**entries[0], "board_sha256": "sha-old", "found_xy": [1.0, 1.0]})
    entries[1] = {**entries[1], "found_xy": [entries[1]["found_xy"][0] + 0.003, entries[1]["found_xy"][1]]}
    corr = holefit.fit_correction(entries, BOARD)
    assert [d["hole"] for d in corr["dropped"]] == [entries[1]["hole"]]
    assert corr["shift_m"][0] == pytest.approx(-0.0005, abs=1e-4)
    assert holefit.fit_correction(entries, "sha-new") is None


def test_correction_matrix_moves_a_board_pose():
    corr = {"shift_m": [-0.0005, 0.0003], "turn_deg": 0.2, "centre_m": [0.5, 0.0]}
    T = np.eye(4)
    T[:3, 3] = [0.48, -0.015, -0.019]
    M = holefit.correction_matrix(corr)
    moved = M @ T
    point = T @ np.array([0.02, 0.01, 0.025, 1.0])
    assert (M @ point)[:2] == pytest.approx(holefit.correct_xy(point[:2], corr), abs=1e-12)
    assert moved[2, 3] == pytest.approx(T[2, 3])


def test_record_and_load(tmp_path):
    holefit.record_found(tmp_path, "round_c3mm", [0.42, -0.1], {"xy": [0.4195, -0.1], "sigma_m": 1e-4, "source": "t"},
                         BOARD, "run1")
    holefit.record_found(tmp_path, "hexagon_c3mm", [0.48, -0.1], {"xy": [0.4795, -0.1], "sigma_m": 1e-4, "source": "t"},
                         BOARD, "run2")
    entries = holefit.load_found(tmp_path)
    assert [e["hole"] for e in entries] == ["round_c3mm", "hexagon_c3mm"]
    corr = holefit.fit_correction(entries, BOARD)
    assert corr["shift_m"][0] == pytest.approx(-0.0005, abs=2e-5)


# ---------------------------------------------------------------- the local residual field
BOARD_HOLES = load_yaml(PIPELINE / "config" / "holes.yaml")["holes"]


def board_entries(rng, field, sigma=5e-5, shift=(-0.0004, 0.0001), names=None):
    """Found holes at the board's real layout (holes.yaml, the board at 0.5, -0.02 in base): a rigid shift + field(xy)
    + noise."""
    out = []
    for name, h in BOARD_HOLES.items():
        if names is not None and name not in names:
            continue
        v = np.array([0.5 + h["x_m"], -0.02 + h["y_m"]])
        f = v + np.asarray(shift) + field(v) + rng.normal(0.0, sigma, 2)
        out.append({"hole": name, "vision_xy": list(v), "found_xy": list(f), "sigma_m": sigma, "board_sha256": BOARD,
                    "run": name})
    return out


def smooth(v):  # a posture-like error: varies over ~0.15 m, 0.4 mm
    return np.array([0.0004 * math.sin(2 * math.pi * (v[0] - 0.5) / 0.3), 0.0003 * math.cos(2 * math.pi * v[1] / 0.25)])


def test_rigid_board_passes_the_model_check():
    rng = np.random.default_rng(20)
    entries = board_entries(rng, lambda v: np.zeros(2))
    corr = holefit.local_correction(entries, BOARD, at_xy=[0.6, 0.0])
    m = corr["model"]
    assert m["chi2"] <= 2.0 * m["dof"] and m["extra_scatter_m"] < 3e-5
    assert corr["sigma_f_m"] < 1e-4  # no field
    assert np.linalg.norm(corr["local_m"]) < 5e-5
    loo = holefit.leave_one_out(entries, BOARD)
    assert loo["rms"]["rigid_m"] < 1.2e-4 and loo["rms"]["local_m"] < 1.3e-4 and loo["rms"]["none_m"] > 3e-4


def test_smooth_field_is_found_and_predicts_held_out_holes():
    rng = np.random.default_rng(21)
    entries = board_entries(rng, smooth)
    corr = holefit.local_correction(entries, BOARD)
    assert corr["model"]["chi2"] > 2.0 * corr["model"]["dof"] and corr["model"]["extra_scatter_m"] > 1e-4
    loo = holefit.leave_one_out(entries, BOARD)
    print(f"smooth field: leave-one-out RMS none {loo['rms']['none_m'] * 1e3:.3f}, rigid {loo['rms']['rigid_m'] * 1e3:.3f}, "
          f"local {loo['rms']['local_m'] * 1e3:.3f} mm")
    assert loo["rms"]["local_m"] < 0.7 * loo["rms"]["rigid_m"]


def test_uncorrelated_scatter_does_no_harm():
    """Per-hole scatter (the grasp: a new peg each hole) is not a field: the local term must not make it worse."""
    rng = np.random.default_rng(22)
    entries = board_entries(rng, lambda v: rng.normal(0.0, 3e-4, 2))
    loo = holefit.leave_one_out(entries, BOARD)
    print(f"uncorrelated: leave-one-out RMS rigid {loo['rms']['rigid_m'] * 1e3:.3f}, local {loo['rms']['local_m'] * 1e3:.3f} mm")
    assert loo["rms"]["local_m"] <= 1.1 * loo["rms"]["rigid_m"]


def test_revisit_follows_the_holes_own_residual():
    rng = np.random.default_rng(23)
    entries = board_entries(rng, lambda v: np.zeros(2), names=list(BOARD_HOLES)[:6])
    odd = dict(entries[2])
    entries[2] = {**odd, "found_xy": list(np.asarray(odd["found_xy"]) + np.array([0.0005, 0.0]))}  # 0.5 mm off the rest
    corr = holefit.local_correction(entries, BOARD, at_xy=entries[2]["vision_xy"], sigma_f_m=5e-4, tau_m=0.0)
    predicted = holefit.correct_xy_full(entries[2]["vision_xy"], corr)
    assert np.linalg.norm(predicted - np.asarray(entries[2]["found_xy"])) < 1.5e-4
    # far from every found hole: the rigid correction, the field's full uncertainty
    far = holefit.local_correction(entries, BOARD, at_xy=[0.9, 0.3], sigma_f_m=5e-4, tau_m=0.0)
    assert np.linalg.norm(far["local_m"]) < 1e-5 and far["local_sigma_m"] == pytest.approx(5e-4, rel=1e-3)


def test_field_split_and_cap_and_default():
    rng = np.random.default_rng(24)
    field = holefit.local_correction(board_entries(rng, smooth), BOARD)
    scatter = holefit.local_correction(board_entries(rng, lambda v: rng.normal(0.0, 3e-4, 2)), BOARD)
    print(f"smooth: sigma_f {field['sigma_f_m'] * 1e3:.3f} tau {field['scatter_m'] * 1e3:.3f} mm; "
          f"scatter: sigma_f {scatter['sigma_f_m'] * 1e3:.3f} tau {scatter['scatter_m'] * 1e3:.3f} mm")
    assert field["sigma_f_m"] > 1e-4 and field["scatter_m"] < 5e-5  # a field (what the rigid fit leaves of it), no scatter
    assert scatter["scatter_m"] > 2e-4 and scatter["sigma_f_m"] < scatter["scatter_m"]  # scatter, little field
    assert holefit.local_correction(board_entries(rng, smooth), BOARD, cap_m=2e-4)["sigma_f_m"] <= 2e-4 + 1e-12
    few = board_entries(rng, lambda v: np.zeros(2), names=list(BOARD_HOLES)[:2])
    assert holefit.local_correction(few, BOARD)["sigma_f_m"] == pytest.approx(3e-4)


def test_full_matrix_matches_correct_xy_full():
    rng = np.random.default_rng(25)
    entries = board_entries(rng, smooth)
    corr = holefit.local_correction(entries, BOARD, at_xy=[0.55, -0.05])
    T = np.eye(4)
    T[:3, 3] = [0.5, -0.02, -0.019]
    p = np.array([0.05, -0.03, 0.025, 1.0])
    moved = holefit.full_matrix(corr) @ (T @ p)
    assert moved[:2] == pytest.approx(holefit.correct_xy_full((T @ p)[:2], corr), abs=1e-12)
