"""The hole's position as a probability grid, updated from every attempt (`pipeline insert --search grid`, 2026-11-21).

Until 2026-11-21 each attempt searched the whole error disc again from line 0, and the only memory was the last hold
or dip (a point and a radius). Here every line the robot slid is evidence:

  missed line   the lead slid along its path without dipping: the hole centres H for which that path ran through the
                dip region (tilt_leads.yaml dip_region_m: the lead positions, from the hole centre, at which it dips;
                at least min_chord inside it) become unlikely (x 1 - p_detect). For flat leads the region is narrowed
                across the slide to the line spacing they are trusted with (robot-tested 1 mm; the simulation's band
                is several mm, but a flat's band shrinks with the peg's yaw error).
  dip           the lead dipped at L: H in L - region (x 1 inside, x 1 - p_true elsewhere).
  hold          the peg stood upright in the hole mouth at xy: a Gaussian about it (sigma half the located radius,
                half the jam radius if it then jammed), mixed with the rest (p_true).
  plunge        an upright plunge at xy on the board: H with xy - H in the free region (tilt_leads.yaml free_m: it would
                have gone in) unlikely; stuck in the hole mouth: H within the free region + 1.5 mm (the chamfer) likely.

The prior is a Gaussian about the (corrected) vision hole, sigma half the error disc. Each attempt's lines are planned
on the posterior (plan): the reference xy is its mean; line offsets across the slide are taken greedily, each the
strip (line spacing wide, the search's along reach long) holding the most probability not yet covered, none within
0.9 of a strip of one already taken, until `target` of it is covered or `max_lines`; peg_search runs them in that
order (--line-offsets). A strip that missed is not ruled out (p_detect), so a later attempt may take it again.

Everything is numpy (the pipeline's interpreter has no shapely); polygons from holefit.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from board_pipeline import holefit

SLIDE, FUNNEL, UPRIGHT, HOLD, PLUNGE = 16, 25, 17, 18, 26


def _unit(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)])


class HoleGrid:
    def __init__(self, centre_xy, error_m: float, lead: dict, slide_rad: float, free_hole, spacing_m: float,
                 step_m: float = 1.0e-4, margin_m: float = 0.0015, p_detect: float = 0.8, p_true: float = 0.9,
                 min_chord_m: float = 0.001, chamfer_m: float = 0.0015):
        self.centre = np.asarray(centre_xy, float)[:2]
        self.error = float(error_m)
        self.lead = lead
        self.slide = float(slide_rad)
        self.d = _unit(self.slide)
        self.a = np.array([-self.d[1], self.d[0]])
        self.spacing = float(spacing_m)
        self.p_detect, self.p_true = float(p_detect), float(p_true)
        self.min_chord, self.chamfer = float(min_chord_m), float(chamfer_m)
        half = self.error + margin_m
        g = np.arange(-half, half + 0.5 * step_m, step_m)
        self.cells = self.centre + np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
        sigma = 0.5 * self.error
        r2 = np.sum((self.cells - self.centre) ** 2, axis=1)
        self.log_p = -0.5 * r2 / sigma ** 2
        # The region turned from the table's hole frame into base axes: the lead's slide direction there is d_deg.
        turn = self.slide - math.radians(float(lead["d_deg"]))
        self.region = self._trusted_region(np.asarray(lead["dip_region_m"], float), lead) @ holefit.rotation(turn).T
        self.free = np.asarray(free_hole, float) @ holefit.rotation(turn).T
        self.events: list[str] = []

    # ---------------------------------------------------------------- the model
    def _trusted_region(self, region_hole: np.ndarray, lead: dict) -> np.ndarray:
        """The dip region narrowed across d (about the effective band's middle) to the strip the line spacing trusts."""
        d = _unit(math.radians(float(lead["d_deg"])))
        a = np.array([-d[1], d[0]])
        band = lead.get("eff_band_m") or lead["band_m"]
        width = float(band[1]) - float(band[0])
        k = min(1.0, self.spacing / width) if width > 0 else 1.0
        mid = 0.5 * (float(band[0]) + float(band[1]))
        along, across = region_hole @ d, region_hole @ a
        across = mid + k * (across - mid)
        return along[:, None] * d + across[:, None] * a

    def probability(self) -> np.ndarray:
        p = np.exp(self.log_p - self.log_p.max())
        return p / p.sum()

    def mean(self) -> np.ndarray:
        return self.probability() @ self.cells

    def spread(self) -> float:
        p = self.probability()
        return float(math.sqrt(p @ np.sum((self.cells - self.mean()) ** 2, axis=1)))

    def mass_within(self, xy, radius: float) -> float:
        return float(self.probability()[np.linalg.norm(self.cells - np.asarray(xy)[:2], axis=1) <= radius].sum())

    def _multiply(self, mask: np.ndarray, factor_in: float, factor_out: float = 1.0) -> None:
        self.log_p += np.where(mask, math.log(max(factor_in, 1e-12)), math.log(max(factor_out, 1e-12)))

    def _region_hits(self, points: np.ndarray, region: np.ndarray) -> np.ndarray:
        """For every cell H: how many of the points p have p - H in the region (cells x points, chunked)."""
        counts = np.zeros(len(self.cells), int)
        for i in range(0, len(self.cells), 2000):
            rel = points[None, :, :] - self.cells[i:i + 2000, None, :]
            counts[i:i + 2000] = np.count_nonzero(holefit.inside(region, rel), axis=1)
        return counts

    # ---------------------------------------------------------------- evidence
    def missed_path(self, lead_points: np.ndarray, label: str = "") -> None:
        """The lead slid along lead_points (base xy, in order) without dipping."""
        points = resample(lead_points, 1.0e-4)
        if len(points) < 2:
            return
        need = max(1, int(round(self.min_chord / 1.0e-4)))
        crossed = self._region_hits(points, self.region) >= need
        self._multiply(crossed, 1.0 - self.p_detect)
        self.events.append(f"missed line {label}: {np.count_nonzero(crossed)} cells less likely")

    def dip(self, lead_xy, label: str = "") -> None:
        point = np.asarray(lead_xy, float)[None, :2]
        # the region dilated a little (the chamfer dips the lead early; the friction lag)
        inside = self._region_hits(point, self.region)[...] > 0
        near = inside | (holefit.boundary_distance(self.region, point[0] - self.cells) <= 0.0015)
        self._multiply(near, 1.0, 1.0 - self.p_true)
        self.events.append(f"dip {label}")

    def hold(self, xy, sigma: float, label: str = "") -> None:
        r2 = np.sum((self.cells - np.asarray(xy, float)[:2]) ** 2, axis=1)
        like = self.p_true * np.exp(-0.5 * r2 / sigma ** 2) + (1.0 - self.p_true) * math.exp(-0.5 * 9.0)
        self.log_p += np.log(like)
        self.events.append(f"hold {label} (sigma {sigma * 1e3:.2f} mm)")

    def plunge(self, xy, result: str, label: str = "") -> None:
        """An upright plunge at xy: 'board' (it did not go in), 'stuck' (in the hole mouth); 'inserted': nothing."""
        rel = np.asarray(xy, float)[:2] - self.cells
        if result == "board":
            self._multiply(holefit.inside(self.free, rel), 1.0 - self.p_true)
        elif result == "stuck":
            near = holefit.inside(self.free, rel) | (holefit.boundary_distance(self.free, rel) <= self.chamfer)
            self._multiply(near, 1.0, 1.0 - self.p_true)
        else:
            return
        self.events.append(f"plunge {label}: {result}")

    # ---------------------------------------------------------------- the plan
    def plan(self, target: float = 0.95, max_lines: int = 15, min_gain: float = 0.01) -> dict:
        """Line offsets (m across the slide, from the reference xy, in order) for the next attempt."""
        p = self.probability()
        ref = self.mean()
        rel = self.cells - ref
        along, across = rel @ self.d, rel @ self.a
        reachable = np.abs(along) <= self.error
        width = self.spacing
        lim = self.error + 0.5 * width
        candidates = np.arange(-lim, lim + 1e-9, 1.0e-4)
        left = p * reachable
        total = float(left.sum())
        in_strip = np.zeros(len(p), bool)
        offsets, detected = [], 0.0
        while len(offsets) < max_lines and total > 0:
            masses = np.array([left[np.abs(across - o) <= 0.5 * width].sum() for o in candidates])
            masses -= 1e-9 * total * np.abs(candidates) / lim  # ties: the one nearer the middle
            for o in offsets:  # not the same strip twice in one run (a line's failures repeat, they are not dice)
                masses[np.abs(candidates - o) < 0.9 * width] = -1.0
            best = int(np.argmax(masses))
            if masses[best] < 0.0 or (masses[best] < min_gain * total and offsets):
                break
            strip = np.abs(across - candidates[best]) <= 0.5 * width
            detected += self.p_detect * float(left[strip].sum())
            left = np.where(strip, left * (1.0 - self.p_detect), left)
            in_strip |= strip
            offsets.append(float(candidates[best]))
            if float((p * reachable)[in_strip].sum()) >= target * total:  # what lines can cover, not p_detect's share
                break
        covered = float((p * reachable)[in_strip].sum())
        return {"hole_xy": [float(v) for v in ref], "offsets_m": offsets,
                "covered": covered / total if total > 0 else 0.0, "detected": detected / total if total > 0 else 0.0,
                "reachable": total,
                "spread_m": self.spread()}

    # ---------------------------------------------------------------- from peg_search's results and logs
    def update_from_run(self, result: dict, csv_path: Path | None, label: str = "", located_radius: float = 0.002,
                        jam_radius: float = 0.003) -> None:
        """A tilt-slide run: its plunge, then per line (the log's segment) the slide, the dip, the hold."""
        if result.get("plunge") in ("on the board", "in the hole") and csv_path is not None and Path(csv_path).exists():
            log = read_log(csv_path)
            rows = log["phase"] == PLUNGE
            if np.any(rows):
                xy = np.array([log["p_x"][rows][-1], log["p_y"][rows][-1]])
                self.plunge(xy, "board" if result["plunge"] == "on the board" else "stuck", f"{label} (tilt-slide)")
        if csv_path is None or not Path(csv_path).exists():
            return
        log = read_log(csv_path)
        if result.get("slide_dir"):
            d = np.asarray(result["slide_dir"][:2], float)
            d = d / np.linalg.norm(d)
        else:
            d = self.d
        a = np.array([-d[1], d[0]])
        offset = float(self.lead["peg_radius_m"]) * d + float(self.lead["lead_across_m"]) * a
        outcomes = result.get("outcomes") or []
        for k in sorted(set(int(v) for v in log["segment"])):
            rows = log["segment"] == k
            phase = log["phase"][rows]
            tip = np.stack([log["p_x"][rows], log["p_y"][rows]], axis=1)
            lead = tip + offset
            outcome = outcomes[k] if k < len(outcomes) else None
            funnel = np.flatnonzero(phase == FUNNEL)
            slide = phase == SLIDE
            if funnel.size:
                before = np.flatnonzero(slide)
                before = before[before < funnel[0]]
                if before.size:
                    self.missed_path(lead[before], f"{label}/{k} before its dip")
                self.dip(lead[funnel[0]], f"{label}/{k}")
            elif np.any(slide) and outcome in ("missed", "shallow"):
                self.missed_path(lead[slide], f"{label}/{k}")
            holds = np.flatnonzero(phase == HOLD)
            if holds.size and outcome != "inserted":
                # a clean hold has the tip on the hole axis (141115: the inserting hold); one that jammed is wedged
                sigma = 0.5 * jam_radius if outcome == "jammed" else 0.25 * located_radius
                self.hold(tip[holds[0]], sigma, f"{label}/{k}")

    def update_from_peck(self, result: dict, label: str = "") -> None:
        first = result.get("first") or {}
        if first.get("xy") and first.get("result") in ("board", "stuck"):
            self.plunge(first["xy"], first["result"], f"{label} (peck)")

    def save(self, path: Path) -> None:
        np.savez_compressed(path, cells=self.cells, probability=self.probability(), centre=self.centre,
                            events=np.array(self.events, dtype=str))


def resample(points: np.ndarray, step: float) -> np.ndarray:
    """A path at even arc-length steps."""
    points = np.asarray(points, float)
    if len(points) < 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return points[:1]
    at = np.arange(0.0, s[-1] + 1e-12, step)
    return np.stack([np.interp(at, s, points[:, 0]), np.interp(at, s, points[:, 1])], axis=1)


def read_log(path: Path) -> dict:
    """peg_search's 1 kHz log: the columns insert needs (tolerates a truncated final line after a disconnect)."""
    want = ["phase", "segment", "p_x", "p_y", "p_z"]
    with open(path) as f:
        header = f.readline().strip().split(",")
        ncols = len(header)
        idx = [header.index(w) for w in want]
        rows = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != ncols:
                continue
            try:
                rows.append([float(parts[i]) for i in idx])
            except ValueError:
                continue
    if not rows:
        raise ValueError("empty log")
    data = np.asarray(rows, float)
    out = {w: data[:, i] for i, w in enumerate(want)}
    out["phase"] = out["phase"].astype(int)
    out["segment"] = out["segment"].astype(int)
    return out
