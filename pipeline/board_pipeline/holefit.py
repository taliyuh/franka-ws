"""Found holes and the board correction they give (`pipeline insert`, 2026-11-21).

The holes vision places are off by a few tenths of a mm (2026-11-21: every inserted hole 0.2-0.9 mm from vision,
base x always negative): an error of the board pose (and of the joint-offset model) that is the same, or nearly,
for every hole. A hole the peg went into is a measurement of where that hole really is:

  wall touches  after the insertion, before the release, `peg_search centre` touches the hole walls in several
                directions; the contacts lie on the outline of the hole's free region (tilt_leads.yaml free_m: the peg
                offsets at which it fits in the hole), so the hole centre is the shift that puts them there
                (fit_centre). ~0.05-0.1 mm.
  seated tip    the tip of the inserted peg (anywhere in that free region: sigma half its reach), when there were no
                touches.

Each found hole is kept in <session>/insert/found_holes.yaml with the vision position insert aimed at (before any
correction) and the board pose's sha256 (a `pipeline locate` that moved the board starts afresh). fit_correction fits
one rigid shift of the board in the base plane, x_true = R(theta) (x_vision - c) + c + t, weighted by the holes'
sigmas, with priors (t 3 mm, theta 0.3 deg) so that one hole gives a shift only and a turn needs holes far apart;
insert applies it to the hole's target (the probe, the plunge and the search all follow).

All 2-D geometry here is numpy only (the pipeline's interpreter has no shapely).
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np
import yaml

FOUND_FILE = "found_holes.yaml"


# ---------------------------------------------------------------- polygons (N x 2, closed implicitly)
def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Even-odd point in polygon, vectorised: pts (..., 2) -> bool (...)."""
    poly = np.asarray(poly, float)
    pts = np.asarray(pts, float)
    x, y = pts[..., 0, None], pts[..., 1, None]
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (np.count_nonzero(crosses & (x < xc), axis=-1) % 2) == 1


def boundary_distance(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance of pts (..., 2) to the polygon's outline."""
    poly = np.asarray(poly, float)
    pts = np.asarray(pts, float)
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ap = pts[..., None, :] - a
    t = np.clip(np.sum(ap * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-30), 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.min(np.linalg.norm(pts[..., None, :] - closest, axis=-1), axis=-1)


def signed_distance(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Negative inside."""
    d = boundary_distance(poly, pts)
    return np.where(inside(poly, pts), -d, d)


def diameter(poly: np.ndarray) -> float:
    poly = np.asarray(poly, float)
    return float(np.max(np.linalg.norm(poly[:, None, :] - poly[None, :, :], axis=-1))) if len(poly) > 1 else 0.0


def reach(poly: np.ndarray) -> float:
    """The largest distance of the outline from the region's vertex mean."""
    poly = np.asarray(poly, float)
    return float(np.max(np.linalg.norm(poly - poly.mean(axis=0), axis=1)))


def to_base(poly_hole: np.ndarray, yaw_base: float) -> np.ndarray:
    """A hole-frame outline turned into base axes (about the hole centre)."""
    return np.asarray(poly_hole, float) @ rotation(yaw_base).T


# ---------------------------------------------------------------- the centre from wall touches
def touch_directions(yaw_base: float, count: int = 6) -> list[float]:
    """Base-frame directions (deg) evenly about the hole, from 15 deg off its x axis (off a polygon's corners)."""
    return [math.degrees(math.remainder(yaw_base + math.radians(15.0) + 2.0 * math.pi * k / count, 2.0 * math.pi))
            for k in range(count)]


def touch_reach(free_hole: np.ndarray) -> float:
    """How far a touch may go: across the whole free region (the peg may start at its far side) + 0.5 mm."""
    return min(0.006, diameter(free_hole) + 0.0005)


def fit_centre(tips_xy, free_hole: np.ndarray, yaw_base: float, guess_xy, search: float = 0.004,
               offset: bool = True, max_delta: float = 2.0e-4, starts: int = 8) -> dict:
    """The hole centre H (base xy) putting the touch contacts on the free region's outline, the region turned to
    base axes: min sum (sd(c_i - H) - delta)^2. delta (offset=True): the outline moved out uniformly, solved per H
    (the mean signed distance) within +-max_delta (a printed hole a little off its CAD; insert widens it for the
    rectangle_c3mm hole, fitted with the 1 mm hole's region). Unbounded, delta trades against H when the contacts
    cluster on a few walls (simulated: an L-shape peg seated in a corner, 0.47 mm off with delta 0.33 mm).
    Coarse grid (step a quarter of the region's reach) over +-search, then the `starts` best separate candidates
    each refined (a single refinement went down the wrong basin: a triangle 1.5 mm off). Returns xy, delta, rms
    (m), the contacts' residuals."""
    tips = np.asarray(tips_xy, float)[:, :2]
    free = to_base(free_hole, yaw_base)
    size = reach(free)
    low = -min(size, max_delta)

    def delta_of(sd):
        return np.clip(sd.mean(axis=-1, keepdims=True), low, max_delta) if offset else np.zeros(sd.shape[:-1] + (1,))

    def costs(cand):
        out = np.empty(len(cand))
        for i in range(0, len(cand), 20000):  # bounded memory
            sd = signed_distance(free, tips[None, :, :] - cand[i:i + 20000, None, :])
            out[i:i + 20000] = np.sum((sd - delta_of(sd)) ** 2, axis=1)
        return out

    def grid(centre, half, step):
        g = np.arange(-half, half + 0.5 * step, step)
        return centre + np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)

    step0 = float(np.clip(0.25 * size, 2.0e-5, 2.5e-4))
    cand = grid(np.asarray(guess_xy, float)[:2], search, step0)
    cost = costs(cand)
    seeds = []
    for i in np.argsort(cost):  # the best candidates at least two coarse steps apart
        if all(np.linalg.norm(cand[i] - s) > 2.0 * step0 for s in seeds):
            seeds.append(cand[i])
        if len(seeds) >= starts:
            break
    best = None
    for centre in seeds:
        half, step = 2.0 * step0, step0 / 5.0
        while True:
            local = grid(centre, half, step)
            c = costs(local)
            centre = local[int(np.argmin(c))]
            if step <= 1.0e-6:
                break
            half, step = 2.0 * step, step / 5.0
        if best is None or c.min() < best[1]:
            best = (centre, float(c.min()))
    centre = best[0]
    residuals = signed_distance(free, tips - centre)
    delta = float(delta_of(residuals)[0]) if offset else 0.0
    residuals = residuals - delta
    return {"xy": [float(v) for v in centre], "delta_m": delta, "rms_m": float(np.sqrt(np.mean(residuals ** 2))),
            "residuals_m": [float(v) for v in residuals]}


def found_from_touches(result: dict, free_hole, yaw_base: float, guess_xy, min_contacts: int = 4,
                       max_rms: float = 1.0e-4, floor: float = 1.0e-4, max_delta: float = 2.0e-4) -> dict | None:
    """The hole from a `peg_search centre` result, or None (too few contacts, all on one side, or they do not fit
    the outline within max_rms, or only with it moved more than max_delta). At least 4 contacts: with the offset
    the fit has 3 unknowns."""
    touches = [t for t in result.get("touches") or [] if t.get("contact")]
    if not result.get("ok") or len(touches) < min_contacts:
        return None
    angles = sorted(math.radians(t["dir_deg"]) % (2.0 * math.pi) for t in touches)
    gaps = np.diff(angles + [angles[0] + 2.0 * math.pi])
    if np.max(gaps) >= math.pi:  # the contacts all on one side: the centre is not pinned
        return None
    fit = fit_centre([t["tip"] for t in touches], free_hole, yaw_base, guess_xy, max_delta=max_delta)
    if fit["rms_m"] > max_rms:
        return None
    return {"xy": fit["xy"], "sigma_m": max(floor, fit["rms_m"]), "source": f"wall touches ({len(touches)})",
            "rms_m": fit["rms_m"], "delta_m": fit["delta_m"]}


def found_from_seat(xy, free_hole, source: str) -> dict:
    """The seated peg's tip: anywhere in the free region about the hole centre."""
    return {"xy": [float(v) for v in np.asarray(xy, float)[:2]], "sigma_m": max(5.0e-5, 0.5 * reach(free_hole)),
            "source": source}


# ---------------------------------------------------------------- the record
def found_path(session_path: Path) -> Path:
    return Path(session_path) / "insert" / FOUND_FILE


def load_found(session_path: Path) -> list[dict]:
    path = found_path(session_path)
    if not path.exists():
        return []
    return list((yaml.safe_load(path.read_text()) or {}).get("holes") or [])


def record_found(session_path: Path, hole: str, vision_xy, found: dict, board_sha256: str, run: str) -> dict:
    entry = {"hole": hole, "vision_xy": [float(v) for v in np.asarray(vision_xy, float)[:2]],
             "found_xy": [float(v) for v in found["xy"]], "sigma_m": float(found["sigma_m"]),
             "source": found["source"], "board_sha256": board_sha256, "run": run,
             "time": datetime.datetime.now().isoformat(timespec="seconds")}
    entries = load_found(session_path) + [entry]
    path = found_path(session_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# pipeline insert: holes the peg went into (board_pipeline/holefit.py). vision_xy: where insert\n"
                    "# aimed (base, before any board correction); found_xy: the hole; sigma_m: its uncertainty.\n" +
                    yaml.safe_dump({"holes": entries}, sort_keys=False))
    return entry


# ---------------------------------------------------------------- the correction
def fit_correction(entries: list[dict], board_sha256: str | None, prior_shift_m: float = 0.003,
                   prior_turn_deg: float = 0.3, outlier_sigmas: float = 3.0, outlier_floor_m: float = 0.0005) -> dict | None:
    """The board's rigid shift in the base plane from the found holes of this board pose (None: none).

    Linear in (tx, ty, theta) for small turns about the holes' weighted centre c:
      found - vision = t + theta J (vision - c),  J = [[0, -1], [1, 0]]
    weighted least squares (1 / sigma^2) with zero-mean priors; the hole worst off its fit is dropped while its
    residual is beyond outlier_sigmas x its combined sigma and outlier_floor_m (at least 2 holes kept)."""
    use = [e for e in entries if board_sha256 is None or e.get("board_sha256") == board_sha256]
    if not use:
        return None
    dropped = []
    while True:
        v = np.array([e["vision_xy"] for e in use], float)
        f = np.array([e["found_xy"] for e in use], float)
        w = 1.0 / np.square(np.array([e["sigma_m"] for e in use], float))
        c = np.sum(v * w[:, None], axis=0) / np.sum(w)
        A = np.zeros((2 * len(use), 3))
        A[0::2, 0] = 1.0
        A[1::2, 1] = 1.0
        A[0::2, 2] = -(v[:, 1] - c[1])
        A[1::2, 2] = v[:, 0] - c[0]
        b = (f - v).reshape(-1)
        W = np.repeat(w, 2)
        P = np.diag([1.0 / prior_shift_m ** 2, 1.0 / prior_shift_m ** 2, 1.0 / math.radians(prior_turn_deg) ** 2])
        N = A.T @ (W[:, None] * A) + P
        cov = np.linalg.inv(N)
        p = cov @ (A.T @ (W * b))
        res = (b - A @ p).reshape(-1, 2)
        err = np.linalg.norm(res, axis=1)
        sig = np.sqrt(1.0 / w + np.trace(cov[:2, :2]) / 2.0)
        worst = int(np.argmax(err / sig))
        if len(use) > 2 and err[worst] > max(outlier_sigmas * sig[worst], outlier_floor_m):
            dropped.append({**use[worst], "residual_m": float(err[worst])})
            use = use[:worst] + use[worst + 1:]
            continue
        break
    return {"shift_m": [float(p[0]), float(p[1])], "turn_deg": math.degrees(float(p[2])),
            "centre_m": [float(c[0]), float(c[1])], "holes": len(use),
            "sigma_shift_m": float(math.sqrt(max(cov[0, 0], cov[1, 1]))),
            "sigma_turn_deg": math.degrees(math.sqrt(cov[2, 2])),
            "residuals_m": {e["hole"] + "@" + str(e.get("run", "")): float(r) for e, r in zip(use, err)},
            "dropped": [{"hole": d["hole"], "run": d.get("run"), "residual_m": d["residual_m"]} for d in dropped],
            "board_sha256": board_sha256}


def correction_matrix(corr: dict) -> np.ndarray:
    """The base-frame 4 x 4 of a correction: a turn about z through its centre, then its shift."""
    R = rotation(math.radians(corr["turn_deg"]))
    c = np.asarray(corr["centre_m"], float)
    t = np.asarray(corr["shift_m"], float)
    M = np.eye(4)
    M[:2, :2] = R
    M[:2, 3] = c + t - R @ c
    return M


def correct_xy(xy, corr: dict) -> np.ndarray:
    M = correction_matrix(corr)
    p = np.asarray(xy, float)[:2]
    return M[:2, :2] @ p + M[:2, 3]


# ---------------------------------------------------------------- the local residual field (2026-11-21)
# The rigid correction assumes one error for the whole board. The robot's kinematic error after the joint-offset model
# varies with the posture (that model alone moves the tip 2.5-3.5 mm, ~1 mm differently from hole to hole), so what
# the rigid fit leaves may not be noise. local_correction models it: rigid fit, then the found holes' residuals as
#   r_i = f(v_i) + e_i + n_i
# f a smooth field over the board (Gaussian process, squared-exponential kernel of length length_m, amplitude
# sigma_f, per axis), e_i a scatter of every insertion of its own (tau: a new peg in the fingers each hole, the
# posture), n_i the hole's measurement (its sigma). sigma_f and tau are the maximum-likelihood split of what the rigid
# fit leaves (4 holes at least; else default_sigma_f, tau 0); sigma_f within [floor, cap]. The correction at a new
# hole is the rigid one + f's mean there: near found holes it follows the field they show, far from all of them (or
# with per-hole scatter only) it is the rigid one. Simulated (tests/test_holefit.py, the 20 holes): a 0.4 mm smooth
# field, leave-one-out 0.37 mm rigid -> 0.13 mm with it; 0.3 mm per-hole scatter: no worse than rigid.

def rigid_residuals(entries: list[dict], corr: dict) -> np.ndarray:
    return np.array([np.asarray(e["found_xy"], float) - correct_xy(e["vision_xy"], corr) for e in entries])


def model_check(entries: list[dict], corr: dict) -> dict:
    """How well one rigid shift explains the found holes: chi^2 of the residuals over their sigmas (dof 2n - 3), and
    the extra scatter s per axis that would make it fit (0: the sigmas explain it)."""
    r = rigid_residuals(entries, corr)
    sig = np.array([e["sigma_m"] for e in entries], float)
    r2 = np.sum(r ** 2, axis=1)
    chi2 = float(np.sum(r2 / sig ** 2))
    dof = max(1, 2 * len(entries) - 3)
    extra = 0.0
    if chi2 > dof:
        lo, hi = 0.0, 0.01
        for _ in range(60):  # sum r^2 / (sigma^2 + s^2) is decreasing in s
            mid = 0.5 * (lo + hi)
            if np.sum(r2 / (sig ** 2 + mid ** 2)) > dof:
                lo = mid
            else:
                hi = mid
        extra = 0.5 * (lo + hi) / math.sqrt(2.0)  # per axis
    return {"chi2": chi2, "dof": dof, "extra_scatter_m": extra, "holes": len(entries),
            "rms_residual_m": float(math.sqrt(np.mean(r2))) if len(r2) else 0.0}


def _kernel(p, q, sigma_f: float, length: float) -> np.ndarray:
    d2 = np.sum((np.asarray(p)[:, None, :] - np.asarray(q)[None, :, :]) ** 2, axis=-1)
    return sigma_f ** 2 * np.exp(-0.5 * d2 / length ** 2)


def field_split(v: np.ndarray, r: np.ndarray, sig: np.ndarray, length: float, floor: float, cap: float) -> tuple:
    """(sigma_f, tau): the maximum-likelihood field amplitude and per-hole scatter for residuals r at v."""
    best = None
    for sigma_f in np.geomspace(floor, cap, 15):
        for tau in np.r_[0.0, np.geomspace(2e-5, cap, 14)]:
            K = _kernel(v, v, sigma_f, length) + np.diag(sig ** 2 + tau ** 2)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            a = np.linalg.solve(L, r)
            ll = -0.5 * float(np.sum(a ** 2)) - r.shape[1] * float(np.sum(np.log(np.diag(L))))
            if best is None or ll > best[0]:
                best = (ll, float(sigma_f), float(tau))
    return best[1], best[2]


def local_correction(entries: list[dict], board_sha256: str | None, at_xy=None, length_m: float = 0.06,
                     sigma_f_m: float | None = None, tau_m: float | None = None, default_sigma_f_m: float = 3.0e-4,
                     floor_m: float = 2.0e-5, cap_m: float = 1.0e-3, **rigid) -> dict | None:
    """The rigid correction plus the local field at at_xy (None: rigid only). Returns the rigid fit's keys and
    local_m (the field there), local_sigma_m (the field's uncertainty there), scatter_m (tau: a new insertion's own),
    sigma_f_m, length_m, model (model_check). sigma_f_m / tau_m given: not fitted."""
    corr = fit_correction(entries, board_sha256, **rigid)
    if corr is None:
        return None
    dropped = {(d["hole"], d.get("run")) for d in corr["dropped"]}
    use = [e for e in entries if (board_sha256 is None or e.get("board_sha256") == board_sha256)
           and (e["hole"], e.get("run")) not in dropped]
    model = model_check(use, corr)
    v = np.array([e["vision_xy"] for e in use], float)
    r = rigid_residuals(use, corr)
    sig = np.array([e["sigma_m"] for e in use], float)
    if sigma_f_m is None or tau_m is None:
        if len(use) >= 4:
            fit_f, fit_tau = field_split(v, r, sig, length_m, floor_m, cap_m)
        else:
            fit_f, fit_tau = default_sigma_f_m, 0.0
        sigma_f_m = fit_f if sigma_f_m is None else sigma_f_m
        tau_m = fit_tau if tau_m is None else tau_m
    sigma_f_m = float(np.clip(sigma_f_m, floor_m, cap_m))
    tau_m = float(tau_m)
    out = {**corr, "local_m": [0.0, 0.0], "local_sigma_m": sigma_f_m, "scatter_m": tau_m, "sigma_f_m": sigma_f_m,
           "length_m": length_m, "model": model,
           "at_xy": None if at_xy is None else [float(x) for x in np.asarray(at_xy)[:2]]}
    if at_xy is None:
        return out
    x = np.asarray(at_xy, float)[:2]
    K = _kernel(v, v, sigma_f_m, length_m) + np.diag(sig ** 2 + tau_m ** 2)
    k = _kernel(v, x[None, :], sigma_f_m, length_m)[:, 0]
    local = k @ np.linalg.solve(K, r)
    var = sigma_f_m ** 2 - k @ np.linalg.solve(K, k)
    out["local_m"] = [float(local[0]), float(local[1])]
    out["local_sigma_m"] = float(math.sqrt(max(var, 0.0)))
    return out


def full_matrix(corr: dict) -> np.ndarray:
    """The base-frame 4 x 4 of a local correction: the rigid one, then the local shift (evaluated at its hole)."""
    M = correction_matrix(corr)
    M[:2, 3] += np.asarray(corr.get("local_m", [0.0, 0.0]), float)
    return M


def correct_xy_full(xy, corr: dict) -> np.ndarray:
    return correct_xy(xy, corr) + np.asarray(corr.get("local_m", [0.0, 0.0]), float)


def leave_one_out(entries: list[dict], board_sha256: str | None, **local) -> dict | None:
    """Each found hole predicted from the others (all its entries left out): the error with no correction, the rigid
    one, and rigid + local field. The model that predicts held-out holes best is the one to trust."""
    use = [e for e in entries if board_sha256 is None or e.get("board_sha256") == board_sha256]
    holes = sorted({e["hole"] for e in use})
    if len(holes) < 2:
        return None
    rows = []
    for hole in holes:
        others = [e for e in use if e["hole"] != hole]
        for e in (e for e in use if e["hole"] == hole):
            corr = local_correction(others, board_sha256, at_xy=e["vision_xy"], **local)
            found = np.asarray(e["found_xy"], float)
            vision = np.asarray(e["vision_xy"], float)
            rows.append({"hole": hole, "run": e.get("run"), "sigma_m": e["sigma_m"],
                         "none_m": float(np.linalg.norm(found - vision)),
                         "rigid_m": float(np.linalg.norm(found - correct_xy(vision, corr))),
                         "local_m": float(np.linalg.norm(found - correct_xy_full(vision, corr)))})
    rms = {k: float(math.sqrt(np.mean([r[k] ** 2 for r in rows]))) for k in ("none_m", "rigid_m", "local_m")}
    return {"rows": rows, "rms": rms}
