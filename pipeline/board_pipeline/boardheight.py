"""The board's height as the robot touches it, modelled once per board pose (`pipeline insert`, 2026-11-23).

Vision's board plane is off the robot's in height by 1-2 mm, and the difference is smooth over the board: the 3-point
touches of 2026-11-23 (9 runs, every shape, one board pose) fit

    z_robot - z_vision = c + gx (x - x0) + gy (y - y0) + o_shape

to 0.07 mm RMS. gx, gy: vision's plane is ~0.25 deg tilted against the robot's (4.2 um/mm along base x). o_shape: the
grasp's, not the peg's (every peg is the same length): the triangle's tip touched 0.42 mm higher than the L's, i.e.
held that much higher in the fingers than the grasp model has it, or tilted in them (an upright touch then meets the
face's lowest edge: extent x tilt, so it differs by shape). It holds while the grasp repeats (the triangle's 4 runs
agreed); one touch per run measures the grasp actually held, which is why board_touches auto touches once.
Each run predicted from the others' touches (leave-one-run-out): 0.12 mm RMS (max 0.25) with no touch of its own, 0.04
mm (max 0.08) once one touch of its own sets c for that grasp -- as good as the 3-point plane (the same spot touched
in two runs repeats to 0.02-0.05 mm).

So insert touches the board once (the slope from here) instead of three times once the model knows the board's slope
and the peg (insert.board_touches auto), and the plunge's contact is classified against the predicted board instead
of vision's (round / hexagon 2026-11-23: the board 2.1 mm above vision's read as "in the air").

Every touch is kept in <session>/insert/board_heights.yaml with vision's height there and the board pose's sha256 (a
`pipeline locate` that moves the board starts afresh); `backfill` collects the touches of earlier runs.
"""

from __future__ import annotations

import datetime
import glob
import json
import math
from pathlib import Path

import numpy as np
import yaml

HEIGHTS_FILE = "board_heights.yaml"


def vision_height(board: np.ndarray, surface_z: float, xy) -> float:
    """Vision's board top (base z) under base xy: the plane through board @ (0, 0, surface_z) along the board's z."""
    board = np.asarray(board, float)
    n = board[:3, 2]
    o = board[:3, :3] @ np.array([0.0, 0.0, surface_z]) + board[:3, 3]
    return float(o[2] - (n[0] * (xy[0] - o[0]) + n[1] * (xy[1] - o[1])) / n[2])


def vision_slope(board: np.ndarray) -> np.ndarray:
    """Vision's board plane slope (dz/dx, dz/dy) in base."""
    n = np.asarray(board, float)[:3, 2]
    return np.array([-n[0] / n[2], -n[1] / n[2]])


def heights_path(session_path: Path) -> Path:
    return Path(session_path) / "insert" / HEIGHTS_FILE


def load(session_path: Path) -> list[dict]:
    path = heights_path(session_path)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return list(data.get("touches") or [])


def _write(session_path: Path, touches: list[dict]) -> None:
    path = heights_path(session_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# pipeline insert: board touches (board_pipeline/boardheight.py). xy, z: the touched board top (base, "
              "m);\n# z_vision: vision's board there; shape: the held peg.\n")
    path.write_text(header + yaml.safe_dump({"touches": touches}, sort_keys=False))


def record(session_path: Path, run: str, shape: str, board_sha256: str | None, points, z_vision) -> list[dict]:
    """Add one run's touches (points: (x, y, z) base m; z_vision: vision's height at each). Replaces the run's own."""
    touches = [t for t in load(session_path) if t.get("run") != run]
    now = datetime.datetime.now().isoformat(timespec="seconds")
    new = [{"run": run, "shape": shape, "xy": [float(p[0]), float(p[1])], "z": float(p[2]), "z_vision": float(zv),
            "board_sha256": board_sha256, "time": now} for p, zv in zip(points, z_vision)]
    _write(session_path, touches + new)
    return new


def backfill(session_path: Path) -> int:
    """The touches of earlier insert runs (peck.json touch_points, vision from targets/<hole>.json of the same board
    pose sha) into board_heights.yaml. Returns how many runs were added."""
    session_path = Path(session_path)
    known = {t.get("run") for t in load(session_path)}
    board_file = session_path / "board_pose.json"
    # targets/ holds the current board pose: runs from before it was written aimed at another one.
    since = datetime.datetime.fromtimestamp(board_file.stat().st_mtime) if board_file.exists() else None
    added = 0
    for run_dir in sorted(glob.glob(str(session_path / "insert" / "2*"))):
        run_dir = Path(run_dir)
        if run_dir.name in known or not (run_dir / "plan.yaml").exists() or not (run_dir / "peck.json").exists():
            continue
        try:
            if since is not None and datetime.datetime.strptime(run_dir.name, "%Y-%m-%d_%H%M%S") < since:
                continue
        except ValueError:
            continue
        try:
            plan = yaml.safe_load((run_dir / "plan.yaml").read_text())
            peck = json.loads((run_dir / "peck.json").read_text())
            target = json.loads((session_path / "targets" / f"{plan['hole']}.json").read_text())
        except (OSError, ValueError, KeyError):
            continue
        points = peck.get("touch_points") or []
        if not points:
            continue
        src = target.get("source") or {}
        board = np.asarray(src.get("T_base_board_true_frame") or target["T_base_board"])
        z_vision = [vision_height(board, float(target["board_surface_z_m"]), p) for p in points]
        record(session_path, run_dir.name, plan["shape"], src.get("board_pose_sha256"), points, z_vision)
        added += 1
    return added


def fit(touches: list[dict], board_sha256: str | None, shape_prior_m: float = 0.0005,
        slope_prior: float = 0.01) -> dict | None:
    """z - z_vision = c + g . (xy - centre) + o_shape (the grasp of that shape) over the touches of this board pose
    (least squares, Gaussian
    priors: o_shape shape_prior_m, g slope_prior; the shapes' offsets sum to zero over the shapes seen, c their mean).
    None without touches."""
    use = [t for t in touches if board_sha256 is None or t.get("board_sha256") == board_sha256]
    if not use:
        return None
    xy = np.array([t["xy"] for t in use], float)
    dz = np.array([t["z"] - t["z_vision"] for t in use], float)
    shapes = sorted({t["shape"] for t in use})
    centre = xy.mean(axis=0)
    n_par = 3 + len(shapes)
    a = np.zeros((len(use), n_par))
    a[:, 0] = 1.0
    a[:, 1:3] = xy - centre
    for i, t in enumerate(use):
        a[i, 3 + shapes.index(t["shape"])] = 1.0
    sigma_z = 5.0e-5  # a touch's own noise (the same spot in two runs: 0.02-0.05 mm)
    # Priors as extra rows: the slope, each shape's offset, and their sum zero (c is the shapes' mean).
    rows = [np.eye(n_par)[1] * sigma_z / slope_prior, np.eye(n_par)[2] * sigma_z / slope_prior]
    rows += [np.eye(n_par)[3 + k] * sigma_z / shape_prior_m for k in range(len(shapes))]
    rows.append(np.r_[0.0, 0.0, 0.0, np.ones(len(shapes))])
    a_all = np.vstack([a, *rows])
    b_all = np.r_[dz, np.zeros(len(rows))]
    w, *_ = np.linalg.lstsq(a_all, b_all, rcond=None)
    resid = dz - a @ w
    runs = sorted({t["run"] for t in use})
    dof = max(1, len(use) - n_par)
    rms = float(math.sqrt(float(resid @ resid) / dof)) if len(use) > n_par else float("nan")
    span = xy.max(axis=0) - xy.min(axis=0) if len(use) > 1 else np.zeros(2)
    return {"board_sha256": board_sha256, "centre": centre.tolist(), "c": float(w[0]), "slope": w[1:3].tolist(),
            "shape_offset": {s: float(w[3 + k]) for k, s in enumerate(shapes)},
            "shape_touches": {s: sum(t["shape"] == s for t in use) for s in shapes},
            "touches": len(use), "runs": len(runs), "span_m": span.tolist(), "rms_m": rms,
            "max_residual_m": float(np.abs(resid).max())}


def slope_known(model: dict | None, min_runs: int = 2, min_span_m: float = 0.04) -> bool:
    """The model's slope rests on touches spread over the board (at least min_span_m along base x and y)."""
    return (model is not None and model["runs"] >= min_runs and min(model["span_m"]) >= min_span_m)


def predict(model: dict, xy, shape: str, z_vision: float) -> tuple[float, bool]:
    """The robot's board height at base xy for the held `shape`, and whether this shape has touches of its own."""
    known = shape in model["shape_offset"]
    d = np.asarray(xy, float)[:2] - np.asarray(model["centre"])
    dz = model["c"] + float(np.dot(model["slope"], d)) + (model["shape_offset"][shape] if known else 0.0)
    return float(z_vision + dz), known


def leave_one_run_out(touches: list[dict], board_sha256: str | None) -> list[dict]:
    """Each run's touches predicted from the other runs': with none of its own, and with its first touch setting its
    grasp's offset (what insert does with board_touches 1)."""
    use = [t for t in touches if board_sha256 is None or t.get("board_sha256") == board_sha256]
    out = []
    for run in sorted({t["run"] for t in use}):
        own = [t for t in use if t["run"] == run]
        model = fit([t for t in use if t["run"] != run], board_sha256)
        if model is None:
            continue
        pred = np.array([predict(model, t["xy"], t["shape"], t["z_vision"])[0] for t in own])
        err = np.array([t["z"] for t in own]) - pred
        one = err - err[0]
        out.append({"run": run, "shape": own[0]["shape"], "shape_known": own[0]["shape"] in model["shape_offset"],
                    "rms_none_m": float(np.sqrt(np.mean(err ** 2))),
                    "rms_one_m": float(np.sqrt(np.mean(one[1:] ** 2))) if len(own) > 1 else float("nan")})
    return out
