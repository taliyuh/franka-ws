#!/usr/bin/env python3
"""Ground-truth-free metrics for the per-image board fit of an eye-in-hand calibration.

    pipeline/tools/fit_metrics.py SESSION [--views 5_fit] [--boot 200]

Run with the pose-refinement interpreter (../pose-refinement/.venv/bin/python).
Uses the single-image fits in calibrate/<views>/b*/ (colour image, fitted
T_cam_board, measured joints) and the session's calibration.json. Writes
SESSION/analysis/fit_metrics.json and fit_residuals.npz.

1 Fit quality   rim residuals at the fitted pose (re-associated with the final
                4 px search window): median, p90, share < 1 px, px -> mm at the
                point's depth, and the per-rim signed mean (systematic part).
2 Precision     a) Gauss-Newton covariance sigma^2 (J^T J)^-1 propagated to the
                   board top corners and centre;
                b) bootstrap over whole rims (rims resampled with replacement,
                   pose re-solved with the edge associations held fixed).
3 Accuracy      a) cross-view consistency: every view mapped into the base
   proxies         frame through calibration.json, distance to the calibrated
                   board (in-sample) plus the solve's leave-one-out numbers;
                b) pairwise: the board from view i vs from view j, against how
                   far the camera moved between them.
All board-pose distances are the largest displacement of the 4 top corners and
the top centre (eye_in_hand.corner_distance_mm), in mm.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
POSE_REFINEMENT = PIPELINE.parent.parent / "pose-refinement" / "boardpose"
sys.path.insert(0, str(POSE_REFINEMENT))

import board_model  # noqa: E402
import bundle_rims  # noqa: E402
import fit_rims6  # noqa: E402
import seed_board  # noqa: E402

from board_pipeline import rimfit  # noqa: E402
from board_pipeline import panda  # noqa: E402
from board_pipeline.eye_in_hand import (POINTS, Calibration, camera_in_base, chain_model,  # noqa: E402
                                        corner_distance_mm, mean_transform, model_terms, scaled)

rimfit.fit_rims6, rimfit.seed_board = fit_rims6, seed_board
rimfit.board_model, rimfit.bundle_rims = board_model, bundle_rims

WINDOW = 4  # px, the last round of fit_rims6.refine's schedule


def pose_vec(T):
    return np.r_[cv2.Rodrigues(np.asarray(T)[:3, :3])[0].ravel(), np.asarray(T)[:3, 3]]


def pose_mat(x):
    R, t = fit_rims6.RT(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def gradients(view, intr):
    K, D = rimfit.k_matrix(intr).astype(np.float32), np.asarray(intr[4:], np.float32)
    image = cv2.undistort(cv2.imread(str(view / "color.png")), K, D, newCameraMatrix=K)
    g = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 1.6)
    return cv2.Scharr(g, cv2.CV_32F, 1, 0), cv2.Scharr(g, cv2.CV_32F, 0, 1)


def residual_fn(P3, N2, edge_xy, normal):
    def res(x):
        R, t = fit_rims6.RT(x)
        projected, _ = fit_rims6.rim_points(P3, N2, R, t)
        return ((projected - edge_xy) * normal).sum(1)
    return res


def jacobian(res, x, h=(1e-6,) * 3 + (1e-6,) * 3):
    cols = []
    for k in range(6):
        d = np.zeros(6)
        d[k] = h[k]
        cols.append((res(x + d) - res(x - d)) / (2 * h[k]))
    return np.stack(cols, 1)


def corner_jacobian(x, h=1e-6):
    """d(corner points in the camera frame, 15 values)/d(pose)."""
    cols = []
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        a = (pose_mat(x + d) @ POINTS)[:3].T.ravel()
        b = (pose_mat(x - d) @ POINTS)[:3].T.ravel()
        cols.append((a - b) / (2 * h))
    return np.stack(cols, 1)


def stats(values):
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max()), "min": float(v.min()), "n": int(len(v))}


def analyse_view(view, P3, N2, lvl, loop, boot, rng):
    intr, _ = rimfit.read_camera_info(view)
    rimfit.use_camera(intr)
    fit = json.loads((view / "board_fit.json").read_text())
    x0 = pose_vec(fit["T_cam_board"])
    gx, gy = gradients(view, intr)
    edges = fit_rims6._associate_edges(gx, gy, P3, N2, lvl, loop, x0, WINDOW, True)
    if edges is None:
        return None, None
    idx, edge_xy, normal = edges
    res = residual_fn(P3[idx], N2[idx], edge_xy, normal)
    r = res(x0)
    T0 = pose_mat(x0)
    z = (P3[idx] @ T0[:3, :3].T + T0[:3, 3])[:, 2]
    fx = intr[0]
    r_mm = r * z / fx * 1e3
    e = -r                              # detected edge minus model, + = into the hole

    # 1 fit quality and its systematic part
    rims = {}
    for lid in np.unique(loop[idx]):
        m = loop[idx] == lid
        rims[int(lid)] = {"level": int(lvl[idx][m][0]), "n": int(m.sum()),
                          "mean_px": float(e[m].mean()),
                          "within_mad_px": float(np.median(np.abs(e[m] - np.median(e[m]))))}
    rim_means = np.array([v["mean_px"] for v in rims.values()])
    quality = {
        "n": int(len(r)), "distance_m": float(np.linalg.norm(T0[:3, 3])),
        "mm_per_px": float(np.median(z) / fx * 1e3),
        "mad_px": float(np.median(np.abs(r))), "p90_px": float(np.percentile(np.abs(r), 90)),
        "rms_px": float(np.sqrt((r ** 2).mean())), "inlier_1px": float((np.abs(r) < 1).mean()),
        "mad_mm": float(np.median(np.abs(r_mm))), "p90_mm": float(np.percentile(np.abs(r_mm), 90)),
        "level_mean_px": {str(k): float(e[lvl[idx] == k].mean()) for k in np.unique(lvl[idx])},
        "rim_mean_abs_px": float(np.median(np.abs(rim_means))),
        "rim_mean_spread_px": float(rim_means.std()),
        "within_rim_mad_px": float(np.median([v["within_mad_px"] for v in rims.values()])),
        "rims": len(rims),
    }

    # 2a covariance
    J = jacobian(res, x0)
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    cov = sigma ** 2 * np.linalg.inv(J.T @ J)
    C = corner_jacobian(x0)
    cc = C @ cov @ C.T
    corner_sd = [np.sqrt(np.trace(cc[3 * i:3 * i + 3, 3 * i:3 * i + 3])) * 1e3 for i in range(POINTS.shape[1])]
    covariance = {"sigma_px": float(sigma),
                  "corner_sd_max_mm": float(max(corner_sd)),
                  "t_sd_mm": (np.sqrt(np.diag(cov)[3:]) * 1e3).tolist(),
                  "rot_sd_mrad": (np.sqrt(np.diag(cov)[:3]) * 1e3).tolist()}

    # 2b bootstrap over whole rims
    rim_ids = np.unique(loop[idx])
    members = {lid: np.nonzero(loop[idx] == lid)[0] for lid in rim_ids}
    disp, tz = [], []
    for _ in range(boot):
        pick = np.concatenate([members[l] for l in rng.choice(rim_ids, len(rim_ids))])
        rb = residual_fn(P3[idx][pick], N2[idx][pick], edge_xy[pick], normal[pick])
        sol = least_squares(rb, x0, loss="soft_l1", f_scale=2.0,
                            x_scale=[0.01, 0.01, 0.01, 0.002, 0.002, 0.002], max_nfev=60)
        Tb = pose_mat(sol.x)
        disp.append(corner_distance_mm(Tb, T0))
        tz.append((Tb[:3, 3] - T0[:3, 3]) * 1e3)
    tz = np.array(tz)
    bootstrap = {"corner_mm": stats(disp), "t_sd_mm": tz.std(0).tolist()}

    per_point = {"xy": P3[idx][:, :2], "level": lvl[idx], "loop": loop[idx], "e_px": e, "e_mm": -r_mm}
    return {"quality": quality, "covariance": covariance, "bootstrap": bootstrap}, per_point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--views", default="5_fit")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--max-mad", type=float, default=1.0)
    ap.add_argument("--refit", action="store_true",
                    help="re-run each view's robot-seeded fit with calibration.json's intrinsics "
                         "first (into analysis/refit/), for sessions whose fit step used other ones")
    ap.add_argument("--redo", action="store_true", help="redo an existing --refit")
    ap.add_argument("--outlier-mm", type=float, default=5.0,
                    help="views whose board is further than this from the others are failed fits")
    a = ap.parse_args()

    cal = Calibration.load(a.session / "calibration.json")
    cal_data = json.loads((a.session / "calibration.json").read_text())
    board_ref = np.asarray(cal_data["T_base_board_during_calibration"])
    P3, N2, lvl, loop = board_model.build_levels()
    rng = np.random.default_rng(0)

    source = a.session / "calibrate" / a.views
    if a.refit:
        intr = np.array([cal_data["intrinsics"][k] for k in rimfit.INTRINSIC_NAMES])
        target = a.session / "analysis" / "refit"
        for view in sorted(source.glob("b*")):
            if not (view / "board_fit.json").exists():
                continue
            dest = target / view.name
            dest.mkdir(parents=True, exist_ok=True)
            if not (dest / "color.png").exists():
                (dest / "color.png").symlink_to((view / "color.png").resolve())
            (dest / "joints.json").write_bytes((view / "joints.json").read_bytes())
            info = json.loads((view / "camera_info.json").read_text())
            rimfit.write_camera_info(dest, intr, info["width"], info["height"], info["frame_id"])
            if (dest / "board_fit.json").exists() and not a.redo:
                continue
            seed = json.loads((view / "board_fit.json").read_text())["T_cam_board"]
            if rimfit.fit_from_pose(dest, intr, seed) is None:
                (dest / "board_fit.json").unlink(missing_ok=True)
                print(f"{view.name}: refit failed")
        source = target

    views, points = {}, {}
    for view in sorted(source.glob("b*")):
        if not (view / "board_fit.json").exists():
            continue
        result, pp = analyse_view(view, P3, N2, lvl, loop, a.boot, rng)
        if result is None:
            print(f"{view.name}: no edges")
            continue
        q = np.asarray(json.loads((view / "joints.json").read_text())["measured_joints"])
        T = np.asarray(json.loads((view / "board_fit.json").read_text())["T_cam_board"])
        result["q"] = q.tolist()
        result["cam_board"] = T.tolist()
        result["base_board"] = cal.base_board(q, T).tolist()
        result["cam_in_base"] = cal.camera_in_base(q).tolist()
        result["used"] = result["quality"]["mad_px"] <= a.max_mad
        result["to_calibrated_board_mm"] = corner_distance_mm(cal.base_board(q, T), board_ref)
        result["loo_mm"] = cal_data["quality"]["per_view_loo_mm"].get(view.name)
        views[view.name] = result
        points[view.name] = pp
        qy, b = result["quality"], result["bootstrap"]["corner_mm"]
        print(f"{view.name}: mad {qy['mad_px']:.3f} px ({qy['mad_mm']:.3f} mm) inl {qy['inlier_1px']:.2f} "
              f"rim|mean| {qy['rim_mean_abs_px']:.2f}  cov {result['covariance']['corner_sd_max_mm']:.3f} mm  "
              f"boot p50 {b['median']:.3f} p90 {b['p90']:.3f} mm  to-board {result['to_calibrated_board_mm']:.2f} mm",
              flush=True)

    used = [k for k, v in views.items() if v["used"]]
    # A failed fit is a view whose board lies more than --outlier-mm from the robust
    # mean of the others (the board is static): drop the worst until all agree.
    kept = list(views)
    while len(kept) > 3:
        mean = mean_transform([np.asarray(views[k]["base_board"]) for k in kept])
        dist = {k: corner_distance_mm(np.asarray(views[k]["base_board"]), mean) for k in kept}
        worst = max(dist, key=dist.get)
        if dist[worst] <= a.outlier_mm:
            break
        kept.remove(worst)
    clean = [k for k in views if k in kept]

    # 3a refit of the chain on the clean views only (same model terms and fixed
    # joint offsets as the session), with leave-one-out
    terms = tuple(t for t in model_terms(cal_data["model"]) if t != "off")
    n, fit_chain, error, unpack = chain_model(terms, cal.dq)
    chain_views = [{"name": k, "q": np.asarray(views[k]["q"]),
                    "cam_board": np.asarray(views[k]["cam_board"])} for k in clean]
    v = fit_chain(chain_views, np.r_[panda.to_vec(cal.x), panda.to_vec(board_ref), np.zeros(n - 12)])
    fit_err = np.array([error(w, v) for w in chain_views])
    loo = np.array([error(chain_views[i], fit_chain(chain_views[:i] + chain_views[i + 1:], v))
                    for i in range(len(chain_views))])
    x, board_clean, dq, s_, c = unpack(v)
    for k in views:
        q, T = np.asarray(views[k]["q"]), np.asarray(views[k]["cam_board"])
        C = camera_in_base(q, x, dq, c)
        views[k]["base_board_clean"] = (C @ scaled(T, s_)).tolist()
        views[k]["cam_in_base_clean"] = C.tolist()
        views[k]["to_clean_board_mm"] = corner_distance_mm(C @ scaled(T, s_), board_clean)
    refit = {"views": clean, "terms": list(terms), "fit_mm": stats(fit_err), "loo_mm": stats(loo),
             "per_view_loo_mm": dict(zip(clean, loo.round(4).tolist())),
             "mount_change_mm": float(np.linalg.norm(x[:3, 3] - cal.x[:3, 3]) * 1e3),
             "board_change_mm": corner_distance_mm(board_clean, board_ref)}
    print(f"clean refit on {len(clean)} views: fit median {np.median(fit_err):.3f} mm, "
          f"LOO median {np.median(loo):.3f} p90 {np.percentile(loo, 90):.3f} max {loo.max():.3f} mm")

    # 3b pairwise consistency vs camera motion, clean views through the clean refit
    pairs = []
    for i, j in combinations(clean, 2):
        Ci, Cj = np.asarray(views[i]["cam_in_base_clean"]), np.asarray(views[j]["cam_in_base_clean"])
        rot = np.degrees(np.linalg.norm(cv2.Rodrigues(Ci[:3, :3].T @ Cj[:3, :3])[0]))
        pairs.append({"a": i, "b": j, "rot_deg": float(rot),
                      "move_mm": float(np.linalg.norm(Ci[:3, 3] - Cj[:3, 3]) * 1e3),
                      "board_mm": corner_distance_mm(np.asarray(views[i]["base_board_clean"]),
                                                     np.asarray(views[j]["base_board_clean"]))})

    suffix = "_refit" if a.refit else ""
    out = a.session / "analysis"
    out.mkdir(exist_ok=True)
    (out / f"fit_metrics{suffix}.json").write_text(json.dumps(
        {"session": a.session.name, "views_dir": str(source), "refit": a.refit, "max_mad_px": a.max_mad, "used": used,
         "outlier_mm": a.outlier_mm, "clean": clean, "clean_refit": refit,
         "calibration_quality": {k: v for k, v in cal_data["quality"].items() if k != "per_view_loo_mm"},
         "views": views, "pairs": pairs}, indent=1))
    np.savez_compressed(out / f"fit_residuals{suffix}.npz", **{
        f"{name}__{key}": value for name, pp in points.items() for key, value in pp.items()})
    print(f"wrote {out / f'fit_metrics{suffix}.json'}")


if __name__ == "__main__":
    main()
