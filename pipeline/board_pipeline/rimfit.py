"""Board fitting through pose-refinement/boardpose (rim fit + bundle adjustment).

pose-refinement keeps the camera model in module globals (fit_rims6.K_4K, ...)
and bundle_rims hard-codes a 5-term distortion model; the Azure needs its
8-term rational model. Both adaptations live here, so the rest of the
pipeline only sees fit_depth_seeded / fit_from_pose / bundle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from board_pipeline.config import Config

INTRINSIC_NAMES = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]

fit_rims6 = seed_board = board_model = bundle_rims = None


def load(cfg: Config) -> None:
    """Import the pose-refinement modules (once)."""
    global fit_rims6, seed_board, board_model, bundle_rims
    if fit_rims6 is not None:
        return
    sys.path.insert(0, str(cfg.pose_refinement / "boardpose"))
    import board_model as _board_model
    import bundle_rims as _bundle_rims
    import fit_rims6 as _fit_rims6
    import seed_board as _seed_board
    fit_rims6, seed_board, board_model, bundle_rims = _fit_rims6, _seed_board, _board_model, _bundle_rims


# ------------------------------------------------------------- camera model
def intrinsics_vector(K, D) -> np.ndarray:
    K = np.asarray(K, float).reshape(3, 3)
    return np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], np.asarray(D, float)]


def k_matrix(intr) -> np.ndarray:
    fx, fy, cx, cy = intr[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def use_camera(intr) -> None:
    K, D = k_matrix(intr), np.asarray(intr[4:], float)
    fit_rims6.K_4K = seed_board.K_4K = K
    fit_rims6.D_4K = seed_board.D_4K = D


def write_camera_info(view: Path, intr, width: int, height: int, frame_id: str) -> None:
    (view / "camera_info.json").write_text(json.dumps({
        "width": width, "height": height, "distortion_model": "rational_polynomial",
        "k": k_matrix(intr).ravel().tolist(), "d": list(map(float, intr[4:])),
        "frame_id": frame_id}, indent=2))


def read_camera_info(view: Path) -> tuple[np.ndarray, dict]:
    info = json.loads((view / "camera_info.json").read_text())
    return intrinsics_vector(info["k"], info["d"]), info


# ------------------------------------------------------------------ fitting
def _transform(result) -> np.ndarray:
    return np.asarray(result["T_cam_board"], float)


def save_fit(view: Path, result: dict, mode: str) -> dict:
    keep = {k: result[k] for k in ("mad_px", "rms_px", "n") if k in result}
    fit = dict(T_cam_board=_transform(result).tolist(), mode=mode, **keep)
    (view / "board_fit.json").write_text(json.dumps(fit, indent=2))
    return fit


def fit_depth_seeded(view: Path, intr, rim_bias=None) -> dict | None:
    """Fit a view with no prior pose; needs color.png + depth.png."""
    use_camera(intr)
    result = fit_rims6.fit_view(view, topk=20, frozen_rim_bias=rim_bias)
    return None if result is None else save_fit(view, result, "depth-seeded")


def fit_from_pose(view: Path, intr, T_cam_board_seed, rim_bias=None) -> dict | None:
    """Refine a view from a predicted board pose (color.png only)."""
    use_camera(intr)
    result = fit_rims6.fit_view_from_pose(view, T_cam_board_seed, constrain=False,
                                          frozen_rim_bias=rim_bias)
    return None if result is None else save_fit(view, result, "robot-seeded")


def overlay(view: Path, intr, T_cam_board, out: Path) -> None:
    """Fitted rims drawn on the undistorted image (green top, red floor), half size."""
    K, D = k_matrix(intr), np.asarray(intr[4:], float)
    image = cv2.undistort(cv2.imread(str(view / "color.png")), K, D, newCameraMatrix=K)
    P3, _, level, _ = board_model.build_levels()
    T = np.asarray(T_cam_board)
    points = P3 @ T[:3, :3].T + T[:3, 3]
    uv = (K @ (points / points[:, 2:3]).T).T[:, :2]
    for (u, v), lvl in zip(uv, level):
        cv2.circle(image, (int(u), int(v)), 3, (0, 0, 255) if lvl else (0, 255, 0), -1)
    cv2.imwrite(str(out), cv2.resize(image, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ------------------------------------------------------ bundle adjustment
def _project_rational(P3, rvec, tvec, intr):
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = intr
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    c = P3 @ R.T + tvec
    z = np.maximum(c[:, 2], 1e-6)
    x, y = c[:, 0] / z, c[:, 1] / z
    r2 = x * x + y * y
    radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], 1)


def bundle(views: list[Path], intr0, free: list[str], out_dir: Path) -> dict:
    """Refine the free intrinsics and every view pose over all rim observations.

    From calibration_diagnostics/board_handeye/bundle_azure.py: pose-refinement's
    bundle_rims (edge association, rim bias, sparse LM) with the Azure's
    rational distortion. Writes out_dir/<view>/{board_fit.json, joints.json}
    and returns the refined intrinsics and rim bias.
    """
    from scipy.optimize import least_squares
    br = bundle_rims
    br.project = _project_rational
    br.NINTR = 12
    br.NSH = br.NINTR + br.NBIAS

    pts3, nrm2, level, loop = board_model.build_levels()
    loaded = []
    for d in views:
        image = cv2.imread(str(d / "color.png"), cv2.IMREAD_GRAYSCALE)
        T = np.asarray(json.loads((d / "board_fit.json").read_text())["T_cam_board"])
        v = br.View()
        v.pose = v.name = v.label = d.name
        v.rvec = cv2.Rodrigues(T[:3, :3])[0].ravel()
        v.tvec = T[:3, 3].copy()
        x0, x1, y0, y1 = br.crop_bounds(_project_rational(pts3, v.rvec, v.tvec, intr0), image.shape)
        v.gx, v.gy = br.gradient_images(image, x0, x1, y0, y1)
        v.ox, v.oy = x0, y0
        loaded.append(v)

    intr0 = np.asarray(intr0, float)
    free_idx = [INTRINSIC_NAMES.index(n) for n in free]
    held = [i for i in range(12) if i not in free_idx]
    bias0 = board_model.RIM_BIAS0.copy()
    free_bias = [0, 1]
    intr, bias = intr0.copy(), bias0.copy()

    def residuals(x, views, obs):
        xi = x.copy()
        xi[held] = intr0[held]
        return br.residuals(xi, views, obs, pts3, nrm2, level, True, intr0, free_bias, bias0)

    for iteration, S in enumerate((10, 6, 4, 4)):
        obs = br.measure(loaded, intr, pts3, nrm2, S, level, loop, bias)
        sparsity = br.sparsity(loaded, obs, True, free_bias).tolil()
        sparsity[:, held] = 0
        solution = least_squares(residuals, br.pack(loaded, intr, bias), jac_sparsity=sparsity,
                                 loss="soft_l1", f_scale=1.5, x_scale="jac", method="trf",
                                 max_nfev=40, args=(loaded, obs))
        fitted_intr, fitted_bias = br.unpack(solution.x, loaded)
        intr = intr0.copy()
        intr[free_idx] = np.asarray(fitted_intr)[free_idx]
        bias = bias0.copy()
        bias[free_bias] = np.asarray(fitted_bias)[free_bias]
        r = solution.fun
        print(f"  bundle iteration {iteration} (search {S:2d} px): {len(r)} residuals, "
              f"rms {np.sqrt((r ** 2).mean()):.3f} px, mad {np.median(np.abs(r)):.3f} px", flush=True)

    for name, a, b in zip(INTRINSIC_NAMES, intr0, intr):
        if INTRINSIC_NAMES.index(name) in free_idx:
            print(f"  {name}: {a:.3f} -> {b:.3f}")

    obs = br.measure(loaded, intr, pts3, nrm2, 4, level, loop, bias)
    for source, v, o in zip(views, loaded, obs):
        out = out_dir / v.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "joints.json").write_bytes((source / "joints.json").read_bytes())
        P = board_model.biased_pts(pts3[o["idx"]], nrm2[o["idx"]], level[o["idx"]],
                                   cv2.Rodrigues(v.rvec)[0], v.tvec, bias)
        projected, _ = br.project_with_normals(P, nrm2[o["idx"]], v.rvec, v.tvec, intr)
        res = ((projected - o["q"]) * o["d"]).sum(1)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(v.rvec)[0]
        T[:3, 3] = v.tvec
        (out / "board_fit.json").write_text(json.dumps(dict(
            T_cam_board=T.tolist(), mad_px=float(np.median(np.abs(res))),
            rms_px=float(np.sqrt((res ** 2).mean())), n=int(len(res)), mode="bundled"), indent=2))
    return {"names": INTRINSIC_NAMES, "factory": intr0.tolist(), "refined": intr.tolist(),
            "free": list(free), "rim_bias": bias.tolist()}
