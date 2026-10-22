"""Eye-in-hand setup: Azure Kinect rigid to fr3_link6, the board is the target.

Chain model for view i of a static board:

    T_base_board = FK_link6(q_i + dq) @ Sag(q_i) @ X @ Scale(T_cam_board_i)

X = T_link6_camera, dq = encoder offsets of joints 2-5, Scale = a camera
translation scale (focal length / board scale), Sag = a small camera rotation
proportional to gravity in the link6 frame (mount compliance). Joint 7 does
not move the camera, so it is never used.

calibrate: look -> seed_fit -> plan -> capture -> fit -> bundle -> solve
    Estimates X, dq, scale, sag, refined intrinsics and the board pose.
locate:    look -> seed_fit -> plan -> capture -> fit -> solve
    Holds the calibration fixed; every view gives one T_base_board, and their
    robust mean is the result.

Ported from calibration_diagnostics/board_handeye (generate_poses.py,
to_views.py, fit_views.py, bundle_azure.py, diagnose_models.py, save_best.py,
export_hole_target.py), where the numbers in the 2026-10-12 notes come from.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from board_pipeline import fr3, rimfit, ros
from board_pipeline.config import PIPELINE_DIR, PipelineError, load_yaml
from board_pipeline.session import BOARD_POSE, CALIBRATION, Context, Step, sha256

BOARD_HALF = np.array([0.160, 0.125])
SURFACE_Z = 0.025
# Board top corners and centre: "accuracy at the board" is measured here.
POINTS = np.array([[sx * BOARD_HALF[0], sy * BOARD_HALF[1], SURFACE_Z, 1.0]
                   for sx in (-1, 1) for sy in (-1, 1)] + [[0, 0, SURFACE_Z, 1.0]]).T
GRAVITY = np.array([0, 0, -1.0])


# ================================================================ chain model
def chain_model(terms):
    """(parameter count, fit, error, unpack) for a set of chain error terms.

    off   encoder offsets joints 2-5
    scale camera translation scale
    sag   camera rotation proportional to gravity in the link6 frame
    """
    n = 12 + 4 * ("off" in terms) + ("scale" in terms) + 3 * ("sag" in terms)

    def unpack(v):
        i, dq, s, c = 12, np.zeros(7), 1.0, np.zeros(3)
        if "off" in terms:
            dq[1:5] = v[i:i + 4]
            i += 4
        if "scale" in terms:
            s = 1 + v[i]
            i += 1
        if "sag" in terms:
            c = v[i:i + 3]
        return fr3.from_vec(v[:6]), fr3.from_vec(v[6:12]), dq, s, c

    def predict(view, v):
        x, base_board, dq, s, c = unpack(v)
        return (camera_in_base(view["q"], x, dq, c) @ scaled(view["cam_board"], s) @ POINTS)[:3], \
            (base_board @ POINTS)[:3]

    def residuals(v, views):
        return np.concatenate([(p - r).ravel() for p, r in (predict(w, v) for w in views)]) * 1e3

    def fit(views, v0):
        return least_squares(residuals, v0, args=(views,), loss="soft_l1", f_scale=1.0,
                             x_scale="jac").x

    def error(view, v):
        p, r = predict(view, v)
        return np.linalg.norm(p - r, axis=0).max() * 1e3

    return n, fit, error, unpack


def model_terms(values) -> tuple:
    """Validated chain terms; an unquoted `off` in YAML arrives as False."""
    terms = tuple("off" if v is False else str(v) for v in values)
    unknown = set(terms) - {"off", "scale", "sag"}
    if unknown:
        raise PipelineError(f"unknown chain model terms {sorted(unknown)}; use off, scale, sag")
    return terms


def scaled(cam_board, s):
    out = np.array(cam_board, float)
    out[:3, 3] *= s
    return out


def camera_in_base(q, x, dq, sag):
    """T_base_camera for measured joints q under the chain model."""
    link6 = fr3.fk(np.asarray(q) + dq, "fr3_link6")
    gravity = link6[:3, :3].T @ GRAVITY
    rotation = np.eye(4)
    rotation[:3, :3] = cv2.Rodrigues(np.asarray(sag * gravity, float))[0]
    return link6 @ rotation @ x


class Calibration:
    """The chain parameters stored in calibration.json."""

    def __init__(self, data: dict):
        self.data = data
        self.x = np.asarray(data["T_link6_camera"])
        self.dq = np.array([data["joint_offsets_rad"][n] for n in fr3.JOINT_NAMES])
        self.scale = float(data["camera_translation_scale"])
        self.sag = np.asarray(data["mount_sag_rad_per_g"], float)
        self.intrinsics = np.array([data["intrinsics"][n] for n in rimfit.INTRINSIC_NAMES])
        self.rim_bias = np.asarray(data["rim_bias"], float)

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        return cls(json.loads(Path(path).read_text()))

    def camera_in_base(self, q):
        return camera_in_base(q, self.x, self.dq, self.sag)

    def base_board(self, q, cam_board):
        return self.camera_in_base(q) @ scaled(cam_board, self.scale)

    def cam_board(self, q, base_board):
        """Predicted (unscaled) camera-frame board pose, to seed a fit."""
        return scaled(np.linalg.inv(self.camera_in_base(q)) @ base_board, 1.0 / self.scale)


# ================================================================ view plans
def plan_views(q0, base_board, x, K, width, height, count, p):
    """Joint poses that view the static board from varied directions.

    Every candidate is checked for joint-limit margin, clearance of every link
    origin above the board top (also along the joint-space path between
    consecutive poses) and full board visibility with an image margin.
    Returns (poses, report lines).
    """
    rng = np.random.default_rng(p.seed)
    centre = base_board[:3, 3] + base_board[:3, :3] @ np.array([0, 0, SURFACE_Z])
    table_z = centre[2]          # reference = board top; the board stands above the foam
    normal = base_board[:3, 2]
    if normal[2] < 0:
        raise PipelineError("board +z points down; the seed fit is upside down")
    corners = np.array([[sx * BOARD_HALF[0], sy * BOARD_HALF[1], SURFACE_Z, 1]
                        for sx in (-1, 1) for sy in (-1, 1)])
    seeds = [q0]
    look_x = (fr3.fk(q0, "fr3_link6") @ x)[:3, 0]

    lo, hi = fr3.LOWER + p.limit_margin, fr3.UPPER - p.limit_margin
    if p.box_deg is not None:
        m = math.radians(p.box_deg)
        lo[:6] = np.maximum(lo[:6], q0[:6] - m)
        hi[:6] = np.minimum(hi[:6], q0[:6] + m)

    def ok_pose(q):
        if np.any(q < lo) or np.any(q > hi):
            return False
        return fr3.frame_origins(q)[1:, 2].min() > table_z + p.clearance

    accepted = []
    for _ in range(4000):
        tilt = math.radians(p.max_tilt_deg) * math.sqrt(rng.uniform())
        az = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(p.min_dist, p.max_dist)
        e1 = np.cross(normal, [1, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        direction = math.cos(tilt) * normal + math.sin(tilt) * (math.cos(az) * e1 + math.sin(az) * e2)
        cam_pos = centre + dist * direction
        aim = centre + base_board[:3, :3] @ np.r_[rng.uniform(-0.03, 0.03, 2), 0]
        z = aim - cam_pos
        z /= np.linalg.norm(z)
        ref = base_board[:3, 0] - z * (base_board[:3, 0] @ z)
        ref /= np.linalg.norm(ref)
        # Roll is sampled around the look pose's own roll: with joint 7 fixed,
        # rolling the camera far from it needs large moves of joints 1-6.
        x0 = look_x - z * (look_x @ z)
        roll0 = math.atan2(np.cross(ref, x0) @ z, ref @ x0)
        roll = roll0 + math.radians(rng.uniform(-p.max_roll_deg, p.max_roll_deg))
        xa = math.cos(roll) * ref + math.sin(roll) * np.cross(z, ref)
        cam = np.eye(4)
        cam[:3, :3] = np.c_[xa, np.cross(z, xa), z]
        cam[:3, 3] = cam_pos
        target = cam @ np.linalg.inv(x)
        best = None
        for s in seeds:
            q, err = fr3.ik_link6(target, s)
            if err < 1e-4 and ok_pose(q):
                d = np.linalg.norm(q - s)
                if best is None or d < best[1]:
                    best = (q, d)
        if best is None:
            continue
        q = best[0]
        cb = np.linalg.inv(fr3.fk(q, "fr3_link6") @ x) @ base_board @ corners.T
        uv = (K @ (cb[:3] / cb[2])).T[:, :2]
        if np.any(uv < p.image_margin) or np.any(uv[:, 0] > width - p.image_margin) \
                or np.any(uv[:, 1] > height - p.image_margin):
            continue
        accepted.append((q, cam))
        if len(accepted) >= max(15 * count, 450):
            break
    if not accepted:
        raise PipelineError("no reachable view of the board passed the checks; "
                            "is the board inside the workspace?")

    # Greedy max-min orientation diversity, starting from the most frontal view.
    chosen = [min(range(len(accepted)), key=lambda i: -accepted[i][1][:3, 2] @ -normal)]
    while len(chosen) < min(count, len(accepted)):
        def spread(i):
            return min(Rotation.from_matrix(accepted[i][1][:3, :3].T @ accepted[j][1][:3, :3]).magnitude()
                       for j in chosen)
        chosen.append(max((i for i in range(len(accepted)) if i not in chosen), key=spread))

    # Order from the look pose: always go to the nearest remaining view whose
    # joint-space path is clean (clearance + step limit). Views no clean path
    # reaches are dropped. (The original generate_poses.py dropped a view as
    # soon as the single nearest one failed, which kept only 3 of 30 here.)
    def segment_ok(qa, qb):
        if np.degrees(np.abs(qb - qa).max()) > p.max_step_deg:
            return False
        return all(fr3.frame_origins(qa + s * (qb - qa))[1:, 2].min() > table_z + p.clearance
                   for s in np.linspace(0, 1, 60))

    order, current, left = [], q0, list(chosen)
    while left:
        ranked = sorted(left, key=lambda i: np.abs(accepted[i][0] - current).max())
        nxt = next((i for i in ranked if segment_ok(current, accepted[i][0])), None)
        if nxt is None:
            break
        order.append(nxt)
        left.remove(nxt)
        current = accepted[nxt][0]
    if not order:
        raise PipelineError("no planned view is reachable from the look pose with clean paths; "
                            "raise plan.max_step_deg or plan.max_roll_deg")
    path_min, prev, max_step = np.inf, q0, 0.0
    for i in order:
        q = accepted[i][0]
        for s in np.linspace(0, 1, 60):
            path_min = min(path_min, fr3.frame_origins(prev + s * (q - prev))[1:, 2].min() - table_z)
        max_step = max(max_step, np.degrees(np.abs(q - prev).max()))
        prev = q

    # The capture tool moves to READY first; READY -> look pose is the path
    # already used to reach the look pose, so the list starts there.
    poses = [q0] + [accepted[i][0] for i in order]
    tilts = [math.degrees(math.acos(np.clip(-accepted[i][1][:3, 2] @ normal, -1, 1))) for i in order]
    dists = [np.linalg.norm(accepted[i][1][:3, 3] - centre) for i in order]
    report = [f"selected {len(poses)} poses (including the look pose) from {len(accepted)} candidates",
              f"view tilt {min(tilts):.0f}-{max(tilts):.0f} deg, distance {min(dists):.2f}-{max(dists):.2f} m",
              f"lowest link origin above the board top along all paths: {path_min * 1000:.0f} mm",
              f"largest single-joint move between consecutive poses: {max_step:.0f} deg"]
    return poses, report


# ======================================================================= steps
def _confirm(ctx: Context, message: str) -> None:
    ros.check_robot_ready(ctx.cfg)
    if not getattr(ctx.args, "yes", False):
        input(f"{message}\nKeep the e-stop at hand. Press Enter to start, Ctrl-C to abort... ")


def _cp(ctx: Context, path) -> str:
    return ros.container_path(ctx.cfg, path)


def _settings(ctx: Context) -> dict:
    return ctx.cfg["eye_in_hand"]


def _capture_tool(ctx: Context, poses: Path, out: Path) -> None:
    e = _settings(ctx)
    ros.ros2_run(ctx.cfg, "fr3_calibration", "run_calibration_capture",
                 "--reuse", _cp(ctx, poses),
                 "--cameras", _cp(ctx, ctx.cfg.path(e["cameras_config"])),
                 "--config", _cp(ctx, ctx.cfg.path(ctx.cfg["robot"]["config"])),
                 "--out", _cp(ctx, out), "--speed-scale", e["speed_scale"], "--no-detect",
                 "--timeout-scale", e.get("timeout_scale", 1.0),
                 own=out.parent, interactive=True)


def _calibration(ctx: Context) -> Calibration:
    return Calibration.load(ctx.session.require(CALIBRATION, "run `pipeline calibrate` first"))


def _load_views(directory: Path, max_mad: float) -> list[dict]:
    views = []
    for view in sorted(directory.iterdir()):
        fit_path = view / "board_fit.json"
        if not fit_path.exists():
            continue
        fit = json.loads(fit_path.read_text())
        if fit.get("mad_px", 0) > max_mad:
            print(f"  {view.name}: rim mad {fit['mad_px']:.2f} px > {max_mad}, excluded")
            continue
        q = np.asarray(json.loads((view / "joints.json").read_text())["measured_joints"], float)
        views.append(dict(name=view.name, q=q, cam_board=np.asarray(fit["T_cam_board"]),
                          mad=fit.get("mad_px")))
    return views


def make_steps(stage: str) -> list[Step]:
    calibrating = stage == "calibrate"

    def camera_model(ctx):
        """(intrinsics, rim bias) used for fitting in this stage."""
        if calibrating:
            return None, None      # factory intrinsics from the capture, no rim bias
        cal = _calibration(ctx)
        return cal.intrinsics, cal.rim_bias

    def look(ctx: Context, out: Path) -> None:
        e = _settings(ctx)
        if getattr(ctx.args, "here", False):
            print("  --here: using the current arm pose")
        else:
            look_pose = ctx.cfg.path(e["look_pose"])
            _confirm(ctx, f"The arm moves to READY, then to the look pose in {look_pose}.")
            _capture_tool(ctx, look_pose, out / "move")
        ros.run(ctx.cfg, ["python3", _cp(ctx, PIPELINE_DIR / "ros_nodes" / "grab_azure_rgbd.py"),
                          _cp(ctx, out / "view"), "--ns", e["namespace"]], own=out)

    def seed_fit(ctx: Context, out: Path) -> None:
        source = ctx.step_dir(stage, "look") / "view"
        view = out / "view"
        shutil.copytree(source, view)
        factory, info = rimfit.read_camera_info(view)
        intr, bias = camera_model(ctx)
        if intr is None:
            intr = factory
        rimfit.write_camera_info(view, intr, info["width"], info["height"], info["frame_id"])
        rimfit.load(ctx.cfg)
        print("  searching the whole image for the board (depth-seeded, ~1 min)")
        fit = rimfit.fit_depth_seeded(view, intr, bias)
        max_mad = _settings(ctx)["max_mad_px"]
        if fit is None or fit["mad_px"] > max_mad:
            detail = "no fit" if fit is None else f"rim mad {fit['mad_px']:.2f} px > {max_mad}"
            raise PipelineError(f"seed fit failed ({detail}); check that the whole board is in "
                                f"{view / 'color.png'}, or move the arm and use --here")
        T = np.asarray(fit["T_cam_board"])
        rimfit.overlay(view, intr, T, out / "overlay.jpg")
        q0 = np.asarray(json.loads((view / "joints.json").read_text())["measured_joints"])
        if calibrating:
            x = np.asarray(json.loads(ctx.cfg.path(_settings(ctx)["prior_handeye"]).read_text())["matrix"])
            base_board = fr3.fk(q0, "fr3_link6") @ x @ T
        else:
            base_board = _calibration(ctx).base_board(q0, T)
        (out / "seed.json").write_text(json.dumps({"T_base_board": base_board.tolist()}, indent=2))
        print(f"  rim mad {fit['mad_px']:.3f} px, camera-board distance {np.linalg.norm(T[:3, 3]):.3f} m")
        print(f"  board centre in base (approx.): {np.round(base_board[:3, 3] * 1000, 1)} mm")
        print(f"  check the overlay: {out / 'overlay.jpg'}")

    def plan(ctx: Context, out: Path) -> None:
        e = _settings(ctx)
        seed = ctx.step_dir(stage, "seed_fit")
        view = seed / "view"
        intr, info = rimfit.read_camera_info(view)
        q0 = np.asarray(json.loads((view / "joints.json").read_text())["measured_joints"])
        base_board = np.asarray(json.loads((seed / "seed.json").read_text())["T_base_board"])
        if calibrating:
            x = np.asarray(json.loads(ctx.cfg.path(e["prior_handeye"]).read_text())["matrix"])
        else:
            x = _calibration(ctx).x
        params = SimpleNamespace(**e["plan"])
        poses, report = plan_views(q0, base_board, x, rimfit.k_matrix(intr), info["width"],
                                   info["height"], e[stage]["views"], params)
        (out / "poses.yaml").write_text(yaml.safe_dump({
            "joint_names": fr3.JOINT_NAMES,
            "poses": [{"joints": [round(float(v), 6) for v in q]} for q in poses]}))
        for line in report:
            print("  " + line)

    def capture(ctx: Context, out: Path) -> None:
        poses = ctx.step_dir(stage, "plan") / "poses.yaml"
        count = len(yaml.safe_load(poses.read_text())["poses"])
        _confirm(ctx, f"The arm moves to READY, then through {count} planned views of the board "
                      f"at speed scale {_settings(ctx)['speed_scale']}.")
        try:
            _capture_tool(ctx, poses, out / "capture")
        except PipelineError:
            # The tool exits non-zero when any single pose fails (an unreachable
            # view, a PTP stall); the views it did record are still good, and the
            # fit step enforces the minimum number of usable views.
            recorded = out / "capture" / "poses.yaml"
            if not recorded.exists():
                raise
            got = len(load_yaml(recorded).get("poses") or [])
            if got < (10 if calibrating else 3):
                raise
            print(f"  {got}/{count} views recorded; the failed ones are skipped")

    def fit(ctx: Context, out: Path) -> None:
        e = _settings(ctx)
        camera = e["camera"]
        capture_dir = ctx.step_dir(stage, "capture") / "capture"
        meta = load_yaml(capture_dir / "meta.yaml")["cameras"][camera]
        intr, bias = camera_model(ctx)
        if intr is None:
            intr = rimfit.intrinsics_vector(meta["K"], meta["D"])
        base_board = np.asarray(json.loads(
            (ctx.step_dir(stage, "seed_fit") / "seed.json").read_text())["T_base_board"])
        if calibrating:
            x = np.asarray(json.loads(ctx.cfg.path(e["prior_handeye"]).read_text())["matrix"])
            predict = lambda q: np.linalg.inv(fr3.fk(q, "fr3_link6") @ x) @ base_board  # noqa: E731
        else:
            cal = _calibration(ctx)
            predict = lambda q: cal.cam_board(q, base_board)  # noqa: E731
        rimfit.load(ctx.cfg)
        good = 0
        poses = load_yaml(capture_dir / "poses.yaml")["poses"]
        for pose in poses:
            index = int(pose["index"])
            image = capture_dir / "images" / f"pose_{index:03d}_{camera}.png"
            if not image.exists():
                print(f"  pose {index}: no image, skipped")
                continue
            view = out / f"b{index:03d}"
            view.mkdir()
            shutil.copyfile(image, view / "color.png")
            rimfit.write_camera_info(view, intr, meta["width"], meta["height"], meta["frame_id"])
            q = np.asarray(pose["measured_joints"], float)
            (view / "joints.json").write_text(json.dumps(
                {"joint_names": pose["joint_names"], "measured_joints": q.tolist()}, indent=2))
            result = rimfit.fit_from_pose(view, intr, predict(q), bias)
            if result is None:
                print(f"  {view.name}: FIT FAILED")
                continue
            rimfit.overlay(view, intr, result["T_cam_board"], view / "overlay.jpg")
            ok = result["mad_px"] <= e["max_mad_px"]
            good += ok
            print(f"  {view.name}: rim mad {result['mad_px']:.3f} px  "
                  f"dist {np.linalg.norm(np.asarray(result['T_cam_board'])[:3, 3]):.3f} m"
                  + ("" if ok else "  (above max_mad_px, will be excluded)"))
        minimum = 10 if calibrating else 3
        if good < minimum:
            raise PipelineError(f"only {good} views fitted well (need {minimum}); "
                                f"look at the overlays under {out}")
        print(f"  {good}/{len(poses)} views usable")

    def bundle(ctx: Context, out: Path) -> None:
        views = sorted(d for d in ctx.step_dir(stage, "fit").iterdir() if (d / "board_fit.json").exists())
        intr0, _ = rimfit.read_camera_info(views[0])
        rimfit.load(ctx.cfg)
        print(f"  {len(views)} views, refining {', '.join(_settings(ctx)['calibrate']['bundle_free'])} "
              "(several minutes)")
        result = rimfit.bundle(views, intr0, _settings(ctx)["calibrate"]["bundle_free"], out)
        (out / "intrinsics.json").write_text(json.dumps(result, indent=2))

    def solve_calibration(ctx: Context, out: Path) -> None:
        e = _settings(ctx)
        bundled = ctx.step_dir(stage, "bundle")
        views = _load_views(bundled, e["max_mad_px"])
        terms = model_terms(e["calibrate"]["model"])
        prior = ctx.cfg.path(e["prior_handeye"])
        x_seed = np.asarray(json.loads(prior.read_text())["matrix"])
        n, fit_chain, error, unpack = chain_model(terms)
        base0 = fr3.fk(views[0]["q"], "fr3_link6") @ x_seed @ views[0]["cam_board"]
        print(f"  fitting {'+'.join(terms) or 'X only'} on {len(views)} views, then leave-one-out")
        v = fit_chain(views, np.r_[fr3.to_vec(x_seed), fr3.to_vec(base0), np.zeros(n - 12)])
        loo = np.array([error(views[i], fit_chain(views[:i] + views[i + 1:], v)) for i in range(len(views))])
        fit_err = np.array([error(w, v) for w in views])
        x, base_board, dq, s, c = unpack(v)
        intr = json.loads((bundled / "intrinsics.json").read_text())
        calibration = {
            "setup": "eye_in_hand",
            "parent_frame": "fr3_link6",
            "child_frame": "rgb_camera_link",
            "model": list(terms),
            "T_link6_camera": x.tolist(),
            "joint_offsets_rad": {name: float(dq[k]) for k, name in enumerate(fr3.JOINT_NAMES)},
            "camera_translation_scale": float(s),
            "mount_sag_rad_per_g": np.asarray(c).tolist(),
            "intrinsics": dict(zip(intr["names"], intr["refined"])),
            "rim_bias": intr["rim_bias"],
            "T_base_board_during_calibration": base_board.tolist(),
            "quality": {"views": len(views), "fit_median_mm": float(np.median(fit_err)),
                        "loo_median_mm": float(np.median(loo)),
                        "loo_p90_mm": float(np.percentile(loo, 90)),
                        "loo_max_mm": float(loo.max()),
                        "per_view_loo_mm": dict(zip([w["name"] for w in views], loo.round(4).tolist()))},
            "source": {"prior_handeye": str(prior), "prior_sha256": sha256(prior)},
        }
        text = json.dumps(calibration, indent=2)
        (out / CALIBRATION).write_text(text)
        (ctx.session.path / CALIBRATION).write_text(text)
        print(f"  board-corner error: fit median {np.median(fit_err):.3f} mm; leave-one-out "
              f"median {np.median(loo):.3f} mm, p90 {np.percentile(loo, 90):.3f} mm, max {loo.max():.3f} mm")
        print("  joint offsets j2-5 (deg): " + str(np.round(np.degrees(dq[1:5]), 4)))
        print(f"  wrote {ctx.session.path / CALIBRATION}")

        # The solve fits the board pose too: export it, so check/probe can run
        # straight after calibrate while the board has not moved.
        cal = Calibration(calibration)
        spread = {w["name"]: corner_distance_mm(cal.base_board(w["q"], w["cam_board"]), base_board)
                  for w in views}
        values = np.array(list(spread.values()))
        board = {
            "schema_version": 1, "units": "m", "base_frame": "base", "board_frame": "board",
            "board_surface_z_m": SURFACE_Z,
            "T_base_board": base_board.tolist(),
            "frame_note": ("true kinematic frame (calibrated joint offsets applied); "
                           "probe targets are corrected per hole by `pipeline check/probe`"),
            "validity_note": ("board pose fitted by `calibrate`; valid only while the board stays "
                              "where it was during calibration. Run `pipeline locate` after it moves."),
            "quality": {"views_used": len(views), "views_fitted": len(views),
                        "spread_median_mm": float(np.median(values)),
                        "spread_max_mm": float(values.max()),
                        "per_view_spread_mm": {k: round(v, 4) for k, v in spread.items()},
                        "calibration_loo_median_mm": float(np.median(loo))},
            "source": {"setup": "eye_in_hand", "stage": "calibrate",
                       "calibration": str(ctx.session.path / CALIBRATION),
                       "calibration_sha256": sha256(ctx.session.path / CALIBRATION)},
        }
        text = json.dumps(board, indent=2)
        (out / BOARD_POSE).write_text(text)
        (ctx.session.path / BOARD_POSE).write_text(text)
        centre = base_board @ np.array([0, 0, SURFACE_Z, 1])
        print(f"  board top centre in base: {np.round(centre[:3] * 1000, 2)} mm")
        print(f"  wrote {ctx.session.path / BOARD_POSE} (valid while the board does not move; "
              "`pipeline check/probe` can run now)")

    def solve_locate(ctx: Context, out: Path) -> None:
        e = _settings(ctx)
        cal = _calibration(ctx)
        views = _load_views(ctx.step_dir(stage, "fit"), e["max_mad_px"])
        if len(views) < 2:
            raise PipelineError(f"only {len(views)} usable views; rerun `pipeline locate --from capture`")
        estimates ={w["name"]: cal.base_board(w["q"], w["cam_board"]) for w in views}
        limit = e["locate"]["max_view_spread_mm"]
        kept = dict(estimates)
        while True:
            mean = mean_transform(list(kept.values()))
            spread = {k: corner_distance_mm(T, mean) for k, T in kept.items()}
            worst = max(spread, key=spread.get)
            if spread[worst] <= limit or len(kept) <= 2:
                break
            print(f"  {worst}: {spread[worst]:.3f} mm from the mean, dropped")
            del kept[worst]
        if len(kept) < 2 or spread[worst] > limit:
            raise PipelineError(f"views disagree by more than {limit} mm; recalibrate or check overlays")
        values = np.array(list(spread.values()))
        board = {
            "schema_version": 1, "units": "m", "base_frame": "base", "board_frame": "board",
            "board_surface_z_m": SURFACE_Z,
            "T_base_board": mean.tolist(),
            "frame_note": ("true kinematic frame (calibrated joint offsets applied); "
                           "probe targets are corrected per hole by `pipeline check/probe`"),
            "quality": {"views_used": len(kept), "views_fitted": len(views),
                        "spread_median_mm": float(np.median(values)),
                        "spread_max_mm": float(values.max()),
                        "per_view_spread_mm": {k: round(v, 4) for k, v in spread.items()},
                        "calibration_loo_median_mm": cal.data["quality"]["loo_median_mm"]},
            "source": {"setup": "eye_in_hand", "stage": "locate",
                       "calibration": str(ctx.session.path / CALIBRATION),
                       "calibration_sha256": sha256(ctx.session.path / CALIBRATION)},
        }
        text = json.dumps(board, indent=2)
        (out / BOARD_POSE).write_text(text)
        (ctx.session.path / BOARD_POSE).write_text(text)
        centre = mean @ np.array([0, 0, SURFACE_Z, 1])
        print(f"  board top centre in base: {np.round(centre[:3] * 1000, 2)} mm")
        print(f"  {len(kept)} views, spread median {np.median(values):.3f} mm, max {values.max():.3f} mm")
        print(f"  wrote {ctx.session.path / BOARD_POSE}")

    steps = [
        Step("look", "move to the look pose and grab one RGB-D frame", look),
        Step("seed_fit", "find the board in that frame", seed_fit),
        Step("plan", "plan views of the board", plan),
        Step("capture", "drive through the views and record images + joints", capture),
        Step("fit", "fit the board in every view", fit),
    ]
    if calibrating:
        steps += [Step("bundle", "refine the camera intrinsics over all views", bundle),
                  Step("solve", "solve camera mount, joint offsets and quality", solve_calibration)]
    else:
        steps += [Step("solve", "combine the views into one board pose", solve_locate)]
    return steps


def mean_transform(transforms):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_matrix([t[:3, :3] for t in transforms]).mean().as_matrix()
    T[:3, 3] = np.mean([t[:3, 3] for t in transforms], axis=0)
    return T


def corner_distance_mm(a, b):
    return float(np.linalg.norm((a @ POINTS)[:3] - (b @ POINTS)[:3], axis=0).max() * 1e3)


# ============================================================== hole targets
def hole_target(session, cfg, hole: dict, name: str) -> Path:
    """Board pose for probe_board, corrected for the joint offsets at this hole.

    probe_board / MoveIt use the nominal FR3 kinematics, but the real arm sits
    at FK(q + dq). For the tool pose over the hole we solve the true joint
    angles q_true, then express the board in the frame the nominal model
    believes in at that configuration:

        C = FK(q_true - dq) @ inv(FK(q_true))     T_nominal_board = C @ T_base_board

    C depends (weakly) on the configuration, hence one target file per hole.
    """
    cal = Calibration.load(session.require(CALIBRATION, "run `pipeline calibrate` first"))
    board_file = session.require(BOARD_POSE, "run `pipeline locate` first")
    board = np.asarray(json.loads(board_file.read_text())["T_base_board"])
    probe = load_yaml(cfg.path(cfg["robot"]["manipulation_config"]))["probe"]
    tip_offset = np.asarray(probe["pen_tip_offset_tcp_m"], float)
    rotation = board[:3, :3] @ Rotation.from_euler("xyz", probe["tcp_orientation_board_rpy"]).as_matrix()
    tip = board @ np.r_[hole["x_m"], hole["y_m"], SURFACE_Z, 1]
    target = np.eye(4)
    target[:3, :3] = rotation
    target[:3, 3] = tip[:3] - rotation @ tip_offset

    # IK seed: the look pose of the stage that produced board_pose.json.
    stage = json.loads(board_file.read_text()).get("source", {}).get("stage", "locate")
    seed_joints = session.step_dir(stage, "seed_fit") / "view" / "joints.json"
    seed = np.asarray(json.loads(seed_joints.read_text())["measured_joints"])
    # A local IK from one seed can stall; try the look pose, READY, and both
    # with joint 7 turned, and keep the solution nearest the look pose.
    seeds = [np.r_[base[:6], base[6] + turn] for base in (seed, fr3.READY)
             for turn in (0.0, -math.pi / 2, math.pi / 2, -math.pi, math.pi)]
    solutions = [fr3.ik(target, np.clip(s, fr3.LOWER, fr3.UPPER)) for s in seeds]
    good = [q for q, r in solutions if r < 1e-4]
    q_true, residual = (min(good, key=lambda q: np.abs(q - seed).max()), 0.0) if good \
        else min(solutions, key=lambda t: t[1])
    if residual > 1e-4:
        raise PipelineError(f"no IK solution over {name} (residual {residual:.2e})")
    correction = fr3.fk(q_true - cal.dq, "fr3_hand_tcp") @ np.linalg.inv(fr3.fk(q_true, "fr3_hand_tcp"))
    shift = (correction @ np.r_[tip[:3], 1])[:3] - tip[:3]
    print(f"joint-offset correction at {name}: tip shift {np.round(shift * 1e3, 2)} mm, "
          f"rotation {math.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude()):.3f} deg")

    out = session.path / "targets" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "schema_version": 1, "units": "m", "base_frame": "base", "board_frame": "board",
        "board_surface_z_m": SURFACE_Z,
        "T_base_board": (correction @ board).tolist(),
        "source": {"setup": "eye_in_hand", "hole": name,
                   "method": "board pose in the nominal-FK frame, joint-offset corrected at this hole",
                   "board_pose": str(board_file), "board_pose_sha256": sha256(board_file),
                   "T_base_board_true_frame": board.tolist(),
                   "q_command_rad": (q_true - cal.dq).tolist()}}, indent=2))
    return out
