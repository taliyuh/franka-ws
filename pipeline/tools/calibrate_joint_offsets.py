#!/usr/bin/env python3
"""Calibrate the Panda's joint encoder offsets from an AprilGrid lying on the table.

The Azure Kinect on panda_link6 photographs the board from ~35 planned views
per board placement (3 placements), then the operator touches 6 marked grid
points per placement with the pen held in the round V-slot, guiding the arm by
hand. One bundle adjustment fits

    joint offsets dq2..dq6 (true angle = measured + dq), X = T_link6_camera,
    one board pose Y_p per placement

to the pixel positions of the grid corners and to the touched points:

    camera:  u = project_K,D( inv(FK_link6(q + dq) @ X) @ Y_p @ P )        sigma 0.5 px
    touch:   FK_tcp(q + dq) @ tip  -  Y_p @ P_touched                        sigma 0.5 mm

What can be seen (camera on link6, board pose and X both free):
  dq1  absorbed by the board pose (a turn of the whole arm about base z looks
       like a turned board); harmless for camera-guided work.
  dq2-5 from the camera views.
  dq6  absorbed by X in the camera views; only the touches see it, and it moves
       the tip ~0.26 mm per mrad.
  dq7  does not move the camera, and with the pen 16.5 mm off the joint-7 axis
       hardly moves the tip; held at 0.
The intrinsics come fixed from pipeline/tools/calibrate_azure_intrinsics.py,
the pen tip fixed from pipeline/config/grasps.yaml (round slot).

Validation is leave-one-placement-out: offsets and X are fitted without one
placement, its board pose is then located from its own camera views only, and
its touches are predicted. The p95 of that tip error is the PASS/FAIL number.

Usage (host, from the workspace; robot.launch and azure_4k.launch running in
the franka_ros container, the pen gripped: ./pipeline/pipeline grasp round):

    ./pipeline/tools/joint_offsets run --intrinsics camera_calibration/azure_<stamp>/intrinsics.json
    ./pipeline/tools/joint_offsets status               # what the newest run has done, what is next
    ./pipeline/tools/joint_offsets run --resume         # continue the newest run where it stopped
    ./pipeline/tools/joint_offsets run --resume kinematic_calibration/<stamp> --from 2:capture
                                                        # redo placement 2 from its capture on
    ./pipeline/tools/joint_offsets plan                 # plan from the current pose, no motion
    ./pipeline/tools/joint_offsets solve [kinematic_calibration/<stamp>] \\
        --compare sessions/current/calibration.json

Every stage (look, plan, capture, touch per placement) is saved as it goes;
capture saves each view and touch each point, so after a stop (the robot lost
the connection, a reflex, Ctrl-C) `run --resume` goes on from the next missing
view or point. When the robot stops during a stage, the program explains the
cause, offers Franka's error recovery and retries without quitting.

Output in kinematic_calibration/<stamp>/: calibration.json (the pipeline's
schema: copy it to sessions/<name>/calibration.json to use it), joint_offsets.json,
report.txt, and per placement the look view, the captured views and the touches.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))
sys.path.insert(0, str(TOOLS))

from board_pipeline import eye_in_hand, grasp, panda, ros  # noqa: E402
from board_pipeline.config import WORKSPACE, Config, PipelineError, load_yaml  # noqa: E402
import calibrate_azure_intrinsics as grid  # noqa: E402

OUT_ROOT = WORKSPACE / "kinematic_calibration"
GRAB = WORKSPACE / "pipeline" / "ros_nodes" / "grab_azure_rgbd.py"
TOUCH = WORKSPACE / "pipeline" / "ros_nodes" / "touch_grid.py"
FRANKA_STATES = "/franka_state_controller/franka_states"

SIGMA_PX = 0.5
SIGMA_TOUCH_M = 0.0005
CAMERA_JOINTS = [1, 2, 3, 4]          # 0-based: joints 2-5, seen by the link6 camera
TOUCH_JOINTS = [1, 2, 3, 4, 5]        # + joint 6, seen only through the touches
# Placement name and the look pose's joint-1 turn (rad) that puts the camera over it.
PLACEMENTS = [("centre", 0.0), ("left", 0.45), ("right", -0.45)]
HAND_POSES = ["pen vertical",
              "pen tilted ~25 deg towards the robot base",
              "pen tilted ~25 deg away from the robot",
              "pen tilted ~25 deg to your left",
              "pen tilted ~25 deg to your right",
              "pen vertical, hand turned ~90 deg about the pen (joint 7)"]
PASS_P95_MM = 2.0


# ===================================================================== camera
class Camera:
    def __init__(self, K, D, size, source, data=None):
        self.K = np.asarray(K, float)
        self.D = np.asarray(D, float)
        self.size = (int(size[0]), int(size[1]))
        self.source = str(source)
        self.data = data or {}

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if "refined" in data:                                   # calibrate_azure_intrinsics
            v = dict(zip(data["names"], data["refined"]))
            K = [[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1]]
            D = [v[n] for n in grid.NAMES[4:]]
            return cls(K, D, data.get("image_size", (3840, 2160)), path, data)
        if "k" in data:                                         # camera_info.json
            K, D, size = grid.factory_from(data)
            return cls(K, D, size, path, data)
        raise PipelineError(f"{path}: neither an intrinsics.json nor a camera_info.json")

    def project(self, pts, K=None):
        uv, _ = cv2.projectPoints(np.asarray(pts, float).reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                  self.K if K is None else K, self.D)
        return uv.reshape(-1, 2)

    def with_pinhole(self, values):
        """A copy with fx, fy, cx, cy replaced (refined in the bundle adjustment)."""
        fx, fy, cx, cy = values
        return Camera([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], self.D, self.size,
                      f"{self.source} (fx fy cx cy refined in the joint fit)", self.data)

    def pinhole(self):
        return np.array([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]])

    def intrinsics_dict(self):
        return dict(zip(grid.NAMES, [self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]]
                        + [float(x) for x in self.D]))


def newest_intrinsics():
    found = sorted((WORKSPACE / "camera_calibration").glob("azure_*/intrinsics.json"))
    return found[-1] if found else None


# ================================================================= kinematics
def full_dq(free, values, base=None):
    dq = np.zeros(7) if base is None else np.array(base, float)
    dq[list(free)] = values
    return dq


def link6(q, dq):
    return panda.fk(np.asarray(q, float) + dq, "panda_link6")


def pen_tip(q, dq, tip_offset):
    t = panda.fk(np.asarray(q, float) + dq, "panda_hand_tcp")
    return t[:3, :3] @ tip_offset + t[:3, 3]


def pen_tip_offset(cfg):
    """TCP -> pen tip in TCP axes: the pen sits in the round peg's V-slot (grasps.yaml)."""
    hole = {"shape": "round", "x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0}
    return grasp.hole_tool(cfg, hole, "round_c3mm", turn=0.0).tip_offset


def prior_chain(cfg):
    """(X, dq, source): the current session calibration if any, else the prior hand-eye."""
    current = cfg.sessions_dir / "current" / "calibration.json"
    if current.exists():
        data = json.loads(current.read_text())
        dq = np.array([data["joint_offsets_rad"][n] for n in panda.JOINT_NAMES])
        return np.asarray(data["T_link6_camera"]), dq, str(current)
    path = cfg.path(cfg["eye_in_hand"]["prior_handeye"])
    return np.asarray(json.loads(path.read_text())["matrix"]), np.zeros(7), str(path)


def grid_pose_in_base(view, cam, q, X, dq):
    """Y = FK_link6(q + dq) @ X @ T_cam_grid from one view (PnP)."""
    rvec, tvec = grid.pnp(view, cam.K, cam.D)
    cam_grid = np.eye(4)
    cam_grid[:3, :3] = cv2.Rodrigues(rvec)[0]
    cam_grid[:3, 3] = tvec.ravel()
    return link6(q, dq) @ X @ cam_grid


# ======================================================================= data
class Obs:
    """One captured view: placement p, measured joints q, grid corners (pixels and board)."""

    def __init__(self, p, q, view, name):
        self.p, self.q, self.name = p, np.asarray(q, float), name
        self.img, self.obj, self.ids = view.img.astype(float), view.obj.astype(float), view.ids


class Touch:
    def __init__(self, p, q, point, label):
        self.p, self.q, self.point, self.label = p, np.asarray(q, float), np.asarray(point, float), label


def detect_file(path, board, min_tags=8):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    tags = grid.detect_view(gray, board)
    return grid.View(Path(path).stem, tags, board) if len(tags) >= min_tags else None


def load_placement(pdir, p, board, cache=True):
    """Captured views (with a detection cache) and touches of one placement directory."""
    obs, touches = [], []
    capture = pdir / "capture"
    poses_file = capture / "poses.yaml"
    if poses_file.exists():
        cache_file = pdir / "detections.json"
        cached = json.loads(cache_file.read_text()) if cache and cache_file.exists() else {}
        changed = False
        for entry in load_yaml(poses_file)["poses"]:
            image = capture / "images" / f"pose_{int(entry['index']):03d}_azure.png"
            key = image.name
            if key not in cached:
                view = detect_file(image, board)
                cached[key] = None if view is None else {
                    "ids": view.ids.tolist(), "img": view.img.tolist()}
                changed = True
            if cached[key] is None:
                continue
            ids = np.array(cached[key]["ids"], int)
            tags = {int(i): np.array(c, np.float32)
                    for i, c in zip(ids, np.array(cached[key]["img"]).reshape(-1, 4, 2))}
            obs.append(Obs(p, entry["measured_joints"], grid.View(image.stem, tags, board),
                           f"p{p + 1}/{image.stem}"))
        if changed:
            cache_file.write_text(json.dumps(cached))
    spec_file, raw_file = pdir / "touch_spec.json", pdir / "touches.json"
    if spec_file.exists() and raw_file.exists():
        spec = json.loads(spec_file.read_text())
        for rec in json.loads(raw_file.read_text()):
            target = spec["targets"][rec["target"]]
            touches.append(Touch(p, rec["q"], target["point_board_m"], f"p{p + 1}/{target['label']}"))
    return obs, touches


# ====================================================================== model
class Model:
    """Parameter vector [dq of `free` joints, fx fy cx cy (if fit_k), X (6, if fit_x),
    Y of each placement in `places` (6 each)]."""

    def __init__(self, free, places, base_dq=None, fit_x=True, fit_k=False):
        self.free, self.places = list(free), list(places)
        self.base_dq = np.zeros(7) if base_dq is None else np.asarray(base_dq, float)
        self.fit_x, self.fit_k = fit_x, fit_k
        self.k = len(self.free)
        self.kk = 4 if fit_k else 0
        self.nx = 6 if fit_x else 0
        self.shared = self.k + self.kk + self.nx
        self.index = {p: self.shared + 6 * i for i, p in enumerate(self.places)}
        self.size = self.shared + 6 * len(self.places)

    def pack(self, dq, X, Ys, cam):
        v = [np.asarray(dq, float)[self.free]]
        if self.fit_k:
            v.append(cam.pinhole())
        if self.fit_x:
            v.append(panda.to_vec(X))
        v += [panda.to_vec(Ys[p]) for p in self.places]
        return np.concatenate(v)

    def unpack(self, v, X_fixed=None):
        """(dq, X, {p: Y}, K or None)."""
        dq = full_dq(self.free, v[:self.k], self.base_dq)
        K = None
        if self.fit_k:
            fx, fy, cx, cy = v[self.k:self.k + 4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        a = self.k + self.kk
        X = panda.from_vec(v[a:a + 6]) if self.fit_x else X_fixed
        Ys = {p: panda.from_vec(v[i:i + 6]) for p, i in self.index.items()}
        return dq, X, Ys, K


def residuals(v, model, obs, touches, cam, tip_offset, X_fixed=None):
    dq, X, Ys, K = model.unpack(v, X_fixed)
    out = []
    for o in obs:
        T = np.linalg.inv(link6(o.q, dq) @ X) @ Ys[o.p]
        out.append(((cam.project(o.obj @ T[:3, :3].T + T[:3, 3], K) - o.img) / SIGMA_PX).ravel())
    for t in touches:
        out.append((pen_tip(t.q, dq, tip_offset) - (Ys[t.p] @ np.r_[t.point, 1.0])[:3]) / SIGMA_TOUCH_M)
    return np.concatenate(out) if out else np.zeros(0)


def sparsity(model, obs, touches):
    rows = sum(2 * len(o.img) for o in obs) + 3 * len(touches)
    s = lil_matrix((rows, model.size), dtype=int)
    r = 0
    for o in obs:
        n = 2 * len(o.img)
        s[r:r + n, :model.shared] = 1
        s[r:r + n, model.index[o.p]:model.index[o.p] + 6] = 1
        r += n
    for t in touches:
        s[r:r + 3, :model.k] = 1
        s[r:r + 3, model.index[t.p]:model.index[t.p] + 6] = 1
        r += 3
    return s


def fit(obs, touches, cam, tip_offset, free, init, base_dq=None, fit_x=True, X_fixed=None, fit_k=False):
    """Robust bundle adjustment; init = (dq, X, {p: Y}). Returns a namespace (cam = the
    camera with the refined pinhole values when fit_k)."""
    places = sorted({o.p for o in obs} | {t.p for t in touches})
    model = Model(free, places, base_dq, fit_x, fit_k)
    dq0, X0, Ys0 = init
    v0 = model.pack(dq0, X0, Ys0, cam)
    res = least_squares(residuals, v0, args=(model, obs, touches, cam, tip_offset, X_fixed),
                        method="trf", loss="soft_l1", f_scale=3.0, x_scale="jac",
                        jac_sparsity=sparsity(model, obs, touches), max_nfev=300)
    dq, X, Ys, K = model.unpack(res.x, X_fixed)
    fitted_cam = cam if K is None else cam.with_pinhole(res.x[model.k:model.k + 4])
    J = res.jac.toarray() if hasattr(res.jac, "toarray") else res.jac
    dof = max(len(res.fun) - len(res.x), 1)
    s2 = float(res.fun @ res.fun) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov = np.full((len(res.x),) * 2, np.nan)
    n_px = sum(2 * len(o.img) for o in obs)
    px = res.fun[:n_px] * SIGMA_PX
    return SimpleNamespace(dq=dq, X=X, Ys=Ys, model=model, cov=cov, J=J, res=res, cam=fitted_cam,
                           rms_px=float(np.sqrt(np.mean(px ** 2))) if n_px else float("nan"))


def locate(obs_p, cam, dq, X, Y0):
    """The board pose of one placement from its own views, offsets and X held fixed."""
    p = obs_p[0].p
    out = fit(obs_p, [], cam, None, [], (dq, X, {p: Y0}), base_dq=dq, fit_x=False, X_fixed=X)
    return out.Ys[p], out.rms_px


def board_spread_mm(obs_p, cam, dq, X, Y, span):
    """Per view: largest distance (mm) between where that view alone puts the grid's
    four outer corners and centre (PnP + chain) and where the placement's pose puts them."""
    pts = np.array([[x, y, 0, 1] for x in (0.0, span) for y in (0.0, span)]
                   + [[span / 2, span / 2, 0, 1]]).T
    out = []
    for o in obs_p:
        rvec, tvec = cv2.solvePnP(o.obj, o.img, cam.K, cam.D, flags=cv2.SOLVEPNP_IPPE)[1:]
        rvec, tvec = cv2.solvePnP(o.obj, o.img, cam.K, cam.D, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)[1:]
        cg = np.eye(4)
        cg[:3, :3] = cv2.Rodrigues(rvec)[0]
        cg[:3, 3] = tvec.ravel()
        T = link6(o.q, dq) @ X @ cg
        out.append(float(np.max(np.linalg.norm((T @ pts)[:3] - (Y @ pts)[:3], axis=0))) * 1e3)
    return out


def tip_errors_mm(touches_p, dq, Y, tip_offset):
    return [float(np.linalg.norm(pen_tip(t.q, dq, tip_offset) - (Y @ np.r_[t.point, 1.0])[:3])) * 1e3
            for t in touches_p]


def init_chain(obs, cam, X, dq):
    """Board pose per placement from the view with the most tags."""
    Ys = {}
    for p in sorted({o.p for o in obs}):
        best = max((o for o in obs if o.p == p), key=lambda o: len(o.img))
        view = SimpleNamespace(obj=best.obj.astype(np.float32), img=best.img.astype(np.float32))
        Ys[p] = grid_pose_in_base(view, cam, best.q, X, dq)
    return Ys


def stats(values):
    v = np.asarray(values, float)
    if not len(v):
        return {}
    return {"n": int(len(v)), "mean": float(v.mean()), "rms": float(np.sqrt(np.mean(v ** 2))),
            "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


# ================================================================== solving
def variants(compare_dq, touches):
    """(name, free joints, fixed offsets, use touches, refine fx fy cx cy)."""
    zero = np.zeros(7)
    out = [("nominal", [], zero, False, False)]
    if compare_dq is not None:
        out.append(("compare", [], compare_dq, False, False))
    out.append(("camera J2-5", CAMERA_JOINTS, zero, False, False))
    if touches:
        out += [("full J2-6", TOUCH_JOINTS, zero, True, False), ("full J2-6 +K", TOUCH_JOINTS, zero, True, True)]
    else:
        out.append(("camera J2-5 +K", CAMERA_JOINTS, zero, False, True))
    return out


def clean_obs(obs, touches, cam, tip_offset, init, min_tags=8, log=print):
    """Drop tags whose corners a preliminary full fit cannot explain (mis-refined corners)."""
    f = fit(obs, touches, cam, tip_offset, TOUCH_JOINTS if touches else CAMERA_JOINTS, init)
    errs = []
    for o in obs:
        T = np.linalg.inv(link6(o.q, f.dq) @ f.X) @ f.Ys[o.p]
        errs.append(np.linalg.norm(cam.project(o.obj @ T[:3, :3].T + T[:3, 3]) - o.img, axis=1))
    limit = max(6.0 * float(np.median(np.concatenate(errs))), 3.0)
    kept, dropped_tags, dropped_views = [], 0, 0
    for o, e in zip(obs, errs):
        good = (e.reshape(-1, 4) <= limit).all(1)
        dropped_tags += int((~good).sum())
        if good.sum() < min_tags:
            dropped_views += 1
            continue
        o.img = o.img.reshape(-1, 4, 2)[good].reshape(-1, 2)
        o.obj = o.obj.reshape(-1, 4, 3)[good].reshape(-1, 3)
        o.ids = o.ids[good]
        kept.append(o)
    if dropped_tags or dropped_views:
        log(f"outliers: dropped {dropped_tags} tags with a corner > {limit:.1f} px off and "
            f"{dropped_views} views left with < {min_tags} tags")
    return kept


def solve(run_dir, cam, cfg, compare=None, log=print):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    board = grid.AprilGrid(*meta["grid"], meta["tag_size_m"], meta["spacing"])
    tip_offset = np.asarray(meta["tip_offset_tcp_m"], float)
    X0, dq0 = np.asarray(meta["prior_X"]), np.asarray(meta["prior_dq"])
    obs, touches = [], []
    for p in range(len(PLACEMENTS)):
        pdir = run_dir / f"placement_{p + 1}"
        if pdir.exists():
            o, t = load_placement(pdir, p, board)
            obs += o
            touches += t
            log(f"placement {p + 1}: {len(o)} views, {len(t)} touches")
    if len(obs) < 10:
        raise PipelineError(f"only {len(obs)} usable views in {run_dir}")
    places = sorted({o.p for o in obs})
    span = board.cols * board.pitch - board.tag * board.spacing
    compare_dq = None
    if compare:
        data = json.loads(Path(compare).read_text())
        compare_dq = np.array([data["joint_offsets_rad"][n] for n in panda.JOINT_NAMES])
    Ys0 = init_chain(obs, cam, X0, dq0)
    init = (dq0 * 0, X0, Ys0)
    obs = clean_obs(obs, touches, cam, tip_offset, init, log=log)

    # ---- leave-one-placement-out
    log("\nleave-one-placement-out validation ...")
    table, settings = {}, {}
    for name, free, base, use_touch, fit_k in variants(compare_dq, touches):
        settings[name] = (free, base, use_touch, fit_k)
        tips, spreads, pxs = [], [], []
        for h in places:
            train_obs = [o for o in obs if o.p != h]
            train_t = [t for t in touches if t.p != h] if use_touch else []
            if not train_obs:
                continue
            f = fit(train_obs, train_t, cam, tip_offset, free, (base, X0, Ys0), base_dq=base, fit_k=fit_k)
            held = [o for o in obs if o.p == h]
            Yh, px = locate(held, f.cam, f.dq, f.X, Ys0[h])
            pxs.append(px)
            spreads += board_spread_mm(held, f.cam, f.dq, f.X, Yh, span)
            tips += tip_errors_mm([t for t in touches if t.p == h], f.dq, Yh, tip_offset)
        table[name] = {"tip_mm": stats(tips), "board_spread_mm": stats(spreads),
                       "heldout_px_rms": float(np.mean(pxs)) if pxs else float("nan")}
        log(f"  {name:12s} held-out tip " + (
            f"mean {table[name]['tip_mm']['mean']:.2f}  p95 {table[name]['tip_mm']['p95']:.2f}  "
            f"max {table[name]['tip_mm']['max']:.2f} mm" if tips else "-  (no touches)")
            + f" | board spread median {table[name]['board_spread_mm'].get('median', float('nan')):.2f} mm")

    # ---- choose: refined intrinsics only if they cut the held-out error by 5 % or more
    base_name = "full J2-6" if touches else "camera J2-5"
    k_name = base_name + " +K"

    def metric(name):
        row = table[name]
        return row["tip_mm"]["p95"] if row["tip_mm"] else row["board_spread_mm"].get("median", np.inf)
    chosen = k_name if len(places) > 1 and metric(k_name) < 0.95 * metric(base_name) else base_name
    free, _, _, fit_k = settings[chosen]

    # ---- final fit on everything
    final = fit(obs, touches, cam, tip_offset, free, init, fit_k=fit_k)
    k = final.model.k
    sigma = np.sqrt(np.clip(np.diag(final.cov)[:k], 0, None))
    corr = final.cov[:k, :k] / np.outer(sigma, sigma)
    # Observability of the offsets: X and the board poses projected out (Schur complement).
    J = final.J
    Jd, Jo = J[:, :k], J[:, k:]
    S = Jd.T @ Jd - Jd.T @ Jo @ np.linalg.solve(Jo.T @ Jo + 1e-12 * np.eye(Jo.shape[1]), Jo.T @ Jd)
    sv = np.sqrt(np.clip(np.linalg.eigvalsh(S), 0, None))[::-1]
    fit_tips = [e for p in places
                for e in tip_errors_mm([t for t in touches if t.p == p], final.dq, final.Ys[p], tip_offset)]
    verdict = None
    if table[chosen]["tip_mm"]:
        verdict = "PASS" if table[chosen]["tip_mm"]["p95"] < PASS_P95_MM else "FAIL"
    result = SimpleNamespace(final=final, sigma=sigma, corr=corr, sv=sv, table=table, chosen=chosen,
                             verdict=verdict, fit_tips=fit_tips, obs=obs, touches=touches,
                             board=board, meta=meta, cam=final.cam, cam_file=cam)
    write_outputs(run_dir, result, cfg, compare)
    return result


def write_outputs(run_dir, r, cfg, compare):
    f, names = r.final, panda.JOINT_NAMES
    free = f.model.free
    lines = [f"joint offset calibration  {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"data: {len(r.obs)} views, {len(r.touches)} touches, intrinsics {r.cam.source}",
             f"final fit: {f.rms_px:.3f} px RMS"
             + (f", touch residuals median {np.median(r.fit_tips):.2f} mm, max {max(r.fit_tips):.2f} mm"
                if r.fit_tips else ""),
             "",
             "offsets (true angle = measured + offset):"]
    for i, j in enumerate(free):
        lines.append(f"  {names[j]}: {f.dq[j] * 1e3:+8.3f} mrad  ({math.degrees(f.dq[j]):+7.4f} deg)"
                     f"  +- {r.sigma[i] * 1e3:.3f} mrad")
    held = [n for n in names if names.index(n) not in free]
    lines.append(f"  held at 0 (not observable here): {', '.join(held)}")
    if f.model.fit_k:
        k = f.model.k
        ks = np.sqrt(np.clip(np.diag(f.cov)[k:k + 4], 0, None))
        lines.append("intrinsics refined in the joint fit (they cut the held-out error):")
        for n, a, b, sd in zip(("fx", "fy", "cx", "cy"), r.cam_file.pinhole(), r.cam.pinhole(), ks):
            lines.append(f"  {n}: {a:9.2f} -> {b:9.2f} px  +- {sd:.2f}")
    strong = [(names[free[a]], names[free[b]], r.corr[a, b]) for a in range(len(free))
              for b in range(a + 1, len(free)) if abs(r.corr[a, b]) > 0.9]
    if strong:
        lines.append("  strongly correlated: " + ", ".join(f"{a}/{b} {c:+.2f}" for a, b, c in strong))
    lines.append("  observability (offsets, X and boards projected out): singular values "
                 + " ".join(f"{s:.3g}" for s in r.sv) + f", condition {r.sv[0] / max(r.sv[-1], 1e-12):.1f}")
    lines += ["", "leave-one-placement-out (fit without a placement, locate it from its own views,",
              "predict its touches):",
              f"  {'model':12s} {'tip mean':>9s} {'rms':>6s} {'p95':>6s} {'max':>6s}   "
              f"{'board spread median/p90 mm':>27s}  {'px':>5s}"]
    for name, row in r.table.items():
        t, b = row["tip_mm"], row["board_spread_mm"]
        tip = (f"{t['mean']:9.2f} {t['rms']:6.2f} {t['p95']:6.2f} {t['max']:6.2f}" if t else f"{'-':>9s}" + " " * 21)
        mark = "  <- result" if name == r.chosen else ""
        lines.append(f"  {name:12s} {tip}   {b.get('median', float('nan')):12.2f} / {b.get('p90', float('nan')):5.2f}"
                     f"       {row['heldout_px_rms']:5.2f}{mark}")
    if compare:
        lines.append(f"  (compare = joint offsets of {compare}, X and boards refitted)")
    if r.verdict:
        lines += ["", f"tip error p95 {r.table[r.chosen]['tip_mm']['p95']:.2f} mm on held-out placements: "
                  f"{r.verdict} (target < {PASS_P95_MM:.0f} mm)"]
    else:
        lines += ["", "no touches: joint 6 and the tip were not measured, so the tip accuracy is unknown"]
    out_cal = run_dir / "calibration.json"
    lines += ["", f"to use it in the pipeline: cp {out_cal} sessions/<session>/calibration.json"]
    report = "\n".join(lines)
    print("\n" + report)
    (run_dir / "report.txt").write_text(report + "\n")

    chosen = r.table[r.chosen]
    camera_data = r.cam.data
    calibration = {
        "setup": "eye_in_hand", "parent_frame": "panda_link6", "child_frame": "rgb_camera_link",
        "model": ["off"],
        "T_link6_camera": f.X.tolist(),
        "joint_offsets_rad": {n: float(f.dq[i]) for i, n in enumerate(names)},
        "camera_translation_scale": 1.0,
        "mount_sag_rad_per_g": [0.0, 0.0, 0.0],
        "intrinsics": r.cam.intrinsics_dict(),
        "rim_bias": camera_data.get("rim_bias", [0.0] * 8),
        "T_base_grid_per_placement": {str(p + 1): Y.tolist() for p, Y in f.Ys.items()},
        "quality": {"views": len(r.obs), "touches": len(r.touches), "fit_px_rms": f.rms_px,
                    "loo_median_mm": chosen["board_spread_mm"].get("median"),
                    "loo_p90_mm": chosen["board_spread_mm"].get("p90"),
                    "loo_max_mm": chosen["board_spread_mm"].get("max"),
                    "heldout_tip_mm": chosen["tip_mm"], "verdict": r.verdict,
                    "note": "loo_* = leave-one-placement-out board spread of the AprilGrid"},
        "source": {"tool": "pipeline/tools/calibrate_joint_offsets.py", "run": str(run_dir),
                   "intrinsics": r.cam.source, "pen_tip_offset_tcp_m": r.meta["tip_offset_tcp_m"]},
    }
    (run_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (run_dir / "joint_offsets.json").write_text(json.dumps({
        "offsets_rad": {names[j]: float(f.dq[j]) for j in range(7)},
        "sigma_rad": {names[j]: float(r.sigma[i]) for i, j in enumerate(free)},
        "validation": r.table, "chosen": r.chosen, "verdict": r.verdict}, indent=2))


# =================================================================== planning
def board_points(board):
    """16 grid points (4x4 over the tag corners) and the 36 tag centres, board frame."""
    span = board.cols * board.pitch - board.tag * board.spacing
    g = np.linspace(0, span, 4)
    pts = np.array([[x, y, 0.0] for x in g for y in g])
    centres = board.object_points(np.arange(board.n_tags)).mean(1)
    return pts, centres


def ik_link6(target, seed, iterations=40):
    """panda_link6 IK by damped least squares with the analytic Jacobian, joint 7 kept.

    The pipeline's panda.ik_link6 (scipy, numerical Jacobian) takes ~50 ms; this
    takes ~1 ms, which the planner needs for thousands of candidates.
    Returns (q, error) with the error measured like panda.ik_link6 (10 x m, rad).
    """
    q = np.array(seed, float)
    for k in range(iterations):
        t, frames = np.eye(4), []
        for i in range(6):
            t = t @ panda.dh(panda.A[i], panda.D[i], panda.ALPHA[i], q[i])
            frames.append(t)
        ep = target[:3, 3] - t[:3, 3]
        er = cv2.Rodrigues(target[:3, :3] @ t[:3, :3].T)[0].ravel()
        e = max(np.abs(ep).max() * 10, np.abs(er).max())
        if e < 1e-6 or (k >= 12 and e > 1e-2):     # converged, or not converging (unreachable)
            break
        J = np.empty((6, 6))
        for i, f in enumerate(frames):         # MDH: joint i turns about z of frame i
            J[:3, i] = np.cross(f[:3, 2], t[:3, 3] - f[:3, 3])
            J[3:, i] = f[:3, 2]
        step = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), np.r_[ep, er])
        q[:6] += step * min(1.0, 0.3 / max(np.abs(step).max(), 1e-12))
    t = panda.fk(q, "panda_link6")
    err = max(np.abs(target[:3, 3] - t[:3, 3]).max() * 10,
              np.abs(cv2.Rodrigues(target[:3, :3] @ t[:3, :3].T)[0]).max())
    return q, float(err)


def plan_placement(q_look, Y, X, dq, cam, board, tip_offset, count, prior_info, settings, rng):
    """Joint poses viewing the board, chosen for information on dq2..dq5.

    Candidates: camera poses around the board (distance, tilt, roll), solved by IK
    and kept when inside the joint limits (and a box around the look pose), clear
    of the table (link origins, pen tip) and of the upper arm, with the grid in view.
    The path starts at the look pose; every next view is the most informative
    candidate (D-optimal on dq2-5 with X and the board pose projected out, on top
    of prior_info from earlier placements) that a clean joint-space segment
    reaches (step limit, clearance checked along it), via READY if none does.
    Returns (poses, information on [dq2..5, X] after this placement, report lines).
    """
    s = settings
    centre = (Y @ np.r_[board.center, 1.0])[:3]
    normal = Y[:3, 2]
    if normal[2] < 0.5:
        raise PipelineError("the grid's +z does not point up; is the board flat and face up?")
    table_z = centre[2]
    pts16, centres = board_points(board)
    lo, hi = panda.LOWER + s.limit_margin, panda.UPPER - s.limit_margin
    box = math.radians(s.box_deg)
    lo[:6] = np.maximum(lo[:6], q_look[:6] - box)
    hi[:6] = np.minimum(hi[:6], q_look[:6] + box)
    w, h = cam.size
    zero = np.zeros(7)

    def clear(q):
        if panda.frame_origins(q)[1:, 2].min() < table_z + s.clearance:
            return False
        return pen_tip(q, zero, tip_offset)[2] > table_z + s.tip_clearance

    def sees(cam_pose):
        pc = (np.linalg.inv(cam_pose) @ Y @ np.c_[centres, np.ones(len(centres))].T)[:3].T
        if np.any(pc[:, 2] <= 0.05):
            return False
        uv = cam.project(pc)
        inside = ((uv[:, 0] > s.image_margin) & (uv[:, 0] < w - s.image_margin)
                  & (uv[:, 1] > s.image_margin) & (uv[:, 1] < h - s.image_margin))
        return inside.mean() >= s.min_visible

    def segment_ok(qa, qb):
        if np.degrees(np.abs(qb - qa).max()) > s.max_step_deg:
            return False
        if not all(clear(qa + t * (qb - qa)) for t in np.linspace(0, 1, 30)):
            return False
        return all(eye_in_hand.camera_arm_clearance(qa + t * (qb - qa), X) > s.arm_clearance
                   for t in np.linspace(0, 1, 6))

    e1 = np.cross(normal, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    look_x = (link6(q_look, dq) @ X)[:3, 0]
    ready = np.r_[panda.READY[:6], q_look[6]]
    accepted, tries = [], 0
    while tries < s.tries and len(accepted) < s.candidates:
        tries += 1
        tilt = math.radians(s.max_tilt_deg) * math.sqrt(rng.uniform())
        az = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(s.min_dist, s.max_dist)
        direction = math.cos(tilt) * normal + math.sin(tilt) * (math.cos(az) * e1 + math.sin(az) * e2)
        pos = centre + dist * direction
        aim = centre + Y[:3, :3] @ np.r_[rng.uniform(-0.05, 0.05, 2), 0]
        z = aim - pos
        z /= np.linalg.norm(z)
        ref = look_x - z * (look_x @ z)
        if np.linalg.norm(ref) < 1e-6:
            continue
        ref /= np.linalg.norm(ref)
        roll = math.radians(rng.uniform(-s.max_roll_deg, s.max_roll_deg))
        xa = math.cos(roll) * ref + math.sin(roll) * np.cross(z, ref)
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = np.c_[xa, np.cross(z, xa), z]
        cam_pose[:3, 3] = pos
        if not sees(cam_pose):
            continue
        wrist = (cam_pose @ np.linalg.inv(X))[:3, 3]
        if np.linalg.norm(wrist - [0, 0, panda.D[0]]) > 0.80:     # beyond the arm's reach
            continue
        # IK of the true arm, then the commanded (measured) angles: true = measured + dq.
        target = cam_pose @ np.linalg.inv(X)
        seeds = [q_look]
        if accepted:     # the candidate whose camera is closest: stays on the same IK branch
            seeds.append(min(accepted, key=lambda a: np.linalg.norm(a[1][:3, 3] - pos))[0] + np.r_[dq[:6], 0])
        for seed in seeds:
            q, err = ik_link6(target, seed)
            q = q - np.r_[dq[:6], 0]
            if err < 1e-4 and np.all(q >= lo) and np.all(q <= hi) and clear(q) \
                    and eye_in_hand.camera_arm_clearance(q, X) > s.arm_clearance:
                accepted.append((q, cam_pose))
                break
    if len(accepted) < 3:
        raise PipelineError(f"only {len(accepted)} reachable views of the board passed the checks "
                            f"({tries} tries); is the board inside the workspace?")

    # Information per candidate on [dq2..5, X, Y] from 16 grid points (numerical Jacobian).
    def info(q):
        def proj(v):
            d = full_dq(CAMERA_JOINTS, v[:4], dq)
            T = np.linalg.inv(link6(q, d) @ panda.from_vec(v[4:10])) @ panda.from_vec(v[10:16])
            return cam.project(pts16 @ T[:3, :3].T + T[:3, 3]).ravel() / SIGMA_PX
        v0 = np.r_[dq[CAMERA_JOINTS], panda.to_vec(X), panda.to_vec(Y)]
        f0 = proj(v0)
        J = np.stack([(proj(v0 + 1e-5 * np.eye(16)[i]) - f0) / 1e-5 for i in range(16)], 1)
        return J.T @ J

    infos = [info(q) for q, _ in accepted]
    prior = np.zeros((10, 10)) if prior_info is None else np.asarray(prior_info)

    def schur(A, keep):
        a, b, c = A[:keep, :keep], A[:keep, keep:], A[keep:, keep:]
        return a - b @ np.linalg.solve(c + 1e-9 * np.eye(len(c)), b.T)

    def score(sum16):
        d = schur(prior + schur(sum16, 10), 4)
        sign, logdet = np.linalg.slogdet(d + 1e-9 * np.eye(4))
        return logdet if sign > 0 else -np.inf

    # Greedy informative path from the look pose over clean segments.
    total = info(q_look)
    poses, current, left, waypoints, used = [q_look], q_look, set(range(len(accepted))), 0, []
    while len(used) < count - 1 and left:
        ranked = sorted(left, key=lambda i: -score(total + infos[i]))
        nxt = next((i for i in ranked if segment_ok(current, accepted[i][0])), None)
        via_ready = False
        if nxt is None and segment_ok(current, ready):
            nxt = next((i for i in ranked if segment_ok(ready, accepted[i][0])), None)
            via_ready = nxt is not None
        if nxt is None:
            break
        if via_ready:
            poses.append(ready)
            waypoints += 1
        poses.append(accepted[nxt][0])
        used.append(nxt)
        left.discard(nxt)
        total = total + infos[nxt]
        current = accepted[nxt][0]
    qs = np.array(poses)
    dists = [np.linalg.norm(accepted[i][1][:3, 3] - centre) for i in used]
    tilts = [math.degrees(math.acos(np.clip(-accepted[i][1][:3, 2] @ normal, -1, 1))) for i in used]
    before, after = score(np.zeros((16, 16))), score(total)
    report = [f"{len(used) + 1} views ({waypoints} READY waypoints) from {len(accepted)} "
              f"reachable candidates ({tries} tries)",
              f"distance {min(dists):.2f}-{max(dists):.2f} m, tilt {min(tilts):.0f}-{max(tilts):.0f} deg",
              "joint ranges (deg): " + "  ".join(f"J{j + 1} {np.degrees(np.ptp(qs[:, j])):.0f}"
                                                for j in range(6)),
              f"dq2-5 information log det: {after:.1f}" + (f" (before this placement {before:.1f})"
                                                           if prior_info is not None else "")]
    if len(used) < count - 1:
        report.append(f"only {len(used) + 1} of {count} views could be chained with clean paths")
    return poses, prior + schur(total, 10), report


# ================================================================== robot run
#
# A run is a sequence of stages per placement -- look, plan, capture, touch --
# then solve. Each stage leaves its result in files, and the state of a run is
# read back from those files, so `run --resume` continues exactly where it
# stopped: inside capture at the next missing view (capture_views.py saves
# every view at once), inside touch at the next missing point. When the robot
# stops (lost connection, reflex), the stage offers to recover and retry
# without leaving the program.

STAGES = ["look", "plan", "capture", "touch"]
CAPTURE = WORKSPACE / "pipeline" / "ros_nodes" / "capture_views.py"
RT_TUNE = WORKSPACE / "docker" / "robot_rt_tune.sh"
COMMUNICATION_HINT = (
    "communication_constraints_violation: this PC missed the robot's 1 kHz control loop "
    "(franka_control answered too late).\n"
    + (f"  Run the real-time tuning on the HOST (again after every start of robot.launch):\n"
       f"    sudo {RT_TUNE.relative_to(WORKSPACE)}\n"
       f"  check it with: {RT_TUNE.relative_to(WORKSPACE)} --status\n" if RT_TUNE.exists() else
       "  Most often the CPU is clocking down: set the governor to performance (as root on the host),\n"
       "    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor\n")
    + "  and keep heavy programs closed while the arm moves; the 4K camera stream also loads the CPU.")


class Stopped(Exception):
    """The operator stopped the run; everything done so far is saved."""


def ctx_for(cfg, yes=False):
    return SimpleNamespace(cfg=cfg, args=SimpleNamespace(yes=yes))


def write_poses(path, poses):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump({"joint_names": panda.JOINT_NAMES,
                                   "poses": [{"joints": [round(float(v), 6) for v in q]} for q in poses]},
                                  sort_keys=False))
    tmp.replace(path)


def grab(cfg, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    ros.run(cfg, ["python3", ros.container_path(cfg, GRAB), ros.container_path(cfg, out),
                  "--ns", "/azure_camera"], own=out.parent)


def placement_dir(run_dir, p):
    return Path(run_dir) / f"placement_{p + 1}"


def look_result(pdir, board):
    """(q_look, view) of the accepted look of a placement, or None."""
    color, joints, done = pdir / "look" / "view" / "color.png", pdir / "look" / "view" / "joints.json", \
        pdir / "look" / "look.json"
    if not (color.exists() and joints.exists()):
        return None
    if not done.exists():          # runs from before look.json: accept a look view that saw the board
        view = detect_file(color, board)
        if view is None or len(view.ids) < 30:
            return None
        q = json.loads(joints.read_text())["measured_joints"]
        try:
            done.write_text(json.dumps({"q": q, "tags": int(len(view.ids))}))
        except OSError:
            pass
        return np.asarray(q, float), view
    return np.asarray(json.loads(done.read_text())["q"], float), detect_file(color, board)


def capture_progress(pdir):
    """(views planned, views captured, kept as is)."""
    planned = len(load_yaml(pdir / "views.yaml")["poses"]) if (pdir / "views.yaml").exists() else 0
    poses = pdir / "capture" / "poses.yaml"
    captured = len({int(e["index"]) for e in load_yaml(poses).get("poses", [])}) if poses.exists() else 0
    return planned, captured, (pdir / "capture" / "KEEP").exists()


def touch_progress(pdir):
    """(targets, touches recorded, rest skipped)."""
    spec, raw = pdir / "touch_spec.json", pdir / "touches.json"
    targets = len(json.loads(spec.read_text())["targets"]) if spec.exists() else 0
    recorded = len({r["target"] for r in json.loads(raw.read_text())}) if raw.exists() else 0
    return targets, recorded, (pdir / "TOUCH_SKIPPED").exists()


def stage_state(run_dir, p, meta, board):
    """{stage: (done, text)} for one placement, read from its files."""
    pdir = placement_dir(run_dir, p)
    look = look_result(pdir, board) is not None
    plan = (pdir / "views.yaml").exists()
    planned, captured, keep = capture_progress(pdir)
    targets, recorded, skipped = touch_progress(pdir)
    state = {"look": (look, "ok" if look else "-"),
             "plan": (plan, f"{planned} views" if plan else "-"),
             "capture": (plan and (captured >= planned or keep),
                         f"{captured}/{planned}" + (" (kept)" if keep and captured < planned else "")
                         if plan else "-")}
    if meta["touches"] <= 0:
        state["touch"] = (True, "off")
    else:
        state["touch"] = (targets > 0 and (recorded >= targets or skipped),
                          f"{recorded}/{targets or meta['touches']}" + (" (rest skipped)" if skipped else ""))
    return state


def next_stage(run_dir, meta, board):
    for p in range(len(PLACEMENTS)):
        state = stage_state(run_dir, p, meta, board)
        for stage in STAGES:
            if not state[stage][0]:
                return p, stage
    return None


def print_status(run_dir, meta, board):
    print(f"\nrun {run_dir}")
    for p, (name, _) in enumerate(PLACEMENTS):
        state = stage_state(run_dir, p, meta, board)
        print(f"  placement {p + 1} ({name:6s})  " + "   ".join(
            f"{stage} {'[x]' if state[stage][0] else '[ ]'} {state[stage][1]:<11s}" for stage in STAGES))
    nxt = next_stage(run_dir, meta, board)
    print(f"  next: placement {nxt[0] + 1}, {nxt[1]}" if nxt else "  next: solve (all data collected)")
    return nxt


def reset_from(run_dir, p, stage):
    """Move the files of `stage` and the later stages of placement p into an old_<stamp> folder."""
    pdir = placement_dir(run_dir, p)
    files = {"look": ["look", "look_pose.yaml"], "plan": ["views.yaml", "plan_info.json"],
             "capture": ["capture", "detections.json"],
             "touch": ["touch_spec.json", "touches.json", "touch_targets.png", "TOUCH_SKIPPED"]}
    backup = pdir / f"old_{datetime.datetime.now().strftime('%H%M%S')}"
    moved = []
    for s in STAGES[STAGES.index(stage):]:
        for name in files[s]:
            if (pdir / name).exists():
                backup.mkdir(parents=True, exist_ok=True)
                (pdir / name).rename(backup / name)
                moved.append(name)
    print(f"placement {p + 1}: redoing from {stage}; moved {', '.join(moved) or 'nothing'}"
          + (f" to {backup}" if moved else ""))


def recovery_hint(cfg):
    try:
        mode, errors = ros.robot_mode(cfg), ros.motion_errors(cfg)
    except PipelineError as error:
        return f"  {error}"
    if mode is None:
        return ("  no robot state arrives: is robot.launch still running? Restart it in the container:\n"
                "    roslaunch panda_calibration robot.launch robot_ip:=172.16.0.2")
    name = ros.ROBOT_MODES.get(mode, str(mode))
    lines = [f"  robot mode {name}" + (f"; last motion errors: {', '.join(errors)}" if errors else "")]
    if "communication_constraints_violation" in errors:
        lines.append("  " + COMMUNICATION_HINT)
    if name != "MOVE":
        lines.append("  r = run Franka's error recovery now (the arm does not move), then try again")
    return "\n".join(lines)


def robot_step(cfg, label, action, keep=None):
    """Run a robot/ROS action; when it fails, explain, offer recovery and retry it.

    The actions resume from their saved progress, so a retry continues where it stopped.
    keep: an optional callable for 'k = keep what was done so far and go on'."""
    while True:
        try:
            return action()
        except PipelineError as error:
            print(f"\n*** {label} stopped: {error}")
            print(recovery_hint(cfg))
            options = "Enter = try again (continues from the last saved step), r = error recovery + try again"
            if keep:
                options += ", k = keep what is done and go on"
            answer = input(options + ", q = quit (resume later) > ").strip().lower()
            if answer == "q":
                raise Stopped()
            if answer == "k" and keep:
                keep()
                return None
            if answer == "r":
                try:
                    ros.run(cfg, ["rostopic", "pub", "-1", "/franka_control/error_recovery/goal",
                                  "franka_msgs/ErrorRecoveryActionGoal", "{}"])
                except PipelineError as error2:
                    print(f"error recovery failed: {error2}")


def look_pose(cfg, p):
    q = np.asarray(load_yaml(cfg.path(cfg["eye_in_hand"]["look_pose"]))["poses"][0]["joints"], float)
    q[0] += PLACEMENTS[p][1]
    return q


def stage_look(cfg, run_dir, p, meta, board, args):
    name, turn = PLACEMENTS[p]
    pdir = placement_dir(run_dir, p)
    look_yaml = pdir / "look_pose.yaml"
    write_poses(look_yaml, [look_pose(cfg, p)])

    def move():
        eye_in_hand._confirm(ctx_for(cfg, args.yes),
                             f"\n=== placement {p + 1}/{len(PLACEMENTS)} ({name}): look ===\n"
                             f"The arm moves to READY, then to the look pose (joint 1 turned "
                             f"{math.degrees(turn):+.0f} deg).")
        eye_in_hand._capture_tool(ctx_for(cfg, args.yes), look_yaml, pdir / "look" / "move")
    robot_step(cfg, "the move to the look pose", move)
    view_dir = pdir / "look" / "view"
    while True:
        input(f"Put the AprilGrid flat, face up, under the camera ({name} placement), "
              "then press Enter to check > ")
        robot_step(cfg, "the camera image", lambda: grab(cfg, view_dir))
        view = detect_file(view_dir / "color.png", board)
        n = 0 if view is None else len(view.ids)
        if n >= 30:
            q = json.loads((view_dir / "joints.json").read_text())["measured_joints"]
            (pdir / "look" / "look.json").write_text(json.dumps({"q": q, "tags": int(n)}))
            print(f"  OK: {n} of 36 tags seen")
            return
        print(f"  only {n} of 36 tags seen; move the board so all of it is in view, keep it flat")


def rough_board(pdir, board, cam, X, dq):
    q_look, view = look_result(pdir, board)
    return q_look, view, grid_pose_in_base(SimpleNamespace(obj=view.obj, img=view.img), cam, q_look, X, dq)


def stage_plan(cfg, run_dir, p, meta, board, cam, args):
    pdir = placement_dir(run_dir, p)
    X, dq = np.asarray(meta["prior_X"]), np.asarray(meta["prior_dq"])
    q_look, view, Y = rough_board(pdir, board, cam, X, dq)
    prior = None
    for k in range(p - 1, -1, -1):           # information of the placements planned before
        f = placement_dir(run_dir, k) / "plan_info.json"
        if f.exists():
            prior = np.asarray(json.loads(f.read_text())["info_after"])
            break
    print(f"\n=== placement {p + 1}: planning views ...")
    poses, info, report = plan_placement(q_look, Y, X, dq, cam, board, np.asarray(meta["tip_offset_tcp_m"]),
                                         meta["views"], prior, plan_settings(args),
                                         np.random.default_rng(20261119 + p))
    print("  " + "\n  ".join(report))
    (pdir / "plan_info.json").write_text(json.dumps({"info_after": info.tolist(), "report": report}))
    write_poses(pdir / "views.yaml", poses)       # last: its presence marks the stage done


def stage_capture(cfg, run_dir, p, meta, args):
    pdir = placement_dir(run_dir, p)
    planned, captured, _ = capture_progress(pdir)
    e = cfg["eye_in_hand"]
    cp = lambda path: ros.container_path(cfg, path)  # noqa: E731

    def go():
        eye_in_hand._confirm(ctx_for(cfg, args.yes),
                             f"\n=== placement {p + 1}: capture ({capture_progress(pdir)[1]}/{planned} views "
                             "saved) ===\nThe arm rejoins the planned path from where it stands and visits the "
                             f"missing views, speed scale {e['speed_scale']}.")
        ros.run(cfg, ["python3", cp(CAPTURE), cp(pdir / "views.yaml"), cp(pdir / "capture"),
                      "--speed-scale", str(e["speed_scale"]), "--timeout-scale", str(e.get("timeout_scale", 1.0)),
                      "--config", cp(cfg.path(cfg["robot"]["config"]))], own=pdir, interactive=True)

    def keep():
        (pdir / "capture").mkdir(parents=True, exist_ok=True)
        (pdir / "capture" / "KEEP").write_text("views kept as captured\n")
    robot_step(cfg, f"the capture of placement {p + 1}", go, keep=keep)


def choose_targets(view, board, count):
    """Grid corners to touch, spread over the board (farthest-point order from the centre)."""
    pts = view.obj[:, :2].astype(float)
    order = [int(np.argmin(np.linalg.norm(pts - board.center[:2], axis=1)))]
    while len(order) < min(count, len(pts)):
        d = np.min(np.linalg.norm(pts[:, None] - pts[order][None], axis=2), axis=1)
        order.append(int(np.argmax(d)))
    return order


def write_touch_spec(cfg, pdir, view, board, Y, tip_offset, count):
    idx = choose_targets(view, board, count)
    color = cv2.imread(str(pdir / "look" / "view" / "color.png"))
    scale = 1600.0 / color.shape[1]
    img = cv2.resize(color, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    targets = []
    for k, i in enumerate(idx):
        uv = view.img[i] * scale
        cv2.circle(img, (int(uv[0]), int(uv[1])), 18, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(img, str(k + 1), (int(uv[0]) + 20, int(uv[1]) - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 255), 3, cv2.LINE_AA)
        point = view.obj[i].astype(float)
        targets.append({"label": str(k + 1), "point_board_m": point.tolist(),
                        "target_base_m": (Y @ np.r_[point, 1.0])[:3].tolist(),
                        "instruction": HAND_POSES[k % len(HAND_POSES)]})
    cv2.putText(img, "touch the X where the tag corner meets the small square", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    picture = pdir / "touch_targets.png"
    cv2.imwrite(str(picture), img)
    spec = {"topic": FRANKA_STATES, "tip_offset_tcp_m": np.asarray(tip_offset).tolist(),
            "board_rotation": Y[:3, :3].tolist(), "image": ros.container_path(cfg, picture), "targets": targets}
    (pdir / "touch_spec.json").write_text(json.dumps(spec, indent=2))


def stage_touch(cfg, run_dir, p, meta, board, cam, args):
    pdir = placement_dir(run_dir, p)
    X, dq = np.asarray(meta["prior_X"]), np.asarray(meta["prior_dq"])
    if not (pdir / "touch_spec.json").exists():     # kept on resume: the touches refer to its targets
        _, view, Y = rough_board(pdir, board, cam, X, dq)
        write_touch_spec(cfg, pdir, view, board, Y, meta["tip_offset_tcp_m"], meta["touches"])
    picture = pdir / "touch_targets.png"
    try:
        subprocess.Popen(["xdg-open", str(picture)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
    targets, recorded, _ = touch_progress(pdir)
    print(f"\n=== placement {p + 1}: touches ({recorded}/{targets} saved) ===\nTouch targets: {picture} "
          "(opened in an image viewer); the numbers mark the points to touch.")
    cp = lambda path: ros.container_path(cfg, path)  # noqa: E731

    def skip():
        (pdir / "TOUCH_SKIPPED").write_text("remaining touches skipped\n")
    while True:
        robot_step(cfg, f"the touches of placement {p + 1}",
                   lambda: ros.run(cfg, ["python3", cp(TOUCH), cp(pdir / "touch_spec.json"),
                                         cp(pdir / "touches.json")], own=pdir, interactive=True), keep=skip)
        targets, recorded, skipped = touch_progress(pdir)
        if recorded >= targets or skipped:
            return
        if input(f"{recorded} of {targets} points touched. Enter = touch the rest now, "
                 "s = skip them > ").strip().lower() == "s":
            skip()
            return


def plan_settings(args):
    return SimpleNamespace(min_dist=0.32, max_dist=0.70, max_tilt_deg=50.0, max_roll_deg=60.0,
                           limit_margin=0.15, box_deg=getattr(args, "box_deg", 60.0), clearance=0.125,
                           tip_clearance=0.05, arm_clearance=0.15, image_margin=60.0, min_visible=0.8,
                           max_step_deg=45.0, tries=4000, candidates=400)


def find_run(value):
    if value in (None, "latest"):
        runs = sorted(d for d in OUT_ROOT.glob("20*") if (d / "run.json").exists())
        if not runs:
            raise PipelineError(f"no runs in {OUT_ROOT}")
        return runs[-1]
    run_dir = Path(value).resolve()
    if not (run_dir / "run.json").exists():
        raise PipelineError(f"{run_dir} has no run.json")
    return run_dir


def new_run(cfg, args):
    intr = Path(args.intrinsics) if args.intrinsics else newest_intrinsics()
    if intr is None:
        raise PipelineError("no --intrinsics given and no camera_calibration/azure_*/intrinsics.json; "
                            "run pipeline/tools/calibrate_azure_intrinsics.py first")
    X, dq, source = prior_chain(cfg)
    run_dir = OUT_ROOT / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True)
    if not (OUT_ROOT / ".gitignore").exists():
        (OUT_ROOT / ".gitignore").write_text("# 4K calibration images; not for git\n*\n")
    meta = {"intrinsics": str(Path(intr).resolve()), "grid": [6, 6], "tag_size_m": args.tag_size,
            "spacing": 0.3, "tip_offset_tcp_m": pen_tip_offset(cfg).tolist(), "prior_X": X.tolist(),
            "prior_dq": dq.tolist(), "prior_source": source, "views": args.views, "touches": args.touches}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def cmd_run(args):
    cfg = Config.template()
    run_dir = find_run(args.resume) if args.resume else new_run(cfg, args)
    meta = json.loads((run_dir / "run.json").read_text())
    ros.reclaim(cfg, run_dir)          # files a crashed container step left owned by root
    board = grid.AprilGrid(*meta["grid"], meta["tag_size_m"], meta["spacing"])
    cam = Camera.load(meta["intrinsics"])
    if args.from_stage:
        try:
            number, stage = args.from_stage.split(":")
            p = int(number) - 1
            assert 0 <= p < len(PLACEMENTS) and stage in STAGES
        except (ValueError, AssertionError):
            raise PipelineError(f"--from {args.from_stage}: use PLACEMENT:STAGE, e.g. 2:capture "
                                f"(stages: {', '.join(STAGES)})")
        reset_from(run_dir, p, stage)
    print(f"intrinsics: {meta['intrinsics']}\nprior chain: {meta['prior_source']}\n"
          f"pen tip (TCP axes, mm): {np.round(np.asarray(meta['tip_offset_tcp_m']) * 1e3, 2)}")
    print_status(run_dir, meta, board)
    try:
        while True:
            nxt = next_stage(run_dir, meta, board)
            if nxt is None:
                break
            p, stage = nxt
            if stage == "look":
                stage_look(cfg, run_dir, p, meta, board, args)
            elif stage == "plan":
                stage_plan(cfg, run_dir, p, meta, board, cam, args)
            elif stage == "capture":
                stage_capture(cfg, run_dir, p, meta, args)
            else:
                stage_touch(cfg, run_dir, p, meta, board, cam, args)
    except (Stopped, KeyboardInterrupt):
        print_status(run_dir, meta, board)
        print(f"\nstopped; everything above is saved. Continue with:\n"
              f"  ./pipeline/tools/joint_offsets run --resume {run_dir}")
        return
    print_status(run_dir, meta, board)
    solve(run_dir, cam, cfg, args.compare)


def cmd_status(args):
    run_dir = find_run(args.run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    board = grid.AprilGrid(*meta["grid"], meta["tag_size_m"], meta["spacing"])
    if print_status(run_dir, meta, board):
        print(f"\ncontinue with: ./pipeline/tools/joint_offsets run --resume {run_dir}")
    else:
        print(f"\nall collected; solve with: ./pipeline/tools/joint_offsets solve {run_dir}")


def cmd_plan(args):
    """Plan from the current arm pose and a fresh image; nothing moves."""
    cfg = Config.template()
    intr = Path(args.intrinsics) if args.intrinsics else newest_intrinsics()
    if intr is None:
        raise PipelineError("give --intrinsics")
    cam = Camera.load(intr)
    board = grid.AprilGrid(6, 6, args.tag_size, 0.3)
    X, dq, source = prior_chain(cfg)
    out = OUT_ROOT / "plan_preview"
    if args.view:
        view_dir = Path(args.view)
    else:
        view_dir = out / "view"
        grab(cfg, view_dir)
    view = detect_file(view_dir / "color.png", board)
    if view is None:
        raise PipelineError(f"no AprilGrid in {view_dir / 'color.png'}")
    q = np.asarray(json.loads((view_dir / "joints.json").read_text())["measured_joints"])
    Y = grid_pose_in_base(SimpleNamespace(obj=view.obj, img=view.img), cam, q, X, dq)
    print(f"{len(view.ids)} tags; grid centre in base: {np.round((Y @ np.r_[board.center, 1])[:3], 3)} m")
    poses, _, report = plan_placement(q, Y, X, dq, cam, board, pen_tip_offset(cfg), args.views, None,
                                      plan_settings(args), np.random.default_rng(20261119))
    print("  " + "\n  ".join(report))
    write_poses(out / "views.yaml", poses)
    print(f"wrote {out / 'views.yaml'} (not executed)")


def cmd_solve(args):
    cfg = Config.template()
    run_dir = find_run(args.run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    cam = Camera.load(args.intrinsics or meta["intrinsics"])
    solve(run_dir, cam, cfg, args.compare)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "plan", "solve", "status"):
        p = sub.add_parser(name)
        if name == "status":
            p.add_argument("run_dir", nargs="?", default="latest", help="run folder (default: the newest)")
            continue
        p.add_argument("--intrinsics", help="intrinsics.json from calibrate_azure_intrinsics.py "
                                            "(default: the newest in camera_calibration/)")
        p.add_argument("--compare", help="a pipeline calibration.json whose offsets to score as well "
                                         "(e.g. sessions/current/calibration.json)")
        p.add_argument("--tag-size", type=float, default=0.055, help="AprilGrid tag side, m")
        if name == "solve":
            p.add_argument("run_dir", nargs="?", default="latest", help="run folder (default: the newest)")
        else:
            p.add_argument("--views", type=int, default=35, help="views per placement (default 35)")
            p.add_argument("--box-deg", type=float, default=60.0,
                           help="joints 1-6 stay within this of the look pose (default 60)")
        if name == "run":
            p.add_argument("--resume", nargs="?", const="latest", metavar="RUN_DIR",
                           help="continue a run where it stopped (default: the newest run)")
            p.add_argument("--from", dest="from_stage", metavar="PLACEMENT:STAGE",
                           help="with --resume: redo from this stage, e.g. 2:capture or 1:touch "
                                "(stages: look, plan, capture, touch; old files are kept in old_<time>/)")
            p.add_argument("--touches", type=int, default=6, help="touches per placement (0 = camera only)")
            p.add_argument("--yes", action="store_true", help="do not ask for Enter before motions")
        if name == "plan":
            p.add_argument("--view", help="a look view folder (color.png, joints.json) instead of grabbing")
    args = ap.parse_args()
    if args.command == "run" and args.from_stage and not args.resume:
        ap.error("--from needs --resume RUN_DIR")
    try:
        {"run": cmd_run, "plan": cmd_plan, "solve": cmd_solve, "status": cmd_status}[args.command](args)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nstopped; continue with `./pipeline/tools/joint_offsets run --resume`", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
