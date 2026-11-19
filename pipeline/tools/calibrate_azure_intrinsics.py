#!/usr/bin/env python3
"""Calibrate the Azure Kinect colour camera (2160P) with a hand-held AprilGrid.

Board: Kalibr AprilGrid, 6x6 AprilTags 36h11 with a 2-bit black border, 5.5 cm
tags, 1.65 cm gaps (0.3 x tag) with small black squares at the gap crossings.
Tag 0 is bottom-left, ids run along a row, rows go up. Every tag corner touches
a corner square, so each corner is an X junction that cv2.cornerSubPix pins to
a small fraction of a pixel.

Live, in the franka_ros container while `roslaunch panda_calibration
azure_4k.launch` runs. Move the BOARD and hold each pose still for ~1 s: a view
is captured automatically when the board is still and the pose is new. The
preview lists the poses still missing (distance band x which board edge is
turned away from the camera), suggests the next one, frames the image regions
without corners in red, and says ALL DONE when the set is complete.
Keys: space = capture now, u = undo last, q = calibrate, Esc = quit.
If the camera launch stops (Ctrl-C, or the driver dies and takes roscore down),
the window says NO FRAMES; start the launch again and the tool re-subscribes
by itself, keeping the views captured so far.

    docker exec -it franka_ros bash -c "source /opt/franka_underlay/install/setup.bash && \\
        python3 /franka_ws/pipeline/tools/calibrate_azure_intrinsics.py"

Offline, again from a previous run's folder (host or container, no ROS):

    python3 pipeline/tools/calibrate_azure_intrinsics.py --images camera_calibration/azure_<stamp>

The fit tries three sets of free parameters: none (factory), fx fy cx cy with
the factory distortion (what the pipeline's rim bundle frees), and all 12. It
keeps the simplest set that predicts held-out views best (k-fold
cross-validation over views). Writes to the output folder:
  intrinsics.json   layout of calibrate/6_bundle/intrinsics.json (names, factory,
                    refined, free, rim_bias) plus fit statistics
  camera_info.json  the driver's factory calibration
  images/view_NNN.png, coverage.png (all corners, residuals x50)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]

NAMES = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
# OpenCV fixes parameters in these groups (one flag each), so they are freed together.
FIX_FLAGS = [
    (("fx", "fy"), cv2.CALIB_FIX_FOCAL_LENGTH),
    (("cx", "cy"), cv2.CALIB_FIX_PRINCIPAL_POINT),
    (("k1",), cv2.CALIB_FIX_K1),
    (("k2",), cv2.CALIB_FIX_K2),
    (("p1", "p2"), cv2.CALIB_FIX_TANGENT_DIST),
    (("k3",), cv2.CALIB_FIX_K3),
    (("k4",), cv2.CALIB_FIX_K4),
    (("k5",), cv2.CALIB_FIX_K5),
    (("k6",), cv2.CALIB_FIX_K6),
]
# Rim-edge offsets of the hole-rim board fitter (pose-refinement
# boardpose/board_model.py RIM_BIAS0). Not a camera parameter and not estimated
# here; written so this file can stand in for calibrate/6_bundle/intrinsics.json.
RIM_BIAS0 = [1.604e-4, -1.124e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
NEW_POSE_DEG = 8.0      # auto-capture only poses this far (rotation) ...
NEW_POSE_M = 0.04       # ... or this far (board centre) from every captured one
CELLS = (4, 3)          # image coverage grid shown in the preview


class AprilGrid:
    def __init__(self, cols, rows, tag, spacing):
        self.cols, self.rows, self.tag, self.spacing = cols, rows, tag, spacing
        self.pitch = tag * (1.0 + spacing)
        self.n_tags = cols * rows
        self.center = np.array([(cols * self.pitch - tag * spacing) / 2,
                                (rows * self.pitch - tag * spacing) / 2, 0.0])

    def object_points(self, ids):
        """(N, 4, 3) board coordinates of the tags' corners, in OpenCV's corner order.

        x right, y up, origin at tag 0's bottom-left corner. Kalibr prints each
        tag turned 180 deg relative to cv2.aruco.drawMarker, so OpenCV's corner 0
        is the tag's bottom-right (checked on a photo of the board).
        """
        ids = np.asarray(ids, int)
        s = self.tag
        x0 = (ids % self.cols) * self.pitch
        y0 = (ids // self.cols) * self.pitch
        xy = np.stack([np.stack([x0 + s, y0], -1), np.stack([x0, y0], -1),
                       np.stack([x0, y0 + s], -1), np.stack([x0 + s, y0 + s], -1)], 1)
        return np.concatenate([xy, np.zeros(xy.shape[:2] + (1,))], -1).astype(np.float32)

    def as_dict(self):
        return {"type": "kalibr_aprilgrid", "family": "36h11", "border_bits": 2,
                "grid": [self.cols, self.rows], "tag_size_m": self.tag, "spacing": self.spacing}


class View:
    def __init__(self, name, tags, board):
        self.name = name
        self.ids = np.array(sorted(tags), int)
        self.img = np.concatenate([tags[i] for i in self.ids]).astype(np.float32)   # (4N, 2)
        self.obj = board.object_points(self.ids).reshape(-1, 3)                      # (4N, 3)


# ------------------------------------------------------------------ detection

def detector_parameters():
    make = getattr(cv2.aruco, "DetectorParameters_create", None)   # OpenCV < 4.7
    p = make() if make is not None else cv2.aruco.DetectorParameters()
    p.markerBorderBits = 2                                 # Kalibr tags: 2-bit black border
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # refined below, on the X junctions
    p.minMarkerPerimeterRate = 0.01
    p.adaptiveThreshWinSizeMax = 53                        # tags up to ~400 px across in 4K
    return p


# Always at full resolution: a half-size image loses most tags of tilted or
# distant boards. Two passes over a 4K frame take ~0.1 s, fine for the preview.
PARAMS = detector_parameters()


def detect_tags(gray, board):
    """Tag id -> coarse (4, 2) corners; they only seed cornerSubPix.

    Two passes: the image, and a copy with the dark regions thinned by a pixel,
    which splits the diagonal contacts between tags and corner squares that
    can merge a tag's outline with them. Per tag the LARGEST quad wins: on big
    tags the adaptive threshold hollows out the 2-bit border, and aruco then
    sometimes returns the ring's inner outline, 10-30 px inside the tag (seen
    with OpenCV 4.2 and 4.5), which cornerSubPix cannot recover from.
    """
    found = {}
    for img in (gray, cv2.dilate(gray, np.ones((3, 3), np.uint8))):
        corners, ids, _ = cv2.aruco.detectMarkers(img, DICTIONARY, parameters=PARAMS)
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            i, q = int(i), c.reshape(4, 2).astype(np.float32)
            if i < board.n_tags and (i not in found or
                                     cv2.contourArea(q) > cv2.contourArea(found[i])):
                found[i] = q
    return found


def refine(gray, tags):
    """cornerSubPix on each tag's corners; drops tags at the image border or that diverge."""
    h, w = gray.shape[:2]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    out = {}
    for i, c in tags.items():
        side = float(np.mean(np.linalg.norm(c - np.roll(c, 1, 0), axis=1)))
        win = int(np.clip(round(0.12 * side), 3, 15))   # inside the corner square (0.3 x side)
        if (c.min() < win + 2 or c[:, 0].max() > w - win - 3 or c[:, 1].max() > h - win - 3):
            continue
        r = cv2.cornerSubPix(gray, c.reshape(-1, 1, 2).copy(), (win, win), (-1, -1), crit)
        r = r.reshape(4, 2)
        if np.linalg.norm(r - c, axis=1).max() <= 0.5 * win + 1.0:
            out[i] = r
    return out


def detect_view(gray, board):
    return refine(gray, detect_tags(gray, board))


# ------------------------------------------------------------------ fitting

def flags_for(free):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL
    for group, flag in FIX_FLAGS:
        if not set(group) & set(free):
            flags |= flag
    return flags


def fit(views, size, k0, d0, free):
    """calibrateCamera from the factory values with only `free` released.

    Returns K, D (8), rvecs, tvecs, sigma of the 12 parameters (0 when fixed).
    """
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    out = cv2.calibrateCameraExtended(
        [v.obj for v in views], [v.img for v in views], tuple(size),
        k0.copy(), d0.reshape(1, 8).copy(), flags=flags_for(free), criteria=crit)
    _, K, D, rvecs, tvecs, std = out[:6]
    return K, D.ravel()[:8], rvecs, tvecs, std.ravel()[:12]


def pnp(view, K, D):
    _, rvec, tvec = cv2.solvePnP(view.obj, view.img, K, D, flags=cv2.SOLVEPNP_IPPE)
    _, rvec, tvec = cv2.solvePnP(view.obj, view.img, K, D, rvec, tvec, True,
                                 cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec


def residuals(view, K, D, rvec=None, tvec=None):
    """projected - detected, (4N, 2); the pose is solved by PnP when not given."""
    if rvec is None:
        rvec, tvec = pnp(view, K, D)
    proj, _ = cv2.projectPoints(view.obj, rvec, tvec, K, D)
    return proj.reshape(-1, 2) - view.img


def rms(errors):
    e = np.concatenate([np.asarray(x).reshape(-1, 2) for x in errors])
    return float(np.sqrt(np.mean(np.sum(e ** 2, 1))))


def cross_validate(views, size, k0, d0, free, folds):
    """RMS on held-out views: fit on the other folds, PnP + reprojection on these."""
    if not free:
        return rms([residuals(v, k0, d0) for v in views])
    order = np.random.RandomState(0).permutation(len(views))
    errors = []
    for f in range(folds):
        test = set(order[f::folds].tolist())
        K, D = fit([views[i] for i in order if i not in test], size, k0, d0, free)[:2]
        errors += [residuals(views[i], K, D) for i in sorted(test)]
    return rms(errors)


def clean(views, size, k0, d0, min_tags):
    """Drop mis-refined tags and bad views, judged by an all-parameter fit."""
    K, D, rvecs, tvecs, _ = fit(views, size, k0, d0, NAMES)
    errs = [np.linalg.norm(residuals(v, K, D, r, t), axis=1) for v, r, t in zip(views, rvecs, tvecs)]
    tag_limit = max(6.0 * float(np.median(np.concatenate(errs))), 1.5)
    view_rms = np.array([float(np.sqrt(np.mean(e ** 2))) for e in errs])
    view_limit = max(3.0 * float(np.median(view_rms)), 1.0)
    kept, dropped_tags = [], 0
    for v, e, vr in zip(views, errs, view_rms):
        if vr > view_limit:
            print(f"  dropped {v.name}: RMS {vr:.2f} px > {view_limit:.2f}")
            continue
        good = (e.reshape(-1, 4) <= tag_limit).all(1)
        dropped_tags += int((~good).sum())
        if good.sum() < min_tags:
            print(f"  dropped {v.name}: {good.sum()} tags left after outlier removal")
            continue
        v.ids, v.img, v.obj = (v.ids[good], v.img.reshape(-1, 4, 2)[good].reshape(-1, 2),
                               v.obj.reshape(-1, 4, 3)[good].reshape(-1, 3))
        kept.append(v)
    if dropped_tags:
        print(f"  dropped {dropped_tags} tags with a corner residual > {tag_limit:.2f} px")
    return kept


def pose_stats(rvec, tvec, board):
    R = cv2.Rodrigues(rvec)[0]
    centre = R @ board.center + np.ravel(tvec)
    tilt = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
    return float(np.linalg.norm(centre)), tilt


def params12(K, D):
    return [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])] + [float(x) for x in D[:8]]


def camera_from12(p):
    K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], float)
    return K, np.array(p[4:12], float)


def load_compare(path):
    data = json.loads(Path(path).read_text())
    if "refined" in data:                     # intrinsics.json
        values = dict(zip(data["names"], data["refined"]))
        return camera_from12([values[n] for n in NAMES])
    if "k" in data:                           # camera_info.json
        return np.array(data["k"], float).reshape(3, 3), np.array((list(data["d"]) + [0] * 8)[:8], float)
    if "intrinsics" in data:                  # pipeline calibration.json
        return camera_from12([data["intrinsics"][n] for n in NAMES])
    raise SystemExit(f"{path}: no 'refined', 'k' or 'intrinsics' to compare")


def calibrate(views, size, k0, d0, board, args, out):
    if len(views) < 10:
        raise SystemExit(f"only {len(views)} usable views; collect at least 20 (40-60 is better)")
    print(f"\n{len(views)} views, {sum(len(v.ids) for v in views)} tags; removing outliers ...")
    views = clean(views, size, k0, d0, args.min_tags)
    folds = max(2, min(args.folds, len(views) // 2))

    variants = [("factory", [])]
    variants.append(("fxfycxcy", NAMES[:4]))
    if args.free:
        variants.append(("custom", list(args.free)))
    variants.append(("all", list(NAMES)))

    results = {}
    print(f"fitting {len(variants)} variants, {folds}-fold cross-validation ...")
    for name, free in variants:
        if free:
            K, D, rvecs, tvecs, std = fit(views, size, k0, d0, free)
        else:
            K, D, std = k0, d0, np.zeros(12)
            poses = [pnp(v, K, D) for v in views]
            rvecs, tvecs = [p[0] for p in poses], [p[1] for p in poses]
        errors = [residuals(v, K, D, r, t) for v, r, t in zip(views, rvecs, tvecs)]
        results[name] = {"free": free, "K": K, "D": D, "std": std, "rvecs": rvecs, "tvecs": tvecs,
                         "errors": errors, "fit_rms": rms(errors),
                         "cv_rms": cross_validate(views, size, k0, d0, free, folds)}

    # The simplest variant within 2 % of the best held-out RMS (variants go simple -> complex).
    best_cv = min(r["cv_rms"] for r in results.values())
    chosen = next(n for n, _ in variants if results[n]["cv_rms"] <= 1.02 * best_cv)
    res = results[chosen]

    compare = {}
    for path in args.compare or []:
        K, D = load_compare(path)
        compare[str(path)] = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                              "rms_px": rms([residuals(v, K, D) for v in views])}

    factory, refined = params12(k0, d0), params12(res["K"], res["D"])
    per_view = []
    for v, r, t, e in zip(views, res["rvecs"], res["tvecs"], res["errors"]):
        dist, tilt = pose_stats(r, t, board)
        per_view.append({"name": v.name, "tags": int(len(v.ids)), "rms_px": round(rms([e]), 4),
                         "distance_m": round(dist, 3), "tilt_deg": round(tilt, 1)})

    # ---- report
    print("\nper view (chosen variant):")
    for p in per_view:
        print(f"  {p['name']:>10}  tags {p['tags']:2d}  RMS {p['rms_px']:.3f} px  "
              f"dist {p['distance_m']:.2f} m  tilt {p['tilt_deg']:4.1f} deg")
    print("\nvariant     free  fit RMS px  held-out RMS px")
    for name, _ in variants:
        r = results[name]
        mark = "   <- chosen" if name == chosen else ""
        print(f"  {name:<9} {len(r['free']):3d}   {r['fit_rms']:9.4f}  {r['cv_rms']:14.4f}{mark}")
    if compare:
        print("\nother intrinsics on these views (PnP per view, nothing fitted to them):")
        for path, c in compare.items():
            print(f"  {c['rms_px']:.4f} px  fx {c['fx']:.2f} fy {c['fy']:.2f}  {path}")
        print(f"  {results['factory']['cv_rms']:.4f} px  factory  |  "
              f"{res['cv_rms']:.4f} px  this calibration ({chosen}, held-out)")
    print(f"\n{'param':<5} {'factory':>14} {'refined':>14} {'delta':>11} {'delta %':>9} {'sigma':>10}")
    for n, f, r, s in zip(NAMES, factory, refined, res["std"]):
        pct = f"{100 * (r - f) / f:+8.3f}%" if abs(f) > 1e-3 else f"{'-':>9}"
        print(f"{n:<5} {f:14.6f} {r:14.6f} {r - f:+11.5f} {pct} {s:10.5f}")
    print(f"\nRMS {res['fit_rms']:.4f} px over {len(views)} views, "
          f"{sum(len(v.ids) for v in views) * 4} corners")

    result = {
        "names": NAMES,
        "factory": factory,
        "refined": refined,
        "free": res["free"],
        "rim_bias": RIM_BIAS0,
        "rim_bias_source": "default RIM_BIAS0 of the rim fitter; not estimated by this calibration",
        "std": [float(s) for s in res["std"]],
        "rms_px": res["fit_rms"],
        "cv_rms_px": res["cv_rms"],
        "variants": {n: {"free": results[n]["free"], "fit_rms_px": results[n]["fit_rms"],
                         "cv_rms_px": results[n]["cv_rms"],
                         "values": params12(results[n]["K"], results[n]["D"]),
                         "std": [float(s) for s in results[n]["std"]]} for n, _ in variants},
        "compare": compare,
        "views_used": len(views),
        "views": per_view,
        "image_size": [int(size[0]), int(size[1])],
        "board": board.as_dict(),
        "method": "AprilGrid, cv2.calibrateCameraExtended (rational model) from the factory "
                  f"values, variant chosen by {folds}-fold cross-validation over views",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out / "intrinsics.json").write_text(json.dumps(result, indent=2))
    coverage_image(views, res["errors"], size, out / "coverage.png", res["fit_rms"])
    print(f"\nwrote {out / 'intrinsics.json'} and {out / 'coverage.png'}")


def coverage_image(views, errors, size, path, rms_px):
    s = 0.5
    canvas = np.full((int(size[1] * s), int(size[0] * s), 3), 255, np.uint8)
    for v, e in zip(views, errors):
        for p, d in zip(v.img, e):
            a = (int(round(p[0] * s)), int(round(p[1] * s)))
            b = (int(round((p[0] + 50 * d[0]) * s)), int(round((p[1] + 50 * d[1]) * s)))
            cv2.line(canvas, a, b, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, a, 2, (200, 80, 0), -1)
    cv2.putText(canvas, f"{len(views)} views, RMS {rms_px:.3f} px, residuals x50",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


# ------------------------------------------------------------------ live capture
#
# The preview asks for the poses that pin the intrinsics down: in every
# distance band, the board facing the camera and with each of its four edges
# turned away from the camera. Tilted views separate the focal length from the
# distance; views spread over the whole image (the region frames) fix the
# distortion, which is largest in the image corners.

BANDS = [(0.42, "0.30-0.42 m"), (0.65, "0.42-0.65 m"), (99.0, "0.65-0.90 m")]   # (upper limit, label)
TURNS = ["front", "left", "right", "top", "bottom"]   # which board edge (as seen) is turned away
NEED = {"front": 3, "left": 2, "right": 2, "top": 2, "bottom": 2}   # views per band
TURN_TEXT = {"front": "board facing the camera (tilt < 15 deg)",
             "left": "turn the LEFT edge away (20-45 deg)",
             "right": "turn the RIGHT edge away (20-45 deg)",
             "top": "turn the TOP edge away (20-45 deg)",
             "bottom": "turn the BOTTOM edge away (20-45 deg)"}
FRONT_DEG, MAX_TILT_DEG = 15.0, 60.0
REGION_NEED = 40          # corners in a preview region (CELLS) before it counts as covered


def pose_class(rvec, tvec, board):
    """(band index, turn, distance m, tilt deg) of a board pose."""
    dist, tilt = pose_stats(rvec, tvec, board)
    band = next(k for k, (upper, _) in enumerate(BANDS) if dist < upper)
    if tilt < FRONT_DEG:
        return band, "front", dist, tilt
    n = cv2.Rodrigues(rvec)[0][:, 2]
    n = -n if n[2] > 0 else n           # board normal, pointing at the camera
    # n_x > 0: the board's right side (in the image) is farther away; n_y > 0: its bottom.
    if abs(n[0]) >= abs(n[1]):
        return band, ("right" if n[0] > 0 else "left"), dist, tilt
    return band, ("bottom" if n[1] > 0 else "top"), dist, tilt


def rotation_deg(r1, r2):
    R = cv2.Rodrigues(r1)[0].T @ cv2.Rodrigues(r2)[0]
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def put(img, text, org, colour=(255, 255, 255), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def keep_subscribed(topic, type_name, status):
    """Re-subscribe to `topic` whenever roscore is replaced (daemon thread).

    azure_4k.launch starts its own roscore. Stopping the launch, or the driver
    dying (it is `required`, so roslaunch takes roscore down with it), replaces
    the master; the new one does not know this node and rospy never registers
    again, so frames would never come back. Register with the new master here
    and hand rospy its publisher list, as the master's publisherUpdate would;
    later driver starts then reach this node the normal way.
    """
    import rospy
    import rosgraph
    from xmlrpc.client import ServerProxy

    name, uri = rospy.get_name(), rospy.get_node_uri()
    master, first = rosgraph.Master(name), True
    while not rospy.is_shutdown():
        try:
            if name not in dict(master.getSystemState()[1]).get(topic, []):
                publishers = master.registerSubscriber(topic, type_name, uri)
                ServerProxy(uri).publisherUpdate("/master", topic, publishers)
                if not first:
                    print(f"new roscore: subscribed to {topic} again", flush=True)
            status["master"] = True
        except (OSError, rosgraph.MasterException):   # no roscore (connection refused)
            status["master"] = False
        first = False
        time.sleep(1.0)


def live(args, board, out, k0, d0, size):
    import threading
    import rospy
    from sensor_msgs.msg import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros_nodes"))
    from grab_azure_rgbd import to_array

    topic = f"{args.ns}/rgb/image_raw"
    latest, status = {}, {"master": True}

    def on_image(m):
        latest["msg"], latest["t"] = m, time.time()

    rospy.Subscriber(topic, Image, on_image, queue_size=1,
                     buff_size=2 ** 27)   # a 4K BGRA frame is ~33 MB
    threading.Thread(target=keep_subscribed, args=(topic, Image._type, status), daemon=True).start()
    started, stale_said = time.time(), False
    (out / "images").mkdir(parents=True, exist_ok=True)

    win = "Azure intrinsics calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    s = 1280.0 / size[0]
    disp = np.zeros((int(round(size[1] * s)), 1280, 3), np.uint8)
    put(disp, f"waiting for {args.ns}/rgb/image_raw ...", (20, 40))
    views, poses, classes, files = [], [], [], []
    counter = 0
    last_msg, prev, still_since, last_capture = None, None, None, 0.0
    frame = gray = None
    tags = {}
    flash, flash_until = "", 0.0

    def corners():
        return np.concatenate([v.img for v in views]) if views else np.zeros((0, 2), np.float32)

    def progress():
        count = {}
        for c in classes:
            count[c] = count.get(c, 0) + 1
        missing = [(b, t) for b in range(len(BANDS)) for t in TURNS if count.get((b, t), 0) < NEED[t]]
        regions = np.zeros(CELLS[::-1], int)
        pts = corners()
        if len(pts):
            cx = np.minimum((pts[:, 0] / size[0] * CELLS[0]).astype(int), CELLS[0] - 1)
            cy = np.minimum((pts[:, 1] / size[1] * CELLS[1]).astype(int), CELLS[1] - 1)
            np.add.at(regions, (cy, cx), 1)
        return count, missing, regions

    def capture():
        nonlocal counter, last_capture, flash, flash_until
        last_capture = time.time()
        refined = refine(gray, tags)
        if len(refined) < args.min_tags:
            flash, flash_until = f"only {len(refined)} usable tags, not saved", time.time() + 1.5
            return
        name = f"view_{counter:03d}"
        counter += 1
        path = out / "images" / f"{name}.png"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        view = View(name, refined, board)
        pose = pnp(view, k0, d0)
        band, turn, dist, tilt = pose_class(pose[0], pose[1], board)
        views.append(view)
        poses.append(pose)
        classes.append((band, turn))
        files.append(path)
        _, missing, regions = progress()
        flash, flash_until = f"captured {name}: {BANDS[band][1]}, {turn}", time.time() + 1.2
        print(f"  {name}: {len(refined)} tags, {dist:.2f} m, tilt {tilt:.0f} deg ({turn}); "
              f"missing poses {len(missing)}, uncovered regions {int((regions < REGION_NEED).sum())}")

    print(f"waiting for images on {topic}")
    while True:
        msg = latest.get("msg")
        if msg is not None and msg is not last_msg:
            if last_msg is None:
                print(f"frames arriving ({msg.width}x{msg.height} {msg.encoding}); the live view "
                      f"and all prompts are in the window '{win}'", flush=True)
            elif stale_said:
                print("frames are back", flush=True)
            stale_said = False
            last_msg = msg
            now = time.time()
            frame = to_array(msg)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = detect_tags(gray, board)
            centres = {i: c.mean(0) for i, c in tags.items()}

            speed = np.inf
            if prev is not None:
                common = set(centres) & set(prev[1])
                if len(common) >= args.min_tags:
                    move = np.median([np.linalg.norm(centres[i] - prev[1][i]) for i in common])
                    speed = move / max(now - prev[0], 1e-3)
            prev = (now, centres)
            still_since = (still_since or now) if speed < args.still_speed else None

            current, new, state = None, False, "no board (needs 8 tags)"
            if len(tags) >= args.min_tags:
                pose = pnp(View("live", tags, board), k0, d0)
                current = pose_class(pose[0], pose[1], board)
                centre = cv2.Rodrigues(pose[0])[0] @ board.center + pose[1].ravel()
                new = all(rotation_deg(pose[0], r) > NEW_POSE_DEG or
                          np.linalg.norm(centre - (cv2.Rodrigues(r)[0] @ board.center + t.ravel()))
                          > NEW_POSE_M for r, t in poses)
                if current[3] > MAX_TILT_DEG:
                    state = f"too steep (> {MAX_TILT_DEG:.0f} deg)"
                elif still_since is None:
                    state = "moving - hold still"
                elif not new:
                    state = "this pose is captured - move on"
                elif now - still_since < args.still_time or now - last_capture < 1.0:
                    state = "hold still ..."
                else:
                    capture()
                    state = "captured - move on"

            # ---- preview
            count, missing, regions = progress()
            disp = cv2.resize(frame, (1280, int(round(size[1] * s))), interpolation=cv2.INTER_AREA)
            pts = np.int32(corners() * s)
            if len(pts):
                for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    disp[np.clip(pts[:, 1] + dy, 0, disp.shape[0] - 1),
                         np.clip(pts[:, 0] + dx, 0, disp.shape[1] - 1)] = (0, 255, 0)
            cw, ch = disp.shape[1] / CELLS[0], disp.shape[0] / CELLS[1]
            for cy in range(CELLS[1]):
                for cx in range(CELLS[0]):
                    if regions[cy, cx] < REGION_NEED:
                        cv2.rectangle(disp, (int(cx * cw) + 2, int(cy * ch) + 2),
                                      (int((cx + 1) * cw) - 3, int((cy + 1) * ch) - 3), (0, 0, 255), 2)
            colour = (0, 255, 0) if still_since is not None else (0, 200, 255)
            for c in tags.values():
                cv2.polylines(disp, [np.int32(c * s)], True, colour, 1, cv2.LINE_AA)

            panel = disp[:250, :560]
            panel[:] = (panel * 0.35).astype(np.uint8)
            if current is not None:
                put(disp, f"views {len(views)}   {current[2]:.2f} m   tilt {current[3]:.0f} deg "
                          f"({current[1]})   {state}", (10, 20))
            else:
                put(disp, f"views {len(views)}   {state}", (10, 20))
            put(disp, "edge away:", (10, 46), (180, 180, 180))
            for k, t in enumerate(TURNS):
                put(disp, t, (140 + 80 * k, 46), (180, 180, 180))
            for b, (_, label) in enumerate(BANDS):
                y = 70 + 22 * b
                put(disp, label, (10, y))
                for k, t in enumerate(TURNS):
                    n, need = count.get((b, t), 0), NEED[t]
                    col = (0, 220, 0) if n >= need else (0, 220, 255) if n else (60, 60, 255)
                    text = f"{n}/{need}"
                    if current is not None and (current[0], current[1]) == (b, t):
                        text = f"[{text}]"
                    put(disp, text, (140 + 80 * k, y), col)
            covered = int((regions >= REGION_NEED).sum())
            put(disp, f"image regions {covered}/{regions.size} (red frames: no corners yet)", (10, 144))
            if missing:
                here = [m for m in missing if current is not None and m[0] == current[0]]
                b, t = (here or missing)[0]
                again = current is not None and (current[0], current[1]) == (b, t) and not new
                put(disp, f"NEXT: {BANDS[b][1]}, {TURN_TEXT[t]}", (10, 170), (255, 255, 0))
                if again:
                    put(disp, "      (this spot is done: move the board elsewhere in the image)",
                        (10, 192), (255, 255, 0))
            elif covered < regions.size:
                put(disp, "NEXT: board into the red frames (image corners/edges)", (10, 170), (255, 255, 0))
            else:
                put(disp, "ALL DONE - press q to calibrate (more views do not hurt)", (10, 170), (0, 255, 0))
            if time.time() < flash_until:
                put(disp, flash, (10, 214), (0, 255, 0))
            put(disp, "space capture   u undo   q calibrate   Esc quit", (10, 238), (180, 180, 180))

        # ---- no new frames: say why, in the window and once in the terminal
        shown = disp
        silent = time.time() - latest.get("t", started)
        if silent > (2.0 if last_msg is not None else 5.0):
            why = ("no roscore: azure_4k.launch stopped (Ctrl-C, or the driver died). Start it "
                   "again; this tool re-subscribes by itself" if not status["master"] else
                   f"is the camera driver running? (rostopic hz {topic})")
            if not stale_said:
                print(f"no frames for {silent:.0f} s - {why}", flush=True)
                stale_said = True
            shown = disp.copy()
            put(shown, f"NO FRAMES for {silent:.0f} s", (10, shown.shape[0] - 50), (0, 0, 255), 1.0)
            put(shown, why, (10, shown.shape[0] - 20), (0, 0, 255), 0.5)

        cv2.imshow(win, shown)
        key = cv2.waitKey(5) & 0xFF
        if key == ord(" ") and len(tags) >= args.min_tags:
            capture()
        elif key == ord("u") and views:
            for lst in (views, poses, classes):
                lst.pop()
            files.pop().unlink()
            flash, flash_until = "removed the last view", time.time() + 1.2
        elif key == ord("q"):
            if len(views) < 10:
                flash, flash_until = f"need at least 10 views (have {len(views)})", time.time() + 2.0
                continue
            cv2.destroyAllWindows()
            return views
        elif key == 27:
            cv2.destroyAllWindows()
            return None


# ------------------------------------------------------------------ main

def camera_info_dict(info):
    return {"width": info.width, "height": info.height, "distortion_model": info.distortion_model,
            "d": list(info.D), "k": list(info.K), "r": list(info.R), "p": list(info.P),
            "frame_id": info.header.frame_id}


def factory_from(info):
    k0 = np.array(info["k"], float).reshape(3, 3)
    d0 = np.array((list(info["d"]) + [0.0] * 8)[:8], float)
    return k0, d0, (int(info["width"]), int(info["height"]))


def make_dir(path):
    """mkdir -p; returns the top-most directory it created (None if it existed)."""
    top = None
    for p in reversed([path] + list(path.parents)):
        if not p.exists():
            top = top or p
    path.mkdir(parents=True, exist_ok=True)
    return top


def give_back(path):
    """Files written in the container belong to root; hand them to the workspace owner."""
    if path is None or os.geteuid() != 0:
        return
    st = WORKSPACE.stat()
    if st.st_uid == 0:
        return
    os.chown(path, st.st_uid, st.st_gid)
    for root, dirs, files in os.walk(path):
        for n in dirs + files:
            os.chown(os.path.join(root, n), st.st_uid, st.st_gid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, help="calibrate offline from this folder "
                    "(images/*.png + camera_info.json of an earlier run)")
    ap.add_argument("--out", type=Path, help="output folder (live default: "
                    "camera_calibration/azure_<stamp>; --images default: that folder)")
    ap.add_argument("--tag-size", type=float, default=0.055, help="tag side, m (default 0.055)")
    ap.add_argument("--spacing", type=float, default=0.3, help="gap / tag side (default 0.3)")
    ap.add_argument("--grid", type=int, nargs=2, default=[6, 6], metavar=("COLS", "ROWS"))
    ap.add_argument("--ns", default="/azure_camera", help="camera namespace (live)")
    ap.add_argument("--min-tags", type=int, default=8, help="tags a view needs (default 8)")
    ap.add_argument("--free", nargs="+", choices=NAMES,
                    help="also try this set of free parameters (fx+fy, cx+cy, p1+p2 go together)")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    ap.add_argument("--compare", type=Path, action="append",
                    help="intrinsics.json / camera_info.json / calibration.json to score on the "
                         "same views (repeatable)")
    ap.add_argument("--still-time", type=float, default=0.7, help="s the board must be still")
    ap.add_argument("--still-speed", type=float, default=12.0,
                    help="px/s (4K) below which the board counts as still")
    args = ap.parse_args()

    if args.free:
        for group, _ in FIX_FLAGS:
            if 0 < len(set(group) & set(args.free)) < len(group):
                ap.error(f"--free: {' and '.join(group)} can only be freed together")
    board = AprilGrid(args.grid[0], args.grid[1], args.tag_size, args.spacing)

    if args.images:
        folder = args.images
        info_path = folder / "camera_info.json"
        if not info_path.exists():
            raise SystemExit(f"{info_path} is missing (factory calibration from the driver)")
        k0, d0, size = factory_from(json.loads(info_path.read_text()))
        out = args.out or folder
        created = make_dir(out)
        if out.resolve() != folder.resolve():
            (out / "camera_info.json").write_text(info_path.read_text())
        paths = sorted((folder / "images").glob("*.png")) or sorted(folder.glob("view_*.png"))
        if not paths:
            raise SystemExit(f"no images in {folder}/images")
        views = []
        for p in paths:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise SystemExit(f"{p}: cannot read")
            if gray.shape[::-1] != tuple(size):
                raise SystemExit(f"{p}: {gray.shape[1]}x{gray.shape[0]}, camera_info says {size}")
            tags = detect_view(gray, board)
            print(f"  {p.name}: {len(tags)} tags" + ("" if len(tags) >= args.min_tags else ", skipped"))
            if len(tags) >= args.min_tags:
                views.append(View(p.stem, tags, board))
    else:
        import rospy
        from sensor_msgs.msg import CameraInfo
        rospy.init_node("calibrate_azure_intrinsics", anonymous=True, disable_signals=True)
        try:
            info = rospy.wait_for_message(f"{args.ns}/rgb/camera_info", CameraInfo, timeout=15.0)
        except rospy.ROSException:
            raise SystemExit(f"no {args.ns}/rgb/camera_info; is "
                             "`roslaunch panda_calibration azure_4k.launch` running?")
        info = camera_info_dict(info)
        k0, d0, size = factory_from(info)
        if size != (3840, 2160):
            print(f"WARNING: camera is {size[0]}x{size[1]}; this calibration is only valid for that mode")
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out = args.out or WORKSPACE / "camera_calibration" / f"azure_{stamp}"
        created = make_dir(out)
        (out / "camera_info.json").write_text(json.dumps(info, indent=2))
        print(f"saving to {out}")
        try:
            views = live(args, board, out, k0, d0, size)
        except KeyboardInterrupt:
            views = None
        finally:
            rospy.signal_shutdown("done")
        if views is None:
            give_back(created or out)
            print(f"stopped without calibrating; images kept in {out}; calibrate later with "
                  f"--images {out}")
            return
    try:
        calibrate(views, size, k0, d0, board, args, out)
    finally:
        give_back(created or out)


if __name__ == "__main__":
    main()
