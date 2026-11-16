"""`pipeline diag [N]`: six tests that narrow down where a constant probing error comes from.

The probe lands the peg tip 2-3 mm from the hole, toward the robot base (-x),
every time. Each test changes one thing and watches how the error reacts:

  1 camera roll     the camera turns about its own optical axis above the
                    board; a hand-eye translation or principal-point error turns
                    with it, the true board does not.               (vision only)
  2 heights         the camera backs off along its optical axis; an angular
                    error grows with distance, a translation error does not.
                                                                     (vision only)
  3 tool turn       touch one hole with the hand turned about the peg; a wrong
                    TCP/peg tip turns with the hand.            (hand-guided touch)
  4 board turn      touch a hole, turn the board 180 deg, locate, touch again;
                    an error of the board model turns with the board.     (touch)
  5 board aside     touch a hole with the board in front and to the sides;
                    a kinematic error points at the base, a frame error stays
                    along base x.                                         (touch)
  6 touch-off       touch 3-4 distant holes; their distances, measured by the
                    arm alone, against the board drawing.                 (touch)

Tests 1 and 2 move the arm through planned camera poses, capture, fit the
board in every image with the calibration held fixed, and compare the board
poses they give. Tests 3-6 stop the trajectory controller so you can guide the
held peg into a hole by hand; the arm's joints then say where the robot thinks
the tip is, and the hole says where it really is.

Sign convention of every touch result: probe error = where the probe sends the
tip (the vision target) minus where the hole really is (the touched tip). The
reported fault, "2-3 mm too close to the base", reads as probe error x = -2..-3 mm.

All touch numbers are in the calibrated ("true") kinematic frame: the touched
tip is FK(q + dq) of the measured joints with the calibration's joint offsets,
the target is the board pose of board_pose.json. That is the frame the probe
aims in after its joint-offset correction.

Results go to sessions/<name>/diagnose/test<N>_<name>/<date>_<time>/.
"""

from __future__ import annotations

import datetime
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from board_pipeline import eye_in_hand, grasp, panda, rimfit, ros
from board_pipeline.config import PIPELINE_DIR, PipelineError, load_yaml
from board_pipeline.session import BOARD_POSE, CALIBRATION, Context, Session

SURFACE_Z = eye_in_hand.SURFACE_Z
CENTRE = np.array([0.0, 0.0, SURFACE_Z, 1.0])     # board top centre, board frame


@dataclass(frozen=True)
class Test:
    number: int
    slug: str
    title: str
    kind: str          # "camera" or "touch"
    needs: str
    question: str


TESTS = {t.number: t for t in (
    Test(1, "camera_roll", "camera roll about its optical axis", "camera",
         "arm moves (planned views), board still, ~3 min",
         "is the camera side (hand-eye translation, principal point) shifting the board?"),
    Test(2, "heights", "board seen from several heights", "camera",
         "arm moves (planned views), board still, ~3 min",
         "does the board pose drift with camera distance (hand-eye rotation, intrinsics, scale)?"),
    Test(3, "tool_turn", "same hole touched with the hand turned about the peg", "touch",
         "hand guiding, 2+ touches of one hole",
         "is the TCP / peg tip where the model says?"),
    Test(4, "board_turn", "board turned 180 deg on the table", "touch",
         "hand guiding; touch, turn the board, `pipeline locate --fresh`, touch again",
         "is the board model (hole positions, origin, print scale) wrong?"),
    Test(5, "board_aside", "board in front and to the sides of the robot", "touch",
         "hand guiding; touch, move the board to y = +-30 cm, locate, touch again",
         "does the error point at the base (kinematics) or stay along base x (frames)?"),
    Test(6, "touch_off", "touch-off of 3-4 distant holes", "touch",
         "hand guiding, one touch per hole (asks to swap the peg if the shape changes)",
         "are the distances the arm measures between holes the board's?"),
)}

# With the camera on link6 (joint 7 does not turn it), the Panda rolls the camera
# about its optical axis over the board only about +-45..60 deg from the look
# pose's roll; 90 deg needs joint 2 past its limit or the arm folded back over
# itself. Rolls it cannot reach are skipped.
DEFAULT_ROLLS = (0.0, 30.0, 60.0, 0.0, -30.0, -60.0, 0.0)
# 30 cm does not fit the whole board in the image; 35 cm only with the board's
# long side along the image width (test 2 picks that roll when it can).
DEFAULT_DISTANCES = (0.35, 0.45, 0.55, 0.65, 0.75)
ROLL_CANDIDATES = (-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0)
DEFAULT_TOUCH_HOLE = "round_c0p2mm"
# Two tight and two 1 mm holes at the corners of a 189 x 104 mm rectangle; the
# 104 mm side runs along the board y axis (about base x with the board as it lay).
DEFAULT_TOUCH_OFF = ("round_c0p03mm", "round_c1mm", "rectangle_c1mm", "rectangle_c0p03mm")
TOUCH_NODE = PIPELINE_DIR / "ros_nodes" / "touch_hole.py"
FRANKA_STATES = "/franka_state_controller/franka_states"


# ===================================================================== helpers
def _mm(v) -> str:
    return "[" + ", ".join(f"{x * 1e3:+7.2f}" for x in np.ravel(v)) + "]"


def _test_dir(session: Session, test: Test) -> Path:
    return session.path / "diagnose" / f"test{test.number}_{test.slug}"


def _new_run(session: Session, test: Test) -> Path:
    run = _test_dir(session, test) / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run.mkdir(parents=True)
    return run


def _latest_run(session: Session, test: Test, chosen: str | None = None) -> Path | None:
    if chosen:
        run = ros.host_path(session.cfg, chosen)
        if not (run / "run.json").exists():
            raise PipelineError(f"{run} is not a `pipeline diag {test.number}` run (no run.json)")
        return run
    runs = sorted(p.parent for p in _test_dir(session, test).glob("*/run.json"))
    return runs[-1] if runs else None


def _save(run: Path, data: dict) -> None:
    (run / "run.json").write_text(json.dumps(data, indent=2) + "\n")


def _board(session: Session) -> tuple[np.ndarray, Path]:
    path = session.require(BOARD_POSE, "run `pipeline locate` (or `calibrate`) first")
    return np.asarray(json.loads(path.read_text())["T_base_board"]), path


def _calibration(session: Session) -> eye_in_hand.Calibration:
    if session.cfg.setup != "eye_in_hand":
        raise PipelineError("`pipeline diag` is written for the eye_in_hand setup "
                            f"(this session is {session.cfg.setup})")
    return eye_in_hand.Calibration.load(session.require(CALIBRATION, "run `pipeline calibrate` first"))


def _look_view(session: Session, board_file: Path) -> Path:
    """The seed view (look pose image + joints) of the stage that produced board_pose.json."""
    stage = json.loads(board_file.read_text()).get("source", {}).get("stage", "locate")
    return session.step_dir(stage, "seed_fit") / "view"


def _rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


# =============================================================== camera tests
def _camera_pose(centre, direction, distance, x_ref, roll_deg) -> np.ndarray:
    """Camera at centre + distance * direction looking at centre, rolled roll_deg from x_ref."""
    z = -np.asarray(direction, float) / np.linalg.norm(direction)
    x0 = x_ref - z * (x_ref @ z)
    x0 /= np.linalg.norm(x0)
    a = math.radians(roll_deg)
    xa = math.cos(a) * x0 + math.sin(a) * np.cross(z, x0)
    cam = np.eye(4)
    cam[:3, :3] = np.c_[xa, np.cross(z, xa), z]
    cam[:3, 3] = centre + distance * np.asarray(direction, float) / np.linalg.norm(direction)
    return cam


class _Planner:
    """IK and safety checks for camera poses (as eye_in_hand.plan_views checks its views)."""

    def __init__(self, board, x, K, width, height, p, q_look, max_step_deg):
        self.board, self.x, self.K, self.size, self.p = board, x, K, (width, height), p
        self.q_look = q_look
        self.max_step = max_step_deg
        self.table_z = (board @ CENTRE)[2]
        self.corners = np.array([[sx * eye_in_hand.BOARD_HALF[0], sy * eye_in_hand.BOARD_HALF[1], SURFACE_Z, 1]
                                 for sx in (-1, 1) for sy in (-1, 1)])
        self.lo, self.hi = panda.LOWER + p.limit_margin, panda.UPPER - p.limit_margin
        self.ready = np.r_[panda.READY[:6], q_look[6]]   # joint 7 does not move the camera
        rng = np.random.default_rng(p.seed + 7)
        self.random_seeds = [np.r_[rng.uniform(self.lo[:6], self.hi[:6]), q_look[6]] for _ in range(24)]

    def clear(self, q) -> bool:
        return panda.frame_origins(q)[1:, 2].min() > self.table_z + self.p.clearance

    def visible(self, q) -> bool:
        cb = np.linalg.inv(panda.fk(q, "panda_link6") @ self.x) @ self.board @ self.corners.T
        if np.any(cb[2] <= 0):
            return False
        uv = (self.K @ (cb[:3] / cb[2])).T[:, :2]
        m = self.p.image_margin
        return not (np.any(uv < m) or np.any(uv[:, 0] > self.size[0] - m) or np.any(uv[:, 1] > self.size[1] - m))

    def why_not(self, q) -> str | None:
        if np.any(q < self.lo) or np.any(q > self.hi):
            return "joint limits"
        if not self.clear(q):
            return "too close to the board/table"
        if not self.visible(q):
            return "board not fully in the image"
        return None

    def segment_ok(self, qa, qb) -> bool:
        if math.degrees(np.abs(qb - qa).max()) > self.max_step:
            return False
        return all(self.clear(qa + s * (qb - qa)) for s in np.linspace(0, 1, 60))

    def follow(self, cam_from, q_from, cam_to, step_deg=1.0, step_m=0.01) -> tuple[list | None, str]:
        """Joints along a walk of the camera from cam_from (arm at q_from) to cam_to in
        small steps, each IK seeded from the last; ([q...], "") or (None, reason).

        Over the board the arm is close to its shoulder singularity (joints 1 and 3
        nearly aligned): rolling the camera by 1 deg turns joints 1 and 3 about
        4-5 deg against each other. A plain IK from a far seed jumps to another
        branch; the walk stays on the arm's branch, and its intermediate poses
        serve as waypoints that keep every move short (route()).
        """
        turn = Rotation.from_matrix(cam_from[:3, :3].T @ cam_to[:3, :3]).magnitude()
        shift = np.linalg.norm(cam_to[:3, 3] - cam_from[:3, 3])
        n = max(1, int(math.ceil(max(math.degrees(turn) / step_deg, shift / step_m))))
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix([cam_from[:3, :3], cam_to[:3, :3]]))
        q, path = np.asarray(q_from, float), []
        for s in np.linspace(0, 1, n + 1)[1:]:
            cam = np.eye(4)
            cam[:3, :3] = slerp([s]).as_matrix()[0]
            cam[:3, 3] = (1 - s) * cam_from[:3, 3] + s * cam_to[:3, 3]
            q, err = panda.ik_link6(cam @ np.linalg.inv(self.x), q)
            if err > 1e-4 or np.any(q < self.lo) or np.any(q > self.hi):
                return None, "out of reach within the joint limits (without folding the arm over)"
            path.append(q)
        reason = self.why_not(q)
        return (None, reason) if reason else (path, "")

    def waypoints(self, current, path) -> list | None:
        """Poses picked from `path` (ending at the view) so that every move from `current`
        is a clean segment; None if a step of the walk itself is not."""
        out, last = [], current
        for i, q in enumerate(path):
            nxt = path[i + 1] if i + 1 < len(path) else None
            if nxt is None or not self.segment_ok(last, nxt):
                if not self.segment_ok(last, q):
                    return None
                out.append(q)
                last = q
        return out

    def route(self, views):
        """Capture order. The capture tool starts at READY; the list starts at the look
        pose (READY -> look pose is the path the calibration already takes). Each view
        is reached along its camera walk, with waypoints from the walk where a direct
        joint move would be too long.

        views: [(label, value, path)]. Returns (poses, kinds, skipped): kinds[i] is
        the (label, value) of pose i, None for a waypoint.
        """
        poses, kinds, skipped, current = [self.q_look.copy()], [None], [], self.q_look
        for label, value, path in views:
            steps = self.waypoints(current, path)
            if steps is None:
                skipped.append(f"{label}: no clean joint path (a link would dip toward the board)")
                continue
            for q in steps[:-1]:
                poses.append(q)
                kinds.append(None)
            poses.append(steps[-1])
            kinds.append((label, value))
            current = steps[-1]
        return poses, kinds, skipped


def _plan_camera_test(test: Test, session: Session, args, cal, board, board_file):
    e = session.cfg["eye_in_hand"]
    p = SimpleNamespace(**e["plan"])
    view = _look_view(session, board_file)
    _, info = rimfit.read_camera_info(view)
    q_look = np.asarray(json.loads((view / "joints.json").read_text())["measured_joints"], float)
    K = rimfit.k_matrix(cal.intrinsics)
    planner = _Planner(board, cal.x, K, info["width"], info["height"], p, q_look,
                       args.max_step if args.max_step else p.max_step_deg)
    centre = (board @ CENTRE)[:3]
    look_cam = panda.fk(q_look, "panda_link6") @ cal.x
    look_dist = float(np.linalg.norm(look_cam[:3, 3] - centre))
    x_ref = look_cam[:3, 0]                        # roll 0 = the look pose's roll

    directions = {"top": board[:3, 2],
                  "look": (look_cam[:3, 3] - centre) / look_dist}
    order = [args.view] if args.view else ["look", "top"]
    if test.number == 1:
        values = args.rolls or list(DEFAULT_ROLLS)
        rolls = [None]
        make = lambda d, v, r: _camera_pose(centre, directions[d], args.distance or 0.50, x_ref, v)  # noqa: E731
        label = lambda v: f"roll {v:+.0f} deg"  # noqa: E731
    else:
        values = args.distances or list(DEFAULT_DISTANCES)
        # One roll for all distances: the reachable one that brings the board's long
        # side (x) closest to the image width, so the near views still see all of it.
        z = -board[:3, 2]
        x0 = x_ref - z * (x_ref @ z)
        aligned = math.degrees(math.atan2(np.cross(x0, board[:3, 0]) @ z, x0 @ board[:3, 0]))
        rolls = sorted(ROLL_CANDIDATES, key=lambda r: abs(math.remainder(math.radians(r - aligned), math.pi)))
        make = lambda d, v, r: _camera_pose(centre, directions[d], v, x_ref, r)  # noqa: E731
        label = lambda v: f"{v * 100:.0f} cm"  # noqa: E731

    report, best = [], None
    for direction in order:
        for roll in rolls:
            views, skipped, near, near_cam = [], [], q_look, look_cam
            for v in values:
                cam = make(direction, v, roll)
                path, reason = planner.follow(near_cam, near, cam)
                if path is None:
                    skipped.append(f"{label(v)}: {reason}")
                    continue
                views.append((label(v), float(v), path))
                near, near_cam = path[-1], cam
            count = len({round(v, 3) for _, v, _ in views})
            if best is None or count > best[0]:
                best = (count, direction, roll, views, skipped)
            if count == len(set(values)):
                break
        # Test 1 keeps the look direction when it works; test 2 wants the widest distance span.
        if best[0] >= 2 and (test.number == 1 or best[0] == len(set(values))):
            break
        report.append(f"camera direction '{direction}': only {best[0]} usable value(s) "
                      f"({'; '.join(best[4]) or '-'})")
    count, direction, roll, views, skipped = best
    if count < 2:
        raise PipelineError("could not plan this test: " + " | ".join(report)
                            + "\nTry --view look, other --rolls/--distances or --distance, or --max-step")
    report = []
    poses, kinds, unreachable = planner.route(views)
    skipped += unreachable
    if len({k[1] for k in kinds if k}) < 2:
        raise PipelineError("fewer than two views have a clean path: " + "; ".join(skipped))
    report.append(f"camera direction: {direction} ({'along the board normal' if direction == 'top' else 'as the look pose sees the board'}), "
                  f"look pose camera {look_dist * 100:.0f} cm from the board centre"
                  + (f"; camera roll {roll:+.0f} deg from the look pose's" if roll is not None else ""))
    report.append(f"{sum(1 for k in kinds if k)} views: " + ", ".join(k[0] for k in kinds if k)
                  + f" (+{kinds.count(None)} waypoint(s): the look pose first, then poses along the camera"
                    f" path that keep every joint move under {planner.max_step:g} deg)")
    report += [f"skipped {s}" for s in skipped]
    return poses, kinds, report, q_look, look_dist, direction


def _fit_captures(session: Session, cal, board, capture_dir: Path, out: Path, kinds) -> list[dict]:
    e = session.cfg["eye_in_hand"]
    camera = e["camera"]
    meta = load_yaml(capture_dir / "meta.yaml")["cameras"][camera]
    rimfit.load(session.cfg)
    views = []
    for pose in load_yaml(capture_dir / "poses.yaml")["poses"]:
        index = int(pose["index"])
        if index >= len(kinds) or kinds[index] is None:
            continue                                   # READY waypoint
        label, value = kinds[index]
        image = capture_dir / "images" / f"pose_{index:03d}_{camera}.png"
        if not image.exists():
            print(f"  {label}: no image, skipped")
            continue
        view = out / f"v{index:03d}"
        view.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, view / "color.png")
        rimfit.write_camera_info(view, cal.intrinsics, meta["width"], meta["height"], meta["frame_id"])
        q = np.asarray(pose["measured_joints"], float)
        result = rimfit.fit_from_pose(view, cal.intrinsics, cal.cam_board(q, board), cal.rim_bias)
        if result is None:
            print(f"  {label}: FIT FAILED")
            continue
        rimfit.overlay(view, cal.intrinsics, result["T_cam_board"], view / "overlay.jpg")
        cam_board = np.asarray(result["T_cam_board"])
        ok = result["mad_px"] <= e["max_mad_px"]
        print(f"  {label}: rim mad {result['mad_px']:.3f} px, camera-board {np.linalg.norm(cam_board[:3, 3]) * 100:.1f} cm"
              + ("" if ok else f"  (above max_mad_px {e['max_mad_px']}, excluded)"))
        if not ok:
            continue
        views.append({"label": label, "value": value, "index": index, "q": q.tolist(),
                      "mad_px": float(result["mad_px"]),
                      "camera_distance_m": float(np.linalg.norm(cam_board[:3, 3])),
                      "R_base_camera": cal.camera_in_base(q)[:3, :3].tolist(),
                      "T_base_board": cal.base_board(q, cam_board).tolist()})
    return views


def run_camera_test(test: Test, session: Session, args) -> Path:
    cal = _calibration(session)
    board, board_file = _board(session)
    poses, kinds, report, q_look, look_dist, direction = _plan_camera_test(test, session, args, cal, board, board_file)
    print(f"\n=== test {test.number}: {test.title} ===")
    for line in report:
        print("  " + line)
    run = _new_run(session, test)
    (run / "poses.yaml").write_text(yaml.safe_dump({
        "joint_names": panda.JOINT_NAMES,
        "poses": [{"joints": [round(float(v), 6) for v in q]} for q in poses]}))
    data = {"test": test.number, "title": test.title, "session": session.name,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "board_pose": str(board_file), "T_base_board_located": board.tolist(),
            "camera_direction": direction, "look_camera_distance_m": look_dist,
            "R_base_camera_look": cal.camera_in_base(q_look)[:3, :3].tolist(),
            "plan": report, "kinds": [list(k) if k else None for k in kinds], "views": []}
    _save(run, data)
    if args.plan_only:
        print(f"\n--plan-only: nothing moved. Planned poses: {run / 'poses.yaml'}")
        return run

    ctx = Context(session, session.cfg, args)
    eye_in_hand._confirm(ctx, f"The arm moves to READY, then through {len(poses)} poses above the board "
                              f"at speed scale {session.cfg['eye_in_hand']['speed_scale']}. The board must not move.")
    try:
        eye_in_hand._capture_tool(ctx, run / "poses.yaml", run / "capture")
    except PipelineError:
        if not (run / "capture" / "poses.yaml").exists():
            raise
        print("  some poses failed; analysing the ones recorded")
    print("\nfitting the board in every view (calibration held fixed)")
    data["views"] = _fit_captures(session, cal, board, run / "capture", run / "fit", kinds)
    _save(run, data)
    return run


# ------------------------------------------------------------ camera analysis
def _centre(T) -> np.ndarray:
    return (np.asarray(T) @ CENTRE)[:3]


def _view_table(views, board_ref) -> list[str]:
    ref = np.asarray(views[0]["T_base_board"])
    lines = [f"  {'view':>14s} {'board top centre in base [mm]':>32s}   {'vs first view [mm] x/y/z':>27s}"
             f"  {'rot':>6s} {'vs locate x/y/z':>24s}  mad"]
    for v in views:
        T = np.asarray(v["T_base_board"])
        c = _centre(T)
        lines.append(f"  {v['label']:>14s} {_mm(c):>32s}   {_mm(c - _centre(ref)):>27s}  "
                     f"{_rotation_deg(ref, T):5.2f}° {_mm(c - _centre(board_ref)):>24s}  {v['mad_px']:.2f}")
    return lines


def analyse_roll(data: dict) -> list[str]:
    views = data["views"]
    if len({round(v["value"]) for v in views}) < 2:
        return ["fewer than two different rolls fitted; nothing to compare"]
    lines = _view_table(views, np.asarray(data["T_base_board_located"]))
    # p_i = c + R_i[:, :2] t: a camera-frame in-plane error t moves the board
    # with the camera's roll; the true board centre c does not move.
    a = np.vstack([np.hstack([np.eye(3), np.asarray(v["R_base_camera"])[:, :2]]) for v in views])
    b = np.concatenate([_centre(v["T_base_board"]) for v in views])
    sol = np.linalg.lstsq(a, b, rcond=None)[0]
    t = sol[3:]
    residual = np.linalg.norm((a @ sol - b).reshape(-1, 3), axis=1)
    r_look = np.asarray(data["R_base_camera_look"])
    shift = r_look[:, :2] @ t
    spread = max(v["value"] for v in views) - min(v["value"] for v in views)
    pairs = [(v, w) for v in views for w in views
             if abs(abs(v["value"] - w["value"]) - 180) < 1]
    lines += ["",
              f"rolls fitted: {spread:.0f} deg apart" + ("" if spread >= 90 else
                                                          "  (less than 90 deg: the estimate below is weak)"),
              f"in-plane camera error t (camera x, y): {_mm(t)} mm, |t| = {np.linalg.norm(t) * 1e3:.2f} mm "
              f"(fit residual per view: max {residual.max() * 1e3:.2f} mm)",
              f"what it does to the located board (camera turned as at the look pose): {_mm(shift)} mm in base x/y/z"]
    if pairs:
        v, w = pairs[0]
        d = _centre(w["T_base_board"]) - _centre(v["T_base_board"])
        lines.append(f"measured between {v['label']} and {w['label']}: {_mm(d)} mm "
                     f"(|xy| {np.linalg.norm(d[:2]) * 1e3:.2f} mm; twice the in-plane error)")
    verdict = abs(shift[0]) * 1e3
    lines += ["", "reading:",
              f"  base-x shift from the camera side: {shift[0] * 1e3:+.2f} mm (the probe error is about -2..-3 mm)."]
    if np.linalg.norm(t) < 0.0005:
        lines.append("  < 0.5 mm: hand-eye translation and principal point are fine in this plane;"
                     " the -x error comes from elsewhere (tests 3-6).")
    elif verdict > 1.5 and shift[0] < 0:
        lines.append("  the camera side explains much of the -x error: a hand-eye translation error, or an"
                     " angular one (hand-eye rotation, principal point) of distance x angle. Test 2 tells them apart.")
    else:
        lines.append("  a camera-side error exists but it does not point along -x at the look pose;"
                     " it is not the main cause. Compare with test 2.")
    lines.append("  (t mixes a translation error with distance x angular error; at "
                 f"{np.mean([v['camera_distance_m'] for v in views]) * 100:.0f} cm, 1 mrad = "
                 f"{np.mean([v['camera_distance_m'] for v in views]):.2f} mm.)")
    return lines


def analyse_heights(data: dict) -> list[str]:
    views = data["views"]
    d = np.array([v["camera_distance_m"] for v in views])
    if np.ptp(d) < 0.05:
        return ["the fitted views span less than 5 cm of distance; nothing to compare"]
    lines = _view_table(views, np.asarray(data["T_base_board_located"]))
    p = np.array([_centre(v["T_base_board"]) for v in views])
    a = np.c_[np.ones_like(d), d]
    coef = np.linalg.lstsq(a, p, rcond=None)[0]     # rows: offset, slope (per m)
    slope = coef[1]
    residual = np.linalg.norm(p - a @ coef, axis=1)
    look = data["look_camera_distance_m"]
    lines += ["",
              f"camera distance {d.min() * 100:.0f}-{d.max() * 100:.0f} cm; board centre drift per 10 cm of "
              f"distance: {_mm(slope * 0.1)} mm in base x/y/z (fit residual max {residual.max() * 1e3:.2f} mm)",
              f"distance-proportional part at the look pose distance ({look * 100:.0f} cm): {_mm(slope * look)} mm",
              "", "reading:"]
    x_part = slope[0] * look * 1e3
    if abs(x_part) < 0.5:
        lines.append(f"  {x_part:+.2f} mm in x: no distance-dependent error worth the name. If the probe error is"
                     " still there it is a constant offset: hand-eye translation (test 1) or after the camera (3-6).")
    elif x_part < -1.0:
        lines.append(f"  {x_part:+.2f} mm in x grows with distance: an angular error (hand-eye rotation, principal"
                     " point cx/cy, sag) or scale. That explains much of the -x probe error.")
    else:
        lines.append(f"  {x_part:+.2f} mm in x: some angular error, but not the main -x cause.")
    lines.append("  Drift along the viewing direction (z for the 'top' view) is the focal-length / translation"
                 " scale term; sideways drift is rotation or principal point.")
    return lines


# ================================================================ touch tests
def _target(session: Session, holes: dict, name: str, tip_from: str | None):
    """(tool, tip source, true-frame tip target, nominal-frame board pose) for one hole."""
    if name not in holes:
        raise PipelineError(f"unknown hole {name!r}; `pipeline holes` lists them")
    board, _ = _board(session)
    # The same peg turn as `pipeline probe` would pick from here, so the target
    # (and its joint-offset correction) is the one the probe drives to.
    start = ros.current_joints(session.cfg)
    tool, _ = grasp.choose_turn(session.cfg, holes[name], name, board,
                                None if start is None else np.asarray(start))
    source = "CAD"
    if tip_from:
        tool, source = grasp.apply_pivot_tip(tool, grasp.load_pivot_tip(ros.host_path(session.cfg, tip_from)))
    nominal_file = eye_in_hand.hole_target(session, session.cfg, holes[name], name, tool)
    nominal = np.asarray(json.loads(nominal_file.read_text())["T_base_board"])
    return tool, source, (board @ np.r_[tool.point, SURFACE_Z, 1.0])[:3], nominal


def touch_hole(session: Session, holes: dict, name: str, tip_from: str | None, run: Path,
               label: str, instructions: str, reference_yaw: float | None = None) -> dict | None:
    """One hand-guided touch of a hole; the record, or None when the operator stopped."""
    cal = _calibration(session)
    board, board_file = _board(session)
    tool, source, target, nominal = _target(session, holes, name, tip_from)
    tag = f"{len(list(run.glob('touch_*.json'))) + 1:02d}"
    spec = run / f"touch_{tag}_spec.json"
    result = run / f"touch_{tag}.json"
    spec.write_text(json.dumps({
        "hole": name, "label": label, "instructions": instructions, "topic": FRANKA_STATES,
        "tip_offset_tcp_m": tool.tip_offset.tolist(),
        "target_nominal_m": (nominal @ np.r_[tool.point, SURFACE_Z, 1.0])[:3].tolist(),
        "board_rotation": nominal[:3, :3].tolist(),
        "reference_yaw_deg": reference_yaw}, indent=2))
    try:
        ros.run(session.cfg, ["python3", ros.container_path(session.cfg, TOUCH_NODE),
                              ros.container_path(session.cfg, spec), ros.container_path(session.cfg, result)],
                interactive=True, own=run)
    except PipelineError as error:
        if not result.exists():
            print(f"  no touch recorded ({error})")
            return None
    raw = json.loads(result.read_text())
    q = np.asarray(raw["q"], float)
    r = np.asarray(raw["R"], float)
    t = np.asarray(raw["t"], float)
    tip_nominal = r @ tool.tip_offset + t
    tcp_true = panda.fk(q + cal.dq, "panda_hand_tcp")
    tip_true = tcp_true[:3, :3] @ tool.tip_offset + tcp_true[:3, 3]
    error = target - tip_true
    tilt = math.degrees(math.acos(np.clip(-tcp_true[:3, 2] @ board[:3, 2], -1, 1)))
    record = {
        "hole": name, "label": label, "shape": tool.shape, "tip_source": source,
        "tip_offset_tcp_m": tool.tip_offset.tolist(), "hole_point_board_m": tool.point.tolist(),
        "q": q.tolist(), "O_T_EE_R": r.tolist(), "O_T_EE_t": t.tolist(),
        "joint_spread_rad": raw.get("joint_spread_rad"),
        "R_base_tcp_true": tcp_true[:3, :3].tolist(),
        "tip_true_m": tip_true.tolist(), "tip_nominal_m": tip_nominal.tolist(),
        "target_true_m": target.tolist(),
        "probe_error_base_m": error.tolist(),
        "probe_error_board_m": (board[:3, :3].T @ error).tolist(),
        "probe_error_no_joint_offsets_base_m": (target - tip_nominal).tolist(),
        "hand_yaw_deg": math.degrees(math.atan2(tcp_true[1, 0], tcp_true[0, 0])),
        "peg_tilt_deg": tilt,
        "T_base_board": board.tolist(), "board_pose": str(board_file),
        "time": raw.get("time"),
    }
    print(f"  {name} ({label}): probe error {_mm(error)} mm in base x/y/z, "
          f"{_mm(board[:3, :3].T @ error)} mm in board axes; hand yaw {record['hand_yaw_deg']:+.1f} deg, "
          f"peg tilt {tilt:.1f} deg")
    if tilt > 3.0:
        print("  the peg is tilted more than 3 deg: the touch is only as good as the tip offset along the peg")
    return record


TOUCH_HELP = ("Guide the held peg into the hole by hand (hold the guiding buttons on the wrist), let it\n"
              "seat by itself, release the buttons so the arm holds still, then press Enter. Keep the peg\n"
              "upright and do not push it sideways in the fingers: the grip, not your hand, sets the tip.\n"
              "Tip: grasp with more force for these tests, e.g. `pipeline grasp round --force 40`.")


def run_touch_test(test: Test, session: Session, args, hand_peg) -> Path:
    holes = load_yaml(session.cfg.path(session.cfg["holes"]))["holes"]
    _calibration(session)
    board, _ = _board(session)
    print(f"\n=== test {test.number}: {test.title} ===\n{TOUCH_HELP}\n")
    ros.check_robot_ready(session.cfg)

    if test.number in (4, 5):
        run = None if args.new else _latest_run(session, test)
        data = json.loads((run / "run.json").read_text()) if run else None
        if data is None:
            run = _new_run(session, test)
            data = {"test": test.number, "title": test.title, "session": session.name,
                    "created": datetime.datetime.now().isoformat(timespec="seconds"), "touches": []}
        else:
            print(f"continuing {run} ({len(data['touches'])} touches so far; --new starts over)")
        phases = list(dict.fromkeys(t["phase"] for t in data["touches"]))
        if phases:
            last = next(t for t in reversed(data["touches"]))
            moved = np.linalg.norm(_centre(last["T_base_board"]) - _centre(board)) * 1e3
            turned = _rotation_deg(np.asarray(last["T_base_board"]), board)
            print(f"board since phase '{last['phase']}': centre moved {moved:.1f} mm, turned {turned:.1f} deg")
            if moved < 5 and turned < 2:
                if input("The board pose is the same as in the last phase. Moved the board and ran "
                         "`pipeline locate --fresh`? Enter = touch anyway, q = stop > ").strip().lower() == "q":
                    return run
        default = {4: ["as placed", "turned 180 deg"], 5: ["in front", "left (+y)", "right (-y)"]}[test.number]
        suggestion = default[len(phases)] if len(phases) < len(default) else f"phase {len(phases) + 1}"
        phase = input(f"name of this board placement [{suggestion}] > ").strip() or suggestion
        names = args.holes or [DEFAULT_TOUCH_HOLE]
    elif test.number == 3:
        run = _new_run(session, test)
        data = {"test": 3, "title": test.title, "session": session.name,
                "created": datetime.datetime.now().isoformat(timespec="seconds"), "touches": []}
        phase = None
        names = [args.holes[0] if args.holes else DEFAULT_TOUCH_HOLE]
    else:
        run = _new_run(session, test)
        data = {"test": 6, "title": test.title, "session": session.name,
                "created": datetime.datetime.now().isoformat(timespec="seconds"), "touches": []}
        phase = None
        names = args.holes or list(DEFAULT_TOUCH_OFF)
        if len(names) < 3:
            raise PipelineError("test 6 needs at least 3 holes (4 is better), not all in one line")
    _save(run, data)

    held = None
    try:
        for name in names:
            if name not in holes:
                raise PipelineError(f"unknown hole {name!r}; `pipeline holes` lists them")
            shape = holes[name]["shape"]
            if shape != held:
                hand_peg(session.cfg, shape, held)
                held = shape
            labels = ["hand as usual", "hand turned ~180 deg about the peg"] if test.number == 3 \
                else ["hand as usual"]
            k = 0
            while True:
                label = labels[k] if k < len(labels) else f"hand turned, touch {k + 1}"
                ref = data["touches"][0]["hand_yaw_deg"] if (test.number == 3 and data["touches"]) else None
                text = f"touch {name}: {label}"
                if test.number == 3 and k > 0:
                    text += (" -- turn the hand about the peg axis, as far toward 180 deg as joint 7 allows "
                             "(90 deg also works); the live line shows the turn")
                record = touch_hole(session, holes, name, args.tip_from, run, label, text, ref)
                if record is None:
                    raise KeyboardInterrupt
                if phase:
                    record["phase"] = phase
                data["touches"].append(record)
                _save(run, data)
                k += 1
                if test.number != 3:
                    break
                if k >= 2 and input("another touch with a different turn? y = yes, Enter = done > ").strip().lower() != "y":
                    break
    except KeyboardInterrupt:
        print("\nstopped; the touches so far are kept")
    if test.number == 4 and len({t.get('phase') for t in data['touches']}) < 2:
        print("\nnext: turn the board 180 deg on the table, `pipeline locate --fresh`, then `pipeline diag 4` again")
    if test.number == 5 and len({t.get('phase') for t in data['touches']}) < 2:
        print("\nnext: move the board to y = +30 cm (and later -30 cm), keep its orientation to the base,\n"
              "`pipeline locate --fresh --here` with the arm jogged so the camera sees it, then `pipeline diag 5`")
    return run


# ------------------------------------------------------------- touch analysis
def _touch_table(touches) -> list[str]:
    lines = [f"  {'touch':28s} {'probe error base x/y/z [mm]':>27s}  {'board x/y/z [mm]':>27s}  {'yaw':>7s}  tilt"]
    for t in touches:
        name = f"{t['hole']} {t.get('phase') or t['label']}"[:28]
        lines.append(f"  {name:28s} {_mm(t['probe_error_base_m']):>27s}  {_mm(t['probe_error_board_m']):>27s}  "
                     f"{t['hand_yaw_deg']:+6.1f}°  {t['peg_tilt_deg']:.1f}°")
    return lines


def analyse_turn(data: dict) -> list[str]:
    touches = data["touches"]
    lines = _touch_table(touches)
    if len(touches) < 2:
        return lines + ["", "need at least two touches with different hand turns"]
    yaws = np.array([t["hand_yaw_deg"] for t in touches])
    span = max(abs(math.remainder(math.radians(a - b), 2 * math.pi)) for a in yaws for b in yaws)
    # e_i = b + R_i Delta: b does not turn with the hand, Delta = real minus model
    # tip in TCP axes does. Only Delta x/y are seen (the hand turns about vertical).
    a = np.vstack([np.hstack([np.eye(3), np.asarray(t["R_base_tcp_true"])[:, :2]]) for t in touches])
    e = np.concatenate([t["probe_error_base_m"] for t in touches])
    sol = np.linalg.lstsq(a, e, rcond=None)[0]
    b, delta = sol[:3], sol[3:]
    lines += ["",
              f"hand turned up to {math.degrees(span):.0f} deg between touches"
              + ("" if math.degrees(span) >= 90 else "  (less than 90 deg: weak separation)"),
              f"part that stays put (not the tool): {_mm(b)} mm in base x/y/z",
              f"part that turns with the hand: peg tip off by {_mm(delta)} mm in TCP x/y "
              f"(|{np.linalg.norm(delta) * 1e3:.2f}| mm; real tip minus model tip)",
              "", "reading:"]
    if np.linalg.norm(delta) > 0.7 * max(np.linalg.norm(b[:2]), 1e-4):
        lines.append("  most of the error turns with the hand: the TCP / peg tip is not where the model puts it."
                     " Redo the pivot calibration (10+ poses) and probe with --tip-from.")
    else:
        lines.append(f"  the error stays at {b[0] * 1e3:+.2f} mm in base x whatever the hand does: the TCP is fine;"
                     " look at the camera side (tests 1-2), the board (4) or the arm (5-6).")
    return lines


def analyse_moves(data: dict) -> list[str]:
    touches = data["touches"]
    lines = _touch_table(touches)
    phases = list(dict.fromkeys(t["phase"] for t in touches))
    if len(phases) < 2:
        return lines + ["", f"only one board placement ('{phases[0] if phases else '-'}') so far; "
                            "move the board, locate, and run the test again"]
    by_phase = {p: [t for t in touches if t["phase"] == p] for p in phases}
    mean = {p: (np.mean([t["probe_error_base_m"] for t in ts], axis=0),
                np.mean([t["probe_error_board_m"] for t in ts], axis=0),
                np.mean([t["tip_true_m"] for t in ts], axis=0)) for p, ts in by_phase.items()}
    lines += ["", f"  {'placement':16s} {'error base x/y/z [mm]':>27s}  {'board x/y/z':>27s}  "
                  f"{'toward base':>11s}  {'sideways':>8s}"]
    for p, (eb, ebo, tip) in mean.items():
        radial = tip[:2] / np.linalg.norm(tip[:2])
        side = np.array([-radial[1], radial[0]])
        lines.append(f"  {p[:16]:16s} {_mm(eb):>27s}  {_mm(ebo):>27s}  {-eb[:2] @ radial * 1e3:+10.2f}  "
                     f"{eb[:2] @ side * 1e3:+8.2f}")
    base = np.array([m[0] for m in mean.values()])
    board = np.array([m[1] for m in mean.values()])
    spread_base = np.linalg.norm(base[:, :2] - base[:, :2].mean(0), axis=1).max() * 1e3
    spread_board = np.linalg.norm(board[:, :2] - board[:, :2].mean(0), axis=1).max() * 1e3
    lines += ["", f"in-plane spread between placements: {spread_base:.2f} mm in base axes, "
                  f"{spread_board:.2f} mm in board axes", "", "reading:"]
    if data["test"] == 4:
        if spread_board < 0.6 * spread_base:
            lines.append("  the error turns with the board: the board model is off (hole positions vs the"
                         " fitted pattern, print scale, origin convention).")
        elif spread_base < 0.6 * spread_board:
            lines.append("  the error stays put in base axes when the board turns: the board model is fine.")
        else:
            lines.append("  no clear winner: both spreads are similar (repeat the touches, use a tighter hole).")
    else:
        tips = np.array([m[2] for m in mean.values()])
        radial = tips[:, :2] / np.linalg.norm(tips[:, :2], axis=1, keepdims=True)
        # e = k * (unit vector toward the base) vs e = constant in base axes
        toward = -radial
        k = float(np.sum(base[:, :2] * toward) / np.sum(toward * toward))
        res_radial = np.linalg.norm(base[:, :2] - k * toward, axis=1).max() * 1e3
        res_const = spread_base
        lines.append(f"  'points at the base' model ({k * 1e3:+.2f} mm toward the base): worst residual "
                     f"{res_radial:.2f} mm; 'constant in base axes': {res_const:.2f} mm")
        if res_radial < 0.6 * res_const:
            lines.append("  the error follows the direction to the base: arm kinematics (joint offsets,"
                         " link lengths, elbow/shoulder deflection).")
        elif res_const < 0.6 * res_radial:
            lines.append("  the error stays along the same base axis wherever the board is: a frame mix-up"
                         " (TF, base frame, a fixed offset in the chain), not kinematics.")
        else:
            lines.append("  no clear winner; touch more placements (the further apart, the better).")
    return lines


def _similarity_2d(src, dst):
    """dst ~ s R src + t (2D); returns (s, angle rad, t, residuals)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd(b.T @ a)
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    s = float(np.trace(np.diag(sv) @ d) / np.sum(a * a))
    t = mu_d - s * r @ mu_s
    return s, math.atan2(r[1, 0], r[0, 0]), t, np.linalg.norm(dst - (s * (r @ src.T).T + t), axis=1)


def analyse_touch_off(data: dict) -> list[str]:
    touches = data["touches"]
    lines = _touch_table(touches)
    if len(touches) < 3:
        return lines + ["", "need at least 3 touched holes"]
    board = np.asarray(touches[0]["T_base_board"])
    holes = np.array([t["hole_point_board_m"] for t in touches])
    # Touched tips in the (vision) board frame: the arm's measurement of the holes.
    measured = np.array([(np.linalg.inv(board) @ np.r_[t["tip_true_m"], 1.0])[:2] for t in touches])
    lines += ["", f"  {'pair':40s} {'drawing':>9s} {'arm':>9s} {'diff':>7s}  along base"]
    for i in range(len(touches)):
        for j in range(i + 1, len(touches)):
            dd = np.linalg.norm(holes[j] - holes[i]) * 1e3
            dm = np.linalg.norm(measured[j] - measured[i]) * 1e3
            direction = board[:2, :2] @ (holes[j] - holes[i])
            axis = "x" if abs(direction[0]) > abs(direction[1]) else "y"
            lines.append(f"  {touches[i]['hole'] + ' - ' + touches[j]['hole']:40s} {dd:8.2f}  {dm:8.2f}  "
                         f"{dm - dd:+6.2f}   mostly {axis}")
    if np.linalg.matrix_rank(holes - holes.mean(0), tol=1e-3) < 2:
        return lines + ["", "the holes lie on one line: distances only, no 2D fit (add a hole off the line)"]
    s, angle, t, res = _similarity_2d(holes, measured)
    shift_base = board[:3, :2] @ t
    lines += ["",
              f"best fit of the touched holes onto the drawing (in board axes): scale {(s - 1) * 1e3:+.2f} per mille, "
              f"rotation {math.degrees(angle) * 60:+.1f} arcmin, offset {_mm(t)} mm board x/y",
              f"  = offset {_mm(shift_base)} mm in base x/y/z between where the arm finds the holes and the vision"
              " board (probe error = minus this)",
              f"  residual per hole after the fit: max {res.max() * 1e3:.2f} mm", "", "reading:"]
    if abs(s - 1) * 1e3 > 3:
        lines.append(f"  the arm measures distances {(s - 1) * 1e3:+.1f} per mille off the drawing: kinematics"
                     " (or board print scale; test 4 tells which) in this part of the workspace.")
    else:
        lines.append("  distances agree: the arm is consistent across the board. A remaining common offset is"
                     " between the camera chain and the arm (tests 1-3), not a scale/kinematic error.")
    return lines


ANALYSES = {1: analyse_roll, 2: analyse_heights, 3: analyse_turn, 4: analyse_moves, 5: analyse_moves,
            6: analyse_touch_off}


def report(run: Path) -> None:
    data = json.loads((run / "run.json").read_text())
    if data.get("test") in (1, 2) and not data.get("views"):
        print(f"{run}: no fitted views (yet)")
        return
    lines = [f"test {data['test']}: {data['title']}  ({run})", ""] + ANALYSES[data["test"]](data)
    text = "\n".join(lines)
    (run / "report.txt").write_text(text + "\n")
    print("\n" + text + f"\n\nwritten to {run / 'report.txt'}")


# ======================================================================== menu
def menu() -> int:
    print("Tests for the constant -x probing error (see pipeline/DETAILS.md, `pipeline diag`):\n")
    for t in TESTS.values():
        print(f"  {t.number}  {t.title}\n     asks: {t.question}\n     needs: {t.needs}")
    print("\nSuggested order: 1, 2 (no touching), 3, 6, then 4 and 5 (they move the board).")
    while True:
        answer = input("test number (q = quit) > ").strip().lower()
        if answer == "q":
            raise KeyboardInterrupt
        if answer.isdigit() and int(answer) in TESTS:
            return int(answer)
        print("  1-6 or q")


def run(session: Session, args, hand_peg) -> None:
    number = args.test or menu()
    test = TESTS[number]
    if args.report is not None:
        chosen = _latest_run(session, test, args.report or None)
        if chosen is None:
            raise PipelineError(f"no test {number} run in {_test_dir(session, test)}")
        report(chosen)
        return
    if test.kind == "camera":
        out = run_camera_test(test, session, args)
        if args.plan_only:
            return
    else:
        out = run_touch_test(test, session, args, hand_peg)
    report(out)
