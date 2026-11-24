"""`pipeline extract --hole H`: pull the peg out of a hole with the empty hand (2026-11-23, for the campaign of
full-board insertions). No key presses.

1. where     the peg: insert's release record (<session>/campaign/state.yaml, or --from-run: the tip pose when the
             hand opened, lift_release.json) gives the hand pose exactly; without one, the vision hole (with the
             board correction of the found holes) and the CAD seat, mirrored for extract.mirrored_shapes (the real
             rectangle and L pegs are the CAD's mirror images).
2. approach  probe_board to extract.approach_height_mm above the hole (ROS), unless --here (the arm is still over
             the peg: the campaign right after the insert). robot.launch is then stopped.
3. pull      peg_search extract (libfranka, rl_manip robot_session): fingers open to the seat width + 2 x
             extract.open_margin_mm, joint moves over the peg and down to extract.hover_mm above it; then a Cartesian
             impedance stiff in z, soft in x, y, tilt and yaw (extract.stiffness): down until the fingers' roof
             touches the peg top (that sets the grasp height, as the fingers' README asks), press, grasp at the seat
             width (finger B pushes the peg into A's V, which pulls the soft hand onto the peg: the hand fits the
             peg, the peg stays put), pull straight up until the tip is out, then a joint move extract.lift_m up.
4. failure   in the hole (no peg felt, grasp, stuck pull): the hand opens and lifts, the peg stays where it was.

Nothing it measures goes into the found holes or the board correction: the pipeline learns the holes from
inserting only. Everything goes to sessions/<name>/extract/<date>_<time>/.
"""

from __future__ import annotations

import datetime
import json
import math
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from board_pipeline import grasp, holefit, insert, ros
from board_pipeline.config import Config, PipelineError, load_yaml

# peg_search's stage when it failed -> where the peg is afterwards (before this module's recovery)
STAGES_NOTHING_HELD = ("options", "connect", "posture", "open", "approach", "descend")
STAGES_IN_HOLE = ("press", "grasp", "settle", "pull")


def settings(session_cfg: Config) -> dict:
    data = session_cfg.data.get("extract")
    return data if data else Config.template()["extract"]


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def equivalent_rotation(r_peg: np.ndarray, r_ref: np.ndarray, period: float) -> np.ndarray:
    """r_peg (the tip frame, z along the peg) turned about its own z by the multiple of the peg's symmetry period
    nearest r_ref (period 0: the round peg, any turn: r_ref's). The same grasp of the same peg."""
    if period <= 0.0:
        rel = r_peg.T @ r_ref
        return r_peg @ _rz(math.atan2(rel[1, 0], rel[0, 0]))
    best, best_angle = r_peg, math.inf
    for k in range(max(1, round(2 * math.pi / period))):
        cand = r_peg @ _rz(k * period)
        angle = math.acos(max(-1.0, min(1.0, (np.trace(cand.T @ r_ref) - 1.0) / 2.0)))
        if angle < best_angle:
            best, best_angle = cand, angle
    return best


def pose_from_record(record: dict) -> np.ndarray:
    return np.asarray(record["tip_pose"], float).reshape(4, 4)


def record_from_run(run_dir: Path) -> dict:
    """The release record of an inserted run (its summary.yaml, else built from lift_release.json and plan.yaml)."""
    summary_file = run_dir / "summary.yaml"
    summary = load_yaml(summary_file) if summary_file.exists() else {}
    if summary.get("release"):
        return summary["release"]
    plan = load_yaml(run_dir / "plan.yaml")
    pose = json.loads((run_dir / "lift_release.json").read_text()).get("start_pose") \
        if (run_dir / "lift_release.json").exists() else None
    if not pose:
        raise PipelineError(f"{run_dir} has no release pose (lift_release.json from a peg_search with lift --result)")
    plane = summary.get("plane")
    board_z = float(plane["point"][2]) if plane else float(plan["hole_base_m"][2])
    return {"hole": plan["hole"], "shape": plan["shape"], "tip_pose": pose, "board_z_m": board_z,
            "tip_offset_tcp_m": [v / 1000.0 for v in plan["tip_offset_tcp_mm"]], "run": run_dir.name,
            "period_rad": None}


def _vision_board(session, data: dict, dq) -> np.ndarray:
    """The board in peg_search's frame (calibrated q + dq when joint offsets are used), with the board correction of
    the found holes (insert's, from inserting only)."""
    src = data.get("source") or {}
    board = np.asarray(src["T_base_board_true_frame"]) if dq is not None else np.asarray(data["T_base_board"])
    s = insert.settings(session.cfg)
    if s.get("board_correction", True):
        hole_xy = (board @ np.array([0.0, 0.0, data["board_surface_z_m"], 1.0]))[:2]
        entries = holefit.load_found(session.path)
        if not entries:
            fallback = Path(session.path) / "campaign" / "extract_found_holes.yaml"
            if fallback.exists():
                entries = list((yaml.safe_load(fallback.read_text()) or {}).get("holes") or [])
        corr = holefit.local_correction(entries, src.get("board_pose_sha256"),
                                        at_xy=hole_xy, **insert.local_options(s))
        if corr is not None:
            board = holefit.full_matrix(corr) @ board
            print(f"board correction from {corr['holes']} found hole(s) applied to the vision hole")
    return board


def run(session, args, probe_target, record: dict | None = None, here: bool = False,
        restart_launch: bool = True) -> dict:
    """Pull the peg out of args.hole. record: insert's release record of that hole (else the vision hole). here: the
    arm is already over the peg (no ROS approach; robot.launch may be stopped). Returns {"ok", "peg" ("hand",
    "hole", "unknown"), "out", "result", "error"}."""
    cfg = session.cfg
    s, e = insert.settings(cfg), settings(cfg)
    holes = load_yaml(cfg.path(cfg["holes"]))["holes"]
    if args.hole not in holes:
        raise PipelineError(f"unknown hole {args.hole!r}; `pipeline holes` lists them")
    shape = holes[args.hole]["shape"]
    if record is not None and record.get("hole") != args.hole:
        raise PipelineError(f"the release record is of {record.get('hole')}, not {args.hole}")
    mirror = record is None and shape in (e.get("mirrored_shapes") or [])
    dq = insert._joint_offsets(session, not args.no_joint_correction)
    if record is not None and "joint_offsets_rad" in record:
        dq = record["joint_offsets_rad"]  # the frame the release pose was measured in
    probe_argv = None
    if here:
        if record is None:
            raise PipelineError("--here needs a release record (the arm over the released peg)")
        tool = SimpleNamespace(tip_offset=np.asarray(record["tip_offset_tcp_m"], float))
        grasp_width = grasp.grasp_width_mm(cfg, shape)
    else:
        hand_travel = insert.travel_heights(s)[1]
        approach_mm = hand_travel * 1e3 if hand_travel is not None else float(e["approach_height_mm"])
        tool, _, target, probe_argv = probe_target(session, holes, args.hole, None, approach_mm,
                                                   0.0, joint_correction=not args.no_joint_correction, mirror=mirror)
        grasp_width = tool.grasp_width_mm
        data = json.loads(Path(target).read_text())
        if record is None and (data.get("source") or {}).get("T_base_board_true_frame") is None:
            dq = None  # insert's rule: offsets only with the calibrated-frame board
        board = _vision_board(session, data, dq)
        r_approach = board[:3, :3] @ tool.rotation
    if record is not None:
        pose = pose_from_record(record)
        if not here:
            # The same grasp, turned by the peg's symmetry to the wrist the approach chose. Only the record's tip
            # geometry (insert's tool) puts the hand where it was.
            period = record.get("period_rad")
            period = float(period) if period is not None else float(grasp.hole_tool(cfg, holes[args.hole], args.hole).period)
            pose[:3, :3] = equivalent_rotation(pose[:3, :3], r_approach, period)
            tool = SimpleNamespace(tip_offset=np.asarray(record["tip_offset_tcp_m"], float),
                                   grasp_width_mm=grasp_width)
        board_z = float(record["board_z_m"])
        vision_z = float(record.get("vision_board_z_m", board_z))
        top_guess = float(pose[2, 3])  # the tip when the hand opened: the peg may drop, not rise
        where = f"the release of run {record.get('run')} (tip {(top_guess - board_z) * 1e3:+.2f} mm from the board)"
    else:
        pose = np.eye(4)
        pose[:3, :3] = r_approach
        pose[:3, 3] = (board @ np.r_[tool.point, data["board_surface_z_m"], 1.0])[:3]
        board_z = float(pose[2, 3])
        vision_z = board_z
        top_guess = board_z
        where = "the vision hole" + (" (the mirrored CAD seat)" if mirror else "")
    hover_z = top_guess + float(e["hover_mm"]) / 1000.0
    min_z = board_z - float(e["max_depth_mm"]) / 1000.0
    open_width = grasp_width + 2.0 * float(e["open_margin_mm"])
    print(f"\nextract {args.hole} ({shape}): the peg at {where}, ({pose[0, 3] * 1e3:.2f}, {pose[1, 3] * 1e3:.2f}) mm; "
          f"board {board_z * 1e3:.2f} mm; descent from {hover_z * 1e3:.2f} to at most {min_z * 1e3:.2f} mm; fingers "
          f"{open_width:.1f} mm open, grasp {grasp_width:.2f} mm")

    now = datetime.datetime.now()
    out = Path(tempfile.mkdtemp(prefix="extract_dry_")) if args.dry_run else \
        session.path / "extract" / now.strftime("%Y-%m-%d_%H%M%S")
    run_dir = insert.CONTAINER_RUN_DIR
    peg_travel, hand_travel = insert.travel_heights(s)

    def extract_argv(grasp_pose: np.ndarray, name: str) -> list:
        return [s["robot_ip"], "extract", f"{run_dir}/{name}.csv", *insert._tool_and_load_args(s, tool, dq),
                *insert._free_speed_args(s),
                "--grasp-pose", *(f"{v:.8f}" for v in grasp_pose.reshape(-1)), *rest,
                "--result", f"{run_dir}/{name}.json", "--yes"]

    rest = [
            "--board-z", f"{board_z:.6f}", "--extract-hover-z", f"{hover_z:.6f}", "--extract-min-z", f"{min_z:.6f}",
            "--open-width", f"{open_width / 1000.0:.5f}", "--grasp-width", f"{grasp_width / 1000.0:.5f}",
            "--grasp-force", f"{float(e['grasp_force_n']):g}",
            "--grasp-eps-inner", f"{float(e['grasp_eps_inner_mm']) / 1000.0:g}",
            "--grasp-eps-outer", f"{float(e['grasp_eps_outer_mm']) / 1000.0:g}",
            "--extract-stiffness", *(f"{float(v):g}" for v in e["stiffness"]),
            *(["--extract-descent-stiffness", *(f"{float(v):g}" for v in e["descent_stiffness"])]
              if e.get("descent_stiffness") else []),
            *(["--extract-pull-stiffness", *(f"{float(v):g}" for v in e["pull_stiffness"])]
              if e.get("pull_stiffness") else []),
            *(["--extract-wiggle-deg", f"{float(e['wiggle_deg']):g}"] if e.get("wiggle_deg") is not None else []),
            *(["--extract-wiggle-hz", f"{float(e['wiggle_hz']):g}"] if e.get("wiggle_hz") else []),
            *(["--extract-wiggle-lag", f"{float(e['wiggle_lag_mm']) / 1000.0:g}"] if e.get("wiggle_lag_mm") else []),
            *(["--extract-stuck-time", f"{float(e['stuck_time_s']):g}"] if e.get("stuck_time_s") else []),
            "--extract-speed", f"{float(e['speed_mm_s']) / 1000.0:g}",
            "--extract-contact-force", f"{float(e['contact_force_n']):g}",
            "--extract-press", f"{float(e['press_n']):g}",
            "--extract-pull-speed", f"{float(e['pull_speed_mm_s']) / 1000.0:g}",
            "--extract-pull-max-lag", f"{float(e['pull_max_lag_mm']) / 1000.0:g}",
            "--extract-clear", f"{float(e['clear_mm']) / 1000.0:g}",
            # out of the hole, up to the travel height with a peg (insert.peg_travel_mm: where insert's probe starts)
            *(["--extract-lift-to-z", f"{vision_z + peg_travel:.5f}"] if peg_travel is not None else
              ["--extract-lift", f"{float(e['lift_m']):g}"]),
            *map(str, e.get("options") or []), *insert._backstop(s)]
    argv = extract_argv(pose, "extract")

    use_libfranka_travel = (not here and s.get("travel", "libfranka") == "libfranka"
                            and insert.extract_ik_seed(probe_argv) is not None)
    if args.dry_run:
        if use_libfranka_travel:
            seed = insert.extract_ik_seed(probe_argv)
            t_z = vision_z + (hand_travel if hand_travel is not None else float(e["approach_height_mm"]) / 1000.0)
            if not insert.move_to_hole(cfg, s, out, seed, "dry_move", travel_z=t_z, target_z=t_z,
                                       tool=tool, dq=dq, dry_run=True):
                shutil.rmtree(out, ignore_errors=True)
                raise PipelineError("peg_search move refused the options (above)")
        elif probe_argv is not None and ros.controller_states(cfg):
            ros.rosrun(cfg, "panda_calibration", "probe_board", *probe_argv, "--dry-run")
        code = insert.peg_search(cfg, s, out, [*argv, "--dry-run"])
        shutil.rmtree(out, ignore_errors=True)
        if code != 0:
            raise PipelineError("peg_search refused the options (above)")
        print(f"\nlooks right? `pipeline extract --hole {args.hole}` moves the arm")
        return {"ok": True, "peg": "hole", "out": None, "result": None, "error": None}

    out.mkdir(parents=True)
    (out / "plan.yaml").write_text(yaml.safe_dump({
        "session": session.name, "created": now.isoformat(timespec="seconds"), "hole": args.hole, "shape": shape,
        "source": "release" if record else "vision", "record": record, "mirrored_seat": mirror,
        "grasp_pose": pose.tolist(), "board_z_m": board_z, "hover_z_m": hover_z, "min_z_m": min_z,
        "open_width_mm": open_width, "grasp_width_mm": grasp_width, "settings": e,
        "joint_offsets_rad": dq, "peg_search": [str(v) for v in argv]}, sort_keys=False))
    insert.robot_session_dir(cfg, s)
    outcome = {"ok": False, "peg": "hole", "out": out, "result": None, "error": None}
    try:
        if not here:
            if use_libfranka_travel:
                if insert.launch_running(cfg, s):
                    insert.stop_launch(cfg, s)
                print("\n== move: over the peg (peg_search move) ==", flush=True)
                seed = insert.extract_ik_seed(probe_argv)
                t_z = vision_z + (hand_travel if hand_travel is not None else float(e["approach_height_mm"]) / 1000.0)
                if not insert.move_to_hole(cfg, s, out, seed, "approach", travel_z=t_z, target_z=t_z, tool=tool, dq=dq):
                    raise PipelineError("peg_search move over the peg failed")
            else:
                ros.check_robot_ready(cfg)
                print("\n== approach: over the peg (probe_board) ==")
                # the approach at the empty hand's travel height itself (1 mm over it, then down to it)
                ros.rosrun(cfg, "panda_calibration", "probe_board",
                           *insert.fast_probe_argv(cfg, s, probe_argv, out, 0.001 if hand_travel is not None else None),
                           "--stay", "--yes")
        if insert.launch_running(cfg, s):
            insert.stop_launch(cfg, s)
        tries = 1 + int(e.get("retries", 3))
        outcome["attempts"] = 0
        for k in range(1, tries + 1):
            name = "extract" if k == 1 else f"extract_{k}"
            print(f"\n== extract{f' (retry {k - 1}/{tries - 1})' if k > 1 else ''}: roof onto the peg, grasp, pull ==",
                  flush=True)
            outcome["attempts"] = k
            code = insert.peg_search(cfg, s, out, argv if k == 1 else extract_argv(pose, name))
            result_file = out / f"{name}.json"
            result = json.loads(result_file.read_text()) if result_file.exists() else \
                {"ok": False, "stage": "unknown", "error": f"peg_search exited {code} without a result"}
            outcome["result"] = result
            stage = result.get("stage")
            if code == 0 and result.get("ok") and result.get("held"):
                outcome.update(ok=True, peg="hand", error=None)
                print(f"\nextracted{f' on retry {k - 1}' if k > 1 else ''}: the roof touched the peg "
                      f"{result['contact_above_board'] * 1e3:+.2f} mm from the board, the fingers moved the hand "
                      f"{np.linalg.norm(result['seated_shift_xy']) * 1e3:.2f} mm and {result['seated_turn_deg']:+.2f} "
                      f"deg; hand {result['width'] * 1e3:.2f} mm ({result['width_off'] * 1e3:+.2f} from the seat); pull "
                      f"lag {result['pull_peak_lag_m'] * 1e3:.2f} mm, {result.get('pull_stalls', 0)} stall(s)")
                break
            outcome["error"] = result.get("error")
            print(f"\nextract failed ({stage}): {result.get('error')}")
            if stage not in STAGES_NOTHING_HELD and stage not in STAGES_IN_HOLE:
                # Out of the hole (the lift failed, or the hand lost the peg after the pull), or it never started.
                outcome["peg"] = "unknown" if stage in ("lift", "done", "unknown") else "hole"
                break
            # The peg in its hole (or never touched): let go of it, if held, and get the hand clear: just over the
            # hover for a retry (the next attempt starts there), insert.lift_m after the last.
            retry = k < tries and stage not in ("options", "connect")
            opened = insert.open_hand(cfg, s, out)
            end_z = (result.get("end_tip") or [None, None, None])[2]
            if retry and opened and end_z is not None:
                to_z = max(float(end_z) + 0.005, hover_z + float(e.get("retry_above_hover_mm", 5.0)) / 1000.0)
                print(f"\n== recovery: hand open, up to tip z {to_z * 1e3:.1f} mm for a retry ==")
                lifted = insert.robot_program(cfg, s, out, "peg_search", [
                    s["robot_ip"], "lift", f"{run_dir}/lift_retry_{k}.csv", "--to-z", f"{to_z:.5f}",
                    *insert._tool_and_load_args(s, tool, dq), *insert._free_speed_args(s),
                    *insert._backstop(s), "--yes"]) == 0
            else:
                retry = False
                print("\n== recovery: hand open, " + (f"up to the empty-hand travel height ({hand_travel * 1e3:g} mm) =="
                                                       if hand_travel is not None else f"lift {s['lift_m'] * 100:g} cm =="))
                lifted = opened and insert.lift(cfg, s, out, float(s["lift_m"]), "lift_recovery", tool=tool, dq=dq,
                                                to_z=vision_z + hand_travel if hand_travel is not None else None)
            outcome["peg"] = "hole" if lifted else "unknown"
            if not lifted:
                outcome["error"] = f"{outcome['error']}; the recovery (open, lift) failed"
                break
            if not retry:
                break
            seated = result.get("seated_tip")
            if seated:  # the fingers found the peg's axis there: the retry aims at it
                print(f"retry aims where the fingers seated on the peg: "
                      f"({(seated[0] - pose[0, 3]) * 1e3:+.2f}, {(seated[1] - pose[1, 3]) * 1e3:+.2f}) mm from before")
                pose = pose.copy()
                pose[0, 3], pose[1, 3] = float(seated[0]), float(seated[1])
    finally:
        (out / "outcome.yaml").write_text(yaml.safe_dump({k: (str(v) if isinstance(v, Path) else v)
                                                          for k, v in outcome.items()}, sort_keys=False))
        if restart_launch:
            insert.start_launch(cfg, s)
    return outcome


def cmd_extract(session, args, probe_target) -> None:
    record = None
    if args.from_run:
        run_dir = Path(args.from_run)
        if not run_dir.is_absolute():
            run_dir = session.path / "insert" / args.from_run
        record = record_from_run(run_dir)
        if record.get("hole") != args.hole:
            raise PipelineError(f"{run_dir.name} inserted {record.get('hole')}, not {args.hole}")
    elif not args.vision:
        from board_pipeline import campaign  # the campaign's peg record, when it has one for this hole
        record = campaign.record_for(session, args.hole)
    if args.here and record is None:
        raise PipelineError("--here needs a release record of the hole (--from-run, or the campaign's)")
    outcome = run(session, args, probe_target, record=record, here=args.here)
    if not outcome["ok"]:
        raise PipelineError(f"{args.hole}: not extracted ({outcome['error']}); the peg is "
                            f"{'in the hole' if outcome['peg'] == 'hole' else 'somewhere unknown'}; "
                            f"results in {outcome['out']}")
    if not args.dry_run:
        from board_pipeline import campaign
        campaign.note_extracted(session, args.hole)
        print(f"\ndone: the {holes_shape(session, args.hole)} peg is in the hand; results in {outcome['out']}")


def holes_shape(session, hole: str) -> str:
    return load_yaml(session.cfg.path(session.cfg["holes"]))["holes"][hole]["shape"]
