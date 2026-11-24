"""`pipeline` command line. See pipeline/README.md and pipeline/DETAILS.md."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from board_pipeline import campaign, diagnose, extract, eye_in_hand, grasp, insert, panda, ros, stationary
from board_pipeline.config import SETUPS, Config, PipelineError, load_yaml
from board_pipeline.session import (BOARD_POSE, CALIBRATION, REUSED, Context, Session,
                                    run_stage, step_status)

STAGES = ("calibrate", "locate")
LIMIT_WARNING_RAD = 0.1    # warn when a planned joint is this close to its limit


def _module(setup: str):
    return eye_in_hand if setup == "eye_in_hand" else stationary


def _steps(setup: str, stage: str):
    return _module(setup).make_steps(stage)


def _holes(cfg: Config) -> dict:
    return load_yaml(cfg.path(cfg["holes"]))["holes"]


# ------------------------------------------------------------------ commands
def cmd_new(cfg: Config, args) -> None:
    source = Session.open(cfg, args.calib_from) if args.calib_from else None
    session = Session.create(cfg, args.setup, args.name, source)
    print(f"created session {session.name} ({args.setup}) and made it current")
    print(f"  {session.path}")
    if source:
        print(f"  calibration reused from {source.name}; next: `pipeline locate`")
    else:
        print("  next: launch the robot and camera(s) in the container, then `pipeline calibrate`")


def cmd_use(cfg: Config, args) -> None:
    session = Session.open(cfg, args.name)
    session.make_current()
    print(f"current session: {session.name} ({session.cfg.setup})")


def cmd_sessions(cfg: Config, args) -> None:
    link = Session.current_link(cfg)
    current = link.resolve().name if link.is_symlink() else None
    found = sorted(p for p in cfg.sessions_dir.glob("*") if not p.is_symlink() and (p / "config.yaml").exists()) \
        if cfg.sessions_dir.exists() else []
    if not found:
        print("no sessions yet; `pipeline new --setup eye_in_hand|stationary`")
    for path in found:
        session = Session(path)
        products = [n for n in (CALIBRATION, BOARD_POSE) if (path / n).exists()]
        mark = "*" if path.name == current else " "
        print(f"{mark} {path.name:32s} {session.cfg.setup:12s} {', '.join(products) or '-'}")


def cmd_status(cfg: Config, args) -> None:
    session = Session.open(cfg, args.session)
    print(f"session {session.name}  setup {session.cfg.setup}\n  {session.path}")
    reused = session.path / REUSED
    for stage in STAGES:
        print(f"\n{stage}:")
        if stage == "calibrate" and reused.exists() and not (session.path / stage).exists():
            print(f"  calibration reused from {reused.read_text().strip()}")
            continue
        for index, step in enumerate(_steps(session.cfg.setup, stage), 1):
            print(f"  {index}. {step.name:9s} {step_status(session.path / stage, index, step)}")
    print()
    for name in (CALIBRATION, BOARD_POSE):
        path = session.path / name
        print(f"{name:17s} {'present' if path.exists() else 'missing'}")
    calibration = session.path / CALIBRATION
    if calibration.exists():
        quality = json.loads(calibration.read_text()).get("quality", {})
        if "loo_median_mm" in quality:
            print(f"  calibration leave-one-out: median {quality['loo_median_mm']:.3f} mm, "
                  f"p90 {quality['loo_p90_mm']:.3f} mm")


def cmd_stage(cfg: Config, args) -> None:
    session = Session.open(cfg, args.session)
    setup = session.cfg.setup
    if args.command == "locate" and not (session.path / CALIBRATION).exists():
        raise PipelineError(f"{session.name} has no {CALIBRATION}; run `pipeline calibrate`, or start "
                            "a session with `pipeline new --setup ... --calib-from <session>`")
    if args.command == "calibrate":
        stale = ("locate", CALIBRATION, BOARD_POSE, "targets", REUSED)
    else:
        stale = (BOARD_POSE, "targets")
    print(f"session {session.name} ({setup})")
    _base_calibration_note(session, args)
    context = Context(session, session.cfg, args)
    finished = run_stage(context, args.command, _steps(setup, args.command), args.fresh,
                         args.from_step, on_rerun=lambda: session.invalidate(*stale),
                         until=args.until)
    if not finished:
        return
    nxt = "pipeline check --hole <name>"
    if args.command == "calibrate":
        nxt = ("pipeline check --hole <name>` if the board has not moved, else `pipeline locate"
               if (session.path / BOARD_POSE).exists() else "pipeline locate")
    print(f"\n{args.command} finished. Next: `{nxt}`")


def _base_calibration_note(session: Session, args) -> None:
    """Say where calibrate's camera parameters and joint offsets come from, and when a --fit flag
    would not reach the step it changes."""
    if args.command != "calibrate" or session.cfg.setup != "eye_in_hand":
        return
    base = session.cfg["eye_in_hand"].get("base_calibration") or {}
    for flag, key, step, what in (("fit_intrinsics", "intrinsics", "fit", "camera parameters"),
                                  ("fit_joint_offsets", "joint_offsets_rad", "solve", "joint offsets")):
        fitting = getattr(args, flag, False)
        if key not in base:
            source = "fitted from the board views (this session's config has no base_calibration)"
        elif fitting:
            source = f"fitted from the board views (--{flag.replace('_', '-')})"
        else:
            source = f"fixed from base_calibration ({base.get(key.replace('_rad', '') + '_source', 'config')})"
        print(f"  {what}: {source}")
        steps = [s.name for s in _steps(session.cfg.setup, "calibrate")]
        index = steps.index(step) + 1
        done = step_status(session.path / "calibrate", index, _steps(session.cfg.setup, "calibrate")[index - 1])
        rerun = args.fresh or (args.from_step in steps[:index])
        if fitting and done.startswith("done") and not rerun:
            print(f"  note: `{step}` is already done, so --{flag.replace('_', '-')} changes nothing; "
                  f"add --from {step}")


def cmd_holes(cfg: Config, args) -> None:
    holes = _holes(cfg)
    board = None
    if args.session or Session.current_link(cfg).is_symlink():
        try:
            path = Session.open(cfg, args.session).path / BOARD_POSE
            if path.exists():
                board = np.asarray(json.loads(path.read_text())["T_base_board"])
        except PipelineError:
            pass
    print(f"{'hole':20s} {'board x':>9s} {'board y':>9s} {'yaw':>8s}"
          + ("   base x/y/z (mm, from board_pose.json)" if board is not None else ""))
    for name, hole in holes.items():
        line = f"{name:20s} {hole['x_m'] * 1e3:8.2f}mm {hole['y_m'] * 1e3:8.2f}mm {hole['yaw_deg']:7.2f}°"
        if board is not None:
            point = board @ np.array([hole["x_m"], hole["y_m"], 0.025, 1.0])
            line += "   " + " ".join(f"{v * 1e3:8.2f}" for v in point[:3])
        print(line)


def _print_turns(tool, candidates, start) -> None:
    """One line per symmetric turn of the peg: joint 7 at the hole, the chosen one marked."""
    shape_turns = "any turn" if tool.period == 0 else \
        "no turn" if len(candidates) == 1 else f"turns every {math.degrees(tool.period):g} deg"
    origin = f"joint 7 now {start[6]:+.2f} rad" if start is not None else \
        "no robot state, keeping joint 7 near mid-range"
    print(f"{tool.shape} peg: {shape_turns}; {origin}")
    if tool.period == 0:
        candidates = [c for c in candidates if c[0].turn == tool.turn]
    for option, q in candidates:
        mark = "->" if option.turn == tool.turn else "  "
        j7 = "unreachable" if q is None else \
            f"j7 {q[6]:+.2f} rad ({grasp.wrist_margin(q[6]):.2f} rad from its limit)" \
            + ("" if start is None else f", turn {q[6] - start[6]:+.2f} rad")
        print(f"  {mark} turn {math.degrees(option.turn):+7.1f} deg: {j7}")


def _planned_joints(target, tool):
    """The joints the arm is meant to have at the hole, as commanded (nominal) angles.

    eye_in_hand writes them to the target file (the joint-offset correction is
    only valid there); otherwise they are the ones choose_turn predicted.
    """
    planned = json.loads(target.read_text()).get("source", {}).get("q_command_rad")
    return np.asarray(planned, float) if planned is not None else tool.joints


def _probe_target(session: Session, holes: dict, name: str, tip_from: str | None, height_mm: float,
                  turn_deg: float = 0.0, joint_correction: bool = True, mirror: bool = False):
    """Write one hole's target and tool files; returns (tool, tip source, target file, probe_board argv).
    mirror: the peg is the CAD's mirror image (grasp.mirror_seat; extract.mirrored_shapes)."""
    if name not in holes:
        raise PipelineError(f"unknown hole {name!r}; `pipeline holes` lists them")
    hole = holes[name]
    board = np.asarray(json.loads(session.require(BOARD_POSE, "run `pipeline locate` first")
                                  .read_text())["T_base_board"])
    # The wrist turn is measured from where the arm stands now; without a
    # robot state, joint 7 is kept near mid-range.
    start = ros.current_joints(session.cfg)
    tool, candidates = grasp.choose_turn(session.cfg, hole, name, board,
                                         None if start is None else np.asarray(start), mirror=mirror)
    _print_turns(tool, candidates, start)
    if turn_deg:
        tool = grasp.turn_further(session.cfg, hole, name, tool, board, turn_deg)
    tool, tip_source = grasp.resolve_tip(tool, tip_from, session.path, lambda p: ros.host_path(session.cfg, p))
    print(f"held peg: {tool.shape} in the V-block fingers (grasp width {tool.grasp_width_mm:.2f} mm), "
          f"peg yaw {math.degrees(tool.peg_yaw):.2f} deg on the board, "
          f"tip {np.round(tool.tip_offset * 1e3, 2)} mm from the TCP ({tip_source})")
    target = _module(session.cfg.setup).hole_target(session, session.cfg, hole, name, tool,
                                                    joint_correction=joint_correction)
    manipulation = grasp.write_manipulation_config(
        session.cfg, tool, session.path / "targets" / f"{name}_tool.yaml")
    argv = ["--board-pose", ros.container_path(session.cfg, target),
            "--offset", float(tool.point[0]), float(tool.point[1]), height_mm / 1000.0,
            "--config", ros.container_path(session.cfg, session.cfg.path(session.cfg["robot"]["config"])),
            "--manipulation-config", ros.container_path(session.cfg, manipulation)]
    # probe_board starts its IK at the planned joints and refuses a solution
    # in another posture: MoveIt, seeded at the current joints, has reached
    # holes with joint 1 on its limit and the shoulder flipped, where neither
    # the wrist margin nor the joint-offset correction computed here holds.
    planned = _planned_joints(target, tool)
    if planned is not None:
        margin = np.minimum(planned - panda.LOWER, panda.UPPER - planned)
        for index in np.flatnonzero(margin < LIMIT_WARNING_RAD):
            print(f"warning: the planned posture has joint {index + 1} only {margin[index]:.2f} rad "
                  "from its limit; start from a pose nearer the look pose if the move stops there")
        argv += ["--ik-seed", *(float(v) for v in planned)]
    return tool, tip_source, target, argv


def cmd_probe(cfg: Config, args) -> None:
    session = Session.open(cfg, args.session)
    _, _, _, argv = _probe_target(session, _holes(session.cfg), args.hole, args.tip_from, args.height,
                                  args.turn, joint_correction=not args.no_joint_correction)
    if args.command == "check":
        argv.append("--dry-run")
    elif args.command == "tip":
        # Only reads the robot state: no controller needed, so no readiness check.
        argv += ["--where", "--watch"] if args.watch else ["--where"]
    else:
        ros.check_robot_ready(session.cfg)
        print("Enter approves the approach, Enter during the descent stops it, "
              "Enter again retracts. Keep the e-stop at hand.")
    interactive = args.command == "probe" or (args.command == "tip" and args.watch)
    ros.rosrun(session.cfg, "panda_calibration", "probe_board", *argv, interactive=interactive)
    if args.command == "check":
        print(f"\nlooks right? `pipeline probe --hole {args.hole}` moves the arm")


def cmd_insert(cfg: Config, args) -> None:
    insert.run(Session.open(cfg, args.session), args, _probe_target)


def cmd_extract(cfg: Config, args) -> None:
    extract.cmd_extract(Session.open(cfg, args.session), args, _probe_target)


def cmd_campaign(cfg: Config, args) -> None:
    session = Session.open(cfg, args.session)
    if args.action == "start":
        campaign.cmd_start(session, args)
    elif args.action == "run":
        campaign.cmd_run(session, args, _probe_target)
    elif args.action == "peg":
        campaign.cmd_peg(session, args)
    else:
        campaign.cmd_status(session, args)


def cmd_found(cfg: Config, args) -> None:
    insert.cmd_found(Session.open(cfg, args.session), args)


def cmd_grasp(cfg: Config, args) -> None:
    # The session's config snapshot if there is one, else the template: the
    # grasp width only needs grasps.yaml, not a board pose.
    if args.session or Session.current_link(cfg).is_symlink():
        cfg = Session.open(cfg, args.session).cfg
    name = args.shape or args.hole
    if not name:
        raise PipelineError("which peg? e.g. `pipeline grasp round` "
                            "(round, hexagon, triangle, rectangle or l_shape)")
    holes = _holes(cfg)
    shape = holes[name]["shape"] if name in holes else name  # a hole name means its peg
    width = grasp.grasp_width_mm(cfg, shape)
    if args.epsilon is not None:
        print(f"closing the hand on the {shape} peg: width {width:.2f} mm "
              f"+-{args.epsilon:g} mm, force {args.force:g} N")
        ros.rosrun(cfg, "panda_calibration", "grasp_peg", "--width", f"{width / 1000.0:.6f}",
                   "--epsilon", args.epsilon / 1000.0, "--force", args.force)
        return
    # Pegs sit deeper in the printed V than the CAD width says, and the finger reading
    # drifts: close until the fingers clamp and accept anything up to the CAD width + 1 mm.
    print(f"closing the hand on the {shape} peg until it clamps (accepted up to {width + 1.0:.2f} mm, "
          f"CAD {width:.2f} mm), force {args.force:g} N")
    ros.rosrun(cfg, "panda_calibration", "grasp_peg", "--width", f"{width / 1000.0:.6f}",
               "--until-contact", "--force", args.force)


def cmd_release(cfg: Config, args) -> None:
    print(f"opening the hand to {args.width:g} mm")
    ros.rosrun(cfg, "panda_calibration", "release_peg", "--width", args.width / 1000.0)


def cmd_repeatability(cfg: Config, args) -> None:
    session_path = None
    if args.session or Session.current_link(cfg).is_symlink():
        opened = Session.open(cfg, args.session)
        cfg, session_path = opened.cfg, opened.path
    if args.resume:
        runs = sorted(path.parent for path in (cfg.workspace / "repeatability").glob("*/run.yaml"))
        out = ros.host_path(cfg, args.resume) if args.resume != "latest" else (runs[-1] if runs else None)
        if out is None or not (out / "run.yaml").exists():
            raise PipelineError(f"no run to resume: {out or 'repeatability/ has no run.yaml'}")
        print(f"resuming {out}")
        argv = ["--resume", ros.container_path(cfg, out)] + (["--no-grasp"] if args.no_grasp else [])
    else:
        holes = _holes(cfg)
        name = next((n for n, hole in holes.items() if hole["shape"] == args.shape), None)
        if name is None:
            raise PipelineError(f"no hole for peg shape {args.shape!r} in {cfg['holes']}")
        # Any hole of the shape will do: the tip offset and the grasp width depend on the peg only.
        tool = grasp.hole_tool(cfg, holes[name], name)
        tool, tip_source = grasp.resolve_tip(tool, args.tip_from, session_path, lambda p: ros.host_path(cfg, p))
        print(f"held peg: {tool.shape}, tip {np.round(tool.tip_offset * 1e3, 2)} mm from the TCP ({tip_source})")
        out = cfg.workspace / "repeatability" / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        # Fixed-point: argparse takes "-2e-18" for an option, not for a negative number.
        argv = ["--out", ros.container_path(cfg, out), "--tip-offset", *(f"{v:.6f}" for v in tool.tip_offset),
                "--tip-source", tip_source.split(";")[0], "--shape", tool.shape,
                "--speed-scale", args.speed_scale, "--dwell", args.dwell]
        if not args.no_grasp:
            argv += ["--grasp-width", tool.grasp_width_mm / 1000.0]
    argv += ["--distances", *args.distances, "--repeats", args.repeats]
    ros.check_robot_ready(cfg)
    try:
        ros.rosrun(cfg, "panda_calibration", "measure_repeatability", *argv, interactive=True)
    finally:
        print(f"\nresults (kept after an interrupted run too): {out}")


# The holes `pipeline accuracy` probes by default: the tightest and the loosest of two shapes.
ACCURACY_HOLES = ("round_c0p03mm", "round_c3mm", "rectangle_c0p03mm", "rectangle_c3mm")
ACCURACY_PLAN = "plan.yaml"
RELEASE_WIDTH_MM = 30.0   # wider than every grasp


def _accuracy_run(session: Session, chosen: str) -> Path:
    """The newest run under the session's accuracy/ for "latest", else the folder given."""
    if chosen == "latest":
        runs = sorted(path.parent for path in (session.path / "accuracy").glob(f"*/{ACCURACY_PLAN}"))
        if not runs:
            raise PipelineError(f"no run in {session.path / 'accuracy'}; start one with `pipeline accuracy`")
        return runs[-1]
    run = ros.host_path(session.cfg, chosen)
    if not (run / ACCURACY_PLAN).exists():
        raise PipelineError(f"{run} is not a `pipeline accuracy` run (it has no {ACCURACY_PLAN})")
    return run


def _save_plan(run: Path, plan: dict) -> None:
    (run / ACCURACY_PLAN).write_text("# `pipeline accuracy`: the holes in order; probe_board writes one "
                                     "<hole>.yaml per probed hole\n" + yaml.safe_dump(plan, sort_keys=False))


def _correction_mm(target: Path, tool, height_mm: float):
    """How far the joint-offset correction moved the tip target, mm; None when there is none."""
    data = json.loads(Path(target).read_text())
    true = data.get("source", {}).get("T_base_board_true_frame")
    if true is None:
        return None
    point = np.r_[tool.point, data["board_surface_z_m"] + height_mm / 1000.0, 1.0]
    shift = (np.asarray(data["T_base_board"]) - np.asarray(true)) @ point
    return [round(float(v) * 1000.0, 3) for v in shift[:3]]


def _hand_peg(cfg: Config, shape: str, held: str | None) -> None:
    """Get the peg of this shape into the fingers; PipelineError when the operator stops."""
    width = grasp.grasp_width_mm(cfg, shape)
    print(f"\nthe next holes need the {shape} peg" + (f"; the hand holds the {held} peg" if held else ""))
    while True:
        answer = input(f"Enter = the {shape} peg is in the fingers, g = put it in now (opens the hand, "
                       "then closes it on the peg), q = stop > ").strip().lower()
        if answer == "q":
            raise PipelineError("stopped before the next hole")
        if answer == "":
            return
        if answer != "g":
            print("  Enter, g or q")
            continue
        try:
            input("Hold the peg that is in the fingers, if any: it drops when the hand opens. "
                  "Enter = open the hand > ")
            ros.rosrun(cfg, "panda_calibration", "release_peg", "--width", RELEASE_WIDTH_MM / 1000.0)
            input(f"Put the {shape} peg in the V-block fingers and hold it there. Enter = close the hand > ")
            ros.rosrun(cfg, "panda_calibration", "grasp_peg", "--width", f"{width / 1000.0:.6f}", "--until-contact")
            return
        except PipelineError as error:
            print(f"  {error}. g = try again, Enter = the peg is held anyway")


def _probe_holes(session: Session, holes: dict, run: Path, plan: dict) -> None:
    held = None
    total = len(plan["holes"])
    for index, entry in enumerate(plan["holes"], 1):
        name = entry["name"]
        record = run / f"{name}.yaml"
        status = load_yaml(record).get("status") if record.exists() else None
        if status == "done":
            print(f"\n{index}/{total} {name}: probed earlier, skipped")
            continue
        print(f"\n=== {index}/{total}: {name} ({entry['shape']} peg, clearance {entry['clearance_mm']:g} mm) ===")
        if status:
            print(f"the earlier try ended {status}; probing it again (its record is replaced)")
        if entry["shape"] != held:
            _hand_peg(session.cfg, entry["shape"], held)
            held = entry["shape"]
        tool, tip_source, target, argv = _probe_target(session, holes, name, plan["tip_from"], plan["height_mm"],
                                                       joint_correction=plan.get("joint_correction", True))
        entry.update(tip_offset_tcp_mm=[round(float(v) * 1000.0, 3) for v in tool.tip_offset],
                     tip_source=tip_source.split(";")[0],
                     correction_mm=_correction_mm(target, tool, plan["height_mm"]), target=str(target))
        _save_plan(run, plan)
        ros.rosrun(session.cfg, "panda_calibration", "probe_board", *argv,
                   "--record", ros.container_path(session.cfg, record), interactive=True, own=run)


def _plot_accuracy(cfg: Config, run: Path) -> None:
    ros.rosrun(cfg, "panda_calibration", "plot_probe_accuracy", ros.container_path(cfg, run), own=run)


def cmd_accuracy(cfg: Config, args) -> None:
    session = Session.open(cfg, args.session)
    if args.plot:
        run = _accuracy_run(session, args.plot)
        _plot_accuracy(session.cfg, run)
        print(f"\nresults: {run}")
        return
    holes = _holes(session.cfg)
    if args.resume:
        if args.holes:
            raise PipelineError("--resume continues the saved list of holes; give no holes with it")
        run = _accuracy_run(session, args.resume)
        plan = load_yaml(run / ACCURACY_PLAN)
        print(f"resuming {run} (its --height, --tip-from and joint correction)")
    else:
        names = args.holes or list(ACCURACY_HOLES)
        unknown = [name for name in names if name not in holes]
        if unknown:
            raise PipelineError(f"unknown hole(s) {', '.join(unknown)}; `pipeline holes` lists them")
        now = datetime.datetime.now()
        run = session.path / "accuracy" / now.strftime("%Y-%m-%d_%H%M%S")
        plan = {"session": session.name, "created": now.isoformat(timespec="seconds"),
                "height_mm": args.height, "tip_from": args.tip_from,
                "joint_correction": not args.no_joint_correction,
                "holes": [{"name": name, "shape": holes[name]["shape"],
                           "clearance_mm": holes[name]["clearance_mm"]} for name in names]}
        run.mkdir(parents=True)
        _save_plan(run, plan)
    ros.check_robot_ready(session.cfg)
    print(f"holes: {', '.join(entry['name'] for entry in plan['holes'])}\n"
          "Each hole: Enter approves the move above it; the tip then goes down in 3 mm steps (Enter stops\n"
          "it) and the arm holds at the bottom for your photos. Type what you see, Enter retracts.\n"
          "The moves between holes go straight in joint space, 2 cm above the board. Keep the e-stop at hand.")
    try:
        _probe_holes(session, holes, run, plan)
    except PipelineError as error:
        raise PipelineError(f"{error}\n`pipeline accuracy --resume` probes the holes not done yet") from None
    except KeyboardInterrupt:
        raise PipelineError("interrupted; `pipeline accuracy --resume` probes the holes not done yet") from None
    finally:
        if any((run / f"{entry['name']}.yaml").exists() for entry in plan["holes"]):
            try:
                _plot_accuracy(session.cfg, run)
            except PipelineError as error:
                print(f"no figure: {error}", file=sys.stderr)
        print(f"\nresults: {run}")


def cmd_diag(cfg: Config, args) -> None:
    diagnose.run(Session.open(cfg, args.session), args, _hand_peg)


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Camera calibration, board localisation and hole probing for the Franka Panda.",
        epilog="Run on the host. Typical run: new -> calibrate -> locate -> grasp -> check -> probe "
               "-> release, or grasp -> insert (robot and cameras launched by hand in the container)")
    parser.add_argument("--session", help="session to use instead of the current one")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("new", help="create a session and make it current")
    p.add_argument("name", nargs="?", help="default: <date>_<setup>")
    p.add_argument("--setup", choices=SETUPS, required=True)
    p.add_argument("--calib-from", metavar="SESSION",
                   help="reuse that session's calibration (cameras not moved since)")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("use", help="make another session current")
    p.add_argument("name")
    p.set_defaults(func=cmd_use)

    sub.add_parser("sessions", help="list sessions (* = current)").set_defaults(func=cmd_sessions)
    sub.add_parser("status", help="show which steps of the session are done").set_defaults(func=cmd_status)
    sub.add_parser("holes", help="list hole names and coordinates").set_defaults(func=cmd_holes)

    for stage, text in (("calibrate", "calibrate the camera(s) against the robot"),
                        ("locate", "measure the board pose in the robot base frame")):
        p = sub.add_parser(stage, help=text,
                           description=f"{text}. Rerunning continues at the first unfinished step.")
        p.add_argument("--fresh", action="store_true", help=f"discard all {stage} results and start over")
        p.add_argument("--from", dest="from_step", metavar="STEP",
                       help="redo this step and everything after it (see `pipeline status`)")
        p.add_argument("--until", metavar="STEP", help="stop after this step (e.g. to inspect it)")
        p.add_argument("--here", action="store_true",
                       help="eye_in_hand: use the current arm pose instead of the look pose")
        p.add_argument("--yes", action="store_true", help="do not ask before moving the arm")
        if stage == "calibrate":
            p.add_argument("--fit-intrinsics", action="store_true",
                           help="eye_in_hand: estimate fx fy cx cy from the board views (bundle) instead of "
                                "using eye_in_hand.base_calibration's camera parameters")
            p.add_argument("--fit-joint-offsets", action="store_true",
                           help="eye_in_hand: estimate joint 2-5 offsets from the board views (model term "
                                "\"off\") instead of using eye_in_hand.base_calibration's offsets")
        p.set_defaults(func=cmd_stage)

    for command, text in (("check", "dry run: validate the target and IK, no motion"),
                          ("probe", "move the held peg's tip onto a hole (guarded descent)"),
                          ("tip", "show where the held peg's tip is relative to a hole, no motion")):
        p = sub.add_parser(command, help=text)
        p.add_argument("--hole", required=True, help="e.g. round_c3mm; `pipeline holes` lists them")
        p.add_argument("--height", type=float, default=0.0, metavar="MM",
                       help="peg tip height above the board surface, mm (default 0)")
        p.add_argument("--tip-from", metavar="FILE",
                       help="use the peg tip measured by `rosrun panda_calibration pivot_calibration` "
                            "(only for the peg shape it was measured on) instead of the CAD tip; cad: the CAD tip "
                            "(default: the session's tips/<shape>.yaml if any, else CAD)")
        p.add_argument("--turn", type=float, default=0.0, metavar="DEG",
                       help="turn the hand this much further about the peg axis than the automatic choice "
                            "(a multiple of the peg's symmetry; any angle for the round peg). "
                            "`pipeline diag 3` is the measured version")
        p.add_argument("--no-joint-correction", action="store_true",
                       help="drive to the board pose as calibrated, without the per-hole joint-offset shift "
                            "(eye_in_hand; compare with the corrected landing)")
        if command == "tip":
            p.add_argument("--watch", action="store_true",
                           help="keep printing twice a second until Ctrl-C (also while hand-guiding)")
        p.set_defaults(func=cmd_probe)

    p = sub.add_parser(
        "insert", help="put the held peg into a hole: probe approach, up to 5 tilt searches, release, lift",
        description="The probe's approach to a few mm above the hole (ROS), then up to --attempts runs of the tilt "
                    "search (rl_manip robot_session peg_search tilt-slide, libfranka, the ROS controller stopped), "
                    "each aimed at the best belief so far (an earlier hold or dip); inserted: open the hand and "
                    "lift; not inserted: lift with the peg. Settings: the config's insert section. DETAILS.md.")
    p.add_argument("--hole", required=True, help="e.g. hexagon_c1mm; `pipeline holes` lists them")
    p.add_argument("--height", type=float, default=None, metavar="MM",
                   help="tip height above the vision board where the probe stops (default insert.start_height_mm)")
    p.add_argument("--attempts", type=int, default=None, help="tilt-search runs at most (default insert.attempts)")
    p.add_argument("--pattern", choices=("lines", "spiral"), default=None,
                   help="search pattern (default insert.pattern; lines is the robot-tested one)")
    p.add_argument("--slide-dir", type=float, default=None, metavar="DEG",
                   help="preferred slide direction in the base frame (default insert.slide_dir_base_deg)")
    p.add_argument("--tip-from", metavar="FILE", help="the peg tip: a pivot_calibration file, or cad (default: the session's tips/<shape>.yaml if any, else CAD)")
    p.add_argument("--turn", type=float, default=0.0, metavar="DEG",
                   help="turn the hand further about the peg axis (a multiple of the peg's symmetry), as probe")
    p.add_argument("--no-joint-correction", action="store_true",
                   help="aim at the board pose as calibrated, without the per-hole joint-offset shift")
    p.add_argument("--here", action="store_true",
                   help="skip the probe's approach: the tip is already within the approach above the board")
    p.add_argument("--method", choices=["plunge", "peck", "tilt"],
                   help="plunge (default, insert.method): one position-controlled plunge, the board touched beside "
                        "the hole, then the tilt search; peck: a grid of plunges; tilt: the tilt search only")
    p.add_argument("--touches", choices=["auto", "0", "1", "3"], default=None,
                   help="board touches after a plunge that did not insert (insert.board_touches): 3 the local plane; "
                        "1 this grasp's height on the board height model's slope; 0 the model only; auto 1 once the "
                        "model knows the board's slope, else 3")
    p.add_argument("--no-board-correction", action="store_true",
                   help="aim at the hole as vision has it, without the correction fitted to the holes found so far "
                        "(<session>/insert/found_holes.yaml)")
    p.add_argument("--no-centre", action="store_true",
                   help="inserted: do not touch the hole walls before the release (the seated tip is recorded)")
    p.add_argument("--search", choices=["grid", "belief"], default=None,
                   help="how later attempts are aimed: grid (default, insert.search): a probability grid of the hole "
                        "from every attempt's log, lines planned on it; belief: the last hold or dip")
    p.add_argument("--dry-run", action="store_true",
                   help="validate the probe target and peg_search's options; no motion, no robot connection")
    p.set_defaults(func=cmd_insert)

    p = sub.add_parser(
        "extract", help="pull a peg out of a hole with the empty hand: roof onto the peg top, grasp, compliant pull",
        description="Where the peg is: the campaign's release record of the hole (--from-run: an insert run's), else "
                    "the vision hole. The probe's approach over it (ROS), then peg_search extract (libfranka, "
                    "robot.launch stopped): a Cartesian impedance stiff in z and soft in x, y, tilt and yaw descends "
                    "until the fingers' roof touches the peg top, grasps (the fingers move the soft hand onto the "
                    "peg), pulls it straight out and lifts. Settings: the config's extract section.")
    p.add_argument("--hole", required=True, help="the hole the peg is in")
    p.add_argument("--from-run", metavar="RUN",
                   help="the insert run (sessions/<name>/insert/RUN) that released the peg there: its hand pose")
    p.add_argument("--vision", action="store_true", help="aim at the vision hole even if the campaign has a record")
    p.add_argument("--here", action="store_true",
                   help="the arm is still over the released peg (right after its insert): no ROS approach")
    p.add_argument("--no-joint-correction", action="store_true",
                   help="aim at the board pose as calibrated, without the per-hole joint-offset shift")
    p.add_argument("--dry-run", action="store_true", help="validate the target and peg_search's options; no motion")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser(
        "campaign", help="full-board insertions, unattended: every peg through its holes, extract in between",
        description="start: every peg in its campaign.start_clearance hole, the hand empty. run: each round takes "
                    "every shape through campaign.clearances (extract, insert, ...); a peg that does not go in is "
                    "parked in its campaign.park_clearance hole. Resumable: state.yaml says where every peg is. "
                    "status: where it is and the results so far. peg: tell it where a peg is after you moved it.")
    csub = p.add_subparsers(dest="action", required=True, metavar="ACTION")
    c = csub.add_parser("start", help="a new campaign (the pegs in their start holes)")
    c.add_argument("--rounds", type=int, default=None, help="full-board rounds (default campaign.rounds)")
    c.add_argument("--shapes", nargs="+", default=None, help="pegs in order (default campaign.shapes)")
    c.add_argument("--fresh", action="store_true", help="replace an existing campaign")
    c = csub.add_parser("run", help="run (or go on with) the campaign")
    c.add_argument("--max-inserts", type=int, default=None, metavar="N", help="stop after N inserts")
    c.add_argument("--no-board-correction", action="store_true",
                   help="every insert aims at the vision hole (insert --no-board-correction)")
    c.add_argument("--dry-run", action="store_true", help="list the inserts left; no motion")
    csub.add_parser("status", help="where the pegs are, and the results so far")
    c = csub.add_parser("peg", help="tell the campaign where a peg is (after moving it by hand, or a stop)")
    c.add_argument("shape")
    c.add_argument("--at", required=True, metavar="HOLE", help="its hole, or hand, or unknown")
    c.add_argument("--from-run", metavar="RUN", help="the insert run that released it there (its hand pose)")
    c.add_argument("--step", nargs=3, metavar=("ROUND", "SHAPE", "CLEARANCE"),
                   help="also go on from this insert (e.g. 3 hexagon c1mm)")
    p.set_defaults(func=cmd_campaign)

    p = sub.add_parser("found", help="the holes insert found, and the board correction they give",
                       description="Lists <session>/insert/found_holes.yaml and the board correction fitted to the "
                                   "holes of the current board pose (what `pipeline insert` applies).")
    p.add_argument("--import-runs", action="store_true",
                   help="add the inserted runs of insert/ not in the record yet (their seated tip; runs after the "
                        "board pose was last written only)")
    p.set_defaults(func=cmd_found)

    p = sub.add_parser("grasp", help="close the hand on a peg (width from config/grasps.yaml)")
    p.add_argument("shape", nargs="?", metavar="SHAPE",
                   help="peg shape: round, hexagon, triangle, rectangle or l_shape")
    p.add_argument("--hole", metavar="NAME",
                   help="instead of SHAPE: a shape, or a hole name (round_c1mm) meaning its peg")
    p.add_argument("--force", type=float, default=20.0, metavar="N",
                   help="grip force, N (default 20; the Franka Hand allows 70 continuously)")
    p.add_argument("--epsilon", type=float, default=None, metavar="MM",
                   help="instead of closing until contact, require the CAD width +- MM (the old check)")
    p.set_defaults(func=cmd_grasp)

    p = sub.add_parser("release", help="open the hand and let go of the peg (it drops if not held)")
    p.add_argument("--width", type=float, default=30.0, metavar="MM",
                   help="open the fingers to this width, mm (default 30, wider than every grasp; max 80)")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("repeatability",
                       help="teach points by hand, return to them from home, plot joint and tip errors "
                            "(no compensation)")
    p.add_argument("--shape", default="round", help="held peg (default round)")
    p.add_argument("--tip-from", metavar="FILE", help="the peg tip: a pivot_calibration file, or cad (default: the session's tips/<shape>.yaml if any, else CAD)")
    p.add_argument("--distances", nargs="+", type=float, default=[25.0, 50.0, 75.0], metavar="CM",
                   help="about how far from the base each point is, one per point (default 25 50 75)")
    p.add_argument("--repeats", type=int, default=3, help="repeats per point (default 3)")
    p.add_argument("--speed-scale", type=float, default=0.3,
                   help="fraction of motion.max_joint_speed (default 0.3)")
    p.add_argument("--dwell", type=float, default=1.0,
                   help="seconds to wait after the arm has settled, before measuring (default 1)")
    p.add_argument("--no-grasp", action="store_true", help="skip the step that closes the hand on the peg")
    p.add_argument("--resume", nargs="?", const="latest", metavar="DIR",
                   help="finish an interrupted run's unfinished points (default: the newest run)")
    p.set_defaults(func=cmd_repeatability)

    p = sub.add_parser("accuracy",
                       help="probe several holes in a row: record target vs measured position and joints "
                            "at every move, hold at the bottom for photos, plot the errors")
    p.add_argument("holes", nargs="*", metavar="HOLE",
                   help=f"holes in order (default: {' '.join(ACCURACY_HOLES)}); "
                        "when the peg shape changes it asks you to swap the peg")
    p.add_argument("--height", type=float, default=0.0, metavar="MM",
                   help="where the descent ends: tip height above the board surface, mm "
                        "(default 0: the tip at the surface, in the mouth of the hole)")
    p.add_argument("--tip-from", metavar="FILE", help="the peg tip: a pivot_calibration file, or cad (default: the session's tips/<shape>.yaml if any, else CAD)")
    p.add_argument("--no-joint-correction", action="store_true",
                   help="drive to the board pose as calibrated, without the per-hole joint-offset shift "
                        "(eye_in_hand; compare with the corrected landing)")
    p.add_argument("--resume", nargs="?", const="latest", metavar="DIR",
                   help="probe the holes an interrupted run did not finish (default: the newest run)")
    p.add_argument("--plot", nargs="?", const="latest", metavar="DIR",
                   help="only draw the figure and the table of a run again (default: the newest run)")
    p.set_defaults(func=cmd_accuracy)

    p = sub.add_parser(
        "diag", help="tests that find where a constant probe error comes from (no number: a menu)",
        description="Six diagnostic tests (pipeline/DETAILS.md, `pipeline diag`): 1 camera roll, 2 heights "
                    "(vision only, the arm moves); 3 tool turn, 4 board turn, 5 board aside, 6 touch-off "
                    "(hand-guided touches of holes). Without a number it lists them and asks.")
    p.add_argument("test", nargs="?", type=int, choices=sorted(diagnose.TESTS), metavar="N",
                   help="test number 1-6")
    p.add_argument("--report", nargs="?", const="", metavar="DIR",
                   help="only print the analysis of the newest run of test N (or of DIR)")
    camera = p.add_argument_group("tests 1 and 2 (camera)")
    camera.add_argument("--rolls", nargs="+", type=float, metavar="DEG",
                        help=f"test 1: camera rolls about its optical axis, deg "
                             f"(default {' '.join(f'{v:g}' for v in diagnose.DEFAULT_ROLLS)}; unreachable ones are skipped)")
    camera.add_argument("--distance", type=float, metavar="M",
                        help="test 1: camera distance from the board centre, m (default 0.50)")
    camera.add_argument("--distances", nargs="+", type=float, metavar="M",
                        help=f"test 2: camera distances from the board centre, m "
                             f"(default {' '.join(f'{v:g}' for v in diagnose.DEFAULT_DISTANCES)})")
    camera.add_argument("--view", choices=("top", "look"),
                        help="camera direction: look = as the look pose sees the board (the view `locate` "
                             "uses), top = along the board normal (default: look, else top)")
    camera.add_argument("--max-step", type=float, metavar="DEG",
                        help="largest single-joint move between poses (default plan.max_step_deg)")
    camera.add_argument("--plan-only", action="store_true", help="plan and print the poses, do not move")
    camera.add_argument("--yes", action="store_true", help="do not ask before moving the arm")
    touch = p.add_argument_group("tests 3-6 (hand-guided touches)")
    touch.add_argument("--holes", nargs="+", metavar="HOLE",
                       help=f"holes to touch (tests 3-5 default {diagnose.DEFAULT_TOUCH_HOLE}; test 6 default "
                            f"{' '.join(diagnose.DEFAULT_TOUCH_OFF)})")
    touch.add_argument("--tip-from", metavar="FILE", help="the peg tip: a pivot_calibration file, or cad (default: the session's tips/<shape>.yaml if any, else CAD)")
    touch.add_argument("--new", action="store_true",
                       help="tests 4 and 5: start a new series instead of adding a board placement to the last one")
    p.set_defaults(func=cmd_diag)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(Config.template(), args)
    except PipelineError as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; rerun the same command to continue", file=sys.stderr)
        return 130
    return 0
