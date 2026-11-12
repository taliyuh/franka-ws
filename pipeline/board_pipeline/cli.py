"""`pipeline` command line. See pipeline/README.md."""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from board_pipeline import eye_in_hand, grasp, ros, stationary
from board_pipeline.config import SETUPS, Config, PipelineError, load_yaml
from board_pipeline.session import (BOARD_POSE, CALIBRATION, REUSED, Context, Session,
                                    run_stage, step_status)

STAGES = ("calibrate", "locate")


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


def cmd_probe(cfg: Config, args) -> None:
    session = Session.open(cfg, args.session)
    holes = _holes(session.cfg)
    if args.hole not in holes:
        raise PipelineError(f"unknown hole {args.hole!r}; `pipeline holes` lists them")
    hole = holes[args.hole]
    board = np.asarray(json.loads(session.require(BOARD_POSE, "run `pipeline locate` first")
                                  .read_text())["T_base_board"])
    # The wrist turn is measured from where the arm stands now (probe_board's
    # IK starts there); without a robot state, joint 7 is kept near mid-range.
    start = ros.current_joints(session.cfg)
    tool, candidates = grasp.choose_turn(session.cfg, hole, args.hole, board,
                                         None if start is None else np.asarray(start))
    _print_turns(tool, candidates, start)
    print(f"held peg: {tool.shape} in the V-block fingers (grasp width {tool.grasp_width_mm:.2f} mm), "
          f"peg yaw {math.degrees(tool.peg_yaw):.2f} deg on the board, "
          f"tip {np.round(tool.tip_offset * 1e3, 2)} mm from the TCP")
    target = _module(session.cfg.setup).hole_target(session, session.cfg, hole, args.hole, tool)
    manipulation = grasp.write_manipulation_config(
        session.cfg, tool, session.path / "targets" / f"{args.hole}_tool.yaml")
    dry = args.command == "check"
    argv = ["--board-pose", ros.container_path(session.cfg, target),
            "--offset", float(tool.point[0]), float(tool.point[1]), args.height / 1000.0,
            "--config", ros.container_path(session.cfg, session.cfg.path(session.cfg["robot"]["config"])),
            "--manipulation-config", ros.container_path(session.cfg, manipulation)]
    if dry:
        argv.append("--dry-run")
    else:
        ros.check_robot_ready(session.cfg)
        print("Enter approves the approach, Enter during the descent stops it, "
              "Enter again retracts. Keep the e-stop at hand.")
    ros.rosrun(session.cfg, "panda_calibration", "probe_board", *argv, interactive=not dry)
    if dry:
        print(f"\nlooks right? `pipeline probe --hole {args.hole}` moves the arm")


def cmd_grasp(cfg: Config, args) -> None:
    # The session's config snapshot if there is one, else the template: the
    # grasp width only needs holes.yaml and grasps.yaml, not a board pose.
    if args.session or Session.current_link(cfg).is_symlink():
        cfg = Session.open(cfg, args.session).cfg
    holes = _holes(cfg)
    if args.hole not in holes:
        raise PipelineError(f"unknown hole {args.hole!r}; `pipeline holes` lists them")
    tool = grasp.hole_tool(cfg, holes[args.hole], args.hole)
    print(f"closing the hand on the {tool.shape} peg: width {tool.grasp_width_mm:.2f} mm "
          f"+-{args.epsilon:g} mm, force {args.force:g} N")
    ros.rosrun(cfg, "panda_calibration", "grasp_peg", "--width", tool.grasp_width_mm / 1000.0,
               "--epsilon", args.epsilon / 1000.0, "--force", args.force)


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Camera calibration, board localisation and hole probing for the Franka Panda.",
        epilog="Run on the host. Typical run: new -> calibrate -> locate -> grasp -> check -> probe "
               "(robot and cameras launched by hand in the container)")
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
        p.set_defaults(func=cmd_stage)

    for command, text in (("check", "dry run: validate the target and IK, no motion"),
                          ("probe", "move the held peg's tip onto a hole (guarded descent)")):
        p = sub.add_parser(command, help=text)
        p.add_argument("--hole", required=True, help="e.g. round_c3mm; `pipeline holes` lists them")
        p.add_argument("--height", type=float, default=0.0, metavar="MM",
                       help="peg tip height above the board surface, mm (default 0)")
        p.set_defaults(func=cmd_probe)

    p = sub.add_parser("grasp", help="close the hand on the peg for a hole (width from config/grasps.yaml)")
    p.add_argument("--hole", required=True, help="e.g. round_c1mm; `pipeline holes` lists them")
    p.add_argument("--force", type=float, default=20.0, metavar="N",
                   help="grip force, N (default 20; the Franka Hand allows 70 continuously)")
    p.add_argument("--epsilon", type=float, default=1.0, metavar="MM",
                   help="accepted deviation from the grasp width, mm (default 1)")
    p.set_defaults(func=cmd_grasp)
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
