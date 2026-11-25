"""`pipeline insert --hole H`: put the held peg into the hole with the tilt search. No key presses: every step
is decided from what the robot measures.

0. correct   the hole as vision sees it moved by the board correction fitted to the holes found so far
             (holefit: <session>/insert/found_holes.yaml, the current board pose's; --no-board-correction).
1. probe     the probe's approach (ROS, probe_board --stay --yes) to insert.start_height_mm above that hole; the
             arm holds there.
2. hand over robot.launch is stopped (insert.robot_launch): the robot takes one FCI client, and peg_search
             connecting killed franka_control (2026-11-21).
3. attempts  up to insert.attempts runs of rl_manip robot_session's `peg_search tilt-slide --yes` (libfranka,
             its own container):
               --plunge       attempt 1: straight down over the hole first, compliant, until it feels contact:
                              15 mm below the vision board is inserted; 3 mm or more below it the peg is in the
                              hole mouth (the hold presses it in); higher is the board (back up, then the search);
               --hole-xy      the hole as vision sees it (the search's disc);
               --located-xy   the best belief so far: an earlier hold (the tip on the hole axis), else a dip;
               --board-plane  the plane the first attempt probed (the same posture rule: arm angle 0);
               --lift-at-end  a run that does not insert lifts back to its start height for the next one.
               --board-z      vision's board top: a descent's contact counts only within 2.5 mm above it (or the
                              probed plane), with the tip stopped (2026-11-21: friction lag "found" the board in the air).
             A run that fails (one of peg_search's guards) is followed by `peg_search lift` straight up or down back
             to the start height (short descents: the free-air friction lag grows with them), then the next attempt.
             A failed posture move does not use up an attempt (the later runs skip it); a run that cannot connect, or
             a lift that fails, ends the attempts.
   With --search grid (insert.search, the default with lines) every attempt's lines are planned on the hole's
             probability grid (beliefgrid: --hole-xy its mean, --line-offsets in order), updated from each
             attempt's result and 1 kHz log; the error disc is insert.error_m (2 mm).
3b. centre   inserted, before the release: `peg_search centre` touches the hole walls; the hole centre fitted to
             the contacts (else the seated tip) is recorded for the board correction (--no-centre).
4. finish    inserted: the hand opens (robot_session gripper) and `peg_search lift` raises it insert.lift_m;
             not inserted: lift with the peg. Then robot.launch is started again in the background (its log in
             <workspace>/log/robot_launch/), and insert waits for the trajectory controller and set_load (NE_T_EE, load).

Everything goes to sessions/<name>/insert/<date>_<time>/: plan.yaml, attempt_<k>.csv (peg_search's 1 kHz log),
attempt_<k>.json (its result and belief), lift_*.csv, summary.yaml.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np
import yaml

from board_pipeline import beliefgrid, boardheight, holefit, ros
from board_pipeline.config import Config, PipelineError, load_yaml
from board_pipeline.session import sha256

CONTAINER_RUN_DIR = "/insert"   # the run folder inside the peg_search container
ROBOT_CONTAINER = "peg_search_robot"
ROBOT_IMAGE = "peg-search-libfranka-0.9.2"
TCP_FROM_FLANGE_M = 0.1034      # panda_hand_tcp along the flange z (Franka Hand)
_ASSUME_LAUNCH_STOPPED = False


def settings(session_cfg: Config) -> dict:
    """The session's insert section, else the template's (sessions from before 2026-11-21 have none)."""
    data = session_cfg.data.get("insert")
    return data if data else Config.template()["insert"]


def leads_table(cfg: Config) -> dict:
    path = cfg.workspace / "pipeline" / "config" / "tilt_leads.yaml"
    return load_yaml(path)["holes"]


def hole_leads(cfg: Config, holes: dict, name: str) -> tuple[str, dict]:
    """The tilt_leads.yaml entry of the hole, or (not simulated) of its shape's nearest clearance."""
    table = leads_table(cfg)
    if name in table:
        return name, table[name]
    shape, clearance = holes[name]["shape"], holes[name]["clearance_mm"]
    same = [n for n in table if holes.get(n, {}).get("shape") == shape]
    if not same:
        raise PipelineError(f"config/tilt_leads.yaml has no {shape} hole (tools/extract_tilt_leads.py)")
    nearest = min(same, key=lambda n: abs(math.log(holes[n]["clearance_mm"] / clearance)))
    print(f"note: {name} is not in the simulation's table; using {nearest}'s search geometry")
    return nearest, table[nearest]


def choose_lead(entry: dict, hole_yaw_base: float, preferred_deg: float, s: dict | None = None,
                error_m: float | None = None) -> tuple[dict, float]:
    """Among the leads that dip: the fewest lines for the error disc (line_plan; with s and error_m), then the
    simulation's group order (group 0: its best lead and its symmetric copies), then the slide nearest the preferred
    base direction (2026-11-21: that direction alone decided among group 0 before). Returns (lead, slide direction in
    base, rad)."""
    best = None
    for lead in entry["leads"]:
        if lead.get("dips") is False:
            continue
        angle = hole_yaw_base + math.radians(lead["d_deg"])
        gap = abs(math.remainder(angle - math.radians(preferred_deg), 2 * math.pi))
        lines = line_plan(lead, s, error_m)["lines"] if s is not None and error_m is not None else 0
        key = (lines, lead.get("group", 0), gap)
        if best is None or key < best[0]:
            best = (key, lead, angle)
    if best is None:
        raise PipelineError("no lead of this hole dips in the simulation (tilt_leads.yaml)")
    return best[1], math.remainder(best[2], 2 * math.pi)


def lead_kind(lead: dict) -> str:
    return lead.get("kind") or ("corner" if str(lead["lead"]).startswith("corner") else "flat")


def _require_region(lead: dict) -> None:
    missing = [k for k in ("near_m", "band_m", "along_m") if k not in lead]
    if missing:
        raise PipelineError(f"config/tilt_leads.yaml lacks {', '.join(missing)} (from before 2026-11-21): regenerate it "
                            "with tools/extract_tilt_leads.py")


def line_plan(lead: dict, s: dict, error_m: float) -> dict:
    """The lines for an error disc of error_m (2026-11-21).

    spacing: the table's (band fraction x the effective dip band); for a flat lead at most insert.max_line_spacing_m
    (robot-tested at 1 mm, hexagon 2026-11-21: a flat's band shrinks with the peg's yaw error), for a corner lead at
    most insert.max_corner_line_spacing_m (null: none; a vertex dips anywhere over the hole, whatever the yaw).
    centre_m: the lines' centre across the slide from the believed hole: the effective band's middle less the lead's
    offset across (the L-shape's corner runs 8.3 mm beside the tip's line: with its band about the hole centre, line
    0 covered hole errors -2.3..+19 mm). lines: 1 + 2k, line k covering spacing / 2 either side of k x spacing, out to
    error_m (tilt_search's 1 + 2 ceil(error / spacing) counted one ring more)."""
    _require_region(lead)
    cap = s.get("max_line_spacing_m") if lead_kind(lead) == "flat" else s.get("max_corner_line_spacing_m")
    spacing = float(lead["line_spacing_m"])
    if cap is not None:
        spacing = min(spacing, float(cap))
    band = lead.get("eff_band_m") or lead["band_m"]
    centre = 0.5 * (float(band[0]) + float(band[1])) - float(lead["lead_across_m"])
    rings = max(0, math.ceil((error_m - 0.5 * spacing) / spacing - 1e-9))
    return {"spacing_m": spacing, "centre_m": centre, "lines": 1 + 2 * rings}


def round_lead(entry: dict, preferred_deg: float) -> tuple[dict, float]:
    """The round hole: any direction; slide along the preferred one."""
    return entry["leads"][0], math.radians(preferred_deg)


# ---------------------------------------------------------------- belief
def dip_hole(dip_lead, slide_dir, lead: dict, prior_xy, prior_radius: float) -> tuple[np.ndarray, float]:
    """The hole centre from a dip (2026-11-21) and the half width of its uncertainty across the slide.

    The dip region (tilt_leads.yaml) holds the lead positions, from the hole centre, at which the lead dips: a lead
    sliding along d enters it, and dips, along_m[0] along d from the centre (the near rim plus the footprint), at an
    offset across d (a = z x d) somewhere within band_m. So the centre is along_m[0] behind the dip lead along d and,
    across, within [lead.a - band hi, lead.a - band lo], intersected with the prior disc's (prior_xy, prior_radius).
    Until 2026-11-21 the centre was the dip lead + peg_radius along d: a corner lead's peg_radius is the peg's
    circumradius, not the hole's near extent (robot, triangle_c1mm 141115: 5.2 mm from the hole, the dip radius 3.5),
    and lead_across was left out (the L-shape: 8.3 mm)."""
    _require_region(lead)
    d = np.asarray(slide_dir, float)[:2]
    d /= np.linalg.norm(d)
    a = np.array([-d[1], d[0]])
    L = np.asarray(dip_lead, float)[:2]
    P = np.asarray(prior_xy, float)[:2]
    along = float(L @ d) - float(lead["along_m"][0])
    band = lead["band_m"]
    lo, hi = float(L @ a) - float(band[1]), float(L @ a) - float(band[0])
    plo, phi = float(P @ a) - prior_radius, float(P @ a) + prior_radius
    if max(lo, plo) <= min(hi, phi):
        lo, hi = max(lo, plo), min(hi, phi)
    across = 0.5 * (lo + hi)
    return along * d + across * a, 0.5 * (hi - lo)


def next_belief(result: dict, lead: dict, s: dict, previous: dict | None, prior_xy=None,
                prior_radius: float | None = None, max_radius: float | None = None) -> dict | None:
    """The located search's centre and radius for the next attempt, from this attempt's result. prior_xy /
    prior_radius: the disc this attempt searched (the dip's across estimate is limited to it); default the previous
    belief's, else the result's expected hole and the dip radius. max_radius: the search's hole error (peg_search
    takes no located radius above it)."""
    cap = (lambda r: min(float(r), float(max_radius))) if max_radius is not None else float
    if result.get("hold_xy"):
        if "jammed" in (result.get("outcomes") or []) and not result.get("inserted"):
            # A hold that jammed is the peg wedged in the hole mouth, not on its axis (robot 2026-11-21, triangle_c1mm
            # 134814: 2.02 mm from the hole, just outside the 2 mm located radius).
            return {"xy": [float(v) for v in result["hold_xy"]], "radius_m": cap(s.get("jam_radius_m", 0.003)),
                    "source": "hold that jammed (the peg wedged in the hole mouth)"}
        return {"xy": [float(v) for v in result["hold_xy"]], "radius_m": cap(s["located_radius_m"]),
                "source": "hold (the peg upright in the hole mouth)"}
    if previous is not None and previous["source"].startswith("hold"):
        return previous  # a hold (the tip on the hole axis, or near it) beats any later dip
    if result.get("dip_lead") and result.get("slide_dir"):
        if prior_xy is None:
            prior_xy = previous["xy"] if previous else result.get("expected_hole") or result["dip_lead"][:2]
        if prior_radius is None:
            prior_radius = previous["radius_m"] if previous else float(s["dip_radius_m"])
        xy, half = dip_hole(result["dip_lead"], result["slide_dir"], lead, prior_xy, prior_radius)
        radius = cap(max(float(s["dip_radius_m"]), half))
        return {"xy": [float(v) for v in xy], "radius_m": radius, "source": "dip (the lead over the hole near its rim)"}
    return previous


# ---------------------------------------------------------------- docker / ROS
def robot_session_dir(cfg: Config, s: dict) -> Path:
    path = cfg.path(s["robot_session"])
    if not (path / "docker-compose.yml").exists():
        raise PipelineError(f"{path} has no docker-compose.yml (insert.robot_session in the config)")
    for program in ("peg_search", "gripper"):
        if not (path / "build" / program).exists():
            raise PipelineError(f"{path / 'build' / program} is missing; build it in its container:\n"
                                f"  cd {path} && docker compose run --rm robot bash -c "
                                "'cmake -S controller -B build && cmake --build build -j'")
    return path


def _container_run_dir(cfg: Config, run_dir: Path) -> str | None:
    """Path of run_dir inside the persistent robot_session container, or None if not mounted there."""
    resolved = Path(run_dir).resolve()
    ws = cfg.workspace.resolve()
    try:
        rel = resolved.relative_to(ws)
        return str(Path(cfg["docker"]["workspace"]) / rel)
    except ValueError:
        pass
    try:
        resolved.relative_to(Path("/tmp").resolve())
        return str(resolved)
    except ValueError:
        return None


def ensure_robot_container(cfg: Config, s: dict) -> str:
    """Ensure the persistent robot_session container is running; returns its container name."""
    name = str(s.get("robot_container") or ROBOT_CONTAINER)
    chk = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                         capture_output=True, text=True)
    if chk.returncode == 0 and chk.stdout.strip() == "true":
        return name
    if chk.returncode == 0:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    session_dir = robot_session_dir(cfg, s)
    image = str(s.get("robot_image") or ROBOT_IMAGE)
    cmd = [
        "docker", "run", "-d", "--rm", "--name", name,
        "--network", "host", "--cap-add", "SYS_NICE",
        "--ulimit", "rtprio=99", "--ulimit", "memlock=-1",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp", "-e", f"ROBOT={s['robot_ip']}", "-e", "LOAD=",
        "-v", f"{session_dir}:/ws",
        "-v", f"{cfg.workspace.resolve()}:{cfg['docker']['workspace']}",
        "-v", "/tmp:/tmp",
        "-w", "/ws",
        image, "sleep", "infinity",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise PipelineError(f"could not start {name} container: {res.stderr.strip()}")
    return name


def _rewrite_run_paths(argv: list, mapped_dir: str) -> list[str]:
    prefix = CONTAINER_RUN_DIR + "/"
    out = []
    for a in map(str, argv):
        if a == CONTAINER_RUN_DIR:
            out.append(mapped_dir)
        elif a.startswith(prefix):
            out.append(mapped_dir + "/" + a[len(prefix):])
        else:
            out.append(a)
    return out


def robot_program(cfg: Config, s: dict, run_dir: Path, program: str, argv: list) -> int:
    """Run build/<program> in the robot_session container with the run folder mounted; its exit code."""
    session_dir = robot_session_dir(cfg, s)
    mapped = _container_run_dir(cfg, run_dir) if s.get("persistent_container", True) else None
    if mapped is not None:
        container = ensure_robot_container(cfg, s)
        exec_argv = _rewrite_run_paths(argv, mapped)
        command = ["docker", "exec", "-i", container, f"./build/{program}", *exec_argv]
        print(f"  $ {program} " + shlex.join(map(str, argv[1:])), flush=True)
        return subprocess.run(command, stdin=subprocess.DEVNULL).returncode
    env = dict(os.environ, UID=str(os.getuid()), GID=str(os.getgid()), ROBOT=str(s["robot_ip"]), LOAD="")
    command = ["docker", "compose", "-f", str(session_dir / "docker-compose.yml"), "run",
               "--rm", "-T", "-v", f"{run_dir}:{CONTAINER_RUN_DIR}", "robot", f"./build/{program}", *map(str, argv)]
    print(f"  $ {program} " + shlex.join(map(str, argv[1:])), flush=True)
    return subprocess.run(command, env=env, stdin=subprocess.DEVNULL).returncode


def peg_search(cfg: Config, s: dict, run_dir: Path, argv: list) -> int:
    return robot_program(cfg, s, run_dir, "peg_search", argv)


def _load_args(s: dict) -> list:
    """Flange and link6 payload arguments for peg_search."""
    args = []
    if s and "load_mass_kg" in s and "load_com_m" in s:
        args += ["--load-mass", f"{s['load_mass_kg']:g}", "--load-com", *(f"{v:g}" for v in s["load_com_m"])]
    if s and float(s.get("link6_load_mass_kg") or 0.0) > 0.0 and "link6_load_com_m" in s:
        args += ["--link6-load-mass", f"{s['link6_load_mass_kg']:g}",
                 "--link6-load-com", *(f"{v:g}" for v in s["link6_load_com_m"])]
    return args


def _joint_offsets(session, enabled: bool) -> list[float] | None:
    """Calibrated joint encoder offsets (rad, panda_joint1..7) from the session's calibration.json, or None."""
    if not enabled or session is None:
        return None
    cal_file = Path(session.path) / "calibration.json"
    if not cal_file.exists():
        return None
    offsets = (json.loads(cal_file.read_text()).get("joint_offsets_rad") or {})
    names = [f"panda_joint{i}" for i in range(1, 8)]
    if not all(n in offsets for n in names):
        return None
    dq = [float(offsets[n]) for n in names]
    return dq if any(abs(v) > 0.0 for v in dq) else None


def _joint_offset_args(dq: list[float] | None) -> list:
    return ["--joint-offsets", *(f"{v:.8f}" for v in dq)] if dq else []


def _tool_and_load_args(s: dict, tool=None, dq: list[float] | None = None) -> list:
    """Peg tip geometry (when given), payload, and calibrated joint-offset arguments for peg_search."""
    args = list(peg_geometry_args(tool)) if tool is not None else []
    args += _load_args(s)
    args += _joint_offset_args(dq)
    return args


def travel_heights(s: dict) -> tuple[float | None, float | None]:
    """insert.peg_travel_mm / hand_travel_mm in m (None: not set, the old lift_m and probe clearance)."""
    peg, hand = s.get("peg_travel_mm"), s.get("hand_travel_mm")
    return (float(peg) / 1000.0 if peg else None), (float(hand) / 1000.0 if hand else None)


def _free_speed_args(s: dict) -> list:
    """Free-air joint move speed arguments for peg_search (lift, move, extract/peck transitions)."""
    args = []
    if s and s.get("free_move_speed_m_s") is not None:
        args += ["--free-move-speed", f"{float(s['free_move_speed_m_s']):g}"]
    if s and s.get("free_joint_speed_rad_s") is not None:
        args += ["--free-joint-speed", f"{float(s['free_joint_speed_rad_s']):g}"]
    return args


def extract_ik_seed(probe_argv: list | None) -> list[float] | None:
    """Extract the 7 planned joint angles after --ik-seed from probe_board's argv, if present."""
    if not probe_argv:
        return None
    argv = list(map(str, probe_argv))
    if "--ik-seed" not in argv:
        return None
    i = argv.index("--ik-seed")
    if i + 7 >= len(argv):
        return None
    return [float(v) for v in argv[i + 1: i + 8]]


def move_to_hole(cfg: Config, s: dict, run_dir: Path, target_q: list[float], name: str,
                 travel_z: float | None = None, target_z: float | None = None,
                 tool=None, dq: list[float] | None = None, dry_run: bool = False) -> bool:
    """peg_search move: joint-space travel to target_q via travel_z and straight down to target_z (zero ROS handover)."""
    log = f"{CONTAINER_RUN_DIR}/{name}.csv"
    argv = [s["robot_ip"], "move", log, "--target-q", *(f"{v:.8f}" for v in target_q),
            *_tool_and_load_args(s, tool, dq), *_free_speed_args(s), *_backstop(s)]
    if travel_z is not None:
        argv += ["--travel-z", f"{travel_z:.6f}"]
    if target_z is not None:
        argv += ["--target-z", f"{target_z:.6f}"]
    if not dry_run:
        argv += ["--result", f"{CONTAINER_RUN_DIR}/{name}.json", "--yes"]
    else:
        argv += ["--dry-run"]
    return robot_program(cfg, s, run_dir, "peg_search", argv) == 0


def lift(cfg: Config, s: dict, run_dir: Path, distance: float, name: str, tool=None,
         dq: list[float] | None = None, result: bool = False, to_z: float | None = None) -> bool:
    """peg_search lift: the tip straight up, down if negative (libfranka's joint position controller; stops if held
    back). Passes tool geometry, load, and joint offsets so NE_T_EE, gravity, and FK match the run. result: write
    <name>.json (the start pose: after a release, where extract finds the peg again)."""
    log = f"{CONTAINER_RUN_DIR}/{name}.csv"
    extra = ["--result", f"{CONTAINER_RUN_DIR}/{name}.json"] if result else []
    if to_z is not None:  # to a tip height (the travel heights) instead of a distance
        extra += ["--to-z", f"{to_z:.5f}"]
    return robot_program(cfg, s, run_dir, "peg_search", [s["robot_ip"], "lift", log, "--lift-m", f"{distance:.5f}",
                                                         *_tool_and_load_args(s, tool, dq), *_free_speed_args(s),
                                                         *_backstop(s), *extra, "--yes"]) == 0


def open_hand(cfg: Config, s: dict, run_dir: Path) -> bool:
    return robot_program(cfg, s, run_dir, "gripper",
                         [s["robot_ip"], "open", f"{float(s['release_width_mm']) / 1000.0:g}"]) == 0


def _container(cfg: Config, script: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "exec", cfg["docker"]["container"], "bash", "-c", script],
                          capture_output=True, text=True, timeout=timeout)


def _pattern(text: str) -> str:
    """A pgrep/pkill -f pattern that does not match the `bash -c` running it ("[r]oslaunch ...")."""
    return shlex.quote(f"[{text[0]}]{text[1:]}")


def _running(cfg: Config, text: str) -> bool:
    return _container(cfg, f"pgrep -f {_pattern(text)} >/dev/null").returncode == 0


def launch_running(cfg: Config, s: dict) -> bool:
    if _ASSUME_LAUNCH_STOPPED:
        return False
    return _running(cfg, s["robot_launch_match"])


@contextlib.contextmanager
def launch_kept_stopped(cfg: Config, s: dict):
    """Stop robot.launch once and suppress redundant launch_running / ROS current_joints probes while stopped."""
    global _ASSUME_LAUNCH_STOPPED
    if launch_running(cfg, s):
        stop_launch(cfg, s)
    prev_launch = _ASSUME_LAUNCH_STOPPED
    prev_joints = ros._SKIP_CURRENT_JOINTS
    _ASSUME_LAUNCH_STOPPED = True
    ros._SKIP_CURRENT_JOINTS = True
    try:
        yield
    finally:
        _ASSUME_LAUNCH_STOPPED = prev_launch
        ros._SKIP_CURRENT_JOINTS = prev_joints


def stop_launch(cfg: Config, s: dict) -> None:
    """Stop robot.launch (SIGINT, as Ctrl-C in its terminal) and wait for franka_control to let go of the FCI."""
    print("\n== hand over: stopping robot.launch (peg_search needs the robot's only FCI connection) ==", flush=True)
    match = _pattern(s["robot_launch_match"])
    _container(cfg, f"pkill -INT -f {match}")
    for _ in range(60):
        if not _running(cfg, "franka_control_node") and not launch_running(cfg, s):
            return
        subprocess.run(["sleep", "0.5"])
    _container(cfg, f"pkill -TERM -f {match}; pkill -TERM -f {_pattern('franka_control_node')}")
    subprocess.run(["sleep", "2"])
    if _running(cfg, "franka_control_node"):
        raise PipelineError("franka_control did not stop; stop robot.launch by hand and rerun with --here")


def start_launch(cfg: Config, s: dict) -> Path | None:
    """Start robot.launch again in the background and wait for the trajectory controller and set_load (which
    resets NE_T_EE to the identity and the load after peg_search set the peg tip). Returns its log (host path)."""
    if launch_running(cfg, s):
        print("robot.launch is already running")
        return None
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log = cfg.workspace / "log" / "robot_launch" / f"{stamp}.log"  # log/ is colcon's (root): made in the container
    inner_log = shlex.quote(f"{cfg['docker']['workspace']}/log/robot_launch/{stamp}.log")
    print(f"\n== starting robot.launch again (log {log}) ==", flush=True)
    inner = (f"mkdir -p {cfg['docker']['workspace']}/log/robot_launch && "
             f"{ros.ROS_SETUP.format(workspace=cfg['docker']['workspace'])} && exec {s['robot_launch']} > {inner_log} 2>&1")
    subprocess.run(["docker", "exec", "-d", cfg["docker"]["container"], "bash", "-c", inner], check=True)
    for i in range(120):
        subprocess.run(["sleep", "1"])
        if i >= 3 and not _running(cfg, s["robot_launch_match"]):
            raise PipelineError(f"robot.launch exited early (check FCI in Desk); see {log}")
        states = ros.controller_states(cfg)
        nodes = ros.output(cfg, ["rosnode", "list"], timeout=5.0).split()
        if states.get(ros.MOTION_CONTROLLER) == "running" and "/set_load" not in nodes and "/franka_control" in nodes:
            print(f"robot.launch up: {ros.MOTION_CONTROLLER} running, NE_T_EE and load set")
            return log
    raise PipelineError(f"robot.launch did not come up within 2 min; see {log}")


def fast_probe_argv(cfg: Config, s: dict, argv: list, out: Path, clearance_m: float | None = None) -> list:
    """probe_board's configs for insert (written to the run folder): the descent in one move instead of 3 mm steps,
    and the settle tolerance insert.probe_tolerance_m / _rad instead of 0.2 mm (each 3 mm step took ~5 s with its
    correction move: the plunge and the search cover millimetres, not tenths)."""
    argv = list(map(str, argv))
    host = lambda value: ros.host_path(cfg, value)
    i, j = argv.index("--config"), argv.index("--manipulation-config")
    robot = load_yaml(host(argv[i + 1]))
    robot["motion"]["max_cartesian_position_error_m"] = float(s["probe_tolerance_m"])
    robot["motion"]["max_cartesian_rotation_error_rad"] = float(s["probe_tolerance_rad"])
    tool = load_yaml(host(argv[j + 1]))
    if clearance_m is not None:  # the approach pose this far above the target (the travel height), one descent
        tool["probe"]["approach_height_m"] = float(clearance_m)
    tool["probe"]["descent_segment_m"] = float(tool["probe"]["approach_height_m"])
    for index, name, data in ((i, "probe_robot.yaml", robot), (j, "probe_tool.yaml", tool)):
        (out / name).write_text(yaml.safe_dump(data, sort_keys=False))
        argv[index + 1] = ros.container_path(cfg, out / name)
    return argv


# ---------------------------------------------------------------- the command
def arm_angle_deg(tool) -> float:
    """The elbow arm angle the hole was planned at (grasp.choose_turn's elbow fallback), deg; 0 normally."""
    return math.degrees(float(getattr(tool, "arm_angle", 0.0) or 0.0))


def posture_args(tool) -> list:
    """peg_search's posture target for the modes that hold the elbow (tilt-slide): nothing at the default arm angle
    0; the planned one when grasp.choose_turn had to turn the elbow (else the posture move would turn it back to 0 and
    joint 7 onto its limit)."""
    psi = arm_angle_deg(tool)
    return [] if abs(psi) < 1e-6 else ["--arm-angle-deg", f"{psi:.3f}"]


def peg_geometry_args(tool) -> list:
    """The held peg's tip for peg_search (hand frame from the flange): TCP + the pipeline's tip offset."""
    tip = np.asarray(tool.tip_offset, float)
    return ["--peg-axis-x", f"{tip[0]:.6f}", "--peg-axis-y", f"{tip[1]:.6f}",
            "--peg-tip-z", f"{TCP_FROM_FLANGE_M + tip[2]:.6f}"]


def search_error(s: dict, entry: dict) -> float:
    """The hole error disc the search is sized for: insert.error_m (2 mm since 2026-11-21: today's inserted holes were
    0.2-0.9 mm from vision), else tilt_leads.yaml's (the simulation's 3 x clearance, 5 mm)."""
    return float(s["error_m"]) if s.get("error_m") is not None else float(entry["error_m"])


def sized_lead(lead: dict, table_error_m: float, error_m: float) -> dict:
    """The lead's search sized for error_m instead of the table's disc: run-up (error + near rim + 3 mm), spiral
    radius (error + hole + 3 mm) less the difference, slide length (run-up + error + far rim + 3 mm) less twice it."""
    shrink = table_error_m - error_m
    out = dict(lead, table_error_m=table_error_m, error_m=error_m)
    out["lead_run_up_m"] = float(lead["lead_run_up_m"]) - shrink
    out["slide_length_m"] = float(lead["slide_length_m"]) - 2.0 * shrink
    out["spiral_radius_m"] = float(lead["spiral_radius_m"]) - shrink
    if out["lead_run_up_m"] <= error_m or out["slide_length_m"] <= 2.0 * error_m:
        raise PipelineError(f"lead {lead['lead']}: the search does not fit a {error_m * 1e3:g} mm error disc")
    return out


def board_ref_travel(lead: dict, s: dict, error_m: float) -> float:
    """tilt-slide --board-ref-travel: the slide's first part where the lead is on the board for certain (its run-up
    less the hole's near rim, the error disc and insert.board_ref_margin_m), at most insert.board_ref_travel_max_m (0:
    none). The located search after a jam shrinks run-up and disc alike, so the same length holds there."""
    cap = float(s.get("board_ref_travel_max_m", 0.002))
    if cap <= 0.0 or "near_m" not in lead:
        return 0.0
    free = float(lead["lead_run_up_m"]) - float(lead["near_m"]) - error_m - float(s.get("board_ref_margin_m", 0.001))
    return max(0.0, min(cap, free))


def search_args(lead: dict, slide_rad: float, s: dict, pattern: str, error_m: float) -> list:
    plan = line_plan(lead, s, error_m)
    spacing, lines = plan["spacing_m"], plan["lines"]
    args = ["--line-centre", f"{plan['centre_m']:.5f}", "--pattern", pattern, "--slide-dir-deg", f"{math.degrees(slide_rad):.3f}",
            "--peg-radius", f"{lead['peg_radius_m']:.5f}", "--lead-across", f"{lead['lead_across_m']:.5f}",
            "--lead-run-up", f"{lead['lead_run_up_m']:.5f}", "--slide-length", f"{lead['slide_length_m']:.5f}",
            "--hole-error", f"{error_m:.4f}", "--located-radius", f"{min(float(s['located_radius_m']), error_m):.4f}"]
    ref = board_ref_travel(lead, s, error_m)
    if ref > 0.0:
        args += ["--board-ref-travel", f"{ref:.5f}"]
    if pattern == "lines":
        args += ["--lines", lines, "--line-spacing", f"{spacing:.5f}"]
    else:
        args += ["--lines", 3, "--spiral-pitch", f"{lead['spiral_pitch_m']:.5f}",
                 "--spiral-radius", f"{lead['spiral_radius_m']:.5f}"]
    return args


def touch_points(holes: dict, name: str, tool, board: np.ndarray, surface_z: float,
                 offset_m: float = 0.022) -> list[np.ndarray]:
    """Three flat-board points forming an L-triangle beside `name` (in base frame): between columns, between rows,
    and diagonally in the clear inter-hole cell (>= 22 mm from every hole centre on the NIST board)."""
    hx, hy = float(holes[name]["x_m"]), float(holes[name]["y_m"])
    sx = -1.0 if hx > 0.001 else 1.0
    sy = -1.0 if hy > 0.0 else 1.0
    deltas = ((sx * offset_m, 0.0), (0.0, sy * offset_m), (sx * offset_m, sy * offset_m))
    return [board @ np.array([tool.point[0] + dx, tool.point[1] + dy, surface_z, 1.0]) for dx, dy in deltas]


def _plunge_then_tilt(cfg: Config, s: dict, out: Path, drop_argv: list, common: list, summary: dict, attempts: int,
                      height_mm: float, hole_base, touch_base, lead: dict, error_m: float, grid=None, tool=None,
                      dq: list[float] | None = None, heights: dict | None = None) -> bool:
    """insert.method plunge: one position-controlled plunge, the board touched (3-point local plane; with the board
    height model one touch and its slope, or none and its height), then the tilt search attempts. grid
    (beliefgrid.HoleGrid, --search grid): the attempts' lines planned on it. heights (boardheight, from run): "slope"
    the robot's board slope in base, "z_hole" its predicted height at the hole (None: no model), "record" a callable
    that keeps the touches."""
    print("\n== plunge: straight down over the vision hole (position control) ==", flush=True)
    code = peg_search(cfg, s, out, drop_argv)
    result_file = out / "peck.json"
    last = json.loads(result_file.read_text()) if result_file.exists() else \
        {"ok": False, "stage": "unknown", "error": f"peg_search exited {code} without a result"}
    summary["attempts"].append({"attempt": 0, "method": "plunge", "exit": code, **{
        key: last.get(key) for key in ("ok", "stage", "error", "inserted", "board_measured", "plane", "first",
                                       "end_tip")}})
    if last.get("inserted"):
        print("\nplunge: inserted")
        summary["inserted_by"] = last
        return True
    if not last.get("ok"):
        print(f"\nplunge failed ({last.get('stage')}): {last.get('error')}")
        summary["stopped"] = f"the plunge failed: {last.get('error')}"
        return False
    max_off = float(s.get("board_max_off_m", 0.003))
    touched = [p for p in (last.get("touch_points") or []) if p is not None]
    if heights is not None and touched:
        try:
            heights["record"](touched)
        except Exception as error:  # never fails the run
            print(f"warning: the board touches were not recorded ({error})")
    measured_plane = last.get("plane")
    if not (isinstance(measured_plane, dict) and measured_plane.get("point")) and heights is not None and \
            heights.get("slope") is not None:
        # Fewer than 3 touches: the model's slope through the one touch, or (none) its predicted height at the hole.
        slope = [float(v) for v in heights["slope"]]
        if touched:
            pt = [float(v) for v in touched[0][:3]]
            if heights.get("z_touch") is not None:
                print(f"\nboard touch {pt[2] * 1e3:.3f} mm, {(pt[2] - heights['z_touch']) * 1e3:+.3f} mm from the "
                      "board height model (this grasp's offset); its slope from the model")
            measured_plane = {"point": pt, "slope": slope}
        elif heights.get("z_hole") is not None:
            measured_plane = {"point": [float(hole_base[0]), float(hole_base[1]), float(heights["z_hole"])],
                              "slope": slope}
            print(f"\nno board touch: the board height model's plane, {heights['z_hole'] * 1e3:.3f} mm at the hole")
    use_plane = None
    if isinstance(measured_plane, dict) and measured_plane.get("point") and measured_plane.get("slope"):
        pt = [float(v) for v in measured_plane["point"][:3]]
        sl = [float(v) for v in measured_plane["slope"][:2]]
        z_at_hole = pt[2] + sl[0] * (float(hole_base[0]) - pt[0]) + sl[1] * (float(hole_base[1]) - pt[1])
        if abs(z_at_hole - float(hole_base[2])) <= max_off:
            use_plane = {"point": pt, "slope": sl}
            board_z = z_at_hole
        else:
            print(f"\nthe touched plane ({z_at_hole * 1e3:.2f} mm at the hole) is "
                  f"{(z_at_hole - hole_base[2]) * 1e3:+.2f} mm from vision's: ignoring it")
    if use_plane is None:
        board_z = float(last["board_measured"]) if last.get("board_measured") is not None else float(hole_base[2])
        if abs(board_z - float(hole_base[2])) > max_off:
            print(f"\nthe touched board ({board_z * 1e3:.2f} mm) is {(board_z - hole_base[2]) * 1e3:+.2f} mm from "
                  "vision's: a false contact; using vision's board")
            board_z = float(hole_base[2])
    first = last.get("first") or {}
    contact_z = first.get("tip_z")
    if contact_z is not None and first.get("result") != "inserted":
        depth = board_z - float(contact_z)
        first = {**first, "result": "stuck" if depth >= 0.0025 else "board"}
    contact = "-" if contact_z is None else \
        f"{contact_z * 1e3:.2f} mm, {(board_z - contact_z) * 1e3:.2f} mm below the board"
    slope_note = (f", slope ({use_plane['slope'][0] * 1e3:+.1f}, {use_plane['slope'][1] * 1e3:+.1f}) um/mm"
                  if use_plane is not None else ", no measured slope (online board plane)")
    print(f"\nplunge: not in. The robot's heights: board {board_z * 1e3:.2f} mm (touched{slope_note}; vision "
          f"{hole_base[2] * 1e3:.2f} mm), the plunge's contact {contact}: {first.get('result')}")
    plane = use_plane if use_plane is not None else \
        {"point": [float(touch_base[0]), float(touch_base[1]), board_z], "slope": [0.0, 0.0]}
    summary["plane"] = plane
    stuck = first.get("result") == "stuck" and contact_z is not None
    if grid is not None:
        grid.update_from_peck({"first": first}, "plunge")
    start_z = board_z + height_mm / 1000.0
    approach = (height_mm + float(s["approach_known_margin_mm"])) / 1000.0
    belief = {"xy": [float(v) for v in first["xy"]], "radius_m": min(float(s["located_radius_m"]), error_m),
              "source": "plunge stuck in the hole mouth"} if stuck else None
    for k in range(1, attempts + 1):
        if k > 1:  # upright and at the start height again (a failed run leaves the peg tilted, anywhere)
            print(f"\n== reset: the peg upright, the tip {height_mm:g} mm above the board ==", flush=True)
            if robot_program(cfg, s, out, "peg_search", [s["robot_ip"], "lift", f"{CONTAINER_RUN_DIR}/reset_{k}.csv",
                                                         "--to-z", f"{start_z:.5f}", "--upright",
                                                         *_tool_and_load_args(s, tool, dq), *_backstop(s), "--yes"]) != 0:
                summary["stopped"] = f"the reset before attempt {k} failed (the peg caught?)"
                return False
        plane_args = (["--board-plane", *(f"{v:.6f}" for v in use_plane["point"]),
                       *(f"{v:.6f}" for v in use_plane["slope"])]
                      if use_plane is not None else ["--probe-points", "0"])
        argv = [s["robot_ip"], "tilt-slide", f"{CONTAINER_RUN_DIR}/attempt_{k}.csv", "--approach", f"{approach:.4f}",
                *common, "--board-z", f"{board_z:.6f}", *plane_args,
                "--posture", "start", "--result", f"{CONTAINER_RUN_DIR}/attempt_{k}.json"]
        plan = None
        if grid is not None:
            plan, extra = grid_args(grid, s, lead, error_m, out, k)
            argv += extra
        elif belief is not None:
            argv += ["--located-xy", f"{belief['xy'][0]:.6f}", f"{belief['xy'][1]:.6f}",
                     "--located-radius", f"{belief['radius_m']:.4f}"]
        if k == 1 and stuck:  # compliant over the same spot: the hold presses the peg in from the hole mouth
            in_depth = max(0.001, min(0.002, board_z - contact_z - 0.0005))
            argv += ["--plunge", f"{board_z:.6f}", "--plunge-in-depth", f"{in_depth:.4f}"]
        if plan is not None:
            where = (("compliant plunge at the stuck spot, then " if k == 1 and stuck else "") + grid_text(grid, plan))
        elif k == 1 and stuck:
            where = f"compliant plunge at the stuck spot, then located search {belief['radius_m'] * 1e3:g} mm about it"
        elif belief:
            where = (f"located search about ({belief['xy'][0] * 1e3:.2f}, {belief['xy'][1] * 1e3:.2f}) mm, from a "
                     f"{belief['source']}")
        else:
            where = "the vision hole's error disc"
        print(f"\n== tilt search attempt {k}/{attempts}: {where} (board plane from the touch) ==", flush=True)
        code = peg_search(cfg, s, out, argv)
        result_file = out / f"attempt_{k}.json"
        last = json.loads(result_file.read_text()) if result_file.exists() else \
            {"ok": False, "stage": "unknown", "error": f"peg_search exited {code} without a result"}
        summary["attempts"].append({"attempt": k, "exit": code, "belief_used": belief, "grid_plan": plan, **{
            key: last.get(key) for key in ("ok", "stage", "error", "plunge", "outcomes", "inserted", "hold_xy",
                                           "dip_lead", "end_tip")}})
        if last.get("plunge") not in (None, "none"):
            print(f"plunge: {last['plunge']}")
        if grid is not None:
            grid_update(grid, s, last, out, k)
        if last.get("inserted"):
            print(f"\ninserted on tilt search attempt {k}")
            summary["inserted_by"] = last
            return True
        if last.get("stage") in ("options", "connect", "unknown"):
            summary["stopped"] = f"peg_search could not start: {last.get('error')}"
            return False
        if not last.get("ok"):
            print(f"attempt {k} failed ({last.get('stage')}): {last.get('error')}")
        if grid is not None:
            continue
        belief = next_belief(last, lead, s, belief, prior_xy=belief["xy"] if belief else hole_base[:2],
                             prior_radius=belief["radius_m"] if belief else error_m, max_radius=error_m)
        summary["belief"] = belief
        if belief is not None:
            print(f"belief now: ({belief['xy'][0] * 1e3:.2f}, {belief['xy'][1] * 1e3:.2f}) mm base, "
                  f"radius {belief['radius_m'] * 1e3:.2f} mm, from a {belief['source']}")
    return False


# ---------------------------------------------------------------- the belief grid (beliefgrid, --search grid)
def make_grid(s: dict, hole_base, error_m: float, lead: dict, slide: float, free, spacing: float):
    return beliefgrid.HoleGrid(hole_base[:2], error_m, lead, slide, free, spacing,
                               p_detect=float(s.get("grid_p_detect", 0.8)), p_true=float(s.get("grid_p_true", 0.9)))


def grid_args(grid, s: dict, lead: dict, error_m: float, out: Path, k: int) -> tuple[dict, list]:
    """The next attempt's lines from the grid: --hole-xy its mean, --line-offsets in their order, --lines with room
    for a jam's located search. Returns (the plan, the peg_search arguments)."""
    plan = grid.plan(target=float(s.get("grid_target", 0.95)), max_lines=int(s.get("grid_max_lines", 15)))
    spacing = grid.spacing
    located = 1 + 2 * math.ceil(min(float(s["located_radius_m"]), error_m) / spacing - 1e-9)
    xy = plan["hole_xy"]
    grid.save(out / f"grid_{k}.npz")
    return plan, ["--hole-xy", f"{xy[0]:.6f}", f"{xy[1]:.6f}",
                  "--line-offsets", ",".join(f"{o:.5f}" for o in plan["offsets_m"]),
                  "--lines", str(len(plan["offsets_m"]) + located)]


def grid_text(grid, plan: dict) -> str:
    return (f"grid: {len(plan['offsets_m'])} line(s) at {', '.join(f'{o * 1e3:+.1f}' for o in plan['offsets_m'])} mm "
            f"across about ({plan['hole_xy'][0] * 1e3:.2f}, {plan['hole_xy'][1] * 1e3:.2f}) mm, covering "
            f"{plan['covered'] * 100:.0f} % of where the hole may be (spread {plan['spread_m'] * 1e3:.2f} mm)")


def grid_update(grid, s: dict, result: dict, out: Path, k: int) -> None:
    """The attempt's evidence: its result and its 1 kHz log (a run that died before its log has none)."""
    csv = out / f"attempt_{k}.csv"
    before = len(grid.events)
    try:
        grid.update_from_run(result, csv if csv.exists() else None, str(k),
                             located_radius=float(s["located_radius_m"]), jam_radius=float(s.get("jam_radius_m", 0.003)))
    except (OSError, ValueError, IndexError) as error:  # a truncated log: the attempt adds nothing
        print(f"grid: attempt {k}'s log not read ({error})")
    for event in grid.events[before:]:
        print(f"grid: {event}")
    mean = grid.mean()
    print(f"grid: the hole now about ({mean[0] * 1e3:.2f}, {mean[1] * 1e3:.2f}) mm, spread {grid.spread() * 1e3:.2f} mm")


# ---------------------------------------------------------------- the found hole (holefit)
def free_region(entry: dict, holes: dict, hole: str, table_name: str) -> tuple[list, float]:
    """The hole's free region (tilt_leads.yaml free_m) and how much larger the hole may be than it (m): the rectangle
    3 mm hole uses the 1 mm's table (not in the simulation)."""
    extra = max(0.0, (float(holes[hole]["clearance_mm"]) - float(holes[table_name]["clearance_mm"])) / 2000.0)
    if not entry.get("free_m"):
        raise PipelineError("config/tilt_leads.yaml has no free_m (from before 2026-11-21): regenerate it")
    return entry["free_m"], extra


def seated_xy(result: dict):
    """Where the inserting run left the tip on the hole axis, or near it: the peck's inserted point, else the run's end
    when it is at least 3 mm into the bore, else the hold's start, else the run's end. (2026-11-23, l_shape_c1mm: the
    hold started 9 mm from the hole and the peg was pushed in by hand; it ended 15.7 mm deep 1.9 mm from the target.)"""
    end_depth = result.get("end_above_board")
    deep = end_depth is not None and math.isfinite(float(end_depth)) and float(end_depth) <= -0.003
    for key in ("inserted_xy", "end_tip" if deep else "hold_xy", "hold_xy", "end_tip"):
        value = result.get(key)
        if value is not None and all(v is not None and math.isfinite(float(v)) for v in value[:2]):
            return [float(v) for v in value[:2]], key
    return None, None


def _centre_and_record(cfg, s, out, session, args, tool, entry, holes, table_name, vision_base, yaw_base, board_sha,
                       summary, hole_base, dq: list[float] | None = None) -> dict | None:
    """Inserted, before the release: the hole's centre from wall touches (peg_search centre), else the seated tip;
    recorded in found_holes.yaml for the board correction of later inserts. Never fails the run."""
    free, extra = free_region(entry, holes, args.hole, table_name)
    seat, seat_key = seated_xy(summary.get("inserted_by") or {})
    found = None
    clearance_mm = float(holes[args.hole]["clearance_mm"])
    min_centre_mm = float(s.get("centre_min_clearance_mm", 0.5))
    if s.get("centre_after_insert", True) and not getattr(args, "no_centre", False):
        if clearance_mm < min_centre_mm:
            print(f"\ncentre: skipped for {args.hole} (clearance {clearance_mm:g} mm < {min_centre_mm:g} mm; "
                  "seated tip already pins the hole centre)")
        else:
            plane = summary.get("plane")
            board_z = float(plane["point"][2]) if plane else float(hole_base[2])
            dirs = holefit.touch_directions(yaw_base, int(s.get("centre_touches", 6)))
            reach = min(0.006, holefit.touch_reach(free) + 2.0 * extra)
            argv = [s["robot_ip"], "centre", f"{CONTAINER_RUN_DIR}/centre.csv", *_tool_and_load_args(s, tool, dq),
                    "--board-z", f"{board_z:.6f}", "--centre-dirs-deg", ",".join(f"{d:.2f}" for d in dirs),
                    "--centre-reach", f"{reach:.5f}", *_backstop(s), *map(str, s.get("centre_options", [])),
                    "--result", f"{CONTAINER_RUN_DIR}/centre.json", "--yes"]
            print(f"\n== centre: {len(dirs)} wall touches (at most {reach * 1e3:.2f} mm each) for the hole's position ==",
                  flush=True)
            code = peg_search(cfg, s, out, argv)
            result_file = out / "centre.json"
            result = json.loads(result_file.read_text()) if result_file.exists() else {"ok": False}
            summary["centre"] = {k: result.get(k) for k in ("ok", "stage", "error", "contacts", "touches")}
            if code == 0 and result.get("ok"):
                guess = seat if seat is not None else hole_base[:2]
                found = holefit.found_from_touches(result, free, yaw_base, guess, max_delta=2.0e-4 + extra)
                if found is None:
                    print("centre: the touches do not pin the hole (too few, one-sided, or off its outline): the seated tip")
            else:
                print(f"centre: failed ({result.get('error')}): the seated tip")
    if found is None:
        if seat is None:
            print("no position of the inserted peg to record")
            return None
        found = holefit.found_from_seat(seat, free, f"seated tip ({seat_key})")
        found["sigma_m"] += extra
    entry_ = holefit.record_found(session.path, args.hole, vision_base[:2], found, board_sha, out.name,
                                  arm_angle_deg=arm_angle_deg(tool))
    off = np.asarray(found["xy"]) - np.asarray(vision_base[:2])
    print(f"found {args.hole} at ({found['xy'][0] * 1e3:.2f}, {found['xy'][1] * 1e3:.2f}) mm, "
          f"({off[0] * 1e3:+.2f}, {off[1] * 1e3:+.2f}) mm from vision, +-{found['sigma_m'] * 1e3:.2f} mm, "
          f"{found['source']}; recorded in {holefit.found_path(session.path)}")
    return entry_


def import_runs(session) -> list[dict]:
    """Found holes from the inserted runs of insert/ not in the record yet (their seated tip), runs after the board
    pose was last written only (an older run aimed at another board pose)."""
    cfg = session.cfg
    holes = load_yaml(cfg.path(cfg["holes"]))["holes"]
    board_file = session.path / "board_pose.json"
    if not board_file.exists():
        raise PipelineError("no board_pose.json in the session")
    sha = sha256(board_file)
    since = datetime.datetime.fromtimestamp(board_file.stat().st_mtime)
    known = {e.get("run") for e in holefit.load_found(session.path)}
    added = []
    for run_dir in sorted((session.path / "insert").glob("*_*")):
        plan_file = run_dir / "plan.yaml"
        if not run_dir.is_dir() or run_dir.name in known or not plan_file.exists():
            continue
        try:
            started = datetime.datetime.strptime(run_dir.name, "%Y-%m-%d_%H%M%S")
        except ValueError:
            continue
        if started < since:
            continue
        plan = yaml.safe_load(plan_file.read_text())
        inserting = None
        for result_file in sorted(run_dir.glob("*.json")):
            result = json.loads(result_file.read_text())
            if result.get("inserted") and result_file.name != "centre.json":
                inserting = result
        if inserting is None:
            continue
        name = plan["hole"]
        table_name, entry = hole_leads(cfg, holes, name)
        free, extra = free_region(entry, holes, name, table_name)
        seat, key = seated_xy(inserting)
        if seat is None:
            continue
        found = holefit.found_from_seat(seat, free, f"seated tip ({key}, imported)")
        found["sigma_m"] += extra
        vision = plan.get("vision_hole_base_m") or plan["hole_base_m"]
        added.append(holefit.record_found(session.path, name, vision[:2], found, sha, run_dir.name,
                                          arm_angle_deg=float(plan.get("arm_angle_deg") or 0.0)))
    return added


def correction_entries(entries: list[dict], s: dict, holes: dict) -> list[dict]:
    """The found holes the board correction is fitted to: not those of insert.correction_exclude_shapes, whose found
    position carries their peg's own tip error (2026-11-23: the L's ~1-2 mm, not the board's, and it depends on the
    tip file the run used), nor those found with the elbow turned (boardheight.at_elbow_zero)."""
    skip = set(s.get("correction_exclude_shapes") or [])
    return [e for e in entries if (holes.get(e["hole"]) or {}).get("shape") not in skip and boardheight.at_elbow_zero(e)]


def _report_heights(session, sha: str | None) -> None:
    """`pipeline found`: the board height model (boardheight) of the current board pose and how well it predicts each
    run's touches from the others'."""
    added = boardheight.backfill(session.path)
    if added:
        print(f"board heights: collected the touches of {added} earlier run(s)")
    touches = boardheight.load(session.path)
    model = boardheight.fit(touches, sha)
    if model is None:
        print("board height model: no touches of the current board pose")
        return
    offsets = ", ".join(f"{k} {v * 1e3:+.2f} ({model['shape_touches'][k]})" for k, v in model["shape_offset"].items())
    print(f"board height model ({model['touches']} touches, {model['runs']} runs): robot - vision "
          f"{model['c'] * 1e3:+.2f} mm at ({model['centre'][0] * 1e3:.0f}, {model['centre'][1] * 1e3:.0f}) mm, slope "
          f"({model['slope'][0] * 1e3:+.2f}, {model['slope'][1] * 1e3:+.2f}) um/mm, per grasp (shape: mm, touches): {offsets}; "
          f"residuals {model['rms_m'] * 1e3:.3f} mm RMS, max {model['max_residual_m'] * 1e3:.3f}; slope "
          f"{'known' if boardheight.slope_known(model) else 'not known yet (touches over 40 mm in x and y, 2 runs)'}")
    rows = boardheight.leave_one_run_out(touches, sha)
    if rows:
        print("  each run's touches from the other runs' (RMS mm): no own touch / one own touch")
        for r in rows:
            print(f"  {r['run']}  {r['shape']:10s} {r['rms_none_m'] * 1e3:.3f} / {r['rms_one_m'] * 1e3:.3f}"
                  + ("" if r["shape_known"] else "  (no other run with this peg)"))


def cmd_found(session, args) -> None:
    if args.import_runs:
        added = import_runs(session)
        print(f"imported {len(added)} inserted run(s)")
    all_entries = holefit.load_found(session.path)
    s = settings(session.cfg)
    entries = correction_entries(all_entries, s, load_yaml(session.cfg.path(session.cfg["holes"]))["holes"])
    for e in all_entries:
        off = np.asarray(e["found_xy"]) - np.asarray(e["vision_xy"])
        print(f"{e['run']}  {e['hole']:18s} ({off[0] * 1e3:+.2f}, {off[1] * 1e3:+.2f}) mm from vision, "
              f"+-{e['sigma_m'] * 1e3:.2f} mm  {e['source']}" + ("" if e in entries else "  (not in the correction)"))
    board_file = session.path / "board_pose.json"
    sha = sha256(board_file) if board_file.exists() else None
    _report_heights(session, sha)
    corr = holefit.local_correction(entries, sha, **local_options(s))
    if corr is None:
        print("no found holes for the current board pose: no correction")
        return
    print(f"board correction ({corr['holes']} holes of the current board pose): rigid shift "
          f"({corr['shift_m'][0] * 1e3:+.2f}, {corr['shift_m'][1] * 1e3:+.2f}) mm +-{corr['sigma_shift_m'] * 1e3:.2f}, "
          f"turn {corr['turn_deg']:+.3f} deg +-{corr['sigma_turn_deg']:.3f} about "
          f"({corr['centre_m'][0] * 1e3:.1f}, {corr['centre_m'][1] * 1e3:.1f}) mm")
    for d in corr["dropped"]:
        print(f"  dropped {d['hole']} ({d['run']}): {d['residual_m'] * 1e3:.2f} mm off the fit")
    m = corr["model"]
    verdict = ("the holes' sigmas explain the residuals: one rigid shift fits" if m["chi2"] <= 2.0 * m["dof"] else
               f"residuals beyond the holes' sigmas: ~{m['extra_scatter_m'] * 1e3:.2f} mm per axis the rigid shift "
               "does not explain (posture-dependent kinematics, or the grasp)")
    print(f"model check: rigid residuals {m['rms_residual_m'] * 1e3:.2f} mm RMS, chi2 {m['chi2']:.1f} for {m['dof']} dof: "
          f"{verdict}; local field {corr['sigma_f_m'] * 1e3:.2f} mm over {corr['length_m'] * 1e3:.0f} mm")
    loo = holefit.leave_one_out(entries, sha, **local_options(s))
    if loo is None:
        print("leave-one-out: needs found holes at two places at least")
        return
    r = loo["rms"]
    print(f"leave-one-out (each hole predicted from the others), RMS: no correction {r['none_m'] * 1e3:.2f} mm, "
          f"rigid {r['rigid_m'] * 1e3:.2f} mm, rigid + local {r['local_m'] * 1e3:.2f} mm "
          f"(the holes' own sigmas {math.sqrt(np.mean([x['sigma_m'] ** 2 for x in loo['rows']])) * 1e3:.2f} mm)")
    for x in loo["rows"]:
        print(f"  {x['run']}  {x['hole']:18s} none {x['none_m'] * 1e3:.2f}  rigid {x['rigid_m'] * 1e3:.2f}  "
              f"local {x['local_m'] * 1e3:.2f} mm  (+-{x['sigma_m'] * 1e3:.2f})")


def local_options(s: dict) -> dict:
    """insert.correction_*: the local residual field's settings (holefit.local_correction)."""
    out = {"length_m": float(s.get("correction_length_m", 0.06)), "cap_m": float(s.get("correction_field_cap_m", 0.001))}
    if s.get("correction_field_m") is not None:
        out["sigma_f_m"] = float(s["correction_field_m"])
    return out


def release_record(result_file: Path, hole: str, shape: str, summary: dict, plane: dict | None, hole_base,
                   board_z_guess: float, tool, dq: list[float] | None = None) -> dict | None:
    """Where the released peg is, for extract (2026-11-23): the tip pose (peg_search's EE frame, calibrated) when
    the hand opened, from lift_release.json, and the robot's board height at the hole (the touched plane, else the
    board height model's or vision's). None: no result (an older peg_search, or the lift failed before it)."""
    if not result_file.exists():
        return None
    result = json.loads(result_file.read_text())
    pose = result.get("start_pose")
    if not pose or len(pose) != 16:
        return None
    if plane and plane.get("point") and plane.get("slope"):
        pt, sl = plane["point"], plane["slope"]
        board_z = float(pt[2]) + float(sl[0]) * (float(hole_base[0]) - pt[0]) + float(sl[1]) * (float(hole_base[1]) - pt[1])
    else:
        board_z = float(board_z_guess)
    return {"hole": hole, "shape": shape, "tip_pose": [float(v) for v in pose], "board_z_m": board_z,
            "tip_offset_tcp_m": [float(v) for v in tool.tip_offset], "period_rad": float(tool.period),
            "run": result_file.parent.name, "tip_below_board_m": board_z - float(pose[11]),
            "vision_board_z_m": float(hole_base[2]),
            "joint_offsets_rad": [float(v) for v in dq] if dq else None, "arm_angle_deg": arm_angle_deg(tool)}


def _backstop(s: dict) -> list:
    """peg_search lift's force-change backstop (insert.lift_max_force_change)."""
    return ["--posture-max-force-change", f"{float(s.get('lift_max_force_change', 50)):g}"]


def run(session, args, probe_target, restart_launch: bool = True, raise_on_fail: bool = True) -> dict:
    """probe_target: cli._probe_target (writes the hole's target and tool files, returns the probe_board argv).
    restart_launch False: robot.launch stays stopped at the end (campaign: extract follows, libfranka only).
    Returns {"inserted", "stopped", "out", "summary"}; raise_on_fail False: also when it did not insert."""
    cfg = session.cfg
    s = settings(cfg)
    holes = load_yaml(cfg.path(cfg["holes"]))["holes"]
    if args.hole not in holes:
        raise PipelineError(f"unknown hole {args.hole!r}; `pipeline holes` lists them")
    attempts = args.attempts or int(s["attempts"])
    height_mm = args.height if args.height is not None else float(s["start_height_mm"])
    tool, tip_source, target, probe_argv = probe_target(session, holes, args.hole, args.tip_from, height_mm,
                                                        args.turn, joint_correction=not args.no_joint_correction)
    data = json.loads(Path(target).read_text())
    # The board correction from the holes found so far (holefit): applied to the target the probe, the plunge and the
    # search all aim with. vision_base: the hole as vision has it (recorded with what this run finds).
    src = data.get("source") or {}
    board_sha = src.get("board_pose_sha256")
    nom_board = np.asarray(data["T_base_board"])
    dq = _joint_offsets(session, not args.no_joint_correction) if src.get("T_base_board_true_frame") is not None else None
    peg_raw_board = np.asarray(src["T_base_board_true_frame"]) if dq is not None else nom_board
    vision_base = peg_raw_board @ np.array([tool.point[0], tool.point[1], data["board_surface_z_m"], 1.0])
    correction = None
    if s.get("board_correction", True) and not getattr(args, "no_board_correction", False):
        correction = holefit.local_correction(correction_entries(holefit.load_found(session.path), s, holes), board_sha,
                                              at_xy=vision_base[:2] if s.get("correction_local", True) else None,
                                              **local_options(s))
    if correction is not None:
        m_corr = holefit.full_matrix(correction)
        board = m_corr @ peg_raw_board
        if dq is not None:
            c_nom = nom_board @ np.linalg.inv(peg_raw_board)
            nom_corrected = c_nom @ board
        else:
            nom_corrected = m_corr @ nom_board
        data = {**data, "T_base_board": nom_corrected.tolist(), "board_correction": correction}
        corrected = Path(target).with_name(Path(target).stem + "_corrected.json")
        corrected.write_text(json.dumps(data, indent=2))
        probe_argv = list(probe_argv)
        probe_argv[probe_argv.index("--board-pose") + 1] = ros.container_path(cfg, corrected)
        shift = holefit.correct_xy_full(vision_base[:2], correction) - vision_base[:2]
        local = correction["local_m"]
        print(f"board correction from {correction['holes']} found hole(s): rigid shift "
              f"({correction['shift_m'][0] * 1e3:+.2f}, {correction['shift_m'][1] * 1e3:+.2f}) mm "
              f"(+-{correction['sigma_shift_m'] * 1e3:.2f}), turn {correction['turn_deg']:+.3f} deg "
              f"(+-{correction['sigma_turn_deg']:.3f}); local here ({local[0] * 1e3:+.2f}, {local[1] * 1e3:+.2f}) mm "
              f"(+-{correction['local_sigma_m'] * 1e3:.2f}); this hole moves ({shift[0] * 1e3:+.2f}, "
              f"{shift[1] * 1e3:+.2f}) mm"
              + (f"; dropped {', '.join(d['hole'] for d in correction['dropped'])}" if correction["dropped"] else ""))
    else:
        board = peg_raw_board
    hole_base = board @ np.array([tool.point[0], tool.point[1], data["board_surface_z_m"], 1.0])
    hole_axis = board[:3, :3] @ np.array([math.cos(math.radians(holes[args.hole]["yaw_deg"])),
                                          math.sin(math.radians(holes[args.hole]["yaw_deg"])), 0.0])
    hole_yaw_base = math.atan2(hole_axis[1], hole_axis[0])
    table_name, entry = hole_leads(cfg, holes, args.hole)
    shape = holes[args.hole]["shape"]
    preferred = args.slide_dir if args.slide_dir is not None else float(s["slide_dir_base_deg"])
    error_m = search_error(s, entry)
    group = (s.get("lead_group") or {}).get(shape)
    if group is not None:  # insert.lead_group: this shape's leads of that group only (the L-shape's outer corner: 1)
        chosen = [l for l in entry["leads"] if int(l.get("group", 0)) == int(group)]
        if not chosen:
            raise PipelineError(f"insert.lead_group {shape}: {group}, but {table_name} has no lead of that group")
        entry = {**entry, "leads": chosen}
    lead, slide = round_lead(entry, preferred) if shape == "round" else \
        choose_lead(entry, hole_yaw_base, preferred, s, error_m)
    lead = sized_lead(lead, float(entry["error_m"]), error_m)
    line = line_plan(lead, s, error_m)
    # The round peg: straight lines always (no corners to lead with, no yaw error to search over).
    pattern = args.pattern or ("lines" if shape == "round" else s.get("pattern") or entry["pattern"])
    # How later attempts are aimed: the belief grid (lines only; the spiral keeps the hold / dip belief).
    search = (getattr(args, "search", None) or s.get("search") or "grid") if pattern == "lines" else "belief"
    grid = make_grid(s, hole_base, error_m, lead, slide, free_region(entry, holes, args.hole, table_name)[0],
                     line["spacing_m"]) if search == "grid" else None
    clearance_mm = float(holes[args.hole]["clearance_mm"])
    min_centre_mm = float(s.get("centre_min_clearance_mm", 0.5))
    insert_depth_m = (float(s["tight_insert_depth_m"])
                      if clearance_mm < min_centre_mm and s.get("tight_insert_depth_m") is not None
                      else float(s.get("peck_insert_depth_m", 0.015)))
    min_grasp_args = (["--min-grasp-width", f"{max(0.0006, float(tool.grasp_width_mm) / 1000.0 - 0.003):.5f}"]
                      if getattr(tool, "grasp_width_mm", None) else [])
    common = [*map(str, s["peg_search_options"]), *_tool_and_load_args(s, tool, dq), *min_grasp_args,
              *posture_args(tool),
              "--probe-radius", f"{s['probe_radius_m']:g}", "--hole-xy", f"{hole_base[0]:.6f}", f"{hole_base[1]:.6f}",
              "--board-z", f"{hole_base[2]:.6f}", "--insert-depth", f"{insert_depth_m:g}",
              "--lift-at-end", "--yes", *map(str, search_args(lead, slide, s, pattern, error_m))]

    # Peck (holes with >= insert.peck_min_clearance_mm): position-controlled pecks on a grid about the vision hole.
    # Robot 2026-11-21, hexagon_c3mm near the base: the tilt search's Cartesian impedance lagged 3-11 mm in z there,
    # the internal joint position controller 0.2-0.6 mm.
    # plunge (insert.method, the default): one position-controlled plunge at the vision hole; not in: the board's
    # height touched beside the hole (where the robot thinks the board is), the plunge's contact classified against it
    # (board / stuck in the hole mouth), then the tilt search attempts from that plane. peck: the grid of plunges.
    # tilt: the tilt search attempts only (the probe, the compliant plunge).
    method = args.method or s.get("method", "plunge")
    # The board's height as the robot touches it (boardheight, 2026-11-23): vision's is 1-2 mm low, smoothly over the
    # board, plus the held peg's own offset. The peck aims at the predicted board (its "in the air" test is 2 mm above
    # it), and once the model knows the board's slope one touch beside the hole replaces the three.
    heights = boardheight.load(session.path)
    if not heights:
        boardheight.backfill(session.path)
        heights = boardheight.load(session.path)
    hmodel = boardheight.fit(heights, board_sha) if s.get("board_height_model", True) else None
    slope_ok = boardheight.slope_known(hmodel, int(s.get("board_height_min_runs", 2)),
                                       float(s.get("board_height_min_span_m", 0.04)))
    z_vision_hole = boardheight.vision_height(peg_raw_board, data["board_surface_z_m"], hole_base[:2])
    z_hole_pred, shape_known = (boardheight.predict(hmodel, hole_base[:2], shape, z_vision_hole)
                                if slope_ok else (None, False))
    want = str(args.touches if getattr(args, "touches", None) is not None else s.get("board_touches", "auto"))
    n_touches = (1 if slope_ok else 3) if want == "auto" else int(want)
    if n_touches not in (0, 1, 3):
        raise PipelineError(f"insert.board_touches / --touches: auto, 0, 1 or 3, not {want}")
    if n_touches < 3 and not slope_ok:
        print(f"note: the board height model does not know this board's slope yet ({hmodel['runs'] if hmodel else 0} "
              "run(s) touched): 3 touches")
        n_touches = 3
    if n_touches == 0 and not shape_known:
        print(f"note: no touches with the {shape} peg yet (its grasp's height offset): 1 touch")
        n_touches = 1
    peck_board_z = z_hole_pred if z_hole_pred is not None else float(hole_base[2])
    heights_ctx = None
    if hmodel is not None:
        robot_slope = (boardheight.vision_slope(peg_raw_board) + np.asarray(hmodel["slope"])).tolist() \
            if slope_ok else None
        print(f"board height model ({hmodel['touches']} touches, {hmodel['runs']} runs, {hmodel['rms_m'] * 1e3:.3f} mm "
              f"RMS): " + (f"the board at the hole {peck_board_z * 1e3:.2f} mm (vision {z_vision_hole * 1e3:.2f}), "
                           f"{'the ' + shape + ' offset known' if shape_known else 'no ' + shape + ' touches yet'}"
                           if z_hole_pred is not None else "the slope not known yet"))
    peck_argv = [s["robot_ip"], "peck", f"{CONTAINER_RUN_DIR}/peck.csv", *_tool_and_load_args(s, tool, dq),
                 *_free_speed_args(s), *min_grasp_args,
                 "--hole-xy", f"{hole_base[0]:.6f}", f"{hole_base[1]:.6f}", "--board-z", f"{peck_board_z:.6f}",
                 # tight holes: tight_insert_depth_m like the tilt search (2026-11-23, round_c0p03mm: the plunge went
                 # 4.5-4.8 mm in, was taken as stuck, and the retry's lift jammed in the bore, twice)
                 "--insert-depth", f"{insert_depth_m:g}",
                 *map(str, s.get("peck_options", [])), "--result", f"{CONTAINER_RUN_DIR}/peck.json", "--yes"]
    # Board touches in clear board space beside the hole, under position control: three forming a 22 mm triangle
    # measure the local plane (height + slope); one sets this grasp's height on the model's slope; none: the model.
    all_touches = touch_points(holes, args.hole, tool, board, data["board_surface_z_m"],
                               float(s.get("touch_offset_m", 0.022)))
    touches = all_touches[:n_touches]
    touch_base = all_touches[0]
    touch_args = [v for pt in touches for v in ("--peck-touch", f"{pt[0]:.6f}", f"{pt[1]:.6f}")]
    drop_argv = [*peck_argv, "--peck-max-points", "1", *touch_args]
    if hmodel is not None or n_touches == 3:
        def record_touches(points):  # `out` (this run's folder) is set below, before the plunge calls this
            z_vision = [boardheight.vision_height(peg_raw_board, data["board_surface_z_m"], p) for p in points]
            boardheight.record(session.path, out.name, shape, board_sha, points, z_vision,
                               arm_angle_deg=arm_angle_deg(tool))
        z_touch = (boardheight.predict(hmodel, touch_base[:2], shape, boardheight.vision_height(
            peg_raw_board, data["board_surface_z_m"], touch_base[:2]))[0] if slope_ok and shape_known else None)
        heights_ctx = {"slope": robot_slope if hmodel is not None else None, "z_hole": z_hole_pred,
                       "z_touch": z_touch, "record": record_touches}

    frame_label = ("calibrated q+dq frame" if dq is not None else
                   "joint-offset corrected" if not args.no_joint_correction else "uncorrected")
    print(f"\nhole {args.hole} ({shape}, clearance {holes[args.hole]['clearance_mm']:g} mm) at base "
          f"({hole_base[0] * 1e3:.2f}, {hole_base[1] * 1e3:.2f}, {hole_base[2] * 1e3:.2f}) mm (vision, {frame_label})")
    if method == "plunge":
        pts_str = ", ".join(f"({pt[0] * 1e3:.1f}, {pt[1] * 1e3:.1f})" for pt in touches)
        board_how = (f"3-point board plane at [{pts_str}] mm" if n_touches == 3 else
                     f"one board touch at [{pts_str}] mm on the height model's slope" if n_touches == 1 else
                     "the board height model's plane (no touch)")
        print(f"plunge: position-controlled at the {'predicted' if z_hole_pred is not None else 'vision'} board "
              f"({peck_board_z * 1e3:.2f} mm); not in: {board_how}, then up to {attempts} tilt search attempts")
    if method == "peck":
        print(f"peck search: position-controlled pecks about the vision hole ({' '.join(map(str, s.get('peck_options', [])))})")
    print(f"tilt search{' (not used)' if method == 'peck' else ''}: lead {lead['lead']} of {table_name}, slide {math.degrees(slide):+.1f} deg in base "
          f"(preferred {preferred:g}), {pattern}, {error_m * 1e3:g} mm hole error disc; peg tip "
          f"{np.round(np.asarray(tool.tip_offset) * 1e3, 2)} mm from the TCP ({tip_source})")
    if pattern == "lines":
        print(f"lines: {line['lines']} x {line['spacing_m'] * 1e3:.2f} mm ({lead_kind(lead)} lead), centred "
              f"{line['centre_m'] * 1e3:+.2f} mm across the slide from the hole")
    print(f"start: the tip {height_mm:g} mm above the vision board; " +
          ("one plunge, then the tilt search" if method == "plunge" else "one peck search" if method == "peck" else
           f"up to {attempts} attempts{', plunge first' if s.get('plunge', True) else ''}"))

    now = datetime.datetime.now()
    out = session.path / "insert" / now.strftime("%Y-%m-%d_%H%M%S")
    approach_first = (height_mm + float(s["approach_margin_mm"])) / 1000.0
    first_argv = [s["robot_ip"], "tilt-slide", f"{CONTAINER_RUN_DIR}/attempt_1.csv", "--approach",
                  f"{approach_first:.4f}", "--probe-points", str(int(s["probe_points"])), *common,
                  "--result", f"{CONTAINER_RUN_DIR}/attempt_1.json"]
    plunge_argv = ["--plunge", f"{hole_base[2]:.6f}",
                   "--plunge-in-depth", f"{float(s.get('plunge_in_depth_mm', 3.0)) / 1000.0:.4f}"]
    if s.get("plunge", True):  # straight down first: a loose hole takes the peg without a search
        first_argv += plunge_argv
    peg_travel, hand_travel = travel_heights(s)
    use_libfranka_travel = s.get("travel", "libfranka") == "libfranka" and extract_ik_seed(probe_argv) is not None
    if args.dry_run:  # nothing is written: peg_search gets a temporary folder to mount
        out = Path(tempfile.mkdtemp(prefix="insert_dry_"))
        if use_libfranka_travel:
            seed = extract_ik_seed(probe_argv)
            t_z = float(hole_base[2]) + height_mm / 1000.0
            tr_z = float(hole_base[2]) + peg_travel if peg_travel is not None else None
            if not move_to_hole(cfg, s, out, seed, "dry_move", travel_z=tr_z, target_z=t_z,
                                tool=tool, dq=dq, dry_run=True):
                shutil.rmtree(out, ignore_errors=True)
                raise PipelineError("peg_search move refused the options (above)")
        elif ros.controller_states(cfg):
            ros.rosrun(cfg, "panda_calibration", "probe_board", *probe_argv, "--dry-run")
        else:
            print("ROS (robot.launch) is not running: the probe's IK is not checked (probe_board --dry-run needs it)")
        dry = drop_argv if method == "plunge" else peck_argv if method == "peck" else first_argv
        print(f"\npeg_search, {'the peck search' if method == 'peck' else 'attempt 1'} (dry run, no connection):")
        if peg_search(cfg, s, out, [*dry[:2], f"{CONTAINER_RUN_DIR}/dry.csv", *dry[3:], "--dry-run"]) != 0:
            raise PipelineError("peg_search refused the options (above)")
        shutil.rmtree(out, ignore_errors=True)
        print(f"\nlooks right? `pipeline insert --hole {args.hole}` moves the arm")
        return

    out.mkdir(parents=True)
    plan = {"session": session.name, "created": now.isoformat(timespec="seconds"), "hole": args.hole,
            "shape": shape, "hole_base_m": [float(v) for v in hole_base[:3]], "hole_yaw_base_deg": math.degrees(hole_yaw_base),
            "lead": lead, "lead_table": table_name, "slide_dir_base_deg": math.degrees(slide), "pattern": pattern,
            "line_plan": line,
            "start_height_mm": height_mm, "attempts": attempts, "tip_offset_tcp_mm": [float(v) * 1e3 for v in tool.tip_offset],
            "tip_source": tip_source, "settings": s, "method": method,
            "vision_hole_base_m": [float(v) for v in vision_base[:3]], "board_correction": correction,
            "joint_offsets_rad": dq, "arm_angle_deg": arm_angle_deg(tool),
            "search": search,
            "board_height": {"touches": n_touches, "z_hole_predicted_m": z_hole_pred, "z_vision_hole_m": z_vision_hole,
                             "model": hmodel},
            "peg_search": [str(v) for v in (drop_argv if method == "plunge" else peck_argv if method == "peck" else common)]}
    (out / "plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=False))

    robot_session_dir(cfg, s)
    # The probe's approach pose at the travel height with a peg (insert.peg_travel_mm, where extract leaves it), then
    # straight down to the start height.
    clearance = peg_travel - height_mm / 1000.0 if peg_travel is not None and peg_travel > height_mm / 1000.0 + 0.005 \
        else None
    if not args.here:
        if use_libfranka_travel:
            if launch_running(cfg, s):
                stop_launch(cfg, s)
            print("\n== move: approach the hole (peg_search move) ==", flush=True)
            seed = extract_ik_seed(probe_argv)
            t_z = float(hole_base[2]) + height_mm / 1000.0
            tr_z = float(hole_base[2]) + peg_travel if peg_travel is not None else None
            if not move_to_hole(cfg, s, out, seed, "approach", travel_z=tr_z, target_z=t_z, tool=tool, dq=dq):
                raise PipelineError("peg_search move to the hole failed")
        else:
            probe_argv = fast_probe_argv(cfg, s, probe_argv, out, clearance)
            ros.check_robot_ready(cfg)
            print("\n== probe: approach the hole ==")
            ros.rosrun(cfg, "panda_calibration", "probe_board", *probe_argv, "--stay", "--yes")
    else:
        print("\n--here: starting from where the arm is (within the approach above the board)")
    if launch_running(cfg, s):
        stop_launch(cfg, s)

    summary = {"attempts": [], "inserted": False}
    belief, plane, inserted, last = None, None, False, {}
    # The plunge waits for a run that gets to it: one that failed before (the posture move) left the tip over the hole.
    plunge_pending = bool(s.get("plunge", True))
    # After a posture move that failed while moving, the later runs skip it (--posture start): the same move fails the
    # same way (robot 2026-11-21: the force estimate's drift in free air tripped the backstop).
    skip_posture = False
    # Where the tip is (base z): from each result's end_tip, plus any lift since; the next approach is its height
    # above the board (the probed plane, else vision's board + its margin).
    start_z = float(hole_base[2]) + height_mm / 1000.0
    tip_z = start_z
    # A probed plane this far from vision's board at the hole is a contact taken in the air: not reused.
    plane_max_off = 0.003

    def approach_for_next() -> float:
        if plane is not None:
            above = tip_z - float(plane["point"][2]) + float(s["approach_known_margin_mm"]) / 1000.0
        else:
            above = tip_z - float(hole_base[2]) + float(s["approach_margin_mm"]) / 1000.0
        return max(above, 0.003)

    try:
        if method == "plunge":
            inserted = _plunge_then_tilt(cfg, s, out, drop_argv, common, summary, attempts, height_mm, hole_base,
                                         touch_base, lead, error_m, grid, tool=tool, dq=dq, heights=heights_ctx)
        elif method == "peck":
            print("\n== peck search: position-controlled pecks about the vision hole ==", flush=True)
            code = peg_search(cfg, s, out, peck_argv)
            result_file = out / "peck.json"
            last = json.loads(result_file.read_text()) if result_file.exists() else \
                {"ok": False, "stage": "unknown", "error": f"peg_search exited {code} without a result"}
            summary["attempts"].append({"attempt": 1, "method": "peck", "exit": code, **{
                key: last.get(key) for key in ("ok", "stage", "error", "inserted", "tried", "board_measured",
                                               "inserted_xy", "end_tip")}})
            inserted = bool(last.get("inserted"))
            if inserted:
                summary["inserted_by"] = last
                print(f"\ninserted by the peck search at {np.round(np.asarray(last['inserted_xy']) * 1e3, 2)} mm")
            elif not last.get("ok"):
                print(f"\npeck search failed ({last.get('stage')}): {last.get('error')}")
                if last.get("stage") in ("options", "connect", "unknown"):
                    summary["stopped"] = f"peg_search could not start: {last.get('error')}"
        else:
            k, tries, posture_refunded = 0, 0, False
            while tries < attempts:
                k += 1  # the run (its files); tries: the insertion attempts (a failed posture move is not one)
                tries += 1
                result_file = out / f"attempt_{k}.json"
                if k == 1:
                    argv = list(first_argv)
                else:
                    argv = [s["robot_ip"], "tilt-slide", f"{CONTAINER_RUN_DIR}/attempt_{k}.csv", "--approach",
                            f"{approach_for_next():.4f}", *common, "--result", f"{CONTAINER_RUN_DIR}/attempt_{k}.json"]
                    if plane is not None:
                        argv += ["--board-plane", *(f"{v:.6f}" for v in plane["point"]),
                                 *(f"{v:.6f}" for v in plane["slope"])]
                    else:
                        argv += ["--probe-points", str(int(s["probe_points"]))]
                    if belief is not None and grid is None:
                        argv += ["--located-xy", f"{belief['xy'][0]:.6f}", f"{belief['xy'][1]:.6f}",
                                 "--located-radius", f"{belief['radius_m']:.4f}"]
                    if plunge_pending:
                        argv += plunge_argv
                    if skip_posture:
                        argv += ["--posture", "start"]
                plan = None
                if grid is not None:
                    plan, extra = grid_args(grid, s, lead, error_m, out, k)
                    argv = argv[:argv.index("--result")] + extra + argv[argv.index("--result"):]
                where = (("plunge, then " if plunge_pending else "") + grid_text(grid, plan)) if plan is not None else \
                    (f"located search {belief['radius_m'] * 1e3:g} mm about ({belief['xy'][0] * 1e3:.2f}, "
                     f"{belief['xy'][1] * 1e3:.2f}) mm, from a {belief['source']}") if belief and k > 1 else \
                    "plunge, then the vision hole's error disc" if plunge_pending else \
                    "the vision hole's error disc"
                print(f"\n== attempt {tries}/{attempts}: {where}"
                      f"{', probed board plane reused' if plane is not None and k > 1 else ''}"
                      f"{', no posture move' if skip_posture and k > 1 else ''} ==", flush=True)
                code = peg_search(cfg, s, out, argv)
                last = json.loads(result_file.read_text()) if result_file.exists() else \
                    {"ok": False, "stage": "unknown", "error": f"peg_search exited {code} without a result"}
                if last.get("end_tip"):
                    tip_z = float(last["end_tip"][2])
                summary["attempts"].append({"attempt": k, "exit": code, "belief_used": belief, "grid_plan": plan, **{
                    key: last.get(key) for key in ("ok", "stage", "error", "plunge", "outcomes", "inserted", "hold_xy",
                                                   "dip_lead", "end_tip", "end_above_board")}})
                if grid is not None:
                    grid_update(grid, s, last, out, k)
                if last.get("plunge") not in (None, "none"):
                    print(f"plunge: {last['plunge']}")
                if last.get("stage") in ("run", "done"):
                    plunge_pending = False  # it plunged, or the run moved the tip away from over the hole
                if last.get("inserted"):
                    inserted = True
                    summary["inserted_by"] = last
                    print(f"\ninserted on attempt {tries}")
                    break
                if last.get("plane") and plane is None:
                    point, slope = last["plane"]["point"], last["plane"]["slope"]
                    at_hole = point[2] + slope[0] * (hole_base[0] - point[0]) + slope[1] * (hole_base[1] - point[1])
                    if abs(at_hole - hole_base[2]) <= plane_max_off:
                        plane = last["plane"]
                    else:
                        print(f"probed plane {(at_hole - hole_base[2]) * 1e3:+.1f} mm from the vision board at the hole: "
                              f"not reused")
                if grid is None:
                    belief = next_belief(last, lead, s, belief, prior_xy=belief["xy"] if belief else hole_base[:2],
                                         prior_radius=belief["radius_m"] if belief else error_m, max_radius=error_m)
                if belief is not None:
                    print(f"belief now: ({belief['xy'][0] * 1e3:.2f}, {belief['xy'][1] * 1e3:.2f}) mm base, "
                          f"radius {belief['radius_m'] * 1e3:.2f} mm, from a {belief['source']}")
                if last.get("ok"):
                    continue  # the run lifted back to its start height
                error = str(last.get("error"))
                print(f"\nattempt {k} failed ({last.get('stage')}): {error}")
                if last.get("stage") in ("options", "connect", "unknown"):
                    summary["stopped"] = f"peg_search could not start: {error}"
                    break
                if last.get("stage") == "posture":
                    skip_posture = True
                    print("the posture move failed: the next attempts skip it", end="")
                    if not posture_refunded:
                        posture_refunded = True
                        tries -= 1
                        print(" (not counted as an attempt)", end="")
                    print()
                if tries >= attempts:
                    break
                # Back to the start height, up (out of the hole, off the board) or down (a posture move or a lift left
                # it higher): the next descent is short (robot 2026-11-21: from 19-40 mm up the friction lag was 5-16 mm).
                dz = start_z - tip_z
                if abs(dz) > 0.001:
                    print(f"back to the start height: {'up' if dz > 0 else 'down'} {abs(dz) * 1e3:.1f} mm")
                    if not lift(cfg, s, out, dz, f"to_start_{k}", tool=tool, dq=dq):
                        summary["stopped"] = f"the move back to the start height after run {k} failed (the peg caught?)"
                        break
                    tip_z = start_z
        if inserted:
            if not getattr(args, "no_record", False):
                found = _centre_and_record(cfg, s, out, session, args, tool, entry, holes, table_name, vision_base,
                                           hole_yaw_base, board_sha, summary, hole_base, dq=dq)
                if found is not None:
                    summary["found"] = found
            # Up to the empty hand's travel height (insert.hand_travel_mm, where extract starts), else lift_m.
            release_z = float(hole_base[2]) + hand_travel if hand_travel is not None else None
            print(f"\n== release: open the hand to {s['release_width_mm']:g} mm, " +
                  (f"up to the empty-hand travel height ({hand_travel * 1e3:g} mm over the board) =="
                   if release_z is not None else f"lift {s['lift_m'] * 100:g} cm =="))
            if not open_hand(cfg, s, out):
                summary["stopped"] = "the hand did not open: not lifting (the peg is in the hole)"
            elif not lift(cfg, s, out, float(s["lift_m"]), "lift_release", tool=tool, dq=dq, result=True,
                          to_z=release_z):
                summary["stopped"] = "the lift after the release failed"
            summary["release"] = release_record(out / "lift_release.json", args.hole, shape, summary,
                                                plane if method != "plunge" else summary.get("plane"),
                                                hole_base, peck_board_z, tool, dq)
        elif "stopped" not in summary or "lift" not in summary["stopped"]:
            end_z = float(hole_base[2]) + peg_travel if peg_travel is not None else None
            print("\n== not inserted: " + (f"up to the travel height with the peg ({peg_travel * 1e3:g} mm over the "
                                           "board) ==" if end_z is not None else f"lifting {s['lift_m'] * 100:g} cm with the peg =="))
            if not lift(cfg, s, out, float(s["lift_m"]), "lift_end", tool=tool, dq=dq, to_z=end_z):
                summary["stopped"] = (summary.get("stopped", "") + "; the final lift failed").lstrip("; ")
    finally:
        summary["inserted"] = inserted
        if method != "plunge":  # _plunge_then_tilt keeps its own (2026-11-21: they were overwritten with null)
            summary.update(belief=belief, plane=plane)
        (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
        if restart_launch:
            start_launch(cfg, s)
    outcome = {"inserted": inserted, "stopped": summary.get("stopped"), "out": out, "summary": summary}
    if inserted and "stopped" not in summary:
        print(f"\ndone: {args.hole} inserted; results in {out}")
        return outcome
    reason = summary.get("stopped") or f"not inserted in {len(summary['attempts'])} attempts"
    message = f"{args.hole}: {'inserted, but ' if inserted else ''}{reason}; results in {out}"
    if raise_on_fail:
        raise PipelineError(message)
    print(f"\n{message}")
    return outcome
