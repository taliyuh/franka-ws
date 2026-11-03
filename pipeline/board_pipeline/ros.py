"""Run ROS (Noetic) commands inside the franka_ros container.

The CLI itself runs on the host (it needs the pose-refinement Python 3.12
stack); everything that talks to the robot or the cameras runs in the Noetic
container through `docker exec`. The workspace is bind-mounted, so host paths
under paths.workspace are translated to docker.workspace.

Processes in the container run as root, so every command is followed by a
chown of the session directory back to the host user; otherwise the host
could not delete or overwrite a step's output on a rerun.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from board_pipeline.config import Config, PipelineError


# The image sources the franka_ros underlay on top of Noetic; the workspace
# overlay is sourced last (see docker/Dockerfile).
ROS_SETUP = ("source /opt/franka_underlay/install/setup.bash"
             " && source {workspace}/install/setup.bash")


def container_path(cfg: Config, path: Path) -> str:
    path = Path(path).resolve()
    workspace = cfg.workspace.resolve()
    try:
        relative = path.relative_to(workspace)
    except ValueError as error:
        raise PipelineError(f"{path} is outside the workspace {workspace}, "
                            "so the container cannot see it") from error
    return str(Path(cfg["docker"]["workspace"]) / relative)


def ensure_container(cfg: Config) -> None:
    name = cfg["docker"]["container"]
    result = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                            capture_output=True, text=True)
    if result.stdout.strip() != "true":
        raise PipelineError(
            f"container {name!r} is not running; start it with\n"
            f"  cd {cfg.workspace / 'docker'} && docker compose up -d")


def reclaim(cfg: Config, path: Path) -> None:
    """chown a host directory written by the container back to the host user."""
    subprocess.run(["docker", "exec", cfg["docker"]["container"], "chown", "-R",
                    f"{os.getuid()}:{os.getgid()}", container_path(cfg, path)],
                   capture_output=True)


def run(cfg: Config, argv: list[str], *, own: Path | None = None,
        interactive: bool = False, stdin: str | None = None) -> None:
    """Run argv in the container with ROS sourced; raise if it fails.

    own: host directory to chown back to the host user afterwards.
    interactive: attach the terminal (prompts, Ctrl-C reaches the process).
    stdin: text fed to the process instead (for scripted prompts).
    """
    ensure_container(cfg)
    workspace = cfg["docker"]["workspace"]
    inner = (f"{ROS_SETUP.format(workspace=workspace)}"
             f" && exec {shlex.join(argv)}")
    command = ["docker", "exec"]
    if interactive and sys.stdin.isatty():
        command.append("-it")
    elif stdin is not None:
        command.append("-i")
    command += [cfg["docker"]["container"], "bash", "-c", inner]
    print("  $ " + shlex.join(argv), flush=True)
    try:
        result = subprocess.run(command, input=stdin, text=True)
    finally:
        if own is not None:
            reclaim(cfg, own)
    if result.returncode != 0:
        raise PipelineError(f"`{argv[0]} {argv[1] if len(argv) > 1 else ''}` exited with "
                            f"status {result.returncode}")


def output(cfg: Config, argv: list[str], timeout: float = 20.0) -> str:
    """Run argv in the container with ROS sourced and return its stdout."""
    ensure_container(cfg)
    workspace = cfg["docker"]["workspace"]
    inner = (f"{ROS_SETUP.format(workspace=workspace)}"
             f" && timeout {timeout:g} {shlex.join(argv)}")
    return subprocess.run(["docker", "exec", cfg["docker"]["container"], "bash", "-c", inner],
                          capture_output=True, text=True).stdout


# The pipeline's motions all go to this controller (franka_ros'
# default_controllers.yaml, started by panda_calibration robot.launch).
MOTION_CONTROLLER = "position_joint_trajectory_controller"
# Command controllers that claim the same joints; while one is running the
# trajectory controller cannot be (franka_example_controllers' are all named
# *_example_controller).
MOTION_BLOCKING = ("effort_joint_trajectory_controller",)


def _blocks_motion(name: str) -> bool:
    return name in MOTION_BLOCKING or name.endswith("_example_controller")


# franka_msgs/FrankaState robot_mode is MOVE while franka_control commands the
# arm. After a reflex or a user stop it is not, but the trajectory controller
# keeps running and accepts goals that then play out with the arm standing still.
FRANKA_STATES = "/franka_state_controller/franka_states"
ROBOT_MODES = {0: "OTHER", 1: "IDLE", 2: "MOVE", 3: "GUIDING", 4: "REFLEX",
               5: "USER_STOPPED", 6: "AUTOMATIC_ERROR_RECOVERY"}
ERROR_RECOVERY = ("rostopic pub -1 /franka_control/error_recovery/goal "
                  "franka_msgs/ErrorRecoveryActionGoal \"{}\"")


def robot_mode(cfg: Config) -> int | None:
    """FrankaState robot_mode, None if no state arrives."""
    echoed = output(cfg, ["rostopic", "echo", "-n1", f"{FRANKA_STATES}/robot_mode"],
                    timeout=5.0).split()
    return int(echoed[0]) if echoed and echoed[0].isdigit() else None


def motion_errors(cfg: Config) -> list[str]:
    """Names of the flags set in FrankaState last_motion_errors."""
    listing = output(cfg, ["rostopic", "echo", "-n1", f"{FRANKA_STATES}/last_motion_errors"],
                     timeout=5.0)
    try:
        flags = yaml.safe_load(listing.split("\n---")[0]) or {}
    except yaml.YAMLError:
        return []
    return [name for name, value in flags.items() if value is True] if isinstance(flags, dict) else []


def controller_states(cfg: Config) -> dict:
    """{controller name: state} from the controller_manager, empty if none answers."""
    listing = output(cfg, ["rosservice", "call", "/controller_manager/list_controllers"])
    try:
        data = yaml.safe_load(listing) or {}
    except yaml.YAMLError:
        return {}
    return {entry["name"]: entry["state"] for entry in data.get("controller") or []
            if isinstance(entry, dict) and "name" in entry}


def check_robot_ready(cfg: Config) -> None:
    """Refuse to start a motion step the robot launch cannot execute."""
    states = controller_states(cfg)
    if not states:
        raise PipelineError("no controller_manager answered; is the robot launch running in the "
                            "container (roslaunch panda_calibration robot.launch robot_ip:=...)?")
    blocking = [name for name, state in states.items() if state == "running" and _blocks_motion(name)]
    if blocking:
        raise PipelineError(
            f"{', '.join(blocking)} is running and claims the arm joints, so the position "
            f"trajectory controller cannot run. Switch controllers:\n"
            f"  rosservice call /controller_manager/switch_controller "
            f"\"{{start_controllers: [{MOTION_CONTROLLER}], stop_controllers: [{blocking[0]}], "
            f"strictness: 2}}\"")
    state = states.get(MOTION_CONTROLLER)
    if state != "running":
        start = (f"rosrun controller_manager spawner {MOTION_CONTROLLER}" if state is None else
                 f"rosservice call /controller_manager/switch_controller "
                 f"\"{{start_controllers: [{MOTION_CONTROLLER}], stop_controllers: [], strictness: 2}}\"")
        raise PipelineError(f"{MOTION_CONTROLLER} is {state or 'not loaded'}; every pipeline motion "
                            f"is sent to it. Start it with\n  {start}")
    mode = robot_mode(cfg)
    name = ROBOT_MODES.get(mode, str(mode))
    if mode is None or name == "MOVE":
        return
    recover = f"recover in the container:\n  {ERROR_RECOVERY}"
    if name == "REFLEX":
        errors = motion_errors(cfg)
        cause = f" after {', '.join(errors)}" if errors else ""
        raise PipelineError(
            f"the robot is in reflex mode{cause}; franka_control does not move the arm until it "
            f"is recovered, so every goal would play out with the arm standing still. Check "
            f"that nothing presses on the arm or its cables, then {recover}")
    if name == "USER_STOPPED":
        raise PipelineError(f"the user stop is pressed; release it, then {recover}")
    raise PipelineError(f"robot mode is {name}, not MOVE, so franka_control is not commanding "
                        f"the arm; check Desk (joints unlocked, FCI active), then {recover}")


def rosrun(cfg: Config, package: str, executable: str, *args, **kwargs) -> None:
    run(cfg, ["rosrun", package, executable, *map(str, args)], **kwargs)
