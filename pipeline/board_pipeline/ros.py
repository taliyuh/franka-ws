"""Run ROS 2 commands inside the franka_ros2 container.

The CLI itself runs on the host (it needs the pose-refinement Python 3.12
stack); everything that talks to the robot or the cameras runs in the Humble
container through `docker exec`. The workspace is bind-mounted, so host paths
under paths.workspace are translated to docker.workspace.

Processes in the container run as root, so every command is followed by a
chown of the session directory back to the host user; otherwise the host
could not delete or overwrite a step's output on a rerun.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from board_pipeline.config import Config, PipelineError


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
    inner = (f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash"
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
    inner = (f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash"
             f" && timeout {timeout:g} {shlex.join(argv)}")
    return subprocess.run(["docker", "exec", cfg["docker"]["container"], "bash", "-c", inner],
                          capture_output=True, text=True).stdout


# Controllers that hold a libfranka control loop. While one is active, the
# native PTP action every capture/probe tool uses fails with "Cannot perform
# this operation while another control or read operation is running".
# moveit.launch.py leaves fr3_arm_controller inactive unless activate_arm_controller:=true.
MOTION_BLOCKING = ("fr3_arm_controller",)


def check_robot_ready(cfg: Config) -> None:
    """Refuse to start a motion step the robot launch cannot execute."""
    listing = output(cfg, ["ros2", "control", "list_controllers"])
    if "active" not in listing:
        # A wedged ros2 CLI daemon ("!rclpy.ok()") makes every query come back
        # empty; restart it once before concluding the robot is not running.
        output(cfg, ["bash", "-c", "ros2 daemon stop; ros2 daemon start"])
        listing = output(cfg, ["ros2", "control", "list_controllers"])
    states = {}
    for line in re.sub(r"\x1b\[[0-9;]*m", "", listing).splitlines():
        fields = line.split()
        if len(fields) >= 2:
            states[fields[0].split("[")[0]] = fields[-1]   # "name[type] state" or "name type state"
    if not states:
        raise PipelineError("no controller_manager answered; is the robot launch running in the "
                            "container (franka_fr3_moveit_config moveit.launch.py)?")
    active = [name for name in MOTION_BLOCKING if states.get(name) == "active"]
    if active:
        raise PipelineError(
            f"{', '.join(active)} is active and holds the robot's control loop, so the PTP motions "
            "would be rejected. Deactivate it (MoveIt keeps working for IK):\n"
            f"  ros2 control set_controller_state {active[0]} inactive")


def ros2_run(cfg: Config, package: str, executable: str, *args, **kwargs) -> None:
    run(cfg, ["ros2", "run", package, executable, *map(str, args)], **kwargs)
