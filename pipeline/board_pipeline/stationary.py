"""Stationary setup: Femto Mega (+ Bolt) fixed in the cell, ArUco tag on the gripper.

calibrate: capture -> solve
    Taught capture of the tag on the gripper, then one hand-eye solve: a
    single-camera eye-to-hand solve, or a joint solve of both cameras with a
    shared tag mount (fr3_calibration solve_handeye / solve_joint_handeye).
locate:    capture -> refine -> export
    One synchronized RGB-D frame per camera, the pose-refinement stationary
    rim fit, and the composition with the hand-eye into the robot base frame.

The board pose is in the nominal-FK base frame the hand-eye was solved in, so
probe targets need no per-hole correction.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from board_pipeline import ros
from board_pipeline.config import PipelineError
from board_pipeline.session import BOARD_POSE, CALIBRATION, Context, Step, sha256


def _settings(ctx: Context) -> dict:
    return ctx.cfg["stationary"]


def _cp(ctx: Context, path) -> str:
    return ros.container_path(ctx.cfg, path)


def _robot_config(ctx: Context) -> str:
    return _cp(ctx, ctx.cfg.path(ctx.cfg["robot"]["config"]))


def _host_python(ctx: Context, script: str, *args) -> None:
    command = [sys.executable, str(ctx.cfg.pose_refinement / "boardpose" / script), *map(str, args)]
    print("  $ " + " ".join(command[1:]), flush=True)
    if subprocess.run(command).returncode != 0:
        raise PipelineError(f"{script} failed")


def make_steps(stage: str) -> list[Step]:
    def calibration_capture(ctx: Context, out: Path) -> None:
        s = _settings(ctx)
        args = ["--poses", _cp(ctx, ctx.cfg.path(s["calibration_poses"])),
                "--cameras", _cp(ctx, ctx.cfg.path(s["cameras_config"])),
                "--config", _robot_config(ctx), "--frame-id", "base",
                "--out", _cp(ctx, out / "capture"), "--speed-scale", s["speed_scale"]]
        if (out / "capture" / "poses.yaml").exists():
            args.append("--resume")
        ros.check_robot_ready(ctx.cfg)
        print("  attach the ArUco tag plate; the tool asks before every move")
        ros.ros2_run(ctx.cfg, "fr3_calibration", "run_taught_capture", *args,
                     own=out, interactive=True)

    def calibration_solve(ctx: Context, out: Path) -> None:
        s = _settings(ctx)
        capture = _cp(ctx, ctx.step_dir(stage, "capture") / "capture")
        cameras = list(s["cameras"])
        result = out / CALIBRATION
        if len(cameras) == 1:
            ros.ros2_run(ctx.cfg, "fr3_calibration", "solve_handeye", capture,
                         "--camera", cameras[0], "--camera-frame", f"{cameras[0]}_camera",
                         "--config", _robot_config(ctx), "--output", _cp(ctx, result), own=out)
        else:
            args = [capture]
            for camera in cameras:
                args += ["--camera", camera, "--camera-frame", f"{camera}={camera}_camera"]
            ros.ros2_run(ctx.cfg, "fr3_calibration", "solve_joint_handeye", *args,
                         "--config", _robot_config(ctx), "--output", _cp(ctx, result), own=out)
        (ctx.session.path / CALIBRATION).write_bytes(result.read_bytes())
        print(f"  wrote {ctx.session.path / CALIBRATION}; inspect its quality section before use")

    def board_capture(ctx: Context, out: Path) -> None:
        s = _settings(ctx)
        if not getattr(ctx.args, "yes", False):
            input("Place the board, move the arm out of the cameras' view, "
                  "and press Enter to capture... ")
        args = ["--output", _cp(ctx, out / "capture")]
        for name, namespace in s["cameras"].items():
            args += ["--camera", f"{name}:{namespace}"]
        # The tool is interactive: one ENTER captures, "q" quits.
        ros.ros2_run(ctx.cfg, "fr3_calibration", "capture_stereo_rgbd", *args,
                     own=out, stdin="\nq\n")
        if not list((out / "capture").glob("capture_*")):
            raise PipelineError("no frame was captured from every camera; are the cameras running?")

    def refine(ctx: Context, out: Path) -> None:
        s = _settings(ctx)
        calibration = ctx.session.require(CALIBRATION, "run `pipeline calibrate` first")
        capture = ctx.step_dir(stage, "capture") / "capture"
        args = ["--data", capture, "--reference-camera", s["reference_camera"],
                "--out", out / "refine.json"]
        for camera in s["cameras"]:
            args += ["--camera-base-calibration", f"{camera}={calibration}"]
        if len(s["cameras"]) > 1 and s.get("rig_calibration"):
            args += ["--optical-calibration", ctx.cfg.path(s["rig_calibration"])]
        _host_python(ctx, "stationary_multicam.py", *args)
        _host_python(ctx, "render_stationary_fit.py", "--data", capture,
                     "--src", out / "refine.json", "--outdir", out / "overlays")
        _host_python(ctx, "render_stationary_scene.py", "--src", out / "refine.json",
                     "--out", out / "overlays" / "scene_3d.png")
        print(f"  overlays: {out / 'overlays'}")

    def export(ctx: Context, out: Path) -> None:
        sys.path.insert(0, str(ctx.cfg.pose_refinement / "boardpose"))
        import export_robot_board_pose as exporter
        calibration_path = ctx.session.path / CALIBRATION
        refinement_path = ctx.step_dir(stage, "refine") / "refine.json"
        calibration = json.loads(calibration_path.read_text())
        camera = _settings(ctx)["reference_camera"] if calibration.get("schema_version") == 2 else None
        try:
            board = exporter.compose(calibration, json.loads(refinement_path.read_text()), None, camera)
        except ValueError as error:
            raise PipelineError(f"export refused: {error}") from error
        board["source"].update({"setup": "stationary",
                                "calibration": str(calibration_path),
                                "calibration_sha256": sha256(calibration_path),
                                "refinement": str(refinement_path),
                                "refinement_sha256": sha256(refinement_path)})
        text = json.dumps(board, indent=2)
        (out / BOARD_POSE).write_text(text)
        (ctx.session.path / BOARD_POSE).write_text(text)
        print(f"  wrote {ctx.session.path / BOARD_POSE}")

    if stage == "calibrate":
        return [Step("capture", "taught capture of the gripper tag", calibration_capture, resumable=True),
                Step("solve", "solve the camera-to-base calibration", calibration_solve)]
    return [Step("capture", "grab one synchronized RGB-D frame per camera", board_capture),
            Step("refine", "fit the board in the camera frame", refine),
            Step("export", "express the board in the robot base frame", export)]


def hole_target(session, cfg, hole: dict, name: str) -> Path:
    return session.require(BOARD_POSE, "run `pipeline locate` first")
