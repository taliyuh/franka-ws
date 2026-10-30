#!/usr/bin/env python3
"""Capture native RGB-D image pairs from stationary cameras.

Each ENTER saves one static-scene capture as lossless PNG files, plus the
intrinsics and timestamps required to interpret them later:

  <output>/capture_0001/
      capture.json
      mega/color.png, mega/depth.png, mega/camera_info.json, mega/meta.json
      bolt/color.png, bolt/depth.png, bolt/camera_info.json, bolt/meta.json

The default topics match panda_calibration's orbbec_4k.launch (Mega under
/mega_camera, Bolt under /wrist_camera, depth registered to colour). The cameras
are not hardware synchronized, so this tool waits for one fresh,
timestamp-compatible RGB-D pair from each camera independently. The board must
remain static while those pairs arrive. The recorded inter-camera timestamps
are retained as diagnostics, but they are not an acceptance gate.

Examples:
  rosrun panda_calibration capture_stereo_rgbd
  rosrun panda_calibration capture_stereo_rgbd --output /franka_ws/captures/run_01
  rosrun panda_calibration capture_stereo_rgbd --camera left:/left_camera \\
      --camera right:/right_camera
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


@dataclass(frozen=True)
class CameraTopics:
    """ROS topic prefix and output folder for one camera."""

    name: str
    prefix: str

    @property
    def color(self) -> str:
        return f"{self.prefix}/color/image_raw"

    @property
    def depth(self) -> str:
        return f"{self.prefix}/depth/image_raw"

    @property
    def info(self) -> str:
        return f"{self.prefix}/color/camera_info"


@dataclass(frozen=True)
class ImageFrame:
    """Decoded image and its ROS and local receive timestamps."""

    image: np.ndarray
    ros_stamp_s: float
    received_at_s: float
    encoding: str


@dataclass(frozen=True)
class CameraCapture:
    """A fresh RGB-D pair and its colour-camera intrinsics."""

    color: ImageFrame
    depth: ImageFrame
    info: CameraInfo


class CameraStream:
    """Buffer recent RGB, aligned-depth, and CameraInfo messages."""

    # One decoded RGB-D pair is about 42 MiB at 3840x2160. Eight frames span
    # more than a second at the slow 5 Hz camera without retaining gigabytes.
    BUFFER_SIZE = 8

    def __init__(self, topics: CameraTopics):
        self.name = topics.name
        self.topics = topics
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._color: deque[ImageFrame] = deque(maxlen=self.BUFFER_SIZE)
        self._depth: deque[ImageFrame] = deque(maxlen=self.BUFFER_SIZE)
        self._info: CameraInfo | None = None

        # rospy runs each subscription on its own thread. Large buffers keep
        # a 4K frame from being split across reads and dropped.
        rospy.Subscriber(topics.color, Image, self._color_callback,
                         queue_size=2, buff_size=2 ** 27)
        rospy.Subscriber(topics.depth, Image, self._depth_callback,
                         queue_size=2, buff_size=2 ** 26)
        rospy.Subscriber(topics.info, CameraInfo, self._info_callback, queue_size=1)

    def _color_callback(self, message: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as error:  # noqa: BLE001 - retain prior usable frame
            print(f"{self.name}: cannot decode RGB frame: {error}")
            return
        with self._lock:
            self._color.append(image_frame(image, message))

    def _depth_callback(self, message: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        except Exception as error:  # noqa: BLE001 - retain prior usable frame
            print(f"{self.name}: cannot decode depth frame: {error}")
            return
        with self._lock:
            self._depth.append(image_frame(image, message))

    def _info_callback(self, message: CameraInfo) -> None:
        with self._lock:
            self._info = message

    def fresh_captures(self, started_at_s: float,
                       max_depth_skew_s: float) -> list[CameraCapture]:
        """Return post-keypress RGB-D candidates with compatible timestamps."""
        with self._lock:
            colors = [frame for frame in self._color
                      if frame.received_at_s >= started_at_s]
            depths = [frame for frame in self._depth
                      if frame.received_at_s >= started_at_s]
            info = self._info
        if info is None or not colors or not depths:
            return []

        captures = []
        for color in colors:
            depth = min(depths, key=lambda frame: abs(
                frame.ros_stamp_s - color.ros_stamp_s))
            if abs(color.ros_stamp_s - depth.ros_stamp_s) <= max_depth_skew_s:
                captures.append(CameraCapture(color, depth, info))
        return captures


def image_frame(image: np.ndarray, message: Image) -> ImageFrame:
    """Build an immutable timestamp wrapper around a decoded ROS image."""
    stamp = message.header.stamp
    return ImageFrame(
        image=image,
        ros_stamp_s=stamp.secs + stamp.nsecs * 1e-9,
        received_at_s=time.monotonic(),
        encoding=message.encoding,
    )


def parse_camera(text: str) -> CameraTopics:
    """Parse NAME:/namespace into the corresponding Orbbec topic group."""
    try:
        name, prefix = text.split(":", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("camera must be NAME:/namespace") from error
    if not name or not prefix.startswith("/"):
        raise argparse.ArgumentTypeError("camera must be NAME:/namespace")
    return CameraTopics(name, prefix.rstrip("/"))


def camera_info_dict(info: CameraInfo) -> dict[str, object]:
    """Return the calibration fields needed by offline pose-refinement tools."""
    return {
        "width": info.width,
        "height": info.height,
        "distortion_model": info.distortion_model,
        "d": list(info.D),
        "k": list(info.K),
        "r": list(info.R),
        "p": list(info.P),
        "binning_x": info.binning_x,
        "binning_y": info.binning_y,
    }


def next_capture_directory(output: Path) -> Path:
    """Choose a new sequential output directory without overwriting captures."""
    index = 1
    while (output / f"capture_{index:04d}").exists():
        index += 1
    return output / f"capture_{index:04d}"


def write_json(path: Path, value: dict[str, object]) -> None:
    """Write readable, stable metadata for offline tools."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_capture(output: Path, captures: dict[str, CameraCapture]) -> Path:
    """Save one multi-camera RGB-D observation without resampling either image."""
    capture_dir = next_capture_directory(output)
    capture_dir.mkdir()
    saved_at_s = time.time()
    capture_metadata: dict[str, object] = {"saved_at_unix_s": saved_at_s, "cameras": {}}

    for name, capture in captures.items():
        camera_dir = capture_dir / name
        camera_dir.mkdir()
        color_path = camera_dir / "color.png"
        depth_path = camera_dir / "depth.png"
        if not cv2.imwrite(str(color_path), capture.color.image):
            raise RuntimeError(f"failed to write {color_path}")
        if not cv2.imwrite(str(depth_path), capture.depth.image):
            raise RuntimeError(f"failed to write {depth_path}")

        write_json(camera_dir / "camera_info.json", camera_info_dict(capture.info))
        metadata = {
            "color_encoding": capture.color.encoding,
            "depth_encoding": capture.depth.encoding,
            "color_stamp_unix_s": capture.color.ros_stamp_s,
            "depth_stamp_unix_s": capture.depth.ros_stamp_s,
            "color_depth_skew_ms": 1000 * abs(capture.color.ros_stamp_s - capture.depth.ros_stamp_s),
        }
        write_json(camera_dir / "meta.json", metadata)
        capture_metadata["cameras"][name] = metadata

    camera_stamps = [capture.color.ros_stamp_s for capture in captures.values()]
    capture_metadata["camera_color_span_ms"] = 1000 * (max(camera_stamps) - min(camera_stamps))
    write_json(capture_dir / "capture.json", capture_metadata)
    return capture_dir


def color_span_s(captures: tuple[CameraCapture, ...] | list[CameraCapture]) -> float:
    """Return the timestamp span across a set of colour frames."""
    stamps = [capture.color.ros_stamp_s for capture in captures]
    return max(stamps) - min(stamps)


def wait_for_captures(streams: list[CameraStream], timeout_s: float,
                      max_depth_skew_s: float) -> dict[str, CameraCapture] | None:
    """Wait until every camera has one fresh, valid RGB-D pair.

    The board is stationary during a capture, so the cameras do not need to
    acquire their colour images at nearly the same instant. Each camera still
    has its own RGB/depth timestamp-skew check, and all returned frames must
    have arrived after this wait started.
    """
    started_at_s = time.monotonic()
    deadline = started_at_s + timeout_s
    while time.monotonic() < deadline:
        candidates = [stream.fresh_captures(started_at_s, max_depth_skew_s)
                      for stream in streams]
        if all(candidates):
            # Select each camera's newest valid pair independently. This
            # avoids waiting forever for an inter-camera timestamp match while
            # preserving a deterministic, post-keypress selection rule.
            return {
                stream.name: max(stream_candidates,
                                 key=lambda capture: capture.color.received_at_s)
                for stream, stream_candidates in zip(streams, candidates)
            }
        time.sleep(0.02)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("/franka_ws/captures/stationary"),
                        help="directory for capture_#### folders")
    parser.add_argument("--camera", type=parse_camera, action="append", default=[],
                        help="NAME:/namespace; repeat for each camera")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="seconds to wait for post-keypress frames")
    parser.add_argument("--max-depth-skew-ms", type=float, default=100.0,
                        help="maximum RGB/depth timestamp difference within one camera")
    parser.add_argument("--settle", type=float, default=1.0,
                        help="seconds to wait after ENTER before accepting new frames")
    args = parser.parse_args(rospy.myargv(sys.argv)[1:])
    if args.timeout <= 0 or args.max_depth_skew_ms < 0 or args.settle < 0:
        parser.error("timeout must be positive; skew and settle limits must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)

    cameras = args.camera or [CameraTopics("mega", "/mega_camera"),
                              CameraTopics("bolt", "/wrist_camera")]
    if len({camera.name for camera in cameras}) != len(cameras):
        parser.error("camera names must be unique")

    rospy.init_node("stationary_rgbd_capture", anonymous=True, disable_signals=True)
    streams = [CameraStream(camera) for camera in cameras]

    print("Watching: " + ", ".join(f"{camera.name} ({camera.prefix})" for camera in cameras))
    print("Let the board and depth temporal filters settle. ENTER captures; q then ENTER quits.")
    try:
        while not rospy.is_shutdown():
            command = input("\n[ENTER] capture > ").strip().lower()
            if command in {"q", "quit", "exit"}:
                break
            time.sleep(args.settle)
            captures = wait_for_captures(
                streams, args.timeout, args.max_depth_skew_ms / 1000.0)
            if captures is None:
                print("No fresh RGB-D pair was received from every camera "
                      f"with per-camera depth skew <= {args.max_depth_skew_ms:.1f} ms "
                      f"within {args.timeout:.1f} s.")
                continue
            capture_dir = save_capture(args.output, captures)
            span_ms = 1000 * color_span_s(list(captures.values()))
            print(f"Saved {capture_dir} (inter-camera RGB span {span_ms:.1f} ms; "
                  "diagnostic only)")
    except KeyboardInterrupt:
        print("\nCapture stopped.")
    except EOFError:
        pass
    finally:
        rospy.signal_shutdown("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
