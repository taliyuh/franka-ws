#!/usr/bin/env python3
"""Interactively assess live RGB and depth streams for static pose metrology.

The tool intentionally scores the *current scene* rather than guessing camera
controls.  Press ENTER after changing a light, background, or camera setting to
analyse fresh frames.  RGB metrics apply to the selected image ROI.  Depth
metrics fit the dominant plane in the table ROI, so the transparent board itself
does not need to produce valid depth.

Examples:
  ros2 run calibration_claude tune_cameras
  ros2 run calibration_claude tune_cameras --camera mega:/mega_camera \
      --camera bolt:/wrist_camera --roi 0.2,0.2,0.6,0.6
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


@dataclass(frozen=True)
class Region:
    """A normalized image rectangle."""

    x: float
    y: float
    width: float
    height: float

    def slices(self, image: np.ndarray) -> tuple[slice, slice]:
        height, width = image.shape[:2]
        x0 = round(self.x * width)
        y0 = round(self.y * height)
        x1 = round((self.x + self.width) * width)
        y1 = round((self.y + self.height) * height)
        return slice(y0, y1), slice(x0, x1)


@dataclass(frozen=True)
class CameraTopics:
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


class CameraStream:
    """Latest synchronized-enough camera messages for a static scene."""

    def __init__(self, node: Node, topics: CameraTopics):
        self.name = topics.name
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._color: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._info: CameraInfo | None = None
        self._color_count = 0
        self._depth_count = 0

        node.create_subscription(Image, topics.color, self._color_callback,
                                 qos_profile_sensor_data)
        node.create_subscription(Image, topics.depth, self._depth_callback,
                                 qos_profile_sensor_data)
        node.create_subscription(CameraInfo, topics.info, self._info_callback,
                                 qos_profile_sensor_data)

    def _color_callback(self, message: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as error:  # noqa: BLE001 - keep the last usable frame
            print(f"{self.name}: cannot decode RGB frame: {error}")
            return
        with self._lock:
            self._color = image
            self._color_count += 1

    def _depth_callback(self, message: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        except Exception as error:  # noqa: BLE001 - keep the last usable frame
            print(f"{self.name}: cannot decode depth frame: {error}")
            return
        with self._lock:
            self._depth = image
            self._depth_count += 1

    def _info_callback(self, message: CameraInfo) -> None:
        with self._lock:
            self._info = message

    def capture_fresh(self, timeout_s: float) -> tuple[np.ndarray, np.ndarray, CameraInfo] | None:
        """Return frames that arrived after this call, or ``None`` on timeout."""
        with self._lock:
            next_color = self._color_count + 1
            next_depth = self._depth_count + 1
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if (self._color_count >= next_color and self._depth_count >= next_depth
                        and self._color is not None and self._depth is not None
                        and self._info is not None):
                    return self._color.copy(), self._depth.copy(), self._info
            time.sleep(0.02)
        return None


def parse_region(text: str) -> Region:
    values = [float(value) for value in text.split(",")]
    if len(values) != 4 or any(value < 0 or value > 1 for value in values):
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height in [0, 1]")
    region = Region(*values)
    if region.x + region.width > 1 or region.y + region.height > 1:
        raise argparse.ArgumentTypeError("ROI must stay within the image")
    return region


def parse_camera(text: str) -> CameraTopics:
    try:
        name, prefix = text.split(":", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("camera must be NAME:/namespace") from error
    if not name or not prefix.startswith("/"):
        raise argparse.ArgumentTypeError("camera must be NAME:/namespace")
    return CameraTopics(name=name, prefix=prefix.rstrip("/"))


def rgb_metrics(image: np.ndarray, region: Region) -> dict[str, float]:
    roi = image[region.slices(image)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, [2, 98])
    p05, p95 = np.percentile(gray, [5, 95])
    tiles = [np.median(tile) for row in np.array_split(gray, 6, axis=0)
             for tile in np.array_split(row, 8, axis=1)]
    tile_medians = np.asarray(tiles)
    return {
        "black_pct": 100 * float(np.mean(gray <= 1)),
        "white_pct": 100 * float(np.mean(gray >= 254)),
        "p02": float(low),
        "p98": float(high),
        "contrast": float(p95 - p05),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_pct": 100 * float(np.mean(cv2.Canny(gray, 50, 150) > 0)),
        "lighting_cv_pct": 100 * float(tile_medians.std() / max(tile_medians.mean(), 1)),
    }


def depth_metrics(depth: np.ndarray, info: CameraInfo, region: Region,
                  plane_threshold_mm: float) -> dict[str, float] | None:
    if depth.ndim != 2 or info.k[0] <= 0 or info.k[4] <= 0:
        return None

    roi_y, roi_x = region.slices(depth)
    cropped = depth[roi_y, roi_x]
    step = max(1, round(np.sqrt(cropped.size / 6000)))
    sampled = cropped[::step, ::step].astype(np.float64)
    valid = np.isfinite(sampled) & (sampled > 100) & (sampled < 10000)
    coverage = 100 * float(np.mean(valid))
    if np.count_nonzero(valid) < 100:
        return {"coverage": coverage, "inlier_pct": 0.0, "rmse_mm": float("nan")}

    rows, columns = np.indices(sampled.shape)
    u = columns * step + roi_x.start
    v = rows * step + roi_y.start
    z = sampled[valid] / 1000.0
    points = np.column_stack(((u[valid] - info.k[2]) * z / info.k[0],
                              (v[valid] - info.k[5]) * z / info.k[4], z))
    if len(points) > 5000:
        points = points[np.linspace(0, len(points) - 1, 5000, dtype=int)]

    rng = np.random.default_rng(0)
    threshold_m = plane_threshold_mm / 1000.0
    best_inliers = np.zeros(len(points), dtype=bool)
    for _ in range(100):
        p0, p1, p2 = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        distances = np.abs((points - p0) @ (normal / norm))
        inliers = distances <= threshold_m
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers.sum() < 3:
        return {"coverage": coverage, "inlier_pct": 0.0, "rmse_mm": float("nan")}
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vectors = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    distances = (inlier_points - centroid) @ vectors[-1]
    return {
        "coverage": coverage,
        "inlier_pct": 100 * len(inlier_points) / len(points),
        "rmse_mm": 1000 * float(np.sqrt(np.mean(distances ** 2))),
    }


def recommendations(rgb: dict[str, float], depth: dict[str, float] | None) -> list[str]:
    advice = []
    if rgb["white_pct"] > 0.02:
        advice.append("reduce RGB exposure; clipped rim pixels cannot be fitted")
    # A dark foam surround (and the dark acrylic surface) legitimately contains
    # black pixels. Treat it as underexposure only when most of the ROI is black
    # and even its highlights remain dim.
    elif rgb["black_pct"] > 15 and rgb["p98"] < 180:
        advice.append("increase diffuse visible light or RGB exposure slightly")
    else:
        advice.append("RGB exposure is usable; keep gain at its minimum")
    if rgb["lighting_cv_pct"] > 18:
        advice.append("diffuse or reposition lights to reduce uneven illumination")
    elif rgb["contrast"] < 35:
        advice.append("increase rim/background contrast; test dark matte foam before wood")
    if depth is None or depth["coverage"] < 40:
        advice.append("depth has too few valid table pixels; use a more IR-reflective matte table surround")
    elif depth["inlier_pct"] < 55 or depth["rmse_mm"] > 4:
        advice.append("table depth is not planar enough; move the table ROI away from board edges/reflections")
    else:
        advice.append("table-depth plane is usable for initialization only")
    return advice[:2]


def print_assessment(name: str, color: np.ndarray, depth: np.ndarray, info: CameraInfo,
                     image_roi: Region, table_roi: Region, threshold_mm: float) -> None:
    rgb = rgb_metrics(color, image_roi)
    depth_result = depth_metrics(depth, info, table_roi, threshold_mm)
    print(f"\n{name}: RGB {color.shape[1]}x{color.shape[0]}, depth {depth.shape[1]}x{depth.shape[0]}")
    print("  RGB: black <=1 {black_pct:.3f}% | white >=254 {white_pct:.3f}% | "
          "P02-P98 {p02:.0f}-{p98:.0f} | contrast {contrast:.1f}".format(**rgb))
    print("       sharpness {sharpness:.1f} | edge pixels {edge_pct:.2f}% | "
          "lighting variation {lighting_cv_pct:.1f}%".format(**rgb))
    if depth_result is None:
        print("  depth: unsupported encoding or missing aligned camera intrinsics")
    elif np.isnan(depth_result["rmse_mm"]):
        print("  depth: valid table pixels {:.1f}% (not enough for a plane)".format(
            depth_result["coverage"]))
    else:
        print("  depth: valid table pixels {coverage:.1f}% | plane inliers {inlier_pct:.1f}% | "
              "plane RMSE {rmse_mm:.2f} mm".format(**depth_result))
    print("  next: " + "; ".join(recommendations(rgb, depth_result)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=parse_camera, action="append",
                        default=[], help="NAME:/namespace; repeat for each camera")
    parser.add_argument("--roi", type=parse_region, default=Region(0.2, 0.2, 0.6, 0.6),
                        help="RGB assessment ROI as x,y,width,height (default: centre 60%%)")
    parser.add_argument("--table-roi", type=parse_region, default=Region(0, 0, 1, 1),
                        help="depth plane ROI as x,y,width,height (default: whole frame)")
    parser.add_argument("--timeout", type=float, default=8.0,
                        help="seconds to wait for fresh frames after ENTER")
    parser.add_argument("--plane-threshold-mm", type=float, default=4.0,
                        help="RANSAC table-plane inlier threshold in millimetres")
    parser.add_argument("--save-dir", type=Path,
                        help="optional directory for each assessed RGB and depth frame")
    args = parser.parse_args()
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    cameras = args.camera or [CameraTopics("mega", "/mega_camera"),
                              CameraTopics("bolt", "/wrist_camera")]

    rclpy.init()
    node = Node("camera_tuning")
    streams = [CameraStream(node, camera) for camera in cameras]
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    print("Watching: " + ", ".join(f"{stream.name} ({camera.prefix})"
                                      for stream, camera in zip(streams, cameras)))
    print("Place the board in the final scene. ENTER assesses fresh frames; q then ENTER quits.")
    try:
        while rclpy.ok():
            command = input("\n[ENTER] assess > ").strip().lower()
            if command in {"q", "quit", "exit"}:
                break
            for stream in streams:
                frame = stream.capture_fresh(args.timeout)
                if frame is None:
                    print(f"{stream.name}: no fresh RGB + depth + CameraInfo within {args.timeout:.1f} s")
                    continue
                print_assessment(stream.name, *frame, args.roi, args.table_roi,
                                 args.plane_threshold_mm)
                if args.save_dir is not None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    color, depth, _ = frame
                    color_path = args.save_dir / f"{timestamp}_{stream.name}_color.png"
                    depth_path = args.save_dir / f"{timestamp}_{stream.name}_depth.png"
                    cv2.imwrite(str(color_path), color)
                    cv2.imwrite(str(depth_path), depth)
                    print(f"  saved: {color_path} and {depth_path}")
    finally:
        executor.shutdown()
        thread.join()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
