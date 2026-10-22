"""Drive the FR3 through the generated calibration poses and record the dataset.

Scope is motion + capture: for every pose this records the measured joint
vector, the resulting ``base -> fr3_hand_tcp`` transform, and one image plus
``camera_info`` per camera.  Solving for ``base -> camera`` is left to whatever
downstream tool you prefer; ``meta.yaml`` states which convention the recorded
transforms use.

An AprilTag detection runs on every captured image, but purely as a check that
the tag really was visible -- it is not part of a calibration solve.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from . import tag as tag_module
from .cameras import PinholeCamera
from .context import build_chain, load_camera_set, load_camera_specs, load_config
from .kinematics import FR3_READY, matrix_to_quaternion
from .pose_generator import PoseGenerator, save_poses
from .trajectory_client import ArmDriver

LATCHED = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class CameraStream:
    """Latest image + camera_info for one camera, with timestamped frames.

    Capture waits for a callback after settling and also checks the image header
    timestamp.  The second check matters on hardware: a frame can arrive after
    the arm stopped even though it was exposed before the stop.
    """

    def __init__(self, node: Node, name: str, image_topic: str, info_topic: str):
        self.node = node
        self.name = name
        self.frame = None
        self.frames_seen = 0
        self.image_stamp_s = None
        self.image_received_monotonic_s = None
        self._latest_record = None
        self.info = None
        self._bridge = CvBridge()
        node.create_subscription(Image, image_topic, self._on_image,
                                 qos_profile_sensor_data)
        node.create_subscription(CameraInfo, info_topic, self._on_info, 1)

    def _on_image(self, msg: Image) -> None:
        try:
            self.frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as error:  # noqa: BLE001 - report and keep going
            self.node.get_logger().warn(f'{self.name}: cv_bridge failed: {error}')
            return
        stamp = msg.header.stamp
        self.image_stamp_s = stamp.sec + stamp.nanosec * 1e-9
        self.image_received_monotonic_s = time.monotonic()
        self._latest_record = {
            'header_stamp_s': float(self.image_stamp_s),
            'received_unix_s': float(time.time()),
            'received_monotonic_s': float(self.image_received_monotonic_s),
            'frame_id': msg.header.frame_id,
        }
        self.frames_seen += 1

    def _on_info(self, msg: CameraInfo) -> None:
        self.info = msg

    def grab_fresh_frame(self, discard: int, timeout: float):
        """Compatibility wrapper returning only the selected image."""
        record = self.grab_fresh_frame_record(discard, timeout)
        return None if record is None else record['image']

    def grab_fresh_frame_record(self, discard: int, timeout: float,
                                minimum_header_stamp_s: float | None = None,
                                minimum_received_monotonic_s: float | None = None):
        """Return one image and its timing metadata after the supplied gates.

        ``discard`` is retained as a small settling buffer, but it is applied
        only to frames that already pass the timestamp and callback-time gates.
        A late callback for an older exposure is therefore rejected rather than
        counted as a fresh frame.
        """
        first_frame_count = self.frames_seen
        processed_frame_count = first_frame_count
        valid_frames = 0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.frames_seen > processed_frame_count:
                processed_frame_count = self.frames_seen
                record = self._latest_record
                if record is None:
                    continue
                if (minimum_received_monotonic_s is not None
                        and record['received_monotonic_s'] <= minimum_received_monotonic_s):
                    continue
                if (minimum_header_stamp_s is not None
                        and record['header_stamp_s'] < minimum_header_stamp_s):
                    self.node.get_logger().warn(
                        f'{self.name}: rejecting late frame with header timestamp '
                        f'{record["header_stamp_s"]:.6f} before settle '
                        f'{minimum_header_stamp_s:.6f}')
                    continue
                if valid_frames < discard:
                    valid_frames += 1
                    continue
                selected = dict(record)
                selected['frames_seen_before_request'] = int(first_frame_count)
                selected['frames_discarded_after_gates'] = int(valid_frames)
                selected['image'] = self.frame.copy()
                return selected
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return None


class CaptureSession(Node):
    def __init__(self, config: dict, args):
        super().__init__('calibration_capture')
        self.config = config
        self.args = args
        self.robot_description = None

        self.create_subscription(String, '/robot_description',
                                 self._on_description, LATCHED)
        self.streams = {}

    def _on_description(self, msg: String) -> None:
        self.robot_description = msg.data

    # -- setup ------------------------------------------------------------- #

    def wait_for_robot_description(self, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while self.robot_description is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.robot_description

    def start_streams(self, camera_specs: dict) -> None:
        for name, spec in camera_specs.items():
            self.streams[name] = CameraStream(self, name, spec['image_topic'],
                                              spec['camera_info_topic'])

    def wait_for_cameras(self, timeout: float = 30.0) -> list:
        """Block until every stream has an image and camera_info.  Returns the missing ones."""
        deadline = time.monotonic() + timeout * self.args.timeout_scale
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(s.frame is not None and s.info is not None for s in self.streams.values()):
                return []
        return [name for name, s in self.streams.items()
                if s.frame is None or s.info is None]


def _pose_record(index: int, target, measured, tcp_pose, joint_names,
                 settled_state: dict | None = None) -> dict:
    """One dataset row.

    ``T_base_tcp`` is computed from the *measured* joints, so a joint that
    settles slightly off its target costs accuracy in tag visibility but not in
    the recorded hand pose.
    """
    return {
        'index': int(index),
        'joint_names': list(joint_names),
        'target_joints': [round(float(v), 6) for v in target],
        'measured_joints': [round(float(v), 6) for v in measured],
        'max_joint_error': round(float(np.max(np.abs(measured - target))), 6),
        'T_base_tcp': {
            'xyz': [round(float(v), 6) for v in tcp_pose[:3, 3]],
            'quat_xyzw': [round(float(v), 6) for v in matrix_to_quaternion(tcp_pose[:3, :3])],
            'matrix': [[round(float(v), 6) for v in row] for row in tcp_pose],
        },
    }
    if settled_state is not None:
        record['settled_state'] = settled_state
    return record


def _state_snapshot_record(snapshot: dict | None) -> dict | None:
    if snapshot is None:
        return None
    record = {
        'header_stamp_s': float(snapshot['header_stamp_s']),
        'received_monotonic_s': float(snapshot['received_monotonic_s']),
        'q_rad': [round(float(value), 9) for value in snapshot['q_rad']],
    }
    if 'window_joint_spread_rad' in snapshot:
        record['window_joint_spread_rad'] = [
            round(float(value), 9) for value in snapshot['window_joint_spread_rad']
        ]
    return record


def _image_timing_record(frame_record: dict, settled_state: dict,
                         joint_bracket: dict | None) -> dict:
    record = {
        'image_stamp_s': float(frame_record['header_stamp_s']),
        'image_received_unix_s': float(frame_record['received_unix_s']),
        'image_received_monotonic_s': float(frame_record['received_monotonic_s']),
        'image_frame_id': frame_record['frame_id'],
        'image_header_minus_settled_header_ms': round(
            (frame_record['header_stamp_s'] - settled_state['header_stamp_s']) * 1000.0, 3),
        'image_received_minus_settled_received_ms': round(
            (frame_record['received_monotonic_s']
             - settled_state['received_monotonic_s']) * 1000.0, 3),
        'frames_discarded_after_timestamp_gate': frame_record['frames_discarded_after_gates'],
    }
    if joint_bracket is not None:
        before, after = joint_bracket
        record['joint_before_image'] = _state_snapshot_record(before)
        record['joint_after_image'] = _state_snapshot_record(after)
        record['joint_bracket_interval_ms'] = round(
            (after['received_monotonic_s'] - before['received_monotonic_s']) * 1000.0, 3)
        record['joint_bracket_max_delta_rad'] = round(float(np.max(
            np.abs(np.asarray(after['q_rad']) - np.asarray(before['q_rad'])))), 9)
    else:
        record['joint_bracket_error'] = 'no state sample after image before timeout'
    return record


def _camera_info_record(info) -> dict:
    return {
        'frame_id': info.header.frame_id,
        'width': int(info.width),
        'height': int(info.height),
        'K': [float(v) for v in info.k],
        'D': [float(v) for v in info.d],
        'P': [float(v) for v in info.p],
        'distortion_model': info.distortion_model,
    }


def _write_yaml(path: str, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=None)


def run(session: CaptureSession, chain, poses, args) -> int:
    config = session.config
    joint_names = config['robot']['joint_names']
    driver = ArmDriver(session, joint_names, config['motion'], args.timeout_scale)

    session.get_logger().info(f"waiting for {config['motion']['controller_action']}")
    if not driver.wait_for_controller():
        session.get_logger().error(
            'no trajectory controller. In sim, launch '
            'calibration_claude/sim_calibration.launch.py; on the real robot make '
            'sure fr3_arm_controller is active.')
        return 1
    if not driver.wait_for_state():
        session.get_logger().error('no /joint_states received')
        return 1

    capture_config = config['capture']
    detect = capture_config['detect'] and not args.no_detect
    timestamp_gate = capture_config.get('require_post_settle_timestamp', False)
    require_all_cameras = capture_config.get('require_all_cameras', False)
    image_dir = os.path.join(args.out, 'images')
    if not args.dry_run:
        os.makedirs(image_dir, exist_ok=True)

    records = []
    detections = []
    failures = []

    sequence = [('ready', FR3_READY)] + [(str(i), p) for i, p in enumerate(poses)]
    for label, target in sequence:
        session.get_logger().info(f'--> pose {label}')
        ok, settle_error, message = driver.move_to(target, args.speed_scale)
        if not ok:
            session.get_logger().error(f'pose {label}: {message}')
            failures.append(label)
            if args.stop_on_failure:
                return 1
            continue
        if message:
            session.get_logger().warn(f'  pose {label}: {message}')
        if label == 'ready':
            continue

        measured = driver.positions
        tcp_pose = chain.fk(measured)
        index = int(label)
        settled_state = driver.last_settled_state
        if settled_state is None:
            session.get_logger().error(f'pose {label}: no settled state snapshot')
            failures.append(label)
            if args.stop_on_failure:
                return 1
            continue
        record = _pose_record(
            index,
            target,
            measured,
            tcp_pose,
            joint_names,
            _state_snapshot_record(settled_state),
        )
        record['settle_error'] = round(float(settle_error), 6)

        if args.dry_run:
            records.append(record)
            session.get_logger().info(
                f'  reached, max joint error {record["max_joint_error"]:.4f} rad')
            continue

        per_camera = {}
        capture_error = None
        for name, stream in session.streams.items():
            frame_record = stream.grab_fresh_frame_record(
                capture_config['discard_frames'],
                capture_config['frame_timeout'] * args.timeout_scale,
                (None if not timestamp_gate else settled_state['header_stamp_s']),
                (None if not timestamp_gate else settled_state['received_monotonic_s']),
            )
            if frame_record is None:
                session.get_logger().warn(f'  {name}: no fresh frame arrived')
                per_camera[name] = {'detected': False, 'error': 'no frames'}
                capture_error = f'{name}: no post-settle frame passed the timestamp gate'
                continue
            image = frame_record['image']
            joint_bracket = driver.wait_for_joint_after(
                frame_record['received_monotonic_s'],
                capture_config['frame_timeout'],
            )
            if joint_bracket is None:
                capture_error = f'{name}: no joint sample bracketed the image'
                per_camera[name] = {
                    'detected': False,
                    'error': 'no joint sample after image before timeout',
                }
                continue
            path = os.path.join(image_dir, f'pose_{index:03d}_{name}.png')
            if not cv2.imwrite(path, image):
                capture_error = f'{name}: failed to save image'
                per_camera[name] = {'detected': False, 'error': 'image save failed'}
                continue
            entry = {'image': os.path.relpath(path, args.out)}
            entry.update(_image_timing_record(
                frame_record,
                settled_state,
                joint_bracket,
            ))
            if detect:
                detection = tag_module.detect(
                    image, int(config['tag']['id']), str(config['tag']['family']))
                entry.update(detection.as_dict())
                if detection.detected:
                    cv2.imwrite(
                        os.path.join(image_dir, f'pose_{index:03d}_{name}_overlay.png'),
                        tag_module.draw(image, detection))
            per_camera[name] = entry

        if capture_error and require_all_cameras:
            session.get_logger().error(f'pose {index}: capture incomplete: {capture_error}')
            failures.append(f'pose {index}: {capture_error}')
            return 1

        detections.append({'index': index, 'cameras': per_camera})
        records.append(record)

        if detect:
            seen = [n for n, e in per_camera.items() if e.get('detected')]
            session.get_logger().info(
                f'  joint error {record["max_joint_error"]:.4f} rad, '
                f'tag seen by: {", ".join(seen) if seen else "NONE"}')

    if args.dry_run:
        session.get_logger().info(
            f'dry run complete: {len(records)}/{len(poses)} poses reached'
            + (f', failures: {failures}' if failures else ''))
        return 1 if failures else 0

    _write_yaml(os.path.join(args.out, 'poses.yaml'),
                {'convention': 'T_base_tcp maps points in fr3_hand_tcp into the '
                               'robot base frame (base_link == world == fr3_link0)',
                 'poses': records})
    _write_yaml(os.path.join(args.out, 'detections.yaml'), {'poses': detections})
    _write_yaml(os.path.join(args.out, 'meta.yaml'), {
        'base_link': config['robot']['base_link'],
        'tip_link': config['robot']['tip_link'],
        'joint_names': joint_names,
        'tag': dict(config['tag']),
        'cameras': {name: _camera_info_record(stream.info)
                    for name, stream in session.streams.items()
                    if stream.info is not None},
        'capture': {
            'speed_scale': args.speed_scale,
            'poses_requested': len(poses),
            'poses_recorded': len(records),
            'failures': failures,
            'require_post_settle_timestamp': capture_config.get(
                'require_post_settle_timestamp', False),
            'require_all_cameras': require_all_cameras,
        },
    })

    if detect:
        undetected = [d['index'] for d in detections
                      if not any(e.get('detected') for e in d['cameras'].values())]
        both = sum(1 for d in detections
                   if all(e.get('detected') for e in d['cameras'].values()))
        session.get_logger().info(
            f'captured {len(records)} poses -> {args.out}; '
            f'tag seen by both cameras in {both}, by none in {len(undetected)}')
        if undetected:
            session.get_logger().warn(f'  no detection at poses: {undetected}')
    else:
        session.get_logger().info(f'captured {len(records)} poses -> {args.out}')
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--cameras', default=None)
    parser.add_argument('--out', default='calibration_dataset',
                        help='directory to write the dataset into')
    parser.add_argument('--poses', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--reuse', default=None,
                        help='replay a pose set previously written by preview_poses')
    parser.add_argument('--dry-run', action='store_true',
                        help='move through the poses without capturing anything')
    parser.add_argument('--speed-scale', type=float, default=1.0,
                        help='scales motion.max_joint_speed; use <1 on hardware')
    parser.add_argument('--timeout-scale', type=float, default=1.0,
                        help='stretch every wall-clock timeout (sim with RTF < 1)')
    parser.add_argument('--no-detect', action='store_true',
                        help='skip the AprilTag visibility check on captured images')
    parser.add_argument('--stop-on-failure', action='store_true',
                        help='abort at the first pose the arm fails to reach')
    args = parser.parse_args(argv if argv is not None else _ros_stripped_argv())

    config = load_config(args.config)
    camera_specs = load_camera_specs(args.cameras)
    # Replayed joint targets do not need static camera poses.  The hardware
    # topic config intentionally contains only image/camera_info topics, and
    # live CameraInfo messages replace these placeholder values before capture.
    # Generated-pose capture still needs full camera geometry for visibility
    # scoring, so retain the strict loader in that path.
    if args.reuse:
        cameras = {
            name: PinholeCamera(
                name=name,
                width=1,
                height=1,
                fx=1.0,
                fy=1.0,
                cx=0.0,
                cy=0.0,
                pose=np.eye(4),
            )
            for name in camera_specs
        }
    else:
        cameras = load_camera_set(args.cameras)

    rclpy.init()
    session = CaptureSession(config, args)
    try:
        urdf = session.wait_for_robot_description()
        if urdf:
            session.get_logger().info('using the live /robot_description')
            if 'calib_tag' not in urdf:
                session.get_logger().warn(
                    'the live /robot_description has no calib_tag link -- something '
                    'else is publishing it (a stale franka launch?), so the robot in '
                    'Gazebo may not be carrying the tag')
        else:
            session.get_logger().warn(
                'no /robot_description; falling back to the packaged xacro')
        chain = build_chain(config, urdf)

        if not args.dry_run:
            session.start_streams(camera_specs)
            missing = session.wait_for_cameras()
            if missing:
                session.get_logger().error(f'no image/camera_info from: {missing}')
                return 1
            # Live intrinsics win over whatever the config guessed.
            for name, stream in session.streams.items():
                cameras[name].apply_camera_info(stream.info)

        if args.reuse:
            from .pose_generator import load_joint_poses
            poses = load_joint_poses(args.reuse)
            session.get_logger().info(f'replaying {len(poses)} poses from {args.reuse}')
        else:
            generator = PoseGenerator(chain, cameras, config)
            generated = generator.generate(count=args.poses, seed=args.seed)
            if not generated:
                session.get_logger().error('pose generation produced nothing')
                return 1
            if not args.dry_run:
                os.makedirs(args.out, exist_ok=True)
                save_poses(os.path.join(args.out, 'generated_poses.yaml'),
                           generated, config, cameras)
            visible = sum(1 for p in generated if len(p.visible_cameras) == len(cameras))
            session.get_logger().info(
                f'generated {len(generated)} poses '
                f'({visible} visible to every camera)')
            poses = [p.joints for p in generated]

        return run(session, chain, poses, args)
    finally:
        session.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _ros_stripped_argv():
    """Drop the ``--ros-args ...`` tail that ros2 run/launch appends."""
    argv = sys.argv[1:]
    return argv[:argv.index('--ros-args')] if '--ros-args' in argv else argv


if __name__ == '__main__':
    sys.exit(main())
