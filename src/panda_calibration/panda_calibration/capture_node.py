"""Drive the Panda through the generated calibration poses and record the dataset.

Scope is motion + capture: for every pose this records the measured joint
vector, the resulting ``panda_link0 -> panda_hand_tcp`` transform, and one image plus
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
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from . import tag as tag_module
from .cameras import PinholeCamera
from .context import build_chain, load_camera_set, load_camera_specs, load_config
from .kinematics import PANDA_READY, matrix_to_quaternion
from .pose_generator import PoseGenerator, save_poses
from .trajectory_client import ArmDriver, spin_once, stamp_seconds


def robot_description(timeout: float = 10.0):
    """The URDF from the ``/robot_description`` parameter, or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not rospy.is_shutdown():
        if rospy.has_param('/robot_description'):
            return rospy.get_param('/robot_description')
        time.sleep(0.1)
    return None


class CameraStream:
    """Latest image + camera_info for one camera, with timestamped frames.

    Capture waits for a callback after settling and also checks the image header
    timestamp.  The second check matters on hardware: a frame can arrive after
    the arm stopped even though it was exposed before the stop.
    """

    def __init__(self, name: str, image_topic: str, info_topic: str):
        self.name = name
        self.frame = None
        self.frames_seen = 0
        self.image_stamp_s = None
        self.image_received_monotonic_s = None
        self._latest_record = None
        self.info = None
        self._bridge = CvBridge()
        # One-deep queue and a large buffer: a 4K BGRA frame is ~33 MB, and
        # only the newest image matters.
        self._subscribers = [
            rospy.Subscriber(image_topic, Image, self._on_image, queue_size=1,
                             buff_size=2 ** 27),
            rospy.Subscriber(info_topic, CameraInfo, self._on_info, queue_size=1),
        ]

    def shutdown(self) -> None:
        for subscriber in self._subscribers:
            subscriber.unregister()

    def _on_image(self, msg: Image) -> None:
        try:
            self.frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as error:  # noqa: BLE001 - report and keep going
            rospy.logwarn(f'{self.name}: cv_bridge failed: {error}')
            return
        self.image_stamp_s = stamp_seconds(msg.header.stamp)
        self.image_received_monotonic_s = time.monotonic()
        self._latest_record = {
            'header_stamp_s': float(self.image_stamp_s),
            'received_unix_s': float(time.time()),
            'received_monotonic_s': float(self.image_received_monotonic_s),
            'frame_id': msg.header.frame_id,
            'image': self.frame,
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
                    rospy.logwarn(
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
                selected['image'] = record['image'].copy()
                return selected
            spin_once(0.01)
        return None


class CaptureSession:
    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.streams = {}

    # -- setup ------------------------------------------------------------- #

    def wait_for_robot_description(self, timeout: float = 10.0):
        return robot_description(timeout)

    def start_streams(self, camera_specs: dict) -> None:
        for name, spec in camera_specs.items():
            self.streams[name] = CameraStream(name, spec['image_topic'],
                                              spec['camera_info_topic'])

    def wait_for_cameras(self, timeout: float = 30.0) -> list:
        """Block until every stream has an image and camera_info.  Returns the missing ones."""
        deadline = time.monotonic() + timeout * self.args.timeout_scale
        while time.monotonic() < deadline:
            spin_once(0.1)
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
    record = {
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
        'K': [float(v) for v in info.K],
        'D': [float(v) for v in info.D],
        'P': [float(v) for v in info.P],
        'distortion_model': info.distortion_model,
    }


def _write_yaml(path: str, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=None)


def run(session: CaptureSession, chain, poses, args) -> int:
    config = session.config
    joint_names = config['robot']['joint_names']
    driver = ArmDriver(joint_names, config['motion'], args.timeout_scale)

    rospy.loginfo(f"waiting for {config['motion']['controller_action']}")
    if not driver.wait_for_controller():
        rospy.logerr(
            f"no {config['motion']['controller_action']}; is the robot launch running?")
        return 1
    if not driver.wait_for_state():
        rospy.logerr('no /joint_states received')
        return 1
    frame_problem = driver.check_measured_tcp_frame(chain)
    if frame_problem:
        rospy.logwarn(frame_problem)

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

    sequence = [('ready', PANDA_READY)] + [(str(i), p) for i, p in enumerate(poses)]
    for label, target in sequence:
        rospy.loginfo(f'--> pose {label}')
        ok, settle_error, message = driver.move_to(target, args.speed_scale)
        if not ok:
            rospy.logerr(f'pose {label}: {message}')
            failures.append(label)
            if args.stop_on_failure:
                return 1
            continue
        if message:
            rospy.logwarn(f'  pose {label}: {message}')
        if label == 'ready':
            continue

        measured = driver.positions
        tcp_pose = chain.fk(measured)
        index = int(label)
        settled_state = driver.last_settled_state
        if settled_state is None:
            rospy.logerr(f'pose {label}: no settled state snapshot')
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
            rospy.loginfo(
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
                rospy.logwarn(f'  {name}: no fresh frame arrived')
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
            rospy.logerr(f'pose {index}: capture incomplete: {capture_error}')
            failures.append(f'pose {index}: {capture_error}')
            return 1

        detections.append({'index': index, 'cameras': per_camera})
        records.append(record)

        if detect:
            seen = [n for n, e in per_camera.items() if e.get('detected')]
            rospy.loginfo(
                f'  joint error {record["max_joint_error"]:.4f} rad, '
                f'tag seen by: {", ".join(seen) if seen else "NONE"}')

    if args.dry_run:
        rospy.loginfo(
            f'dry run complete: {len(records)}/{len(poses)} poses reached'
            + (f', failures: {failures}' if failures else ''))
        return 1 if failures else 0

    _write_yaml(os.path.join(args.out, 'poses.yaml'),
                {'convention': 'T_base_tcp maps points in panda_hand_tcp into the '
                               'robot base frame (base_link == panda_link0)',
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
        rospy.loginfo(
            f'captured {len(records)} poses -> {args.out}; '
            f'tag seen by both cameras in {both}, by none in {len(undetected)}')
        if undetected:
            rospy.logwarn(f'  no detection at poses: {undetected}')
    else:
        rospy.loginfo(f'captured {len(records)} poses -> {args.out}')
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
                        help='joint poses to drive through (yaml with a poses list)')
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

    rospy.init_node('calibration_capture', anonymous=True, disable_signals=True)
    session = CaptureSession(config, args)
    try:
        urdf = session.wait_for_robot_description()
        if urdf:
            rospy.loginfo('using the live /robot_description')
        else:
            rospy.logwarn(
                'no /robot_description; falling back to the Panda xacro')
        chain = build_chain(config, urdf)

        if not args.dry_run:
            session.start_streams(camera_specs)
            missing = session.wait_for_cameras()
            if missing:
                rospy.logerr(f'no image/camera_info from: {missing}')
                return 1
            # Live intrinsics win over whatever the config guessed.
            for name, stream in session.streams.items():
                cameras[name].apply_camera_info(stream.info)

        if args.reuse:
            from .pose_generator import load_joint_poses
            poses = load_joint_poses(args.reuse)
            rospy.loginfo(f'replaying {len(poses)} poses from {args.reuse}')
        else:
            generator = PoseGenerator(chain, cameras, config)
            generated = generator.generate(count=args.poses, seed=args.seed)
            if not generated:
                rospy.logerr('pose generation produced nothing')
                return 1
            if not args.dry_run:
                os.makedirs(args.out, exist_ok=True)
                save_poses(os.path.join(args.out, 'generated_poses.yaml'),
                           generated, config, cameras)
            visible = sum(1 for p in generated if len(p.visible_cameras) == len(cameras))
            rospy.loginfo(
                f'generated {len(generated)} poses '
                f'({visible} visible to every camera)')
            poses = [p.joints for p in generated]

        return run(session, chain, poses, args)
    finally:
        for stream in session.streams.values():
            stream.shutdown()
        rospy.signal_shutdown('done')


def _ros_stripped_argv():
    """Drop the ``name:=value`` remappings that rosrun/roslaunch append."""
    return rospy.myargv(sys.argv)[1:]


if __name__ == '__main__':
    sys.exit(main())
