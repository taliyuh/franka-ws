"""Replay hand-picked Cartesian poses and capture camera data after each move."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .capture_node import CameraStream, LATCHED
from .context import build_chain, load_camera_specs, load_config
from .kinematics import (
    make_transform,
    pose_error,
    quaternion_to_matrix,
)
from .trajectory_client import ArmDriver


POSE_PATTERN = re.compile(
    r'pose\s+(\d+):.*?Translation:\s*\[([^]]+)\].*?'
    r'Quaternion\s*\(xyzw\)\s*\[([^]]+)\]',
    re.DOTALL | re.IGNORECASE,
)


def load_cartesian_poses(path: str) -> list[tuple[int, list[float], list[float]]]:
    """Read pose blocks with Translation and Quaternion lines."""
    with open(path, 'r', encoding='utf-8') as handle:
        content = handle.read()

    poses = []
    for index, translation, quaternion in POSE_PATTERN.findall(content):
        xyz = [float(value) for value in translation.split(',')]
        xyzw = [float(value) for value in quaternion.split(',')]
        if len(xyz) != 3 or len(xyzw) != 4:
            raise ValueError(f'pose {index} has an invalid translation or quaternion')
        poses.append((int(index), xyz, xyzw))
    if not poses:
        raise ValueError('no poses found; expected Translation and Quaternion (xyzw) lines')
    return poses


def load_records(path: str) -> list[dict]:
    """Return committed records from an interrupted taught-capture session."""
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    records = payload.get('poses')
    if not isinstance(records, list):
        raise ValueError(f'{path} does not contain a poses list')
    return records


def save_records(path: str, records: list[dict], complete: bool) -> None:
    """Atomically checkpoint the valid samples collected so far."""
    temporary_path = f'{path}.tmp'
    payload = {
        'schema_version': 1,
        'status': 'complete' if complete else 'incomplete',
        'poses': records,
    }
    with open(temporary_path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    os.replace(temporary_path, path)


class TaughtCapture(Node):
    def __init__(self, config: dict, camera_specs: dict, args):
        super().__init__('taught_capture')
        self.args = args
        self.config = config
        self.local_chain = None
        self.use_moveit_ik = None
        self.robot_description = None
        self.joint_names = config['robot']['joint_names']
        self.driver = ArmDriver(self, self.joint_names, config['motion'])
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.create_subscription(String, '/robot_description',
                                 self._on_robot_description, LATCHED)
        self.streams = {
            name: CameraStream(self, name, spec['image_topic'], spec['camera_info_topic'])
            for name, spec in camera_specs.items()
        }

    def _on_robot_description(self, message: String) -> None:
        self.robot_description = message.data

    def wait_for_robot_description(self, timeout: float = 10.0) -> str | None:
        deadline = time.monotonic() + timeout
        while self.robot_description is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.robot_description

    def wait_for_inputs(self) -> bool:
        if not self.args.manual and not self.driver.wait_for_controller():
            self.get_logger().error('trajectory controller is unavailable')
            return False
        if not self.driver.wait_for_state():
            self.get_logger().error('no /joint_states received')
            return False
        self.use_moveit_ik = (
            False if self.args.manual
            else self.ik_client.wait_for_service(timeout_sec=2.0)
        )
        urdf = self.wait_for_robot_description()
        if urdf:
            self.get_logger().info('using the live /robot_description for TCP FK')
            self.kinematic_chain = build_chain(self.config, urdf)
        else:
            self.get_logger().warn(
                'no /robot_description; falling back to the packaged xacro for TCP FK')
            self.kinematic_chain = build_chain(self.config)
        if not self.use_moveit_ik:
            self.get_logger().warn('MoveIt IK unavailable; using local FR3 IK')
            self.local_chain = self.kinematic_chain

        if self.args.test:
            return True

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(stream.frame is not None and stream.info is not None
                   for stream in self.streams.values()):
                return True
        missing = [name for name, stream in self.streams.items()
                   if stream.frame is None or stream.info is None]
        self.get_logger().error(f'no image or camera_info from: {missing}')
        return False

    def validate_ik_target(self, joints) -> list[float]:
        """Reject a distant redundant IK branch before it reaches the controller."""
        target = [float(value) for value in joints]
        maximum_step = self.config['motion'].get('max_ik_joint_step')
        if maximum_step is None:
            return target
        maximum_delta = float(np.max(np.abs(np.asarray(target) - self.driver.positions)))
        if maximum_delta > float(maximum_step):
            raise RuntimeError(
                f'IK solution changes a joint by {maximum_delta:.3f} rad '
                f'(limit {float(maximum_step):.3f}); refusing to move'
            )
        return target

    def solve_ik(self, xyz: list[float], xyzw: list[float]) -> list[float]:
        if not self.use_moveit_ik:
            target = make_transform(quaternion_to_matrix(xyzw), xyz)
            motion = self.config['motion']
            joints, converged = self.local_chain.ik_with_restarts(
                target,
                seed=self.driver.positions,
                position_tolerance=float(motion.get('max_cartesian_position_error_m', 1e-4)),
                rotation_tolerance=float(motion.get('max_cartesian_rotation_error_rad', 8.0e-4)),
            )
            if not converged:
                raise RuntimeError('local IK could not solve this pose')
            return self.validate_ik_target(joints)
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'fr3_arm'
        request.ik_request.ik_link_name = 'fr3_hand_tcp'
        request.ik_request.avoid_collisions = True
        request.ik_request.robot_state.joint_state = JointState()
        request.ik_request.robot_state.joint_state.name = self.joint_names
        request.ik_request.robot_state.joint_state.position = [
            float(value) for value in self.driver.positions
        ]
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = self.args.frame_id
        pose = request.ik_request.pose_stamped.pose
        pose.position.x, pose.position.y, pose.position.z = xyz
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = xyzw

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done():
            raise RuntimeError('timed out waiting for MoveIt IK')
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'MoveIt IK failed with error code {response.error_code.val}')

        positions = dict(zip(response.solution.joint_state.name,
                             response.solution.joint_state.position))
        missing = [name for name in self.joint_names if name not in positions]
        if missing:
            raise RuntimeError(f'MoveIt IK response is missing joints: {missing}')
        return self.validate_ik_target([positions[name] for name in self.joint_names])

    def capture(self, pose_index: int, target: list[float] | None,
                xyz: list[float] | None, xyzw: list[float] | None) -> dict:
        measured_joints = self.driver.positions
        if measured_joints is None:
            raise RuntimeError('joint state disappeared before capture was recorded')
        tcp_pose = self.kinematic_chain.fk(measured_joints)
        images, cameras = {}, {}
        for name, stream in self.streams.items():
            image_path = os.path.join('images', f'pose_{pose_index:03d}_{name}.png')
            image = stream.grab_fresh_frame(
                self.config['capture']['discard_frames'],
                self.config['capture']['frame_timeout'],
            )
            if image is None:
                raise RuntimeError(f'{name}: no fresh image arrived after the move')
            images[name] = image_path
            cameras[name] = {
                'frame_id': str(stream.info.header.frame_id),
                'width': int(stream.info.width),
                'height': int(stream.info.height),
                'K': [float(value) for value in stream.info.k],
                'D': [float(value) for value in stream.info.d],
                'image_stamp_s': None if stream.image_stamp_s is None else float(stream.image_stamp_s),
                'image_received_monotonic_s': (
                    None if stream.image_received_monotonic_s is None
                    else float(stream.image_received_monotonic_s)
                ),
            }
            if not self.args.test:
                if not cv2.imwrite(os.path.join(self.args.out, image_path), image):
                    raise RuntimeError(f'{name}: could not write {image_path}')
        return {
            'index': pose_index,
            'requested_tcp': None if xyz is None else {
                'frame_id': self.args.frame_id, 'xyz': xyz, 'quat_xyzw': xyzw,
            },
            'joint_names': self.joint_names,
            'target_joints': (
                None if target is None
                else [float(value) for value in target]
            ),
            'measured_joints': [float(value) for value in measured_joints],
            'T_base_tcp': {
                'matrix': [[float(value) for value in row] for row in tcp_pose],
            },
            'joint_state_stamp_s': self.driver.state_stamp_s,
            'joint_state_received_monotonic_s': self.driver.state_received_monotonic_s,
            'images': images,
            'cameras': cameras,
        }


def run_manual_capture(node: TaughtCapture, records: list[dict],
                       manifest_path: str) -> int:
    """Capture manually positioned samples until the operator finishes."""
    next_index = max(
        (int(record['index']) for record in records),
        default=0,
    ) + 1
    settle_timeout = float(node.config['capture']['frame_timeout'])

    while True:
        command = input(
            f'Move the arm to capture pose {next_index}, then press Enter '
            '(or type q to finish): '
        ).strip().lower()
        if command in {'q', 'quit', 'done'}:
            break

        if not node.driver.wait_until_stationary(settle_timeout):
            node.get_logger().error(
                f'pose {next_index}: arm did not remain stationary for '
                f'{float(node.config["motion"]["settle_hold"]):.3f} s')
            continue
        try:
            record = node.capture(next_index, None, None, None)
        except RuntimeError as error:
            node.get_logger().error(f'pose {next_index}: {error}')
            continue
        records.append(record)
        save_records(manifest_path, records, complete=False)
        node.get_logger().info(f'captured manual pose {next_index}')
        next_index += 1

    save_records(manifest_path, records, complete=True)
    node.get_logger().info(f'completed manual capture with {len(records)} poses')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poses', help='text file containing hand-picked poses')
    parser.add_argument('--cameras', help='camera YAML with image and camera_info topics')
    parser.add_argument('--out', default='taught_capture', help='directory for poses.yaml and images')
    parser.add_argument('--config', default=None)
    parser.add_argument('--frame-id', default='fr3_link0')
    parser.add_argument('--speed-scale', type=float, default=0.15)
    parser.add_argument('--resume', action='store_true',
                        help='skip samples already committed in poses.yaml')
    parser.add_argument('--manual', action='store_true',
                        help='move the arm by hand and press Enter to capture each pose')
    parser.add_argument('--test', action='store_true',
                        help='move and validate live inputs without writing any data')
    args = parser.parse_args(argv)

    if args.manual and args.test:
        parser.error('--manual and --test cannot be combined')
    if not args.manual and not args.poses:
        parser.error('--poses is required unless --manual is used')
    if args.manual:
        poses = []
    else:
        try:
            poses = load_cartesian_poses(args.poses)
        except (OSError, ValueError) as error:
            print(f'error: {error}', file=sys.stderr)
            return 2

    config = load_config(args.config)
    if not args.test and not args.cameras:
        parser.error('--cameras is required unless --test is used')
    camera_specs = {} if args.test else load_camera_specs(args.cameras)
    if not args.test:
        os.makedirs(os.path.join(args.out, 'images'), exist_ok=True)

    rclpy.init()
    node = TaughtCapture(config, camera_specs, args)
    manifest_path = os.path.join(args.out, 'poses.yaml')
    try:
        records = load_records(manifest_path) if args.resume else []
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f'error: could not load --resume manifest: {error}', file=sys.stderr)
        return 2
    committed_indices = {record.get('index') for record in records}
    if len(committed_indices) != len(records) or None in committed_indices:
        print('error: --resume manifest has missing or duplicate pose indices', file=sys.stderr)
        return 2
    try:
        if not node.wait_for_inputs():
            return 1
        if args.manual:
            return run_manual_capture(node, records, manifest_path)
        for position, (pose_index, xyz, xyzw) in enumerate(poses):
            if pose_index in committed_indices:
                node.get_logger().info(f'skipping committed pose {pose_index}')
                continue
            input(f'Press Enter to move to pose {pose_index} ({position + 1}/{len(poses)}), '
                  'or Ctrl-C to stop... ')
            try:
                target = node.solve_ik(xyz, xyzw)
                ok, _, message = node.driver.move_to(
                    target,
                    args.speed_scale,
                )
            except RuntimeError as error:
                node.get_logger().error(f'pose {pose_index}: {error}')
                return 1
            if not ok:
                node.get_logger().error(f'pose {pose_index}: {message}')
                return 1
            requested_pose = make_transform(quaternion_to_matrix(xyzw), xyz)
            measured_pose = node.kinematic_chain.fk(node.driver.positions)
            position_error, rotation_error = pose_error(requested_pose, measured_pose)
            node.get_logger().info(
                f'pose {pose_index}: measured TCP differs from requested pose by '
                f'{position_error * 1000.0:.3f} mm and '
                f'{np.degrees(rotation_error):.4f} deg; using measured pose')
            try:
                records.append(node.capture(pose_index, target, xyz, xyzw))
            except RuntimeError as error:
                node.get_logger().error(f'pose {pose_index}: {error}')
                return 1
            if not args.test:
                save_records(manifest_path, records, complete=False)
            node.get_logger().info(f'captured pose {pose_index}')

        if not args.test:
            save_records(manifest_path, records, complete=True)
        return 0
    except KeyboardInterrupt:
        if not args.test:
            save_records(manifest_path, records, complete=False)
        node.get_logger().warn(
            f'capture interrupted; {len(records)} committed poses remain in '
            f'{manifest_path}')
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
