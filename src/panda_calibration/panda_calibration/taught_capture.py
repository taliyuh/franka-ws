"""Replay hand-picked Cartesian poses and capture camera data after each move."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import cv2
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState

from .capture_node import CameraStream, robot_description
from .context import build_chain, load_camera_specs, load_config
from .kinematics import (
    make_transform,
    pose_error,
    quaternion_to_matrix,
)
from .trajectory_client import ArmDriver, spin_once


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


class TaughtCapture:
    def __init__(self, config: dict, camera_specs: dict, args):
        self.args = args
        self.config = config
        self.local_chain = None
        self.use_moveit_ik = None
        self.joint_names = config['robot']['joint_names']
        self.driver = ArmDriver(self.joint_names, config['motion'])
        self.ik_service = config['robot'].get('ik_service', '/compute_ik')
        self.ik_client = rospy.ServiceProxy(self.ik_service, GetPositionIK)
        self.streams = {
            name: CameraStream(name, spec['image_topic'], spec['camera_info_topic'])
            for name, spec in camera_specs.items()
        }

    def shutdown(self) -> None:
        self.driver.shutdown()
        for stream in self.streams.values():
            stream.shutdown()

    def wait_for_robot_description(self, timeout: float = 10.0) -> str | None:
        return robot_description(timeout)

    def ik_available(self, timeout: float) -> bool:
        try:
            rospy.wait_for_service(self.ik_service, timeout=timeout)
            return True
        except rospy.ROSException:
            return False

    def wait_for_inputs(self) -> bool:
        if not self.args.manual and not self.driver.wait_for_controller():
            rospy.logerr(f"trajectory controller {self.config['motion']['controller_action']} "
                         'is unavailable; is position_joint_trajectory_controller running?')
            return False
        if not self.driver.wait_for_state():
            rospy.logerr('no /joint_states received')
            return False
        self.use_moveit_ik = (
            False if self.args.manual
            else self.ik_available(2.0)
        )
        urdf = self.wait_for_robot_description()
        if urdf:
            rospy.loginfo('using the live /robot_description for TCP FK')
            self.kinematic_chain = build_chain(self.config, urdf)
        else:
            rospy.logwarn(
                'no /robot_description; falling back to the packaged xacro for TCP FK')
            self.kinematic_chain = build_chain(self.config)
        if not self.use_moveit_ik:
            rospy.logwarn('MoveIt IK unavailable; using local Panda IK')
            self.local_chain = self.kinematic_chain

        frame_problem = self.driver.check_measured_tcp_frame(self.kinematic_chain)
        if frame_problem:
            rospy.logwarn(frame_problem)

        if self.args.test:
            return True

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            spin_once(0.1)
            if all(stream.frame is not None and stream.info is not None
                   for stream in self.streams.values()):
                return True
        missing = [name for name, stream in self.streams.items()
                   if stream.frame is None or stream.info is None]
        rospy.logerr(f'no image or camera_info from: {missing}')
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
        request = GetPositionIKRequest()
        request.ik_request.group_name = self.config['robot'].get('move_group', 'panda_manipulator')
        request.ik_request.ik_link_name = self.config['robot']['tip_link']
        request.ik_request.timeout = rospy.Duration(0.5)
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

        try:
            response = self.ik_client(request)
        except rospy.ServiceException as error:
            raise RuntimeError(f'MoveIt IK call failed: {error}') from error
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
                'K': [float(value) for value in stream.info.K],
                'D': [float(value) for value in stream.info.D],
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
            rospy.logerr(
                f'pose {next_index}: arm did not remain stationary for '
                f'{float(node.config["motion"]["settle_hold"]):.3f} s')
            continue
        try:
            record = node.capture(next_index, None, None, None)
        except RuntimeError as error:
            rospy.logerr(f'pose {next_index}: {error}')
            continue
        records.append(record)
        save_records(manifest_path, records, complete=False)
        rospy.loginfo(f'captured manual pose {next_index}')
        next_index += 1

    save_records(manifest_path, records, complete=True)
    rospy.loginfo(f'completed manual capture with {len(records)} poses')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poses', help='text file containing hand-picked poses')
    parser.add_argument('--cameras', help='camera YAML with image and camera_info topics')
    parser.add_argument('--out', default='taught_capture', help='directory for poses.yaml and images')
    parser.add_argument('--config', default=None)
    parser.add_argument('--frame-id', default='panda_link0')
    parser.add_argument('--speed-scale', type=float, default=0.15)
    parser.add_argument('--resume', action='store_true',
                        help='skip samples already committed in poses.yaml')
    parser.add_argument('--manual', action='store_true',
                        help='move the arm by hand and press Enter to capture each pose')
    parser.add_argument('--test', action='store_true',
                        help='move and validate live inputs without writing any data')
    args = parser.parse_args(argv if argv is not None else rospy.myargv(sys.argv)[1:])

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

    rospy.init_node('taught_capture', anonymous=True, disable_signals=True)
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
                rospy.loginfo(f'skipping committed pose {pose_index}')
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
                rospy.logerr(f'pose {pose_index}: {error}')
                return 1
            if not ok:
                rospy.logerr(f'pose {pose_index}: {message}')
                return 1
            requested_pose = make_transform(quaternion_to_matrix(xyzw), xyz)
            measured_pose = node.kinematic_chain.fk(node.driver.positions)
            position_error, rotation_error = pose_error(requested_pose, measured_pose)
            rospy.loginfo(
                f'pose {pose_index}: measured TCP differs from requested pose by '
                f'{position_error * 1000.0:.3f} mm and '
                f'{np.degrees(rotation_error):.4f} deg; using measured pose')
            try:
                records.append(node.capture(pose_index, target, xyz, xyzw))
            except RuntimeError as error:
                rospy.logerr(f'pose {pose_index}: {error}')
                return 1
            if not args.test:
                save_records(manifest_path, records, complete=False)
            rospy.loginfo(f'captured pose {pose_index}')

        if not args.test:
            save_records(manifest_path, records, complete=True)
        return 0
    except KeyboardInterrupt:
        if not args.test:
            save_records(manifest_path, records, complete=False)
        rospy.logwarn(
            f'capture interrupted; {len(records)} committed poses remain in '
            f'{manifest_path}')
        return 130
    finally:
        node.shutdown()
        rospy.signal_shutdown('done')


if __name__ == '__main__':
    sys.exit(main())
