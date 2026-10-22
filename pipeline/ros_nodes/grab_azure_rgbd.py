"""Save one Azure RGB + depth_to_rgb + camera_info + measured joints, for the board fitter.

Writes <out>/color.png, depth.png (uint16 mm, aligned to RGB), camera_info.json
(pose-refinement layout, all distortion terms kept) and joints.json.
Run while the robot is stationary.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState


def to_array(msg):
    dtype = np.uint16 if msg.encoding in ('16UC1', 'mono16') else np.uint8
    channels = {'bgra8': 4, 'rgba8': 4, 'bgr8': 3, 'rgb8': 3}.get(msg.encoding, 1)
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
    if msg.encoding in ('bgra8', 'rgba8'):
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR if msg.encoding == 'bgra8' else cv2.COLOR_RGBA2BGR)
    elif msg.encoding == 'rgb8':
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr.squeeze()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('--ns', default='/azure_camera')
    p.add_argument('--timeout', type=float, default=15.0)
    a = p.parse_args()

    rclpy.init()
    node = Node('grab_azure_rgbd')
    start = time.time()
    got = {}

    def keep(key):
        def cb(msg):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else 0
            if key in ('color', 'depth') and stamp < start + 0.5:
                return  # only frames exposed after we started (robot already stationary)
            got[key] = msg
        return cb

    node.create_subscription(Image, f'{a.ns}/rgb/image_raw', keep('color'), qos_profile_sensor_data)
    node.create_subscription(Image, f'{a.ns}/depth_to_rgb/image_raw', keep('depth'), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, f'{a.ns}/rgb/camera_info', keep('info'), qos_profile_sensor_data)
    node.create_subscription(JointState, '/franka_robot_state_broadcaster/measured_joint_states',
                             keep('joints'), qos_profile_sensor_data)
    while len(got) < 4 and time.time() - start < a.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    missing = {'color', 'depth', 'info', 'joints'} - set(got)
    if missing:
        raise SystemExit(f'timed out waiting for {sorted(missing)}')

    a.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out / 'color.png'), to_array(got['color']))
    depth = to_array(got['depth'])
    if depth.dtype != np.uint16:
        raise SystemExit(f'unexpected depth encoding {got["depth"].encoding}')
    cv2.imwrite(str(a.out / 'depth.png'), depth)
    info = got['info']
    (a.out / 'camera_info.json').write_text(json.dumps({
        'width': info.width, 'height': info.height, 'distortion_model': info.distortion_model,
        'd': list(info.d), 'k': list(info.k), 'r': list(info.r), 'p': list(info.p),
        'frame_id': info.header.frame_id}, indent=2))
    j = got['joints']
    pos = dict(zip(j.name, j.position))
    (a.out / 'joints.json').write_text(json.dumps({
        'joint_names': [f'fr3_joint{i}' for i in range(1, 8)],
        'measured_joints': [pos[f'fr3_joint{i}'] for i in range(1, 8)],
        'color_stamp_s': got['color'].header.stamp.sec + got['color'].header.stamp.nanosec * 1e-9,
    }, indent=2))
    print(f'saved {a.out}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
