"""Save one Azure RGB + depth_to_rgb + camera_info + measured joints, for the board fitter.

Writes <out>/color.png, depth.png (uint16 mm, aligned to RGB), camera_info.json
(pose-refinement layout, all distortion terms kept) and joints.json.
Run while the robot is stationary.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image, JointState

JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]


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
    p.add_argument('--joints', default='/franka_state_controller/joint_states')
    p.add_argument('--timeout', type=float, default=15.0)
    a = p.parse_args(rospy.myargv(sys.argv)[1:])

    rospy.init_node('grab_azure_rgbd', anonymous=True, disable_signals=True)
    start = time.time()
    got = {}

    def keep(key):
        def cb(msg):
            if key in ('color', 'depth') and msg.header.stamp.to_sec() < start + 0.5:
                return  # only frames exposed after we started (robot already stationary)
            if key == 'joints' and not set(JOINT_NAMES) <= set(msg.name):
                return
            got[key] = msg
        return cb

    # buff_size: a 4K BGRA frame is ~33 MB; the default buffer would drop it.
    rospy.Subscriber(f'{a.ns}/rgb/image_raw', Image, keep('color'), queue_size=1, buff_size=2 ** 27)
    rospy.Subscriber(f'{a.ns}/depth_to_rgb/image_raw', Image, keep('depth'), queue_size=1,
                     buff_size=2 ** 26)
    rospy.Subscriber(f'{a.ns}/rgb/camera_info', CameraInfo, keep('info'), queue_size=1)
    rospy.Subscriber(a.joints, JointState, keep('joints'), queue_size=1)
    while len(got) < 4 and time.time() - start < a.timeout:
        time.sleep(0.05)
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
        'd': list(info.D), 'k': list(info.K), 'r': list(info.R), 'p': list(info.P),
        'frame_id': info.header.frame_id}, indent=2))
    j = got['joints']
    pos = dict(zip(j.name, j.position))
    (a.out / 'joints.json').write_text(json.dumps({
        'joint_names': JOINT_NAMES,
        'measured_joints': [pos[name] for name in JOINT_NAMES],
        'color_stamp_s': got['color'].header.stamp.to_sec(),
    }, indent=2))
    print(f'saved {a.out}')
    rospy.signal_shutdown('done')


if __name__ == '__main__':
    main()
