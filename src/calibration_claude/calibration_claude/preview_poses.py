"""Offline dry-run: generate calibration poses and report predicted tag visibility.

Runs without ROS, Gazebo or a robot -- the whole point is to find out whether a
pose set is worth executing *before* anything moves.  Prints a per-pose table and
optionally renders, for each camera, the tag quads it is predicted to see.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .context import build_chain, load_camera_set, load_config
from .pose_generator import PoseGenerator, save_poses


def _print_table(poses: list, camera_names: list) -> None:
    header = f"{'#':>3}  {'tag xyz (m)':<22}"
    for name in camera_names:
        header += f'  {name:<26}'
    print(header)
    print('-' * len(header))
    for pose in poses:
        xyz = pose.tag_pose[:3, 3]
        row = f'{pose.index:>3}  [{xyz[0]:5.2f} {xyz[1]:5.2f} {xyz[2]:5.2f}]      '
        for name in camera_names:
            view = pose.views[name]
            mark = 'OK ' if view.visible else '-- '
            row += f'  {mark}{view.summary():<23}'
        print(row)


def _print_summary(poses: list, camera_names: list, config: dict) -> bool:
    print()
    ok = True
    # Asking for fewer poses than the per-camera minimum is a legitimate request,
    # not a failure -- cap the bar at what was actually asked for.
    minimum = min(int(config['pose_generation']['min_poses_per_camera']), len(poses))
    for name in camera_names:
        seen = [p for p in poses if name in p.visible_cameras]
        sizes = [p.views[name].apparent_px for p in seen]
        status = 'ok' if len(seen) >= minimum else f'LOW (want >= {minimum})'
        detail = (f'apparent {min(sizes):.0f}-{max(sizes):.0f} px' if sizes else '')
        print(f'  {name:<14} {len(seen):>2}/{len(poses)} poses  {detail:<24} {status}')
        ok = ok and len(seen) >= minimum

    both = sum(1 for p in poses if len(p.visible_cameras) == len(camera_names))
    none = sum(1 for p in poses if not p.visible_cameras)
    print(f'  {"both cameras":<14} {both:>2}/{len(poses)} poses')
    if none:
        print(f'  WARNING: {none} pose(s) visible to no camera')
        ok = False

    rotations = [p.tcp_pose[:3, :3] for p in poses]
    if len(rotations) > 1:
        from .kinematics import rotation_angle_between
        spreads = [np.degrees(rotation_angle_between(a, b))
                   for i, a in enumerate(rotations) for b in rotations[i + 1:]]
        print(f'  rotation spread: min {min(spreads):.1f} deg, '
              f'median {np.median(spreads):.1f} deg, max {max(spreads):.1f} deg')
        if min(spreads) < 5.0:
            print('  WARNING: two poses are nearly identical in orientation')

    smallest = min((p.views[n].apparent_px for p in poses for n in p.visible_cameras),
                   default=0.0)
    if 0.0 < smallest < 45.0:
        print(f'  NOTE: smallest visible tag is {smallest:.0f} px; detection still '
              'works down to ~35 px but corner noise grows quickly below 45 px')
    return ok


def _render_overlays(poses: list, cameras: dict, directory: str) -> None:
    import cv2
    os.makedirs(directory, exist_ok=True)
    for name, camera in sorted(cameras.items()):
        canvas = np.full((camera.height, camera.width, 3), 32, np.uint8)
        cv2.rectangle(canvas, (0, 0), (camera.width - 1, camera.height - 1), (80, 80, 80), 1)
        for pose in poses:
            view = pose.views[name]
            if view.corners_px is None:
                continue
            colour = (80, 220, 80) if view.visible else (70, 70, 200)
            quad = view.corners_px.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [quad], True, colour, 1)
            centre = view.corners_px.mean(axis=0).astype(int)
            cv2.putText(canvas, str(pose.index), tuple(centre),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
        cv2.putText(canvas, f'{name}: predicted tag quads', (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        path = os.path.join(directory, f'predicted_{name}.png')
        cv2.imwrite(path, canvas)
        print(f'  wrote {path}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None, help='calibration.yaml to use')
    parser.add_argument('--cameras', default=None, help='cameras yaml to use')
    parser.add_argument('--poses', type=int, default=None, help='number of poses')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', default=None,
                        help='write the generated pose set to this yaml')
    parser.add_argument('--overlay-dir', default=None,
                        help='render predicted tag quads per camera into this directory')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    cameras = load_camera_set(args.cameras)
    chain = build_chain(config)

    generator = PoseGenerator(chain, cameras, config)
    poses = generator.generate(count=args.poses, seed=args.seed)
    if not poses:
        print('no feasible poses found -- widen pose_generation.extents or relax '
              'the visibility thresholds', file=sys.stderr)
        return 1

    names = sorted(cameras)
    _print_table(poses, names)
    ok = _print_summary(poses, names, config)

    if args.save:
        save_poses(args.save, poses, config, cameras)
        print(f'\n  wrote {args.save}')
    if args.overlay_dir:
        _render_overlays(poses, cameras, args.overlay_dir)

    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
