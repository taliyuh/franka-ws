# calibration_claude

Drives the Franka FR3 through a set of poses suitable for **eye-to-hand**
calibration (two stationary cameras, AprilTag 36h11 ID 0 with a **106 mm black
square** on the gripper) and records the data a hand-eye solver needs.

Scope is **motion + capture + offline hand-eye solving**. The capture node
writes a dataset and tells you, per pose and per camera, whether the tag was
actually detectable. See [Solving from the dataset](#solving-from-the-dataset)
for the conventions and the pitfalls that cost real time.

## Why the poses are generated rather than hard-coded

At the distances in this cell a 106 mm tag is only **30-90 px** across, so "the
tag is somewhere in the image" is not a useful criterion. Pose generation
samples tag poses, runs them through IK, and keeps only those that survive a
projection model checking, per camera:

* all four corners inside the frame with a margin,
* a minimum apparent size (mean edge length in pixels),
* a bounded incidence angle between the tag normal and the line of sight,
* a sane range, and
* **no line-of-sight blocked by the robot's own links** (approximated as
  capsules along the kinematic chain).

It then selects a subset that maximises the spread of `base -> tcp` rotations,
which is what conditions the `AX = XB` problem, while guaranteeing a minimum
number of views per camera.

In Gazebo this model was checked against reality: over 30 camera-views, the
predicted and actually-detected visibility agreed **30/30**, and predicted
apparent size matched the detected size to within ~1 px.

## Quick start (simulation)

```bash
# 1. Look at the poses before anything moves. No ROS, no Gazebo needed.
ros2 run calibration_claude preview_poses --poses 15 --overlay-dir /tmp/preview

# 2. Bring up Gazebo with the tag welded to the gripper.
ros2 launch calibration_claude sim_calibration.launch.py
#    headless:
#    ros2 launch calibration_claude sim_calibration.launch.py \
#        gz_args:="-r -s --headless-rendering $(ros2 pkg prefix --share franka_gazebo_bringup)/worlds/fr3_dual_camera_calibration.sdf"

# 3. Rehearse the motion without capturing anything.
ros2 run calibration_claude run_calibration_capture --dry-run --speed-scale 0.5

# 4. Capture.
ros2 run calibration_claude run_calibration_capture --out ~/handeye_sim --poses 15
```

A 15-pose run takes about 75 s of simulated time and reaches every pose to
within ~0.005 rad.

### Why this launch file and not `fr3_dual_camera_calibration.launch.py`

It reuses that world SDF and those camera bridges unchanged, but spawns the
robot itself, because:

* the robot description has to carry the AprilTag (`urdf/fr3_with_apriltag.gazebo.xacro`
  wraps franka's xacro and welds a textured quad to `fr3_hand_tcp`), and
* `gazebo_franka_arm_example_controller.launch.py` always loads one of the franka
  example controllers, and the default `gravity_compensation_example_controller`
  claims the same effort interfaces the trajectory controller needs.

`franka_gazebo_bringup` has no `joint_trajectory_controller` in its config, and
that file is not ours to edit, so `fr3_arm_controller` is spawned with
`spawner -p config/fr3_arm_controller_effort.yaml`.

> **Note on the stock launch file.** Its two `static_transform_publisher` nodes
> publish the camera *body* poses (X forward, Z up) under child frames named
> `*_color_optical_frame` (Z forward, X right, Y down). Those TFs are rotated
> ~90/90 deg away from the true optical frames. Nothing here reads them --
> camera geometry comes from `config/cameras_sim.yaml`, and this launch
> additionally publishes correct `gt_*_color_optical_frame` frames for RViz.

## Running on the real robot

1. Bring up Franka's native PTP action. Do not start or spawn
   `fr3_arm_controller`; the generic `JointTrajectoryController` can send a
   discontinuous one-point trajectory to libfranka. Use the position-controller
   configuration so only the state broadcasters are started:

```bash
ros2 launch franka_bringup franka.launch.py \
    robot_type:=fr3 \
    robot_ip:=172.16.0.2 \
    load_gripper:=true \
    controllers_yaml:=/franka_ws/fr3_ros_controllers_position.yaml
```

   Verify the runtime before moving:

```bash
ros2 control list_controllers
ros2 action list | grep ptp_motion
```

   `probe_board` uses `config/calibration_hardware.yaml` by default and refuses
   a configuration that does not select `/action_server/ptp_motion`. The
   native action's result must be `SUCCEEDED` and its `target_status` must be
   `TARGET_REACHED`; a controller-side success message alone is not accepted.

2. Write a cameras yaml with the measured extrinsics and the real topic names.
   Use `optical_xyz`/`optical_rpy` there (`body_xyz`/`body_rpy` is the Gazebo
   model-pose convention and gets the body->optical rotation applied for you).
   Extrinsics only need to be roughly right -- they are used for pose generation,
   not for the answer.
3. **Measure `tag.mount_xyz` / `tag.mount_rpy` on the actual plate.** The shipped
   default puts the tag normal along `fr3_hand_tcp` +X (so with the arm at `ready`,
   gripper pointing down, the tag faces world +X) at a 120 mm standoff. The
   standoff matters: at 50 mm the gripper clips the black square at oblique angles.
4. Preview, then run slowly:

```bash
ros2 run calibration_claude preview_poses --cameras my_cameras.yaml --save poses.yaml
ros2 run calibration_claude run_calibration_capture \
    --cameras my_cameras.yaml --reuse poses.yaml --speed-scale 0.2 --out ~/handeye_real
```

There is **no collision checking** -- no SRDF or planning scene is involved. Poses
are screened for joint limits with margin, links staying above the floor, and the
wrist staying out of the base cylinder. Preview and dry-run first, and keep a hand
on the e-stop.

The hardware capture is not required to use both cameras. For a Bolt-only
session, pass the supplied one-camera topic file; the `require_all_cameras`
check then applies only to Bolt. The topic-only hardware YAML is intended for
`run_taught_capture` (or `run_calibration_capture --reuse`), because generated
poses additionally need approximate camera extrinsics:

```bash
ros2 run calibration_claude run_taught_capture \
    --poses /franka_ws/poses.txt \
    --cameras /franka_ws/src/calibration_claude/config/cameras_bolt.yaml \
    --config /franka_ws/src/calibration_claude/config/calibration_hardware.yaml \
    --out /franka_ws/session_bolt_YYYYMMDD
```

## Verifying a refined board pose

After calibration and board refinement produce a JSON artifact, use its board
axes directly. The artifact is intentionally small so it can be regenerated for
each setup session:

```json
{
  "base_frame": "fr3_link0",
  "T_base_board": [[...], [...], [...], [0, 0, 0, 1]],
  "board_surface_z_m": 0.025
}
```

For example, to probe 5 cm along board +X from its origin:

```bash
ros2 run calibration_claude probe_board \
  --board-pose board_pose_robot.json \
  --config config/calibration_hardware.yaml \
  --offset 0.05 0.0 0.0 --dry-run
```

Without `--dry-run`, the command asks before moving to a pose 15 cm above the
target, then descends in guarded segments until Enter is pressed. The approach
height is measured from the board plane using the configured TCP-to-pen-tip
vector, rather than from the TCP origin. It refuses to move if the physical
pen tip would be below the board or if an explicitly requested staging height
does not leave the configured clearance. It never commands below the target
plane and retracts after confirmation (or automatically if the descent limit
is reached). Measure the full `probe.pen_tip_offset_tcp_m` vector and
`probe.tcp_orientation_board_rpy` in `config/board_manipulation.yaml` before
using a different pen. Keep the physical e-stop available throughout.

## Dataset layout

```
<out>/
  meta.yaml              robot frames, tag id/size/mount, live camera_info per camera
  generated_poses.yaml   the pose set, with predicted per-camera visibility
  poses.yaml             per pose: target joints, measured joints, T_base_tcp
  detections.yaml        per pose/camera: detected, corner pixels, apparent size
  images/pose_NNN_<camera>.png            (+ _overlay.png where the tag was found)
```

`T_base_tcp` is computed from the **measured** joints, not the commanded ones, so
a joint that settles slightly off target costs tag visibility but not hand-pose
accuracy. `detections.yaml` is a visibility check, not a calibration product.

For manually selected calibration views, omit `--poses` and add `--manual`:

```bash
ros2 run calibration_claude run_taught_capture \
    --manual \
    --cameras my_cameras.yaml \
    --config config/calibration_hardware.yaml \
    --frame-id base \
    --out ~/handeye_manual
```

Move the arm to a stationary view and press Enter for each sample. Type `q` to
finish; Ctrl-C leaves the manifest resumable with `--resume`. Each sample stores
the measured joints and measured-joint FK TCP pose, not a requested pose.

## Solving from the dataset

After each calibration capture, create the portable base-to-camera artifact:

```bash
ros2 run calibration_claude solve_handeye calibration_dataset \
  --camera mega --camera-frame mega_camera \
  --output session_mega_calibration.json
```

Use the camera name recorded in `meta.yaml` for `--camera`.  Set
`--camera-frame` to the refinement reference frame: the stationary board
pipeline uses `mega_camera` for `--reference-camera mega` and `bolt_camera` for
`--reference-camera bolt`.  This explicit alias prevents accidentally composing
a driver body-frame transform with the optical-frame board fit.  The output
contains `T_base_camera`, PnP reprojection RMS, rejected-view count, and the
PARK-versus-ANDREFF disagreement; inspect those figures before using it.

Verified against Gazebo ground truth (the camera poses are known exactly from the
world SDF): **6.6 mm / 0.19 deg** for `femto_bolt` over 11 views, **14.3 mm / 1.14 deg**
for `femto_mega` over 14 views. Two things that are easy to get wrong:

* **The eye-to-hand substitution.** `cv2.calibrateHandEye` solves the eye-in-hand
  problem. For a fixed camera and a target on the gripper, pass the *inverse* of
  the recorded hand pose as `gripper2base` (i.e. `T_tcp_base`) together with
  `T_cam_tag` from PnP; the returned "cam2gripper" is then `T_base_cam`, which is
  what you want.
* **The method matters.** `CALIB_HAND_EYE_TSAI` diverged badly on this data
  (161 deg off) while `PARK` and `ANDREFF` both landed within a centimetre.
  Cross-check at least two methods.

Object points for PnP are `calibration_claude.cameras.tag_corners(size)`, which
matches the corner order `cv2.aruco.detectMarkers` returns. Use
`SOLVEPNP_IPPE_SQUARE`.

For a dual-camera taught capture, fit both camera transforms in one
reproducible solve instead of producing two independent hand-eye files:

```bash
ros2 run calibration_claude solve_joint_handeye session_dual \
  --camera mega --camera bolt \
  --camera-frame mega=mega_camera \
  --camera-frame bolt=bolt_camera \
  --config config/calibration_hardware.yaml \
  --output session_dual_joint_calibration.json
```

The joint artifact contains `T_base_camera[camera]` for every selected camera
and one fitted `T_tcp_tag`. The final residual is the direct distorted
image-corner error: robot FK predicts the tag pose, that pose is projected with
the captured camera intrinsics/distortion, and the four predicted pixels are
compared with the saved detector corners. Camera names and sample indices are
sorted before fitting, each camera contributes equally despite different
detection counts, and the configured mount is only the initialization; the
shared mount is estimated from all streams. PARK/PnP is used only to seed each
camera's deterministic initialization. Initial outlier selection is based on
each camera's residual median and MAD (three robust standard deviations), not
on a fixed millimetre or degree cutoff; the final answer comes from one
joint robust least-squares fit.

Inspect `quality.per_camera` and the rejected sample list before using the
artifact. `direct_corner_*_px` reports the residual of the actual fitted
observation model, while `pnp_*` is only a diagnostic for the initialization.
The command also performs leave-one-pose-out validation: every pose is removed
from both camera streams, the calibration is refit, and the omitted corners
are predicted without filtering. The downstream stationary pipeline accepts a
joint artifact only when every fold completes and every attempted camera image
contains the tag. If a tag is not detected, that image is recorded but omitted
from fitting, reported under `validation.undetected_indices`, and the artifact
is marked `incomplete` rather than silently treated as validated.

## Notes from getting this working in Gazebo

Recorded because each cost a debugging cycle and none is obvious:

* **`fr3_joint7` reports a constant ~2.93 rad/s velocity while stationary** (its
  position moves < 0.003 rad over 40 samples). Two consequences: JTC folds
  `stopped_velocity_tolerance` into its goal tolerance, so a goal is never
  reported complete; and a D term turns that reading into a constant ~15 Nm
  disturbance. Hence `stopped_velocity_tolerance: 0.0`, `fr3_joint7.d: 0.0`, and
  settle detection based on position stability rather than reported velocity.
* **The `position` command interface does not actuate this hardware plugin.**
  `gz_ros2_control`'s position path writes `JointVelocityCmd`, which
  `FrankaGazeboHardwareInterface`'s wrapped system never applies -- the controller
  reports "Goal successfully reached!" while the joint sits exactly where it was.
  Effort is the only interface that moves the arm.
* **Integral gains are needed**, unlike franka's own config: its gravity model does
  not know about Gazebo's joint damping, so pure P/D leaves a standing error.
* **`CORNER_REFINE_APRILTAG` rejects markers** in OpenCV 4.5.4 -- it returned
  nothing on a clean, fully visible 84 px tag that every other refinement mode
  found. This package uses `CORNER_REFINE_SUBPIX`.
* **`package://` meshes become `model://` when `ros_gz_sim create` converts URDF to
  SDF**, and colcon's isolated install layout means every package with referenced
  meshes needs its share *parent* on `IGN_GAZEBO_RESOURCE_PATH`.

## Layout

| path | what |
|---|---|
| `calibration_claude/kinematics.py` | URDF chain parsing, FK, Jacobian, damped-least-squares IK |
| `calibration_claude/cameras.py` | pinhole model, projection, visibility and occlusion tests |
| `calibration_claude/pose_generator.py` | sampling, IK, scoring, diversity selection, ordering |
| `calibration_claude/tag.py` | 36h11 asset generation and `cv2.aruco` detection |
| `calibration_claude/trajectory_client.py` | native PTP goals and settle detection |
| `calibration_claude/capture_node.py` | `run_calibration_capture` entry point |
| `calibration_claude/solve_handeye.py` | `solve_handeye` and `solve_joint_handeye` entry points |
| `calibration_claude/preview_poses.py` | `preview_poses` entry point |
| `config/calibration_hardware.yaml` | real-robot frames, native PTP action, motion limits |
| `config/calibration.yaml` | simulation calibration frames and motion settings |
| `config/cameras_sim.yaml` | camera poses/intrinsics for the Gazebo scene |
| `urdf/`, `meshes/` | FR3 + tag description and the generated tag quad |

Regenerate the tag assets (they are committed, so this is rarely needed):

```bash
ros2 run calibration_claude make_tag_assets --output meshes --id 0 --size 0.106
```
