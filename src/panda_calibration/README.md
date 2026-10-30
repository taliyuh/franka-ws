# fr3_calibration

Franka FR3 tools used by the board pipeline (`pipeline/`): they move the arm
with Franka's native PTP action, record camera frames together with the
measured joints, solve eye-to-hand calibrations, and probe board holes. The
pipeline calls them for you; this file is for running or changing them
directly.

| executable | what | used by |
|---|---|---|
| `run_calibration_capture` | drive through a joint-pose list (`--reuse`), record images + measured joints | eye-in-hand `look`, `capture` |
| `run_taught_capture` | replay taught poses (or `--manual`), record tag views | stationary `calibrate` |
| `capture_stereo_rgbd` | one synchronized RGB-D frame per camera | stationary `locate` |
| `solve_handeye` / `solve_joint_handeye` | base-to-camera from a tag capture (one / several cameras) | stationary `calibrate` |
| `probe_board` | approach a board point and descend in guarded steps | `check`, `probe` |

## Robot runtime

Every motion goes through `/action_server/ptp_motion` (libfranka's
`AsyncPositionControlHandler`), never through `fr3_arm_controller`: that
effort JTC holds libfranka's control loop while active, so PTP goals are
rejected. `franka_fr3_moveit_config moveit.launch.py` leaves it inactive by
default (`activate_arm_controller:=false`). Check before moving:

```bash
ros2 control list_controllers          # fr3_arm_controller must be inactive
ros2 action list | grep ptp_motion
```

The patched action server in `franka_ros2/franka_hardware` ends a goal when
the arm stops moving: reached if it settled near the goal, "never started"
(the client resubmits, up to 3 times) or "stalled" otherwise. Without that
patch a goal libfranka never reports as reached blocks the server.

Motion settings are in `config/calibration_hardware.yaml` (`motion:`): joint
speed, minimum move duration, goal tolerance, settle detection and the
Cartesian checks `probe_board` applies.

## Probing

```bash
ros2 run fr3_calibration probe_board \
  --board-pose board_pose.json \
  --config config/calibration_hardware.yaml \
  --offset 0.05 0.0 0.0 --dry-run
```

`--offset` is in the board frame (origin at the centre of the bottom face).
Without `--dry-run` it asks before moving to 15 cm above the point (measured
from the board plane at the pen tip), then descends in 3 mm segments until
Enter is pressed. Each move is checked against the measured TCP (0.1 mm /
0.05 deg) and corrected up to 3 times when the residual is inside the
correction window. It never commands the pen tip below the board plane, and
retracts after confirmation. Measure `probe.pen_tip_offset_tcp_m` and
`probe.tcp_orientation_board_rpy` in `config/board_manipulation.yaml` for the
actual tool. Keep the e-stop at hand.

## Dataset layout

```
<out>/
  meta.yaml        robot frames, tag id/size/mount, live camera_info per camera
  poses.yaml       per pose: target joints, measured joints, T_base_tcp
  detections.yaml  per pose/camera: detected, corner pixels, apparent size
  images/pose_NNN_<camera>.png
```

`T_base_tcp` comes from the **measured** joints, so a joint that settles
slightly off target costs nothing in accuracy.

For manually chosen tag views, `run_taught_capture --manual` records one
sample per Enter (`q` finishes, `--resume` continues an interrupted run).

## Solving (stationary cameras)

```bash
ros2 run fr3_calibration solve_handeye <capture> \
  --camera bolt --camera-frame bolt_camera --output calibration.json

ros2 run fr3_calibration solve_joint_handeye <capture> \
  --camera azure --camera bolt \
  --camera-frame azure=azure_camera --camera-frame bolt=bolt_camera \
  --config config/calibration_hardware.yaml --output calibration.json
```

`--camera` is the camera name in `meta.yaml`; `--camera-frame` names the frame
the board refinement uses for that camera, so a driver body frame is never
composed with an optical-frame fit. Inspect the `quality` section (reprojection,
rejected views, method disagreement) before using the result.

The joint solve fits every camera and one shared tag mount in a single robust
least-squares problem on the distorted image corners, then validates by
leaving each pose out. An image without a detected tag is recorded under
`validation.undetected_indices` and marks the artifact `incomplete`.

Pitfalls that cost real time:

* **Eye-to-hand substitution.** `cv2.calibrateHandEye` solves eye-in-hand. For
  a fixed camera and a tag on the gripper, pass `T_tcp_base` (the inverse hand
  pose) as `gripper2base`; the returned "cam2gripper" is then `T_base_cam`.
* **Method.** `CALIB_HAND_EYE_TSAI` diverged (161 deg) on this data while `PARK`
  and `ANDREFF` agreed within a centimetre. Cross-check two methods.
* **Corner refinement.** `CORNER_REFINE_APRILTAG` rejects clean markers in
  OpenCV 4.5.4; this package uses `CORNER_REFINE_SUBPIX`. PnP object points are
  `cameras.tag_corners(size)` (the `detectMarkers` corner order) with
  `SOLVEPNP_IPPE_SQUARE`.
* **Tag mount.** Measure `tag.mount_xyz` / `tag.mount_rpy` on the actual plate;
  at a 50 mm standoff the gripper clips the black square at oblique angles.

Offline (no robot launch) the kinematics come from
`franka_description/robots/fr3/fr3.urdf.xacro`, identical to the live
`/robot_description`.

## Layout

| path | what |
|---|---|
| `fr3_calibration/kinematics.py` | URDF chain, FK, Jacobian, damped-least-squares IK |
| `fr3_calibration/cameras.py` | pinhole model, camera yaml loading, tag corners |
| `fr3_calibration/context.py` | config loading, chain from the live or packaged description |
| `fr3_calibration/trajectory_client.py` | native PTP goals, retries, settle detection |
| `fr3_calibration/capture_node.py` | `run_calibration_capture` |
| `fr3_calibration/pose_generator.py` | tag-visibility pose generation (when `--reuse` is not given) |
| `fr3_calibration/taught_capture.py` | `run_taught_capture` |
| `fr3_calibration/capture_stereo_rgbd.py` | `capture_stereo_rgbd` |
| `fr3_calibration/solve_handeye.py` | `solve_handeye`, `solve_joint_handeye` |
| `fr3_calibration/board_manipulation.py`, `board_pose.py` | `probe_board` |
| `config/calibration_hardware.yaml` | robot frames, tag, motion limits, capture settings |
| `config/board_manipulation.yaml` | probe geometry and speeds |
| `config/cameras_*.yaml` | camera names and topics per setup |
