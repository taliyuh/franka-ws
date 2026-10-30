# panda_calibration

Franka Panda (FER, system 4.2.x, ROS Noetic) tools used by the board pipeline
(`pipeline/`): they move the arm with franka_ros' joint position trajectory
controller, record camera frames together with the measured joints, solve
eye-to-hand calibrations, and probe board holes. The
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

```bash
roslaunch panda_calibration robot.launch robot_ip:=172.16.0.2
```

starts franka_control (libfranka 0.9.2, the last release for Panda system
4.2.x), franka_state_controller at 250 Hz, the gripper,
`position_joint_trajectory_controller` and MoveIt's move_group (for
`/compute_ik`, group `panda_manipulator`). `moveit:=false` leaves move_group
out; the tools then fall back to their local numeric IK.

Every motion is one `FollowJointTrajectory` goal to
`/position_joint_trajectory_controller/follow_joint_trajectory`: a single point
with zero end velocity and acceleration, so the controller interpolates a
quintic and libfranka sees no acceleration jump (franka_control's rate limiter
is on as well). This is franka_ros' own `move_to_start` pattern.
`probe_board` refuses any other controller: the effort trajectory controller
settles with a load-dependent error far above its 0.1 mm tolerance. Check
before moving:

```bash
rosservice call /controller_manager/list_controllers   # position_joint_trajectory_controller: running
rostopic hz /franka_state_controller/joint_states      # ~250 Hz
```

The driver decides a move is finished from the measured joints (settle
detection), not from the controller's result: franka_control's default goal
tolerance (0.05 rad) is far coarser than what calibration needs. A reflex
aborts the goal; recover with

```bash
rostopic pub -1 /franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"
```

Cartesian checks use the robot's own measured end-effector pose (`O_T_EE` in
`/franka_state_controller/franka_states`). That is the end effector set in
Desk; with the Franka Hand selected it coincides with `panda_hand_tcp`. Every
tool compares the two once at startup and warns (`probe_board` refuses) when
they differ by more than 2 mm / 0.57 deg.

Motion settings are in `config/calibration_hardware.yaml` (`motion:`): joint
speed, minimum move duration, settle detection and the Cartesian checks
`probe_board` applies.

## Cameras

```bash
roslaunch panda_calibration azure_4k.launch     # wrist Azure Kinect, /azure_camera
roslaunch panda_calibration orbbec_4k.launch    # Femto Mega /mega_camera, Femto Bolt /wrist_camera
```

Both run the drivers built into the image (Azure_Kinect_ROS_Driver, ROS 1
branch; OrbbecSDK_ROS1) at 3840x2160 colour with depth registered to colour,
and publish the topics `config/cameras_*.yaml` name.

## Probing

```bash
rosrun panda_calibration probe_board \
  --board-pose board_pose.json \
  --config config/calibration_hardware.yaml \
  --offset 0.05 0.0 0.0 --dry-run
```

`--offset` is in the board frame (origin at the centre of the bottom face).
Without `--dry-run` it asks before moving to 15 cm above the point (measured
from the board plane at the pen tip), then descends in 3 mm segments until
Enter is pressed. Each move is checked against the measured TCP (0.1 mm /
0.05 deg) and corrected up to 3 times when the residual is inside the
correction window (in Gazebo the first move lands ~0.25 mm off and one
correction brings it to ~0.03 mm). It never commands the pen tip below the board plane, and
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
sample per Enter (`q` finishes, `--resume` continues an interrupted run). The
running position trajectory controller holds the arm stiff; stop it first so
the arm can be hand-guided with the pilot (wrist) buttons, and start it again
afterwards:

```bash
rosservice call /controller_manager/switch_controller "{start_controllers: [], stop_controllers: [position_joint_trajectory_controller], strictness: 2}"
```

## Solving (stationary cameras)

```bash
rosrun panda_calibration solve_handeye <capture> \
  --camera bolt --camera-frame bolt_camera --output calibration.json

rosrun panda_calibration solve_joint_handeye <capture> \
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
`franka_description/robots/panda/panda.urdf.xacro` (hand:=true), identical to
the live `/robot_description` parameter.

## Layout

| path | what |
|---|---|
| `panda_calibration/kinematics.py` | URDF chain, FK, Jacobian, damped-least-squares IK |
| `panda_calibration/cameras.py` | pinhole model, camera yaml loading, tag corners |
| `panda_calibration/context.py` | config loading, chain from the live or packaged description |
| `panda_calibration/trajectory_client.py` | trajectory goals, settle detection, measured TCP |
| `panda_calibration/capture_node.py` | `run_calibration_capture` |
| `panda_calibration/pose_generator.py` | tag-visibility pose generation (when `--reuse` is not given) |
| `panda_calibration/taught_capture.py` | `run_taught_capture` |
| `panda_calibration/capture_stereo_rgbd.py` | `capture_stereo_rgbd` |
| `panda_calibration/solve_handeye.py` | `solve_handeye`, `solve_joint_handeye` |
| `panda_calibration/board_manipulation.py`, `board_pose.py` | `probe_board` |
| `scripts/` | the `rosrun` executables (thin wrappers around the modules) |
| `launch/robot.launch` | franka_control + position trajectory controller + move_group |
| `launch/azure_4k.launch`, `launch/orbbec_4k.launch` | the 4K camera setups |
| `config/calibration_hardware.yaml` | robot frames, tag, motion limits, capture settings |
| `config/board_manipulation.yaml` | probe geometry and speeds |
| `config/cameras_*.yaml` | camera names and topics per setup |
