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
| `probe_board` | approach a board point and descend in guarded steps; `--record FILE` saves the commanded and measured pose and joints of every move and holds at the bottom for photos | `check`, `probe`, `accuracy` |
| `plot_probe_accuracy` | figure, table and summary of an `accuracy` run | `accuracy` |
| `grasp_peg` / `release_peg` | close the hand on a peg (`--until-contact`: until it clamps) / open it (franka_gripper grasp / move actions) | `grasp`, `release`, `accuracy` |
| `pivot_calibration` | measure the held tool's tip by pivoting it in a dimple | `check`/`probe --tip-from` |
| `measure_repeatability` | return to hand-taught joint angles from home, uncompensated; save and plot the errors | `repeatability` |

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
they differ by more than 2 mm / 0.57 deg (usually `NE_T_EE` is not the
identity; reset: `pipeline/DETAILS.md`, Robot checks).

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
Without `--dry-run` it asks b`--offset` is in the board frame (origin at the centre of the bottom face).
Without `--dry-run` it asks, moves 20 mm above the point, then descends in 3 mm
steps until Enter or the target. Each move is corrected up to 3 times until it
is within 0.2 mm of the measured TCP. The tip never goes below the board plane.

* `--where` (`--watch`): print where the tip is, no motion.
* `--record FILE`: save every move for `pipeline accuracy` and hold at the
  bottom for photos.

Tool and speeds: `config/board_manipulation.yaml`. Keep the e-stop at hand.

lue:

```bash
rosrun panda_calibration pivot_calibration          # robot.launch running
```

It stops `position_joint_trajectory_controller` so the arm can be guided by
hand, and restarts it at the end. Put the tip in one fixed point, ideally a
small conical dimple so it cannot slide. Release the guiding buttons and press
Enter to record a pose (the arm must be still for 0.5 s), and repeat from other
orientations: tilt 30–60° in different directions and turn about the tool axis.
After 4 poses every new one prints the current fit.

The fit solves `R_i · tip + t_i = pivot` for all poses (measured `O_T_EE`) by
least squares. The result goes to `/franka_ws/tool_calibration/pivot_<date>_<time>.yaml`:
`tip_in_tcp_m` is the TCP → tip vector in `panda_hand_tcp` axes, the frame of
`probe.pen_tip_offset_tcp_m` and of the `tip [...] mm from the TCP` line that
`pipeline check/probe` print; `tip_in_flange_m` is the same point in
`panda_link8`. The file also keeps the residuals, a leave-one-out error and
every pose, so `--resolve FILE --drop 3,7` solves again without poses that
slipped. Only the tip position is measured, not the direction of the tool axis.
`--shape` (default `round`) records which peg the tool stands in for;
`pipeline check/probe --tip-from FILE` then uses the measured tip for that
shape only (see `pipeline/DETAILS.md`).

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
| `panda_calibration/probe_accuracy.py` | `probe_board --record`, `plot_probe_accuracy` |
| `panda_calibration/gripper.py` | `grasp_peg` (incl. `--until-contact`), `release_peg` |
| `panda_calibration/pivot_calibration.py` | `pivot_calibration` |
| `panda_calibration/repeatability.py` | `measure_repeatability` |
| `scripts/` | the `rosrun` executables (thin wrappers around the modules) |
| `launch/robot.launch` | franka_control + position trajectory controller + move_group |
| `launch/azure_4k.launch`, `launch/orbbec_4k.launch` | the 4K camera setups |
| `config/calibration_hardware.yaml` | robot frames, tag, motion limits, capture settings |
| `config/board_manipulation.yaml` | probe geometry and speeds |
| `config/cameras_*.yaml` | camera names and topics per setup |
