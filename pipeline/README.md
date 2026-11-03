# Board pipeline

Calibrates a camera against the Franka Panda (FER, system version 4.2.x, ROS
Noetic, libfranka 0.9.2), measures the pose of the NIST ATB board in the robot
base frame, and drives the tip of a held peg onto any of its 20 holes.

Two camera setups are supported, with the same commands:

| setup         | camera                                   | calibration target           |
|---------------|------------------------------------------|------------------------------|
| `eye_in_hand` | Azure Kinect on the wrist (rigid to `panda_link6`) | the board itself    |
| `stationary`  | Femto Mega (+ Bolt) fixed in the cell    | ArUco tag plate on the gripper |

## Quick start

```bash
cd ~/franka_ws                                   # wherever the workspace lives
docker compose -f docker/compose.yaml up -d      # once per boot; container name franka_ros
docker exec -it franka_ros bash -ic "cd /franka_ws && colcon build"   # after code changes (defaults: colcon_defaults.yaml)
# in container terminals (docker exec -it franka_ros bash), leave these running:
#   roslaunch panda_calibration robot.launch robot_ip:=172.16.0.2
#   roslaunch panda_calibration azure_4k.launch       # eye_in_hand
#   roslaunch panda_calibration orbbec_4k.launch      # stationary
./pipeline/pipeline new --setup eye_in_hand      # creates sessions/<date>_eye_in_hand
./pipeline/pipeline calibrate                    # ~15 min, the arm moves
./pipeline/pipeline locate                       # ~5 min, the arm moves; skip it (eye_in_hand)
                                                 # if the board has not moved since calibrate
./pipeline/pipeline check --hole round_c3mm      # dry run: prints the target, no motion
./pipeline/pipeline probe --hole round_c3mm      # moves the held round peg's tip onto the hole
```

`alias pipeline=$PWD/pipeline/pipeline` (run from the workspace) saves typing. The
command runs on the host: anything that touches the robot or cameras runs in
the `franka_ros` container automatically. The board fitting uses the
Python 3.12 venv of the pose-refinement checkout, expected next to the
workspace (`../pose-refinement`); set `paths.pose_refinement` in
`config/pipeline.yaml` or `PIPELINE_POSE_REFINEMENT` if it lives elsewhere.

Use `docker compose up -d`, not `docker compose run`: the CLI finds the
container by the name `franka_ros`.

### Robot and camera launch (in the container)

| launch | starts |
|---|---|
| `panda_calibration robot.launch robot_ip:=...` | franka_control (libfranka 0.9.2), franka_state_controller at 250 Hz, `position_joint_trajectory_controller`, the gripper, and MoveIt's move_group for IK (`moveit:=false` skips it; `use_rviz:=true` opens RViz) |
| `panda_calibration azure_4k.launch` | Azure Kinect under `/azure_camera`: 3840x2160 BGRA colour at 15 fps, NFOV unbinned depth registered to colour (`depth_to_rgb`), no point cloud |
| `panda_calibration orbbec_4k.launch` | Femto Mega under `/mega_camera`, Femto Bolt under `/wrist_camera`: 3840x2160 colour, 640x576 depth registered to colour; `mega_serial:=`, `bolt_serial:=`, `mega_ip:=`, `bolt:=false` |

Before moving the arm: unlock the joints and activate FCI in Desk, and set the
end effector in Desk to the Franka Hand. Every pipeline motion is a single
`FollowJointTrajectory` goal to `position_joint_trajectory_controller`; the
tools refuse to start if it is not running, if another command controller
(e.g. `effort_joint_trajectory_controller`) holds the joints, or if the robot
is in reflex or user-stop mode. After a reflex (collision, joint limit, a push
on the arm), franka_control stops the arm until it is recovered; the trajectory
controller keeps running, but no goal moves the arm. Recover with

```bash
rostopic pub -1 /franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"
```

The Orbbec cameras need Orbbec's udev rules on the host, once:

```bash
docker cp franka_ros:/opt/franka_underlay/src/OrbbecSDK_ROS1/scripts/99-obsensor-libusb.rules /tmp/
sudo cp /tmp/99-obsensor-libusb.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Coming from the FR3 (ROS 2) setup

* Delete the Humble build products before the first Noetic build; the
  container sources `install/setup.bash` and a ROS 2 one breaks every command:
  `rm -rf build install log` (they are root-owned; run it in the container).
* `docker compose -f docker/compose.yaml build` builds the new image
  (`franka_ros:noetic`); stop the old `franka_ros2` container first, both use
  the host network.
* FR3 calibrations do not transfer (different arm, other joint names):
  start new sessions and run `calibrate`. `--calib-from` an FR3 session is
  refused.
* The ROS 2 checkouts in `src/` (franka_ros2, libfranka, OrbbecSDK_ROS2, ...)
  can stay; colcon only builds the paths in `colcon_defaults.yaml`.

## What each step does

### `calibrate`

**eye_in_hand.** Finds where the camera sits on the wrist, using the board
as the target. The board must stay still for the whole stage.

1. `look`: moves to the look pose (`config/azure_look_pose.yaml`) and grabs one RGB-D frame.
   `--here` skips the move and uses the current arm pose. The look pose comes
   from the FR3 cell; check the board is fully in view on the Panda and
   replace it if not.
2. `seed_fit`: finds the board in that frame from depth, with no prior pose. Check
   `calibrate/2_seed_fit/overlay.jpg`: the green dots must sit on the hole rims.
3. `plan`: plans 30 views around the board (joint limits, table clearance
   along every path, whole board in the image).
4. `capture`: drives through the views and stores image + measured joints for each.
5. `fit`: refines the board pose in every image, seeded from the robot.
6. `bundle`: refines the camera intrinsics over all views.
7. `solve`: fits the camera mount, joint 2-5 encoder offsets, scale and mount sag.
   It prints the leave-one-out board-corner error. The 2026-10-22 run (FR3)
   gave a median of 0.50 mm and p90 0.87 mm; the Panda has not been measured
   yet. The board pose fitted along the way is
   written to `board_pose.json`, so `check`/`probe` can run straight away as
   long as the board has not moved.

**stationary.** Finds where each fixed camera is relative to the robot base.

1. `capture`: taught capture of the gripper tag over the poses in
   `config/stationary_calibration_poses.txt`. It asks before each move. If it
   is interrupted, running it again resumes it.
2. `solve`: hand-eye solve: one camera, or a joint solve of both cameras with
   a shared tag mount.

Output: `sessions/<name>/calibration.json`. It stays valid until a camera
moves, which for eye-in-hand includes being bumped on its mount.

### `locate`

Measures the board where it currently lies. Run it after every board move.

* **eye_in_hand:** `look`, `seed_fit`, `plan` (8 views), `capture`, `fit`, then
  `solve`, which turns every view into one board pose with the calibration
  held fixed. Views more than 1.5 mm from the mean are dropped. The printed
  spread tells you how consistent the result is.
* **stationary:** `capture` (one synchronized frame per camera; move the arm
  out of view first), `refine` (rim fit in the camera frame; overlays go to
  `locate/2_refine/overlays`), `export` (into the base frame; refused if the
  refinement is not verified).

Output: `sessions/<name>/board_pose.json` holds `T_base_board`. The board
origin is the centre of its bottom face, and the top surface is at z = 25 mm.

### `check` / `probe`

`check --hole NAME` validates the target and IK without moving. `probe --hole NAME`
approaches 15 cm above the hole with the peg, then descends in guarded steps:

* Enter approves the approach.
* Enter during the descent stops it.
* Enter again retracts.

It never brings the peg tip below the board surface. `--height MM` targets a
point above the surface instead. Approach and descent settings are in
`src/panda_calibration/config/board_manipulation.yaml`.

**The held peg.** The gripper carries the printed V-block fingers
(franka_fr3_fingers v2: finger A, the V-block, on `panda_leftfinger`), which hold
every peg off the TCP axis. The peg is assumed to be the one of the hole's
shape, seated in the V with its top against the roof (gripper width as in
`config/grasps.yaml`). Where each peg sits relative to `panda_hand_tcp`:

| peg       | peg axis from the TCP (hand x, y) | tip below the TCP | turns allowed |
|-----------|-----------------------------------|-------------------|---------------|
| round     | 0, +16.5 mm                       | 39 mm             | any           |
| hexagon   | 0, +15.2 mm                       | 39 mm             | 60°           |
| triangle  | 0, +17.0 mm                       | 39 mm             | 120°          |
| rectangle | +0.5, +15.5 mm                    | 39 mm             | 180°          |
| L         | +4.5, +17.0 mm                    | 39 mm             | none          |

The peg frame is lined up with the hole frame. Of the allowed turns,
the one that keeps the hand closest to the yaw of `tcp_orientation_board_rpy`
is used. The resulting tool (TCP orientation and TCP-to-peg-tip vector) is
written to `targets/<hole>_tool.yaml` and replaces the pen settings of
`board_manipulation.yaml` for that hole. The offsets are nominal CAD values
from `grasp_poses_v2.json`, registered onto the hole-frame convention by
`tools/extract_grasps.py` (run it in the rl_manip checkout, like
`extract_holes.py`).

For eye_in_hand, the board pose is corrected for the calibrated joint
offsets at the chosen hole, because MoveIt plans with the nominal kinematics.
The corrected target is written to `targets/<hole>.json`, next to the
peg tool `targets/<hole>_tool.yaml`.

`pipeline holes` lists the hole names (`<shape>_c<clearance>mm`, e.g.
`hexagon_c0p2mm`) with board coordinates. Once a session has a board pose, it
also lists their position in the base frame. The coordinates come from the
board STL (`config/holes.yaml`, regenerated by `tools/extract_holes.py`).

## Sessions, reruns and starting over

Every result lives in one folder, `sessions/<name>/`. The CLI keeps a
`sessions/current` link, so you never type paths:

```
sessions/2026-10-22_eye_in_hand/
  config.yaml          settings snapshot (copied from config/pipeline.yaml)
  calibrate/1_look/ 2_seed_fit/ ...   one folder per step
  locate/1_look/ ...
  calibration.json     board_pose.json     targets/
```

| situation                                    | command                                   |
|----------------------------------------------|-------------------------------------------|
| a step failed or you pressed Ctrl-C          | rerun the same command; it continues at the failed step |
| redo one step and everything after it        | `pipeline locate --from fit`              |
| throw the stage away and start over          | `pipeline locate --fresh`                 |
| board moved, cameras did not                 | `pipeline locate --fresh` (same session)  |
| new day, keep yesterday's calibration        | `pipeline new --setup eye_in_hand --calib-from <old session>` |
| where am I / what is done                    | `pipeline status`, `pipeline sessions`    |
| go back to an older session                  | `pipeline use <name>` or `--session <name>` on any command |

Completed steps are skipped, and a rerun step invalidates everything after it.
Rerunning `calibrate` deletes the session's `locate` results and
`board_pose.json`, because they depended on the old calibration (eye_in_hand
then writes a new `board_pose.json` from the calibration itself).

## Settings

`config/pipeline.yaml` holds the defaults: robot IP, container name, view
counts, speed scales, rejection thresholds, camera list for `stationary`. A new
session copies it to `sessions/<name>/config.yaml`. To change one session,
edit that copy; to change future sessions, edit the template.

## Layout

| path | what |
|---|---|
| `pipeline` | entry point (bash; picks the pose-refinement interpreter) |
| `board_pipeline/cli.py` | commands |
| `board_pipeline/session.py` | session folders, resumable step runner |
| `board_pipeline/eye_in_hand.py` | Azure steps, chain model, view planner, hole correction |
| `board_pipeline/stationary.py` | Mega/Bolt steps (wraps panda_calibration + pose-refinement) |
| `board_pipeline/rimfit.py` | adapter to pose-refinement's rim fitter and bundle adjustment |
| `board_pipeline/ros.py` | `docker exec` bridge, host/container path mapping, controller check |
| `board_pipeline/panda.py` | Panda DH kinematics and IK |
| `ros_nodes/grab_azure_rgbd.py` | single RGB-D frame grabber (runs in the container) |
| `board_pipeline/grasp.py` | held peg: TCP pose and tool tip that put it on a hole |
| `config/` | settings template, hole and grasp tables, look pose, prior hand-eye, calibration poses |

## Troubleshooting

* **`container 'franka_ros' is not running`**: run `docker compose -f docker/compose.yaml up -d`.
* **`no controller_manager answered`**: `robot.launch` is not running, or the
  ROS master it started is gone.
* **`position_joint_trajectory_controller is stopped`** / **`... is running and
  claims the arm joints`**: the message prints the `switch_controller` call
  that fixes it.
* **A move fails with `trajectory goal ended in state ABORTED`**: usually a
  reflex; check franka_control's output, run the error recovery above, rerun.
* **`the robot is in reflex mode after cartesian_reflex`**: something pressed on
  the arm, the camera or its cable (or a move collided), and franka_control
  stopped the arm. Free it, run the error recovery above, rerun. If it keeps
  happening during moves, enter the wrist camera and its mount as extra load in
  Desk: weight the robot does not know about eats into the 20 N Cartesian
  reflex threshold (`O_F_ext_hat_K` in `/franka_state_controller/franka_states`
  shows it while the arm stands still).
* **`Error in XmlRpcClient::writeRequest: write error (Connection refused)`** in
  the robot launch output: a node exited without unregistering (e.g. a camera
  driver that crashed). Harmless; `rosnode cleanup` clears it.
* **`seed fit failed`**: the board is not fully in view from the look pose.
  Jog the arm until it is, then run `pipeline locate --here`.
* **A capture step times out waiting for images**: are the robot and camera
  launch files running? Check with `rostopic list` / `rostopic hz` in the container.
* **`export refused ... unverified`** (stationary): the two cameras disagree.
  The rig moved; recalibrate.
