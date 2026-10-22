# Board pipeline

Calibrates a camera against the FR3, measures the pose of the NIST ATB board
in the robot base frame, and drives a probe tip onto any of its 20 holes.

Two camera setups are supported, with the same commands:

| setup         | camera                                   | calibration target           |
|---------------|------------------------------------------|------------------------------|
| `eye_in_hand` | Azure Kinect on the wrist (rigid to `fr3_link6`) | the board itself      |
| `stationary`  | Femto Mega (+ Bolt) fixed in the cell    | ArUco tag plate on the gripper |

## Quick start

```bash
cd ~/franka_ws                                   # wherever the workspace lives
docker compose -f docker/compose.yaml up -d      # once per boot; container name franka_ros2
docker exec -it franka_ros2 bash -ic "cd /franka_ws && colcon build"   # after code changes (defaults: colcon_defaults.yaml)
./pipeline/pipeline new --setup eye_in_hand      # creates sessions/<date>_eye_in_hand
./pipeline/pipeline calibrate                    # ~15 min, the arm moves
./pipeline/pipeline locate                       # ~5 min, the arm moves; skip it (eye_in_hand)
                                                 # if the board has not moved since calibrate
./pipeline/pipeline check --hole round_c3mm      # dry run: prints the target, no motion
./pipeline/pipeline probe --hole round_c3mm      # moves the probe tip onto the hole
```

`alias pipeline=$PWD/pipeline/pipeline` (run from the workspace) saves typing. The
command runs on the host: anything that touches the robot or cameras runs in
the `franka_ros2` container automatically. The board fitting uses the
Python 3.12 venv of the pose-refinement checkout, expected next to the
workspace (`../pose-refinement`); set `paths.pose_refinement` in
`config/pipeline.yaml` or `PIPELINE_POSE_REFINEMENT` if it lives elsewhere.

Use `docker compose up -d`, not `docker compose run`: the CLI finds the
container by the name `franka_ros2`.

## What each step does

### `calibrate`

**eye_in_hand.** Finds where the camera sits on the wrist, using the board
as the target. The board must stay still for the whole stage.

1. `look`: moves to the look pose (`config/azure_look_pose.yaml`) and grabs one RGB-D frame.
   `--here` skips the move and uses the current arm pose.
2. `seed_fit`: finds the board in that frame from depth, with no prior pose. Check
   `calibrate/2_seed_fit/overlay.jpg`: the green dots must sit on the hole rims.
3. `plan`: plans 30 views around the board (joint limits, table clearance
   along every path, whole board in the image).
4. `capture`: drives through the views and stores image + measured joints for each.
5. `fit`: refines the board pose in every image, seeded from the robot.
6. `bundle`: refines the camera intrinsics over all views.
7. `solve`: fits the camera mount, joint 2-5 encoder offsets, scale and mount sag.
   It prints the leave-one-out board-corner error. The 2026-10-22 run gave
   a median of 0.50 mm and p90 0.87 mm. The board pose fitted along the way is
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
approaches 15 cm above the hole, then descends in guarded steps:

* Enter approves the approach.
* Enter during the descent stops it.
* Enter again retracts.

It never goes below the board surface. `--height MM` targets a point above
the surface instead. Tool geometry (pen-tip offset, orientation) is in
`src/fr3_calibration/config/board_manipulation.yaml`.

For eye_in_hand, the board pose is corrected for the calibrated joint
offsets at the chosen hole, because MoveIt plans with the nominal kinematics.
The corrected target is written to `targets/<hole>.json`.

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
| `board_pipeline/stationary.py` | Mega/Bolt steps (wraps fr3_calibration + pose-refinement) |
| `board_pipeline/rimfit.py` | adapter to pose-refinement's rim fitter and bundle adjustment |
| `board_pipeline/ros.py` | `docker exec` bridge, host/container path mapping |
| `board_pipeline/fr3.py` | FR3 DH kinematics and IK |
| `ros_nodes/grab_azure_rgbd.py` | single RGB-D frame grabber (runs in the container) |
| `config/` | settings template, hole table, look pose, prior hand-eye, calibration poses |

## Troubleshooting

* **`container 'franka_ros2' is not running`**: run `docker compose -f docker/compose.yaml up -d`.
* **`seed fit failed`**: the board is not fully in view from the look pose.
  Jog the arm until it is, then run `pipeline locate --here`.
* **A capture step times out waiting for images**: are the robot and camera
  launch files running? Check with `ros2 topic list` in the container.
* **`export refused ... unverified`** (stationary): the two cameras disagree.
  The rig moved; recalibrate.
