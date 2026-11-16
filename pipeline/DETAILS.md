# Board pipeline: details

The reference behind [README.md](README.md): what every step does, the files it
writes, and the settings. Start with the README.

## Setup

Two camera setups use the same commands:

| setup | camera | calibration target |
|---|---|---|
| `eye_in_hand` | Azure Kinect on the wrist (rigid to `panda_link6`) | the board itself |
| `stationary` | Femto Mega (+ Bolt) fixed in the cell | ArUco tag plate on the gripper |

Robot: Franka Panda (FER, system 4.2.x), ROS Noetic, libfranka 0.9.2.

The `pipeline` command runs on the host; everything that touches the robot or
the cameras runs in the `franka_ros` container through `docker exec`. Start
the container with `docker compose up -d` (not `run`: the CLI finds it by the
name `franka_ros`). The board fitting uses the Python 3.12 venv of the
pose-refinement checkout next to the workspace (`../pose-refinement`); set
`paths.pose_refinement` in `config/pipeline.yaml` or `PIPELINE_POSE_REFINEMENT`
if it lives elsewhere. `alias pipeline=$PWD/pipeline/pipeline` saves typing.

### Launch files (in the container)

| launch | starts |
|---|---|
| `panda_calibration robot.launch robot_ip:=...` | franka_control (libfranka 0.9.2), franka_state_controller at 250 Hz, `position_joint_trajectory_controller`, the gripper, and MoveIt's move_group for IK (`moveit:=false` skips it; `use_rviz:=true` opens RViz) |
| `panda_calibration azure_4k.launch` | Azure Kinect under `/azure_camera`: 3840x2160 BGRA colour at 15 fps, NFOV unbinned depth registered to colour (`depth_to_rgb`), no point cloud |
| `panda_calibration orbbec_4k.launch` | Femto Mega under `/mega_camera`, Femto Bolt under `/wrist_camera`: 3840x2160 colour, 640x576 depth registered to colour; `mega_serial:=`, `bolt_serial:=`, `mega_ip:=`, `bolt:=false` |

### Robot checks

**Realtime kernel.** On the generic kernel franka_control dies with `Running
kernel does not have realtime capabilities`. GRUB hides its menu: hold Shift at
boot and pick the realtime kernel under *Advanced options*, or make it the
default in `/etc/default/grub`
(`GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux <version>-realtime"`,
then `sudo update-grub`).

**`NE_T_EE`** must be the identity:
`rostopic echo -n1 /franka_state_controller/franka_states/NE_T_EE`. If it holds
the peg tip, every probe stops with `O_T_EE ... differs from the FK of
panda_hand_tcp by 42.34 mm`. Reset it (the services need the controller stopped):

```bash
rosservice call /controller_manager/switch_controller "{start_controllers: [], stop_controllers: [position_joint_trajectory_controller], strictness: 2}"
rosservice call /franka_control/set_EE_frame "NE_T_EE: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]"
rosservice call /controller_manager/switch_controller "{start_controllers: [position_joint_trajectory_controller], stop_controllers: [], strictness: 2}"
```

**Camera load.** The wrist camera is not in Desk's end-effector model. Enter
it the same way (controller stopped), mass in kg, centre of mass from the
flange in m, then check `.../franka_states/m_load`:

```bash
rosservice call /franka_control/set_load "{mass: 0.68, F_x_center_load: [0.02, -0.10, -0.12], load_inertia: [0.001,0,0, 0,0.001,0, 0,0,0.001]}"
```

With it, moves settle 0.3–0.6 mm from the target instead of 1.5–1.8 mm.

Every pipeline motion is a single
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
   along every path, whole board in the image). They stay within `plan.box_deg`
   of the look pose, which sees the board from its far side, so they all look
   back toward the robot base. A bias along that direction then does not average
   out. Therefore 4 more views (`calibrate.robot_side_views`, 0 turns them off)
   look at the board from the robot's side. In those the camera is between the
   base and the board, looking away from the robot: 30–50° from vertical
   (oblique views pin down distance and scale), spread up to 80° left and right
   of the direction to the base. They stay within 40° of READY on joints 2-6
   (80° on joint 1, which turns the arm about the base), keep the camera at
   least 20 cm from the base and the upper arm, and the capture visits them
   first: READY → robot-side views → READY → look pose → the rest. READY sees
   the board too, so its image is one more robot-side view. The limits are
   the `ROBOT_SIDE_*` constants in `eye_in_hand.py`.
4. `capture`: drives through the views and stores image + measured joints for each.
   If a reflex stops it halfway, the views recorded so far are kept (at least
   10 are needed). There is no resume: `pipeline calibrate --from capture`
   drives through all views again.
5. `fit`: refines the board pose in every image, seeded from the robot.
6. `bundle`: refines the camera intrinsics over all views.
7. `solve`: fits the camera mount, joint 2-5 encoder offsets, scale and mount sag.
   It prints the leave-one-out board-corner error: Panda calibrations of
   2026-11-12…16 gave medians of 0.4–1.1 mm and p90 of 0.6–2.6 mm. Joints 1, 6
   and 7 cannot be estimated (a joint-6 offset is indistinguishable from the
   camera mount; joint 7 does not move the camera). The fitted offsets have
   been J3 ≈ 7–10 mrad and J5 ≈ 9–11 mrad; a value far from the earlier ones
   (J5 = 18 mrad on 2026-11-16) means the fit absorbed something else, so
   recalibrate. The board pose fitted along the way is
   written to `board_pose.json`, so `check`/`probe` can run straight away as
   long as the board has not moved. **Temporary test hack:** the board written
   there is moved 3 mm along −x of the base, toward the robot
   (`CALIBRATE_BOARD_SHIFT_BASE_M` in `eye_in_hand.py`; set it to 0 to turn it
   off). The fitted pose is kept as `T_base_board_fitted`. `locate` does not
   shift.

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

### `grasp` / `release`

`grasp SHAPE` closes the Franka Hand on a peg: `round`, `hexagon`, `triangle`,
`rectangle` or `l_shape` (the shapes in `config/grasps.yaml`); `--hole round`
or `--hole round_c1mm` (a hole means its peg) work too. Hold the peg in
the V-block, then run it. The hand closes until the fingers clamp the peg
(`grasp_peg --until-contact`). It accepts any width up to the peg's width in
`config/grasps.yaml` plus 1 mm, because pegs sit deeper in the printed V than
the CAD says and the finger reading drifts. It prints the width it ended at.
Fingers that close on nothing (below 0.5 mm) count as a failed grasp. The hand
keeps squeezing with `--force N` (default 20 N) until it is released.
`--epsilon MM` gives the old check instead: the fingers must stop within MM of
the CAD width.

Release before switching pegs: otherwise the fingers are still on the old peg
and cannot close on the new one.

`release` opens the fingers to `--width MM` (default 30 mm, wider than every
peg's grasp; at most 80) with `franka_gripper`'s move action, so the peg drops
unless something holds it: keep a hand under it, or release it once it sits in
a hole. Fingers already open at least that wide are left alone.

Both move only the fingers, never the arm. Do not run `/franka_gripper/homing`
while holding a peg: homing closes and reopens the fingers fully.

### `repeatability`

Shows how well the arm returns to commanded joint angles, with nothing
compensated. It closes the hand until it clamps the peg and records the finger
width it measures (`--no-grasp` skips that), then for each point: you guide the arm there by hand, it goes home, comes back with one
ordinary trajectory goal to the taught joint angles and records where it
stopped, `--repeats` times (default 3). `--distances` gives one rough distance
from the base per point (default 25 50 75 cm).

Results go to `repeatability/<date>_<time>/`: `measurements.csv`, `run.yaml` and
`repeatability.png` (tip offset in x, y, z and the error of each joint, measured −
commanded). The tip is `panda_hand_tcp` plus the peg's tip offset. All of it comes
from the robot's own joint readings, so it shows how the arm settles against its
commands, not where the tip is in the real world.

`repeatability --resume` finishes the newest interrupted run (or `--resume DIR`):
finished points are kept, and an unfinished one can reuse its saved joint angles
or be taught again.

### `accuracy`

Probes several holes in a row. For every move it records where the probe sent
the arm and where the arm settled: the commanded TCP pose and joints, and the
measured ones. The default holes are `round_c0p03mm round_c3mm rectangle_c0p03mm
rectangle_c3mm`, the tightest and the loosest hole of two shapes. Give other
holes as arguments, in the order to probe them.

For each hole:

1. **Peg.** Before the first hole, and whenever the shape changes, it asks for
   the peg. Press Enter if the peg is already in the fingers. Press `g` to swap
   it: the hand opens, you put the peg in the V, and the hand closes until it
   clamps (`grasp_peg --until-contact`).
2. **Probe.** The usual `probe`, recording as it goes. Enter approves the
   approach, then the tip goes down in 3 mm steps. It stops when it reaches the
   board surface, or when you press Enter. With `--height MM` it stops that
   high above the surface.
3. **Photos.** At the bottom the arm holds still for your photos, and the
   terminal prints the hole and the time. Type what you see (optional), then
   press Enter to retract.

The results go to `sessions/<name>/accuracy/<date>_<time>/`:

* `plan.yaml`: the holes in order, their peg, the tip offset used, and how far
  the joint-offset correction moved each target;
* `<hole>.yaml`: every attempt of every move (the approach, each descent step,
  the retract), plus the pose held for the photos with your note;
* `measurements.csv`: one row per attempt and per hold. Each row has the target
  and the measured tip and TCP, their difference (in base axes, and the tip
  also in board axes), and the commanded and measured joints and their
  difference;
* `probe_accuracy.png`: one column per hole. The rows are the tip error above
  the hole, the tip error at the bottom, and the joint errors at the bottom.
  Each shows the first try and the result after the correction.

**Reading it.** The probe corrects every move. If the arm settles more than
0.2 mm from the target, it shifts the command by the error and tries again (at
most 3 times). The *first try* shows how far the arm settles on its own (about
0.3–0.6 mm above the hole with the camera load entered, 1.5–1.8 mm without
it). *After the correction* is where the probe accepted the move, so it is
within 0.2 mm by design.

Both come from the robot's own joint readings through the nominal model. They
cannot show an error in the calibration, the board pose or the peg. Suppose the
plot says 0.1 mm but the photo shows the peg 2 mm beside the hole. Then the
motion is fine, and the error is in the board pose, the joint-offset model or
the peg. The `joint-offset correction moved it` line in each column title shows
how far the target was shifted for the calibrated joint offsets. Compare it
with what the photo shows.

`accuracy --resume` probes the holes that an interrupted run did not finish.
`accuracy --plot` draws the figure and the table of a run again. Both use the
newest run unless you give its folder.

### `check` / `probe`

`check --hole NAME` validates the target and IK without moving. `probe --hole NAME`
approaches `probe.approach_height_m` above the hole with the peg
(`board_manipulation.yaml`), then descends in guarded steps:

* Enter approves the approach.
* Enter during the descent stops it.
* Enter again retracts.

It never brings the peg tip below the board surface. `--height MM` targets a
point above the surface instead. `--turn DEG` turns the hand further about the
peg axis (a multiple of the peg's symmetry; any angle for the round peg), to
see whether an error turns with the hand. Approach and descent settings are in
`src/panda_calibration/config/board_manipulation.yaml`: 20 mm approach height,
3 mm descent steps. Every move is checked against the robot's measured TCP and
corrected up to 3 times until it is within 0.2 mm.

Both print where the peg tip is now, from the robot's measured TCP plus the tip
offset: at the start, after the approach and at every descent step, as the
distance from the target in board axes (`x`/`y` beside it, height above it).
`tip --hole NAME` prints only that, without moving; `--watch` repeats it twice a
second until Ctrl-C, also while you guide the arm by hand.

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

**Measured tip instead of CAD.** `check` and `probe` accept `--tip-from FILE`,
a result of the pivot calibration (`rosrun panda_calibration pivot_calibration`,
see `src/panda_calibration/README.md`). For that one run it replaces the CAD
TCP-to-tip vector of the peg the file was measured on; other shapes keep their
CAD tip. The `held peg:` line prints both tips and how far apart they are, and
the run is refused if the measured tip is more than 5 mm from the CAD one in
distance from the TCP axis or along it (wrong file or wrong peg):

```bash
./pipeline/pipeline probe --hole round_c1mm --tip-from tool_calibration/pivot_2026-11-05_143012.yaml
```

For eye_in_hand, the board pose is corrected for the calibrated joint
offsets at the chosen hole, because MoveIt plans with the nominal kinematics.
The corrected target is written to `targets/<hole>.json`, next to the
peg tool `targets/<hole>_tool.yaml`; the uncorrected board is kept in it as
`source.T_base_board_true_frame`. The correction (2–5 mm) is computed for the
arm configuration the host's IK picks. MoveIt, which the probe uses, solves
the same TCP pose from the arm's current joints and can pick another elbow
(on 2026-11-12 joint 1 differed by 25–117°), so at such a hole the correction
is only approximate.

`pipeline holes` lists the hole names (`<shape>_c<clearance>mm`, e.g.
`hexagon_c0p2mm`) with board coordinates. Once a session has a board pose, it
also lists their position in the base frame. The coordinates come from the
board STL (`config/holes.yaml`, regenerated by `tools/extract_holes.py`).

### `diag`: where does a constant probe error come from?

Six tests for the case "the peg lands 2-3 mm from the hole, always the same way".
Each test changes one thing and watches how the error reacts. `pipeline diag`
lists them and asks which to run; `pipeline diag N` runs test N directly, and
`pipeline diag N --report` prints the analysis of its newest run again.
Results go to `sessions/<name>/diagnose/test<N>_<name>/<date>_<time>/`
(`run.json` with everything, `report.txt`). They need an eye_in_hand session
with `calibration.json` and `board_pose.json`.

| test | what changes | it separates | how |
|---|---|---|---|
| 1 camera roll | the camera turns about its optical axis over the board | hand-eye translation / principal point (turns with the camera) from the true board (does not) | arm moves, vision only |
| 2 heights | the camera backs off along its optical axis, 35-75 cm | angular errors (grow with distance) from translation errors (constant) | arm moves, vision only |
| 3 tool turn | the hand turns about the peg, same hole | a wrong TCP / peg tip (turns with the hand) from everything else | hand-guided touch |
| 4 board turn | the board turns 180° on the table | a wrong board model (turns with the board) | touch, locate, touch |
| 5 board aside | the board moves to y = ±30 cm | kinematics (error points at the base) from a frame mix-up (stays along base x) | touch, locate, touch |
| 6 touch-off | 3-4 distant holes | the arm's distances between holes against the drawing (scale, rotation, common offset) | hand-guided touches |

Suggested order: 1 and 2 (nothing to touch, the board stays), then 3 and 6,
then 4 and 5 (they move the board).

**Tests 1 and 2 (camera).** They plan camera poses around the look pose's
view of the board, drive through them (READY, then the look pose first, as
`calibrate` does), fit the board in every image with the calibration held
fixed, and compare the board poses they give. `--plan-only` plans and prints
without moving. The camera sits on link 6, so only joints 1-6 can roll it: the
Panda reaches about ±45-60° of roll over the board, not 180° (90° would need
joint 2 past its limit). Rolls it cannot reach are skipped. Near these poses
the arm is close to its shoulder singularity (joints 1 and 3 almost aligned),
so 1° of camera roll turns joints 1 and 3 by 4-5° against each other. The
planner therefore walks the camera from view to view in 1° steps and puts
waypoints from that walk between the views, so no joint moves more than
`plan.max_step_deg` (45°) at a time. The waypoints are captured too, but not used.

* Test 1 fits `board centre = c + R_camera · t` over the views: `t` is the
  in-plane camera error (hand-eye translation, or distance × angle for a
  rotation / principal-point error). The report prints what `t` does to the
  located board in base axes at the look pose's camera orientation, which you
  compare with the -x probe error. |t| < 0.5 mm means the camera side is fine
  in that plane.
* Test 2 fits a straight line of the board centre over camera distance. Its
  slope × the look pose distance is the part of the board pose that an
  angular error (hand-eye rotation, cx/cy, sag) or the scale term puts in
  there. A slope near zero means the camera chain has no angular error, and a
  remaining offset is a translation or comes after the camera.

Options: `--rolls DEG...` (default `0 30 60 0 -30 -60 0`), `--distance M` (test 1,
default 0.50), `--distances M...` (test 2, default `0.35 0.45 0.55 0.65 0.75`),
`--view look|top` (look = along the look pose's view, the one `locate` uses;
top = straight down), `--max-step DEG`, `--yes`.

**Tests 3-6 (touch).** You guide the held peg into a real hole by hand; the
robot's joints then say where it thinks the tip is. Each touch:

1. stops `position_joint_trajectory_controller` (asks first), so the arm can
   be guided with the buttons on the wrist;
2. prints a live line: the tip relative to the probe target in board axes,
   and the hand yaw (for test 3 also how far it is turned from the first touch);
3. Enter (arm still, buttons released) averages 1 s of robot state; it is
   refused if the arm moved;
4. restarts the controller once you let go of the arm. The peg can stay in
   the hole.

The result of a touch is the **probe error** = where the probe sends the tip
(the vision target) − where the hole really is (the touched tip), in base axes
and in board axes. "2-3 mm too close to the base" reads as x ≈ −2.5 mm. Both
are in the calibrated frame: the touched tip is FK(q + dq) with the
calibration's joint offsets, the frame the probe aims in after its correction.
Use a tight hole (default `round_c0p2mm`, ±0.1 mm). Grasp firmly
(`pipeline grasp round --force 40`), keep the peg upright, and let it seat on
its own: a peg pushed sideways in the V-block fingers moves the tip.

* Test 3 (`--holes HOLE`): touch 1 with the hand as usual, touch 2 with the
  hand turned about the peg (toward 180°, as far as joint 7 allows; 90° also
  works), more if you like. The fit splits the error into a part that stays put
  in base axes (not the tool) and a peg tip offset in TCP axes (real − model
  tip), which turns with the hand. `pipeline check/probe/tip --turn DEG` drives
  the same turned hand with the probe instead, for photos.
* Test 4: touch, turn the board 180° on the table, `pipeline locate --fresh`,
  `pipeline diag 4` again. Each run adds a board placement to the newest series
  (`--new` starts another). If the error stays put in board axes, the board
  model is off; if it stays put in base axes, the board model is fine.
* Test 5: the same with the board in front, at y = +30 cm and at y = -30 cm
  (same orientation to the base). The look pose may not see the board there:
  jog the arm until the camera sees it and run `pipeline locate --fresh --here`.
  The report compares "error points at the base" (kinematics) with "error
  constant in base axes" (frames).
* Test 6 (`--holes A B C D`, default `round_c0p03mm round_c1mm rectangle_c1mm
  rectangle_c0p03mm`, the corners of a 189 × 104 mm rectangle): one touch per
  hole. It asks you to swap the peg when the shape changes, as `accuracy`
  does. It prints the distance between every pair as the drawing has it and
  as the arm measured it, and a 2D fit of the touched holes onto the drawing:
  scale (per mille), rotation and common offset. A scale error points at
  kinematics or the board print (test 4 tells which). A common offset with the
  right scale points between the camera chain and the arm (tests 1-3).

`--tip-from FILE` uses a pivot-calibrated tip in tests 3-6, as in `probe`.

## Sessions, reruns and starting over

Every result lives in one folder, `sessions/<name>/`. The CLI keeps a
`sessions/current` link, so you never type paths:

```
sessions/2026-10-22_eye_in_hand/
  config.yaml          settings snapshot (copied from config/pipeline.yaml)
  calibrate/1_look/ 2_seed_fit/ ...   one folder per step
  locate/1_look/ ...
  calibration.json     board_pose.json     targets/
  accuracy/<date>_<time>/     `pipeline accuracy` runs
  diagnose/test<N>_<name>/    `pipeline diag` runs
```

`pipeline repeatability` results go to `repeatability/<date>_<time>/` in the
workspace, pivot calibrations to `tool_calibration/`: they belong to the arm
and the tool, not to a session.

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
| `board_pipeline/diagnose.py` | `pipeline diag`: the six error-source tests, their planning and analysis |
| `ros_nodes/touch_hole.py` | hand-guided touch of one hole (runs in the container), for `diag` 3-6 |
| `src/panda_calibration/panda_calibration/gripper.py` | `grasp_peg`, `release_peg`: gripper action clients behind `pipeline grasp` / `release` (run in the container) |
| `config/` | settings template, hole and grasp tables, look pose, prior hand-eye, calibration poses |

## Troubleshooting (all known messages)

* **`position_joint_trajectory_controller is stopped`** / **`... is running and
  claims the arm joints`**: the message prints the `switch_controller` call
  that fixes it.
* **`trajectory goal ended in state ABORTED`**: usually a reflex; check
  franka_control's output, recover, rerun.
* **`reflex mode after cartesian_reflex`**: something pressed on the arm, the
  camera or its cable, or a move collided. Free it, recover, rerun. If it keeps
  happening, enter the camera load (Robot checks).
* **`reflex mode after communication_constraints_violation`**: the PC missed the
  1 kHz control deadline; apply the network fix, recover, rerun.
* **`reflex mode after joint_motion_generator_position_limits_violation`**: a
  joint (often 7, with the rectangle or L peg) reached its limit. Recover; try
  the hole with `--turn` (a turn the peg's symmetry allows).
* **`Connection to FCI refused`**: FCI is off in Desk (the robot still answers
  ping). Desk turns it off when someone uses Desk or after a robot restart.
* **`O_T_EE ... differs from the FK of panda_hand_tcp by 42.34 mm`**: `NE_T_EE`
  holds the peg tip; reset it (Robot checks).
* **`Error in XmlRpcClient::writeRequest: write error (Connection refused)`**: a
  node exited without unregistering (e.g. a crashed camera driver). Harmless;
  `rosnode cleanup` clears it.
* **`seed fit failed`**: the board is not fully in view from the look pose. Jog
  the arm until it is, then `pipeline locate --here`.
* **A capture step times out waiting for images**: are the robot and camera
  launches running? Check with `rostopic hz` in the container.
* **`export refused ... unverified`** (stationary): the two cameras disagree;
  the rig moved, recalibrate.
