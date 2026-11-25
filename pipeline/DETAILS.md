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

Camera parameters and joint offsets come from `eye_in_hand.base_calibration`
in `config/pipeline.yaml` (2026-11-19: AprilGrid intrinsics, fx 1848.9, and the
`tools/joint_offsets` result, J2–J6). A new session copies them into its
`config.yaml`, and `calibrate` keeps them fixed: `bundle` only refines the view
poses, `solve` fits only the camera mount and the board around them (scale and
sag are off by default; add them to `calibrate.model` to fit them).
`--fit-intrinsics` estimates fx fy cx cy from the board views instead
(`calibrate.bundle_free`), and `--fit-joint-offsets` estimates the joint 2-5
offsets (model term `off`). If those steps are already done, add `--from fit` or
`--from solve`. `calibration.json` → `source` says which was used. Sessions
without `base_calibration` (from before 2026-11-19) fit both, as before.

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
   long as the board has not moved. A test shift of the written board (`CALIBRATE_BOARD_SHIFT_BASE_M` in `eye_in_hand.py`) is off (0); when set, the fitted pose is kept as `T_base_board_fitted`.

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
peg axis than the automatic choice (below), to see whether an error turns with
the hand. Approach and descent settings are in
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

The peg frame is lined up with the hole frame, turned about the peg axis by
one of the allowed turns: they all fit the hole, but each puts the wrist
(joint 7) somewhere else. `check`/`probe` solve the IK for every allowed turn
(every 10° for the round peg) and use the one that turns joint 7 least from
where the arm stands now, among those that keep it at least 0.35 rad
(`WRIST_MARGIN_RAD` in `grasp.py`) inside its ±2.897 rad limits. Without a
robot state, joint 7 is kept as close to mid-range as possible. The candidates
are printed, the chosen one marked `->`. The L has no other turn: if its one
placement leaves joint 7 too close to a limit, `check` warns, and turning the
board on the table is the fix. `--turn DEG` turns the hand further from that
choice (it warns when joint 7 gets close to its limit). The resulting tool
(TCP orientation and TCP-to-peg-tip vector) is
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
is only approximate. `--no-joint-correction` (`check`/`probe`/`tip`/`accuracy`)
skips the correction and drives to the board pose as calibrated; it still
prints the shift it skipped.

`pipeline holes` lists the hole names (`<shape>_c<clearance>mm`, e.g.
`hexagon_c0p2mm`) with board coordinates. Once a session has a board pose, it
also lists their position in the base frame. The coordinates come from the
board STL (`config/holes.yaml`, regenerated by `tools/extract_holes.py`).

### `insert`

`insert --hole NAME` puts the held peg into the hole. Vision and the arm are a few mm off, so the
peg is not driven straight in: it runs the tilt search of rl_manip `robot_session` (`peg_search
tilt-slide`, libfranka, its own container `peg-search-libfranka-0.9.2`; robot_session/README.md 4b).
Settings: the `insert` section of `config/pipeline.yaml` (sessions from before 2026-11-21 have none
and use the template's).

No key presses: every step is decided from what the robot measures.

**Default flow (`insert.method: plunge`).** One position-controlled plunge at the vision hole (the
peck below, one point). Not in: the peg tip touches the board 22 mm beside the hole (`insert.touch_offset_m`,
along the board's x, between the hole columns): that is the board height the robot measures, the plunge's
contact is classified against it (on the board, or stuck in the hole mouth), and it is the plane of up to
`insert.attempts` tilt-search attempts (`--board-plane`, no probe, `--posture start`). Stuck in the mouth: the
first attempt is the compliant plunge at that spot (its hold presses the peg in). Before every later attempt
`peg_search lift --to-z --upright` turns the peg upright at the start height. `--method peck` is the grid
below; `--method tilt` the old attempts (steps 4-5).

**Board height model (2026-11-23, `board_pipeline/boardheight.py`).** Every board touch is kept in
`<session>/insert/board_heights.yaml` with vision's height there and the board pose's sha256. Robot − vision
height fits a plane over the board plus one offset per shape's grasp: 9 runs of 2026-11-23, 0.07 mm RMS;
vision 1.1-1.8 mm low and 0.25° tilted. The pegs are all one length, yet the triangle's tip touched 0.42 mm
higher than the L's: the grasp (held higher in the fingers than the grasp model says, or tilted in them, which
an upright touch reads as the face's lowest edge). Each run predicted from the others: 0.12 mm RMS with no
touch of its own, 0.04 mm (max 0.08) with one; the one touch measures the grasp actually held.
So once the model knows the board's slope (`insert.board_touches: auto`) insert touches **once** beside the hole
(this grasp's height, the model's slope) instead of three times, and the plunge aims at the model's board
instead of vision's (round / hexagon 0.03 mm that day: the board 2.1 mm above vision's, the contact read as "in
the air", the retry pressed into the board and faulted). `--touches 0|1|3` overrides; `pipeline found` prints
the model and the leave-one-run-out check (it also collects the touches of runs from before the file existed).

The **peck search** (`--method peck`) instead of
step 4's tilt-search attempts: one `peg_search peck` run, libfranka's
internal joint position controller (0.2-0.6 mm tracking near the base, where the tilt search's impedance
lagged 3-11 mm, 2026-11-21). Over each point of a 1.7 mm hexagonal grid within 6 mm of the vision hole
(nearest first), 6 mm above the board with the peg upright, straight down at 3 mm/s until contact (back
up, next point) or the tip 15 mm below the board (inserted, then step 5). `insert.peck_options`.

1. **Probe approach.** As `probe`, with `--height` = `insert.start_height_mm` (6 mm above the vision
   board; the vision hole moved by the board correction of step 5 once holes are found),
   `probe_board --stay --yes`: the descent ends there and the arm holds. Its clearance check is
   against the board surface (until 2026-11-21 against the target height, so a settle correction just
   below a 6 mm target was refused). `--here` skips it (the tip already within the approach above the board).
2. **Hand over.** robot.launch is stopped (SIGINT to `insert.robot_launch_match` in the container, as
   Ctrl-C in its terminal). The robot takes one FCI client: on 2026-11-21 peg_search connecting killed
   franka_control and the whole launch. The arm holds where it is.
3. **Plunge** (the first attempt that gets past the posture move, `insert.plunge`). After the posture
   move and the tare, peg_search lowers the upright peg straight down over the hole at 2 mm/s under its
   impedance control (compliant) until the descent's contact test (the tip lags 2.5 mm and the force
   changes 6 N). The tip 15 mm below the vision board: inserted, done. Stopped `insert.plunge_in_depth_mm`
   (3 mm) or more below it: the peg is in the hole mouth, and the hold presses it in (wiggle, yaw sweep,
   retries); if that jams, the search goes on about that point. Higher: it was the board; back up to the
   start height, then the search.
4. **Attempts** (`insert.attempts`, 5). Each is one `peg_search tilt-slide --yes` run. It turns the elbow
   to arm angle 0 first (robot_session/POSTURE_FINDINGS.md; its start check allows 15 N, the force estimate
   read 6-13 N on a free tip), tares the force in the air, then searches:
   * `--hole-xy`: the hole as vision sees it (the joint-offset corrected target, the probe's).
     Every run searches its 5 mm error disc: lines across the slide direction (`insert.pattern`,
     default `lines`, robot-tested on the hexagon; the round peg always uses lines). Spacing (2026-11-21):
     0.7 x the lead's effective dip band (`tilt_leads.yaml`), at most `insert.max_line_spacing_m` (1 mm)
     for a flat lead (round, hexagon: robot-tested, yaw-sensitive) and `insert.max_corner_line_spacing_m`
     (null: none) for a corner lead (triangle, rectangle, L-shape: 5 mm, 3 lines instead of 11). Lines
     1 + 2k, line k covering half a spacing either side of k x spacing out to the error, centred
     (`--line-centre`) on the dip band's middle: the L-shape's corner runs 8.3 mm beside the tip's line,
     so its lines sit 8.5 mm across (line 0 covered hole errors -2.3..+19 mm before).
   * Lead and slide direction: `config/tilt_leads.yaml`, generated from the simulation
     (`tools/extract_tilt_leads.py`, as `extract_holes.py`): per hole the lead (a flat, a corner),
     its offset from the peg axis along and across the slide, run-up, slide length, spiral, and the
     dip region (the lead positions from the hole centre where it dips: `near_m`, `band_m`, `along_m`,
     `eff_band_m`, the band whose lines run at least 1 mm inside the region). Of the leads, the fewest
     lines, then the simulation's group order, then the slide nearest base +y
     (`insert.slide_dir_base_deg`: tilting about base y vibrated the wrist); the round peg
     slides along it exactly. The 3 mm rectangle hole is not in the simulation: it uses the 1 mm's.
   * The first run that searches probes the board (4 points 25 mm about the hole); later attempts reuse
     that plane (`--board-plane`), the same posture rule, so the same kinematic error.
   * Belief. A hold (the peg upright in the hole mouth: the tip on the hole axis) makes the next
     attempt start with a located search of 2 mm about it (`insert.located_radius_m`), 3 mm if that
     hold jammed (`insert.jam_radius_m`: the peg wedged in the mouth, robot 2026-11-21 2.02 mm from the
     hole); a dip only (the lead fell over the hole near its rim): the hole centre `along_m[0]` behind
     the dip lead along the slide and, across, within the dip band intersected with the searched disc,
     radius at least 3.5 mm (`insert.dip_radius_m`), at most the hole error. Until 2026-11-21 the dip
     put the hole at the lead + its distance from the axis: 5.2 mm off at triangle_c1mm (now 1.3).
     A hold is replaced only by a newer hold. After the
     located search misses, the run falls back to the vision disc.
   * A run that does not insert lifts back to its start height (`--lift-at-end`).
   * Every run gets `--board-z` (the vision board top): a descent's contact (the plunge, the probe,
     the lines) counts only with the tip stopped and within 2.5 mm above the board (the probed plane
     once known). In free air the tip trails a descending target by 3-5 mm of joint friction and the
     force estimate moves with it (2026-11-21: "contacts" 15-40 mm above the board). A descent ends on
     the tip's travel, its target at most 9 mm below the tip, and a tip lagging in the air gets peg_search's
     air push (up to 8 N; robot_session README): after a jam, 9 mm of lag used up the target's travel with
     the tip 1.9 mm above the board, "no board contact" four times (triangle_c1mm, 2026-11-21).
   * A run that fails (peg_search's guards: no board, a sink, a lag, a speed limit, the posture move) is
     followed by `peg_search lift` straight up or down back to the start height (libfranka's joint
     position controller, it stops if the peg is held back) and the next attempt. A failed posture
     move does not use up an attempt (the later runs skip it). A probed plane more than 3 mm from
     the vision board is not reused. A run that cannot connect, or a lift that fails, ends the attempts.
   * **Error disc** (`insert.error_m`, 2 mm since 2026-11-21): the search is sized for it, not for
     `tilt_leads.yaml`'s simulated 5 mm: run-up and spiral radius 3 mm shorter, slide 6 mm shorter, lines
     out to 2 mm (a flat lead 5 lines of 1 mm, a corner lead 1 line); every located radius at most it.
     The holes inserted that day were 0.2-0.9 mm from vision.
   * **Belief grid** (`insert.search: grid`, the default with lines; `--search belief` the hold / dip
     belief above; board_pipeline/beliefgrid.py). A probability grid of the hole, 0.1 mm cells (prior:
     the corrected vision hole, sigma half the error disc), updated after every attempt from its result
     and its 1 kHz log: a line whose lead slid without dipping makes the hole centres where its path ran
     at least 1 mm through the dip region (`tilt_leads.yaml dip_region_m`, narrowed for a flat lead to the
     1 mm strip it is trusted with) less likely (x 0.2, `grid_p_detect` 0.8); a dip, a hold (sigma a
     quarter of the located radius, half the jam radius if it jammed) and a plunge (on the board: not where
     it would have gone in, the hole's `free_m`; stuck: near it) weigh in (`grid_p_true` 0.9). Each attempt's
     lines are planned on it: `--hole-xy` its mean, `--line-offsets` in order, greedily the strip holding the
     most probability not yet covered (none twice in a run), until 95 % is covered (`grid_target`). A run
     that died after two lines goes on with the strips not tried, instead of from line 0 again. Simulated
     (tests/test_beliefgrid.py, runs aborting after 30 % of lines): the hexagon's 1 mm hole found as often in
     ~14 % fewer lines; a one-line corner plan gains nothing from misses, only from holds, dips and plunges.
     `grid_<k>.npz` in the run folder: the grid before attempt k.
5. **Centre** (inserted, before the release; `insert.centre_after_insert`, `--no-centre`). `peg_search
   centre`: from where the peg sits, the tip moves straight in 6 directions (`insert.centre_touches`)
   within the hole, libfranka's joint position controller at 1 mm/s, until it touches a wall (a 5 N fast
   force change or the tip 0.3 mm held back), and back. The hole centre is the shift that puts the
   contacts on the outline of the hole's free region (`tilt_leads.yaml free_m`: the peg offsets at which
   it fits; the outline may move out by 0.2 mm, a printed hole off its CAD; simulated 20 um contact noise:
   at most 54 um off over 520 cases). Too few contacts, all on one side, or a poor fit: the seated tip
   instead (sigma half the free region's reach). Recorded in `<session>/insert/found_holes.yaml` with the
   vision position insert aimed at and the board pose's sha256.
   **Board correction** (`insert.board_correction`, `--no-board-correction`; board_pipeline/holefit.py):
   every insert fits one rigid shift of the board in the base plane to the found holes of the current
   board pose (weighted by their sigmas; priors 3 mm and 0.3 deg, so one hole gives a shift and a turn needs
   holes far apart; a hole 3 sigma and 0.5 mm off the fit is dropped) and aims at the corrected target
   (`targets/<hole>_corrected.json`: the probe, the plunge and the search follow). `pipeline found` lists
   the record and the correction; `pipeline found --import-runs` adds the inserted runs of `insert/` since
   the board pose was written (their seated tips). The seven of 2026-11-21 give (-0.34, +0.10) mm +-0.30,
   0.12 deg +-0.16.
   **Local field** (`insert.correction_local`, 2026-11-21): the kinematic error after the joint-offset model
   varies with the posture, so what one rigid shift leaves need not be noise. The rigid fit's residuals at
   the found holes are modelled as a smooth field over the board (Gaussian process, `correction_length_m`
   60 mm) plus a scatter of each insertion's own (a new peg in the fingers each hole), the two split by
   maximum likelihood (4 holes at least; the field at most `correction_field_cap_m` 1 mm). The correction
   at a hole is the rigid one + the field there: near found holes it follows them, far from all of them, or
   when the residuals are per-hole scatter, it is the rigid one. Simulated on the 20 holes: a 0.4 mm smooth
   field, each hole predicted from the others 0.37 mm off with the rigid shift, 0.13 mm with the field; 0.3
   mm per-hole scatter: the field stays off (no worse than rigid). `pipeline found` prints the model check
   (chi^2 of the rigid residuals over the holes' sigmas, the scatter beyond them) and leave-one-out: every
   found hole predicted from the others with no correction, rigid, and rigid + field. 2026-11-21's seven
   seated tips: 0.58 / 0.40 / 0.40 mm (the rigid shift helps; seated tips, +-0.5-1.5 mm, cannot show a
   field: wall touches can).
6. **Finish** (libfranka). Inserted (the tip 15 mm below the board top): robot_session `gripper open`
   to 30 mm, then `peg_search lift` `insert.lift_m` (6 cm: clear of the pegs standing in holes). Not inserted: the same lift with the peg, and the command
   fails. Then robot.launch is started again in the background (`insert.robot_launch`, log in
   `log/robot_launch/`), and `insert` waits for the trajectory controller and set_load, which resets
   NE_T_EE to the identity and the load (peg_search sets the peg tip). A robot.launch you start yourself
   later must not run next to it: stop that one first (`docker exec franka_ros pkill -INT -f roslaunch`).

`--dry-run` checks the probe target and IK (needs robot.launch) and peg_search's options for
attempt 1, without connecting. Other options: `--height`, `--attempts`, `--pattern`, `--slide-dir`,
`--tip-from`, `--turn`, `--no-joint-correction`, `--no-board-correction`, `--no-centre`, `--search`.

Results: `sessions/<name>/insert/<date>_<time>/`: `plan.yaml` (hole, lead, settings, the common
peg_search options), `attempt_<k>.csv` (peg_search's 1 kHz log, `robot_session/analyse_log.py`),
`attempt_<k>.json` (outcome, the plunge's, where the tip held and dipped, the board plane), `grid_<k>.npz`,
`centre.json` / `centre.csv` (the wall touches), `lift_*.csv`, `summary.yaml` (each attempt's grid plan, the
found hole).

peg_search's tip comes from the pipeline's held peg (`--peg-axis-x/-y`, `--peg-tip-z` = TCP 103.4 mm
+ the tip offset), so the believed hole and the search use the probe's tip. Its build must be current:
`cd ../rl_manip/robot_session && docker compose run --rm robot bash -c 'cmake --build build -j'`.

### `extract`

`pipeline extract --hole H` pulls the peg out of hole H with the empty hand (`board_pipeline/extract.py`, rl_manip
robot_session `peg_search extract`). Settings: the config's `extract` section.

1. **Where the peg is.** The campaign's release record of the hole (or `--from-run RUN`, an insert run): the peg
   tip pose when the hand opened there, from `lift_release.json` (peg_search `lift --result`, written by every
   insert since 2026-11-23). Returning the hand to that pose is exact whatever the grasp. Without one (a peg placed
   by hand), the vision hole with the board correction and the CAD seat; for `extract.mirrored_shapes` (the real
   rectangle and L pegs are the CAD's mirror images) the mirrored seat (`grasp.mirror_seat`: y negated, yaw ->
   180 deg - yaw).
2. **Approach.** probe_board to `approach_height_mm` above the hole (ROS), unless `--here` (the arm still over the
   released peg). robot.launch is then stopped.
3. **Pull** (one peg_search run, libfranka). The fingers open to the seat width + 2 x `open_margin_mm`; joint moves
   over the peg and down to `hover_mm` above where its tip was, where the fingers close to the descent width. Then a
   Cartesian impedance at the (would-be) peg tip, stiff in every axis until the roof is on the peg
   (`descent_stiffness`; soft from the start, the empty hand drifted 6 mm in free air on the model's ~2 N error);
   from the grasp on `stiffness` [x y z roll pitch yaw] = 300 300 4000 N/m, 20 20 2 Nm/rad plus the stiff spring's
   mean wrench during the press held as a feed-forward: stiff in z, so the grasp height is set by the contact, soft
   in x, y and yaw, moderately soft in tilt. Down at `speed_mm_s` until the roof of finger A
   touches the peg top (the z force's change faster than its 0.25 s mean, `contact_force_n`, or the tip held back
   1 mm), as the fingers' README asks; pressing `press_n` the hand grasps at the seat width (35 N, accepted
   -3 / +1 mm): finger B pushes the peg into A's V, and since the hole holds the peg, the soft hand is pulled onto
   it. The press off and the targets moved to where the fingers put the hand, the pull goes along the seated peg axis
   (the hole's) at `pull_speed_mm_s`, the tilt held at the seated orientation (`pull_stiffness`, 200 Nm/rad: with
   the grasp's soft tilt the hand pitched up to 1.9 deg and jammed the 0.03 mm hole), wiggling the tilt
   (`wiggle_deg` at `wiggle_hz`) while the tip lags `wiggle_lag_mm`; lagging `pull_max_lag_mm` the target stops
   (the pull held) until the tip catches up, `stuck_time_s` without progress is stuck. Out when the tip is
   `clear_mm` above the board, then a joint move `lift_m` up. Contact is the tip held back 1 mm (the force change
   only past `contact_force_n`: a free-air blip was taken for the peg). Guards: the hand more than 6 mm from the
   vision hole or 12 deg about the peg until the pull, 6 mm off the pull's line while the tip is in the hole (not
   after: the held force lets go as the peg comes free), no peg top down to `max_depth_mm` below the board, the
   peg lost after the pull (gripper width).
4. **Failure in the hole** (no peg felt, grasp, stuck): the hand opens and goes back up over the hover, and
   up to `retries` (3) more attempts follow in place, aimed where the fingers seated on the peg if they got that
   far; after the last the hand lifts, the peg stays. After a successful pull the hand lifts `lift_m` (45 mm: the
   tip about a peg length over the board).

Nothing extract measures goes into the found holes, the board correction or the board heights.
Results: `sessions/<name>/extract/<date>_<time>/` (plan.yaml, extract.csv 1 kHz log with phases 97-98 the joint
moves, 1 descend, 2 press, 3 grasp, 4 settle, 5 pull, 99 lift; extract.json; outcome.yaml).

**Travel heights** (`insert.peg_travel_mm` 55, `insert.hand_travel_mm` 40; the peg tip, or where it would be,
above the vision board at the hole): extract ends and insert's probe approach starts at the first (holding a peg),
insert ends after the release and extract's approach starts at the second (empty hand), so the two hand over at
the same height. Both clear the pegs standing in holes (~28 mm out, ~46 if not fully in; the empty hand's
fingertips are ~30 mm above the tip). Without them: `lift_m` and the probe's 20 mm clearance.

### `campaign`

Full-board rounds, unattended (`board_pipeline/campaign.py`, the config's `campaign` section).

```bash
./pipeline/pipeline campaign start            # pegs in their 0.2 mm holes, hand empty
./pipeline/pipeline campaign run --dry-run    # the inserts left
./pipeline/pipeline campaign run              # resumable; --max-inserts N to stop early
./pipeline/pipeline campaign status           # where the pegs are, success per hole
```

Each round goes row-wise through c3mm -> c1mm -> c0p2mm across all shapes: extract from where the peg is, insert
into the next hole (robot.launch stays stopped for the campaign). A peg that does not go in is parked in its c3mm hole and taken from there for the
next clearance. A failed extract stops the campaign (`on_extract_failure: stop`; with skip_shape a peg left in its hole
blocks only its shape). Anything else that leaves a peg somewhere unknown, or a run stopped mid-action (Ctrl-C),
stops the campaign: look, then `campaign peg SHAPE --at HOLE|hand [--from-run RUN]` and run again.
`sessions/<name>/campaign/`: state.yaml, results.csv (one row per insert, park, extract, skip).

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
| `board_pipeline/insert.py` | `pipeline insert`: probe approach, tilt-search attempts with a belief, release, lift |
| `config/tilt_leads.yaml`, `tools/extract_tilt_leads.py` | tilt-search lead and search size per hole, from rl_manip's simulation |
| `board_pipeline/holefit.py`, `board_pipeline/beliefgrid.py` | found holes, the wall-touch centre fit and the board correction; the hole's probability grid and the attempts' line plans |
| `tests/test_holefit.py`, `tests/test_beliefgrid.py` | their offline tests (centre fits for every hole, the correction, the grid's evidence, a simulated search, the 2026-11-21 replay) |
| `tests/test_insert.py` | offline tests of insert's lines and belief (`python3 -m pytest tests`; `PEG_SEARCH_BIN=` also dry-runs peg_search) |
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
