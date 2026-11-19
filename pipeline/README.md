# Board pipeline

Finds the NIST board in front of the Franka Panda with the wrist camera, then
puts the tip of a held peg into any of its holes. Details:
[DETAILS.md](DETAILS.md).

## Before you start

After every reboot of the PC or the robot:

1. The PC runs the **realtime kernel**: `uname -r` ends in `-realtime`.
2. **Network fix** (lost on reboot):
   ```bash
   echo 1 | sudo tee /sys/class/net/eno1/threaded
   sudo chrt -f -p 95 $(pgrep 'napi/eno1')
   ```
3. **Desk**: joints unlocked, FCI active, end effector = Franka Hand.
4. **Container and launches** (each launch in its own terminal):
   ```bash
   docker compose -f docker/compose.yaml up -d
   docker exec -it franka_ros bash
   roslaunch panda_calibration robot.launch robot_ip:=172.16.0.2
   roslaunch panda_calibration azure_4k.launch
   ```
5. `NE_T_EE` is the identity and the camera load is entered:
   [DETAILS.md → Robot checks](DETAILS.md#robot-checks).

After changing code in `src/`, rebuild: `docker exec -it franka_ros bash -ic "cd /franka_ws && colcon build"`.

## Typical run

```bash
./pipeline/pipeline new --setup eye_in_hand   # new session
./pipeline/pipeline calibrate                 # ~15 min, the arm moves
./pipeline/pipeline grasp round               # seat the peg in the V first
./pipeline/pipeline check --hole round_c3mm   # dry run, no motion
./pipeline/pipeline probe --hole round_c3mm   # peg tip onto the hole
./pipeline/pipeline release
```

Keep the e-stop at hand whenever the arm moves.

## Commands

| command | what it does |
|---|---|
| `new --setup eye_in_hand` | start a session (a folder in `sessions/`) |
| `calibrate` | where the camera sits on the wrist; also measures the board. Camera parameters and joint offsets come from the config (`--fit-intrinsics`, `--fit-joint-offsets` estimate them instead) |
| `locate` | measure the board again after it moved |
| `holes` | list the hole names |
| `grasp SHAPE` / `release` | close the hand on a peg / open it |
| `check --hole H` | show the target, no motion |
| `probe --hole H` | move the peg tip onto the hole (Enter stops, Enter retracts) |
| `tip --hole H` | where the peg tip is now |
| `accuracy` | probe 4 holes, record target vs measured, hold for photos |
| `repeatability` | how well the arm returns to taught points |
| `diag` | tests that find where a constant probe error comes from |
| `status`, `sessions`, `use NAME` | what is done, list sessions, switch session |

Every command has `--help`.

## Results

Everything goes to `sessions/<name>/`: `calibration.json`, `board_pose.json`
(the board in the robot base frame), `targets/`, `accuracy/`, `diagnose/`.

| situation | command |
|---|---|
| a step failed | run the same command again |
| redo a step and what follows | `--from STEP` |
| start a stage over | `--fresh` |
| the board moved | `locate --fresh` |

## When something goes wrong

| message | fix |
|---|---|
| `Running kernel does not have realtime capabilities` | reboot into the realtime kernel |
| `Connection to FCI refused` | activate FCI in Desk |
| `reflex mode after communication_constraints_violation` | network fix (step 2), then recover |
| `reflex mode after ...` (anything else) | free the arm, then recover |
| `O_T_EE ... differs ... by 42.34 mm` | reset `NE_T_EE` (DETAILS.md) |
| `grasp failed` | `release` first, seat the peg in the V |
| `container 'franka_ros' is not running` | `docker compose -f docker/compose.yaml up -d` |
| `no controller_manager answered` | start `robot.launch` |
| `seed fit failed` | move the arm until the camera sees the board, then `locate --here` |

Recover from a reflex (in the container):

```bash
rostopic pub -1 /franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"
```
