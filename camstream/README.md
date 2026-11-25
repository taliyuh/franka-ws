# camstream — live view and recording of the robot cell

Runs on 827 (prorocpc2). A single MediaMTX container reads the camera, stamps
the UTC time on each frame, and does three things with the stream:

| what | where (Tailscale only) |
|---|---|
| live view, under 1 s delay (WebRTC) | `http://100.117.27.103:8889/cam` |
| live view, a few s delay (HLS; via Funnel with a password) | `http://100.117.27.103:8888/cam` |
| list recordings / download a time range | `http://100.117.27.103:9996/list?path=cam` |
| | `http://100.117.27.103:9996/get?path=cam&start=2026-11-25T02:10:00Z&duration=120` |
| recordings, 1 h fMP4 files, deleted after `RECORD_KEEP` | `camstream/recordings/cam/` |
| latest still image, every `SNAPSHOT_EVERY` s | `camstream/snapshots/latest.jpg` |

Nothing listens on the university address (enp3s0); RTSP and the control
API are loopback only. 2 Mbit/s is about 0.9 GB/h, so a 12 h run is ~11 GB.

The overlay and the file names are UTC, the same clock as the ROS logs in
the `franka_ros` container, so a failure time from the logs can be used
directly in a `/get?start=...` link. Use 827's clock for those times: as of
2026-11-24 it was not NTP-synced and ran ~6 min behind real time
(`timedatectl`), so a time read off another machine will miss.

## Setup (once)

```bash
sudo usermod -aG docker $USER      # then log out and back in
cd ~/camstream
cp .env.example .env               # defaults to the test pattern
docker compose up -d
docker compose logs -f
```

The container restarts on its own after a reboot or a crash; ffmpeg is
restarted by MediaMTX within ~5 s if it dies or the camera drops out (a new
1 h segment starts then).

## Switching to the RealSense

1. Plug it into a **USB 3** port, ideally not on the same controller as the
   robot's USB-Ethernet adapter (`lsusb -t` shows which ports share a bus).
2. `scripts/list-cams.sh` and pick the colour node (the RealSense also
   exposes depth, IR and metadata nodes). Use its `/dev/v4l/by-id/...` path.
3. In `.env`: `CAM_SOURCE=v4l2` and `CAM_DEVICE=<that path>`, then
   `docker compose up -d`.

No librealsense is needed: the colour stream is read as a plain UVC webcam.

## After a run

```bash
scripts/timelapse.sh 2026-11-24T20:00 2026-11-25T08:00 night.mp4    # 120x: 12 h -> 6 min
scripts/archive.sh /media/$USER/EXT/camstream                        # copy to the external drive
scripts/archive.sh /media/$USER/EXT/camstream --delete-source        # ... and free the local disk
```

Times are UTC. A fourth argument to `timelapse.sh` sets seconds of real time
per output frame (default 4 → 120x; use multiples of 2, the keyframe interval).

## Sharing with colleagues

- Inside the tailnet: share the machine from the Tailscale admin console
  (they see only 827), then send them the 8889 link.
- Public link: `sudo tailscale funnel --bg http://100.117.27.103:8888`,
  then share `https://<827's MagicDNS name>/cam/` (the bare root gives 404).
  Funnel viewers must log in (user `viewer`, password set with
  `scripts/set-password.sh`, stored only as a hash in `.env`); tailnet
  devices get in without it. The `gate` service (Caddy, `Caddyfile`) does
  this, so unauthenticated internet requests never reach MediaMTX.
  `sudo tailscale funnel reset` turns the public link off.

## Tuning (`.env`)

`CAM_SIZE`, `CAM_FPS`, `CAM_BITRATE`, `RECORD_KEEP`, `RECORDINGS_DIR`
(e.g. record straight to the external drive), `CAM_CPUSET` (keep the
encoder off the cores the robot control loop is pinned to).
