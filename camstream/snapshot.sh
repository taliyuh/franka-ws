#!/bin/sh
# Runs inside the mediamtx container (paths.cam.runOnReady): keeps
# /snapshots/latest.jpg refreshed every SNAPSHOT_EVERY seconds, so a log
# watcher (or a person with scp) can look at the cell without decoding video.
set -eu
EVERY="${SNAPSHOT_EVERY:-5}"
exec ffmpeg -hide_banner -loglevel error -nostats \
  -rtsp_transport tcp -i "rtsp://127.0.0.1:${RTSP_PORT}/${MTX_PATH}" \
  -vf "fps=1/${EVERY}" -q:v 4 -update 1 -atomic_writing 1 -y /snapshots/latest.jpg
