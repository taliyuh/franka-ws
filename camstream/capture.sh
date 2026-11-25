#!/bin/sh
# Runs inside the mediamtx container (paths.cam.runOnInit): grabs the camera,
# stamps UTC time on each frame, encodes H.264 and publishes to the local
# RTSP server. MediaMTX then serves it to browsers and records it.
#
# CAM_SOURCE=test  -> moving test pattern (dry run, no camera needed)
# CAM_SOURCE=v4l2  -> CAM_DEVICE, e.g. the RealSense colour node (see scripts/list-cams.sh)
set -eu

SIZE="${CAM_SIZE:-1280x720}"
FPS="${CAM_FPS:-15}"
BITRATE="${CAM_BITRATE:-2M}"
GOP=$((FPS * 2))   # keyframe every 2 s: browser join time, seek and time-lapse granularity

case "${CAM_SOURCE:-test}" in
  test)
    INPUT="-re -f lavfi -i testsrc2=size=${SIZE}:rate=${FPS}"
    ;;
  v4l2)
    DEV="${CAM_DEVICE:?set CAM_DEVICE in .env}"
    # RealSense D4xx colour node offers YUYV at 1280x720 @ 6/15/30 fps.
    INPUT="-f v4l2 -input_format ${CAM_INPUT_FORMAT:-yuyv422} -video_size ${SIZE} -framerate ${FPS} -i ${DEV}"
    ;;
  *)
    echo "unknown CAM_SOURCE=${CAM_SOURCE}" >&2; exit 1
    ;;
esac

LABEL="${CAM_LABEL:-827}"
FONT="${CAM_FONT:-/fonts/DejaVuSansMono.ttf}"
# UTC, to line up with the ROS logs in the franka_ros container (also UTC).
DRAW="drawtext=fontfile=${FONT}:text='${LABEL}  %{gmtime\:%Y-%m-%d %H\\\\\:%M\\\\\:%S} UTC':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6"

# shellcheck disable=SC2086
exec ffmpeg -hide_banner -loglevel warning -nostats \
  $INPUT \
  -vf "${DRAW},format=yuv420p" \
  -c:v libx264 -preset veryfast -tune zerolatency -profile:v main \
  -b:v "$BITRATE" -maxrate "$BITRATE" -bufsize "$BITRATE" \
  -g "$GOP" -keyint_min "$GOP" -sc_threshold 0 \
  -an \
  -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:${RTSP_PORT}/${MTX_PATH}"
