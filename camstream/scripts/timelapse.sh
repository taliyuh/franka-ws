#!/usr/bin/env bash
# Make a time-lapse from the recordings between two times.
#
#   scripts/timelapse.sh START END [OUT.mp4] [STEP_S]
#   scripts/timelapse.sh 2026-11-24T20:00 2026-11-25T08:00 night.mp4
#
# START/END are UTC (same clock as the overlay and the ROS logs), anything
# `date -d` understands. One frame is taken every STEP_S seconds of real
# time (default 4, a multiple of the 2 s keyframe interval) and played at
# 30 fps, so the default speed-up is 120x: 12 h -> 6 min.
# Only keyframes are decoded, so a 12 h night takes about a minute.
set -euo pipefail
cd "$(dirname "$0")/.."

START=${1:?start time (UTC)}; END=${2:?end time (UTC)}
OUT=${3:-timelapse_$(date -u -d "$START" +%Y%m%d_%H%M).mp4}
STEP=${4:-4}

set -a; . ./.env; set +a
REC=${RECORDINGS_DIR:-./recordings}/cam
IMAGE=$(sed -n 's/^ *image: *//p' docker-compose.yml)

s=$(date -u -d "$START" +%s); e=$(date -u -d "$END" +%s)
[ "$e" -gt "$s" ] || { echo "END must be after START" >&2; exit 1; }

# Segments are named by their UTC start time and are at most 1 h long, so
# take every segment that starts before END and less than 1 h before START.
list=$(mktemp -p "$REC" .tl_XXXX.txt)
trap 'rm -f "$list"' EXIT
n=0; first=
for f in $(ls "$REC"/*.mp4 2>/dev/null | sort); do
  b=$(basename "$f" .mp4)                       # 2026-11-24_20-00-00-123456
  t=$(date -u -d "${b:0:10} ${b:11:2}:${b:14:2}:${b:17:2}" +%s)
  if [ "$t" -lt "$e" ] && [ "$t" -ge $((s - 3600)) ]; then
    echo "file '/rec/$(basename "$f")'" >> "$list"; n=$((n + 1))
    [ -n "$first" ] || first=$t
  fi
done
[ "$n" -gt 0 ] || { echo "no recordings in $REC for that range" >&2; exit 1; }
STEP=$(( STEP < 2 ? 2 : STEP / 2 * 2 ))
echo "$n segment(s) -> $OUT  (1 frame / ${STEP}s, $((STEP * 30))x)"

# Trim to [START, END]: seek from the first segment's start (its name).
# Gaps between segments (camera or container down) are skipped, so the
# end can run slightly past END in that case.
skip=$(( s > first ? s - first : 0 ))
# Only keyframes (one every 2 s, see capture.sh) are decoded; keep every K-th.
K=$(( STEP / 2 > 0 ? STEP / 2 : 1 ))
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$(realpath "$REC")":/rec:ro -v "$(realpath "$(dirname "$OUT")")":/out \
  --entrypoint ffmpeg "$IMAGE" -hide_banner -loglevel warning -stats \
  -skip_frame nokey -f concat -safe 0 -ss "$skip" -t $((e - s)) -i "/rec/$(basename "$list")" \
  -vf "select=not(mod(n\\,${K})),setpts=N/30/TB" -r 30 \
  -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -movflags +faststart \
  -y "/out/$(basename "$OUT")"
echo "wrote $OUT"
