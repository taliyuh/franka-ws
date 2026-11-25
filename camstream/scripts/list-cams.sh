#!/usr/bin/env bash
# List V4L2 video nodes with their driver name and stable by-id path.
# A RealSense D4xx shows several nodes (depth, IR, colour, metadata); the
# colour one is usually named "...RGB Camera" or "...Color" — put its
# /dev/v4l/by-id path in .env as CAM_DEVICE.
shopt -s nullglob
nodes=(/sys/class/video4linux/video*)
if [ ${#nodes[@]} -eq 0 ]; then
  echo "no /dev/video* devices (camera not plugged in, or not on a USB 3 port?)"
  exit 1
fi
for n in "${nodes[@]}"; do
  dev=/dev/$(basename "$n")
  name=$(cat "$n/name")
  byid=$(for l in /dev/v4l/by-id/*; do [ "$(readlink -f "$l")" = "$dev" ] && echo "$l"; done)
  printf '%-12s %-40s %s\n' "$dev" "$name" "${byid:--}"
done
if command -v v4l2-ctl >/dev/null; then
  echo; echo "formats per node: v4l2-ctl -d /dev/videoN --list-formats-ext"
fi
