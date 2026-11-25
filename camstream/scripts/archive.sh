#!/usr/bin/env bash
# Copy finished recordings to another drive (e.g. the external disk) before
# MediaMTX deletes them after RECORD_KEEP. Safe to re-run: rsync skips
# files already there. The segment still being written is left out.
#
#   scripts/archive.sh /media/$USER/EXT/camstream
#   scripts/archive.sh /media/$USER/EXT/camstream --delete-source   # free the local disk too
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:?destination directory}
set -a; . ./.env; set +a
REC=${RECORDINGS_DIR:-./recordings}/cam

mkdir -p "$DEST/cam"

# Newest file is the one MediaMTX is writing.
mapfile -t done < <(ls "$REC"/*.mp4 2>/dev/null | sort | head -n -1)
[ ${#done[@]} -gt 0 ] || { echo "nothing finished to archive"; exit 0; }

rsync -a --info=progress2 "${done[@]}" "$DEST/cam/"
echo "archived ${#done[@]} segment(s) to $DEST/cam"

if [ "${2:-}" = "--delete-source" ]; then
  for f in "${done[@]}"; do
    cmp -s "$f" "$DEST/cam/$(basename "$f")" && rm -f "$f"
  done
  echo "removed local copies that match the archive"
fi
