#!/usr/bin/env bash
# Set the login for the public (Funnel) link. Stores only a bcrypt hash in
# .env, then restarts the gate.
#   scripts/set-password.sh [USER]      # prompts for the password
set -euo pipefail
cd "$(dirname "$0")/.."
USER_NAME=${1:-viewer}
read -rsp "password for '$USER_NAME': " PW; echo
[ ${#PW} -ge 8 ] || { echo "use at least 8 characters" >&2; exit 1; }
IMAGE=$(sed -n '/^  gate:/,/^  [a-z]/s/^ *image: *//p' docker-compose.yml)
HASH=$(docker run --rm "$IMAGE" caddy hash-password --plaintext "$PW")
# Single quotes: compose must not expand the $ signs in the bcrypt hash.
sed -i '/^VIEWER_USER=/d; /^VIEWER_PASS_HASH=/d' .env
printf "VIEWER_USER=%s\nVIEWER_PASS_HASH='%s'\n" "$USER_NAME" "$HASH" >> .env
docker compose up -d
echo "set; public link now asks for $USER_NAME / <password>"
