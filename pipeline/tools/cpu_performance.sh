#!/bin/sh
# Real-time robot PC: every core's cpufreq governor to "performance" now and at every boot.
# 2026-11-25: the governor was "powersave" (intel_pstate) and the camera container used ~1 core; two free-air
# force-change trips mid-transit looked like control-loop jitter. Run once with sudo:
#   sudo pipeline/tools/cpu_performance.sh            (install + apply)
#   sudo pipeline/tools/cpu_performance.sh --undo     (back to Ubuntu's default: ondemand.service at boot)
set -eu
UNIT=/etc/systemd/system/cpu-performance.service

if [ "$(id -u)" -ne 0 ]; then
  echo "run with sudo" >&2
  exit 1
fi

if [ "${1:-}" = "--undo" ]; then
  systemctl disable --now cpu-performance.service 2>/dev/null || true
  rm -f "$UNIT"
  systemctl daemon-reload
  systemctl enable ondemand.service 2>/dev/null || true
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo powersave > "$g"; done
  echo "governor back to powersave; ondemand.service re-enabled"
  exit 0
fi

cat > "$UNIT" <<'EOF'
[Unit]
Description=cpufreq governor performance on every core (real-time robot control)
After=ondemand.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
# Ubuntu's ondemand.service sets the governor ~1 min after boot; it would undo ours.
systemctl disable ondemand.service 2>/dev/null || true
systemctl daemon-reload
systemctl enable --now cpu-performance.service
echo "governors now:"; sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq -c
