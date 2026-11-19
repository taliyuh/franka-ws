#!/usr/bin/env bash
# Real-time tuning of this laptop for the Panda FCI link (franka_control's 1 kHz loop).
#
# Fixes "motion aborted by reflex: communication_constraints_violation".
# Causes measured on 2026-11-19:
#   * CPU governor "powersave", and the deepest idle state (C3) needs ~1 ms to
#     wake up -- a whole control cycle;
#   * franka_control's 1 kHz thread ran on an E-core (CPUs 12-15 are E-cores);
#   * the Realtek NIC (r8169) gets no IRQ thread on the PREEMPT_RT kernel, so
#     robot packets were processed by ksoftirqd at normal priority, competing
#     with the Azure driver, GNOME, Firefox...
#
# Settings last until reboot. Run on the HOST, not in the container:
#   sudo docker/robot_rt_tune.sh        # after boot, before robot.launch
#   sudo docker/robot_rt_tune.sh        # again after each start of robot.launch
#   docker/robot_rt_tune.sh --status    # show the current state (no sudo)
#
# Defaults can be overridden, e.g.  sudo RT_CPU=9 docker/robot_rt_tune.sh

set -u

IFACE=${IFACE:-eno1}        # NIC cabled to the Control box (172.16.0.1)
NET_CPU=${NET_CPU:-10}      # P-core for the NIC interrupt and packet processing
RT_CPU=${RT_CPU:-11}        # P-core for franka_control's 1 kHz thread (hyper-thread pair of 10)
NAPI_PRIO=${NAPI_PRIO:-95}  # SCHED_FIFO priority of the packet thread (franka_control: 99, IRQ threads: 50)
MAX_IDLE_LATENCY_US=10      # idle states slower to wake than this get disabled (POLL and C1 stay)

franka_pid() { pgrep -f 'lib/franka_control/franka_control_node' | head -1; }
franka_rt_tids() { ps -L -o tid=,cls= -p "$1" | awk '$2 == "FF" { print $1 }'; }
napi_pids() { pgrep "^napi/${IFACE}-"; }
nic_irqs() { awk -v d="$IFACE" '$NF == d { sub(":", "", $1); print $1 }' /proc/interrupts; }
affinity() { taskset -pc "$1" | sed 's/.*: //'; }
policy() { ps -o cls=,rtprio= -p "$1" | xargs; }

show_status() {
  echo "CPU governor:      $(sort -u /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | xargs)"
  local off=0 on=0 s
  for s in /sys/devices/system/cpu/cpu*/cpuidle/state*; do
    [ "$(cat "$s/latency")" -gt "$MAX_IDLE_LATENCY_US" ] || continue
    if [ "$(cat "$s/disable")" = 1 ]; then off=$((off + 1)); else on=$((on + 1)); fi
  done
  echo "deep idle states:  $off disabled, $on enabled"
  echo "irqbalance:        $(systemctl is-active irqbalance 2>/dev/null)"
  local irq p t
  for irq in $(nic_irqs); do
    echo "$IFACE IRQ $irq:      CPU $(cat "/proc/irq/$irq/smp_affinity_list")"
  done
  echo "threaded NAPI:     $(cat "/sys/class/net/$IFACE/threaded")"
  for p in $(napi_pids); do
    echo "  $(cat "/proc/$p/comm"): $(policy "$p"), CPU $(affinity "$p")"
  done
  p=$(franka_pid)
  if [ -z "$p" ]; then
    echo "franka_control:    not running"
  fi
  for t in ${p:+$(franka_rt_tids "$p")}; do
    echo "franka_control 1 kHz thread $t: $(policy "$t"), CPU $(affinity "$t")"
  done
}

if [ "${1:-}" = "--status" ]; then show_status; exit 0; fi
[ "$(id -u)" = 0 ] || { echo "Run with sudo (or use --status)."; exit 1; }
[ -d "/sys/class/net/$IFACE" ] || { echo "No network interface $IFACE."; exit 1; }

# 1. Full CPU speed. GNOME's power profile first, so it does not lower it again.
command -v powerprofilesctl >/dev/null && powerprofilesctl set performance 2>/dev/null
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance > "$g"
done

# 2. No deep sleep: a core in C2/C3 needs 0.1-1 ms to wake up for the next packet.
for s in /sys/devices/system/cpu/cpu*/cpuidle/state*; do
  if [ "$(cat "$s/latency")" -gt "$MAX_IDLE_LATENCY_US" ]; then echo 1 > "$s/disable"; fi
done

# 3. irqbalance would keep moving the NIC interrupt away from NET_CPU.
systemctl stop irqbalance 2>/dev/null

# 4. NIC interrupt on a fixed P-core.
for irq in $(nic_irqs); do
  echo "$NET_CPU" > "/proc/irq/$irq/smp_affinity_list"
done

# 5. Received packets get processed in a dedicated kernel thread (napi/eno1-*)
#    with real-time priority, instead of ksoftirqd at normal priority.
echo 1 > "/sys/class/net/$IFACE/threaded"
for p in $(napi_pids); do
  taskset -pc "$NET_CPU" "$p" >/dev/null
  chrt -f -p "$NAPI_PRIO" "$p"
done

# 6. franka_control's 1 kHz thread (its SCHED_FIFO thread) onto a P-core.
p=$(franka_pid)
if [ -n "$p" ]; then
  for t in $(franka_rt_tids "$p"); do taskset -pc "$RT_CPU" "$t" >/dev/null; done
else
  echo "franka_control is not running: after starting robot.launch run this script again."
  echo
fi

show_status
