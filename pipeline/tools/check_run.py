#!/usr/bin/env python3
"""Quick post-run inspector for `pipeline insert` runs.

Usage:
    ./pipeline/tools/check_run.py                  # inspects the newest insert run in the active session
    ./pipeline/tools/check_run.py <run_dir>        # inspects a specific insert/<timestamp> directory
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
PHASES = {
    10: "move", 11: "tilt", 12: "descend", 13: "press", 14: "settle", 15: "preload",
    16: "slide", 17: "upright", 18: "hold", 19: "release", 20: "lift", 21: "done",
    22: "tare", 23: "spiral", 24: "retry", 25: "funnel", 26: "plunge",
    90: "pos_up", 91: "pos_turn", 92: "pos_down", 94: "centre_back", 95: "centre_touch",
    96: "lift_pos", 97: "peck_move", 98: "peck_down",
}


def latest_run() -> Path:
    link = WORKSPACE / "sessions" / "current"
    if link.exists():
        ins = link.resolve() / "insert"
    else:
        runs = sorted((WORKSPACE / "sessions").glob("*/insert/*_*"))
        runs = [p for p in runs if p.is_dir() and (p / "plan.yaml").exists()]
        if not runs:
            raise SystemExit(f"No insert runs found under {WORKSPACE / 'sessions'}")
        return runs[-1]
    runs = sorted(p for p in ins.iterdir() if p.is_dir() and (p / "plan.yaml").exists())
    if not runs:
        raise SystemExit(f"No insert runs found in {ins}")
    return runs[-1]


def load_csv(path: Path) -> tuple[dict[str, int], np.ndarray]:
    with open(path) as f:
        hdr = f.readline().strip().split(",")
        ncols = len(hdr)
        rows = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != ncols:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    col = {k: i for i, k in enumerate(hdr)}
    return col, np.asarray(rows, float) if rows else np.empty((0, ncols))


def inspect_csv(path: Path) -> None:
    col, d = load_csv(path)
    if len(d) == 0:
        print(f"  {path.name:18s}: EMPTY")
        return
    t = d[:, col["time"]]
    ph = d[:, col["phase"]].astype(int)
    pz = d[:, col["p_z"]] * 1e3
    pdz = d[:, col["p_d_z"]] * 1e3
    fx, fy, fz = d[:, col["f_ext_fx"]], d[:, col["f_ext_fy"]], d[:, col["f_ext_fz"]]
    tx, ty, tz = d[:, col["f_ext_tx"]], d[:, col["f_ext_ty"]], d[:, col["f_ext_tz"]]
    fn = np.sqrt(fx**2 + fy**2 + fz**2)
    tn = np.sqrt(tx**2 + ty**2 + tz**2)
    psi = np.degrees(d[:, col["psi"]])
    valid_psi = psi[np.isfinite(psi)]
    psi_str = f"psi={valid_psi[0]:+5.1f}->{valid_psi[-1]:+5.1f}deg" if len(valid_psi) else "psi=?"
    q7 = math.degrees(d[0, col["q6"]])
    print(f"  {path.name:18s}: {t[-1]-t[0]:5.1f}s  {psi_str}  q7={q7:+6.1f}deg  "
          f"z={pz[0]:6.2f}->{pz[-1]:6.2f}mm (min {pz.min():6.2f})  "
          f"start |F|={fn[0]:4.1f}N |T|={tn[0]:4.2f}Nm  max|F|={fn.max():5.1f}N")

    # Show key phase breakdown for peck / attempt / centre
    if path.name.startswith(("peck", "attempt_", "centre")):
        idx = np.where(np.r_[True, ph[1:] != ph[:-1]])[0]
        idx = np.r_[idx, len(ph)]
        for i in range(len(idx) - 1):
            a, b = idx[i], idx[i + 1]
            pcode = int(ph[a])
            if pcode in (12, 13, 14, 16, 17, 18, 22, 25, 26, 95, 98):
                pname = PHASES.get(pcode, str(pcode))
                ez = pz[a:b] - pdz[a:b]
                print(f"    t={t[a]:5.2f}..{t[b-1]:5.2f}s {pname:12s} "
                      f"z={pz[a]:6.2f}->{pz[b-1]:6.2f} (ez {ez.min():+.2f}..{ez.max():+.2f}mm) "
                      f"fz={fz[a]:+5.1f}->{fz[b-1]:+5.1f}N |T|={tn[a]:4.2f}Nm")


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_run()
    plan = yaml.safe_load((run_dir / "plan.yaml").read_text())
    summary = yaml.safe_load((run_dir / "summary.yaml").read_text()) if (run_dir / "summary.yaml").exists() else {}
    hb = plan["hole_base_m"]
    dq = plan.get("joint_offsets_rad")
    print(f"=== {run_dir.name} | {plan['hole']} | method={plan.get('method')} | inserted={summary.get('inserted')} ===")
    print(f"  target ({'calibrated q+dq' if dq else 'nominal'}): "
          f"({hb[0]*1e3:.2f}, {hb[1]*1e3:.2f}, {hb[2]*1e3:.2f}) mm, slide={plan.get('slide_dir_base_deg', 0):.1f} deg")
    if summary.get("stopped"):
        print(f"  STOPPED: {summary['stopped']}")

    peck_j = run_dir / "peck.json"
    if peck_j.exists():
        pj = json.loads(peck_j.read_text())
        print(f"\n[peck.json] ok={pj.get('ok')} inserted={pj.get('inserted')} "
              f"board_measured={None if pj.get('board_measured') is None else round(pj['board_measured']*1e3, 3)}mm "
              f"first={pj.get('first')}")
        if pj.get("error"):
            print(f"  ERROR: {pj['error']}")
        if pj.get("plane"):
            pl = pj["plane"]
            sl = pl["slope"]
            tilt_deg = math.degrees(math.atan(math.hypot(sl[0], sl[1])))
            print(f"  3-pt plane: z(hole)={pl['point'][2]*1e3:.3f}mm (vs vision {hb[2]*1e3:.2f}mm), "
                  f"slope=({sl[0]*1e3:+.2f}, {sl[1]*1e3:+.2f}) um/mm ({tilt_deg:.3f} deg)")
        if pj.get("touch_points"):
            pts = ", ".join(f"({p[0]*1e3:.1f},{p[1]*1e3:.1f},{p[2]*1e3:.3f})" for p in pj["touch_points"])
            print(f"  touch_points (mm): {pts}")

    for af in sorted(run_dir.glob("attempt_*.json")):
        ad = json.loads(af.read_text())
        print(f"\n[{af.name}] ok={ad.get('ok')} stage={ad.get('stage')} inserted={ad.get('inserted')} "
              f"outcomes={ad.get('outcomes')} bias={ad.get('bias_n')}N")
        if ad.get("error"):
            print(f"  ERROR: {ad['error']}")
        if ad.get("dip_lead") or ad.get("hold_xy"):
            print(f"  dip_lead={ad.get('dip_lead')}  hold_xy={ad.get('hold_xy')}")

    centre_j = run_dir / "centre.json"
    if centre_j.exists():
        cj = json.loads(centre_j.read_text())
        print(f"\n[centre.json] ok={cj.get('ok')} contacts={cj.get('contacts')} error={cj.get('error')}")

    if summary.get("found"):
        f = summary["found"]
        off = (np.asarray(f["found_xy"]) - np.asarray(f["vision_xy"])) * 1e3
        print(f"\n[found] {f['source']}  err_from_vision=({off[0]:+.2f}, {off[1]:+.2f}) mm  "
              f"sigma={f['sigma_m']*1e3:.2f} mm")

    print("\n[telemetry]")
    for cf in sorted(run_dir.glob("*.csv")):
        inspect_csv(cf)


if __name__ == "__main__":
    main()
