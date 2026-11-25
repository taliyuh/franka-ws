"""`pipeline campaign`: full-board insertions, unattended (2026-11-23: 20 rounds x 15 holes to validate insert).

Each peg starts in its tightest campaign hole (campaign.start_clearance, c0p2mm). A round goes row-wise through
campaign.clearances (3 -> 1 -> 0.2 mm) across all shapes in campaign.shapes: extract the peg from where it is,
insert it into the target hole, and so on; the round ends with every peg back in its 0.2 mm hole, where the next
round starts. With travel=libfranka, robot.launch stays stopped for the whole campaign.

A peg that does not go in (insert lifts it with the peg) is parked in its 3 mm hole (campaign.park_clearance, skipping
wall-touching `centre` and `found_holes.yaml` recording) and the next clearance takes it from there. A peg that cannot
be extracted (the hand let go, the arm clear) blocks its shape until `campaign peg SHAPE --at HOLE` when
campaign.on_extract_failure is skip_shape (default: stop). Anything that leaves a peg somewhere unknown stops the
campaign.

Within each round (episode), insert learns about the holes from inserting only (found holes, board correction) and
shares that knowledge across the 15 insertions of that round; at the end of every round (and on `campaign start`),
<session>/insert/found_holes.yaml is archived into <session>/campaign/ so the next episode starts without XY hole
corrections from previous episodes. A pre-campaign snapshot is kept at <session>/campaign/extract_found_holes.yaml so
`extract` can still use it while insert starts clean. The Z board height model (<session>/insert/board_heights.yaml) is
kept across rounds.

Everything goes to sessions/<name>/campaign/: state.yaml (where the pegs are, how far it got; rewritten after every
step), results.csv (one row per insert, park and extract), attempts.csv (one row per individual attempt).
"""

from __future__ import annotations

import contextlib
import csv
import datetime
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from board_pipeline import extract, holefit, insert
from board_pipeline.config import Config, PipelineError, load_yaml

EXTRACT_FOUND_FILE = "extract_found_holes.yaml"
RESULT_FIELDS = ["time", "round", "shape", "hole", "role", "ok", "method_succeeded",
                 "attempts", "retries", "duration_s", "run", "note"]
ATTEMPT_FIELDS = ["time", "round", "shape", "hole", "role", "run", "attempt_index",
                  "method", "ok", "inserted", "stage", "plunge", "outcomes", "error"]


def settings(session_cfg: Config) -> dict:
    data = session_cfg.data.get("campaign")
    return data if data else Config.template()["campaign"]


def _dir(session) -> Path:
    return session.path / "campaign"


def load_state(session) -> dict | None:
    path = _dir(session) / "state.yaml"
    return load_yaml(path) if path.exists() else None


def save_state(session, state: dict) -> None:
    path = _dir(session) / "state.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(state, sort_keys=False))
    tmp.replace(path)


def log_result(session, row: dict) -> None:
    path = _dir(session) / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def _attempt_method(entry: dict) -> str:
    return str(entry.get("method") or "tilt")


def _method_succeeded(summary: dict, inserted: bool) -> str:
    if not inserted:
        return ""
    for entry in (summary or {}).get("attempts") or []:
        if entry.get("inserted"):
            return _attempt_method(entry)
    return "tilt"


def _retries(summary: dict) -> int:
    return max(0, _attempts(summary) - 1)


def log_attempts(session, round_num: int, shape: str, hole: str, role: str, run_name: str, summary: dict) -> None:
    entries = (summary or {}).get("attempts") or []
    if not entries:
        return
    path = _dir(session) / "attempts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ATTEMPT_FIELDS)
        if new:
            writer.writeheader()
        for idx, entry in enumerate(entries):
            outcomes = entry.get("outcomes")
            if isinstance(outcomes, (list, tuple)):
                outcomes_str = ";".join(map(str, outcomes))
            else:
                outcomes_str = "" if outcomes is None else str(outcomes)
            writer.writerow({
                "time": now,
                "round": round_num,
                "shape": shape,
                "hole": hole,
                "role": role,
                "run": run_name,
                "attempt_index": entry.get("attempt", idx),
                "method": _attempt_method(entry),
                "ok": int(bool(entry.get("ok"))),
                "inserted": int(bool(entry.get("inserted"))),
                "stage": entry.get("stage") or "",
                "plunge": entry.get("plunge") or "",
                "outcomes": outcomes_str,
                "error": entry.get("error") or "",
            })


def _archive_path(dst_dir: Path, stem: str, suffix: str) -> Path:
    cand = dst_dir / f"{stem}{suffix}"
    n = 2
    while cand.exists():
        cand = dst_dir / f"{stem}_{n}{suffix}"
        n += 1
    return cand


def _archive_file(src: Path, dst_dir: Path, stem: str, suffix: str) -> Path | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = _archive_path(dst_dir, stem, suffix)
    src.replace(dst)
    return dst


def _reset_found_holes(session, label: str, save_for_extract: bool = False) -> Path | None:
    src = holefit.found_path(session.path)
    if not src.exists():
        return None
    if save_for_extract:
        _dir(session).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, _dir(session) / EXTRACT_FOUND_FILE)
    return _archive_file(src, _dir(session), f"found_holes_{label}", ".yaml")


def record_for(session, hole: str) -> dict | None:
    """The release record of the campaign's peg in this hole, if the campaign has one there."""
    state = load_state(session)
    if not state:
        return None
    for peg in state["pegs"].values():
        if peg["at"] == hole:
            return peg.get("record")
    return None


def note_extracted(session, hole: str) -> None:
    """`pipeline extract` by hand: the campaign's peg in that hole is in the hand now."""
    state = load_state(session)
    if not state:
        return
    for shape, peg in state["pegs"].items():
        if peg["at"] == hole:
            state["pegs"][shape] = {"at": "hand", "record": None}
            state["over"] = None
            save_state(session, state)
            print(f"campaign: the {shape} peg is in the hand now")


def new_state(holes: dict, c: dict, iterations: int, shapes: list[str]) -> dict:
    clearances = list(c["clearances"])
    for shape in shapes:
        for clearance in clearances + [c["start_clearance"], c["park_clearance"]]:
            if f"{shape}_{clearance}" not in holes:
                raise PipelineError(f"no hole {shape}_{clearance} in holes.yaml")
    return {"created": datetime.datetime.now().isoformat(timespec="seconds"), "rounds": iterations,
            "shapes": shapes, "clearances": clearances, "park_clearance": c["park_clearance"],
            "round": 1, "shape_index": 0, "step": 0, "found_reset_round": 1,
            "over": None, "pending": None, "blocked": {},
            "pegs": {shape: {"at": f"{shape}_{c['start_clearance']}", "record": None} for shape in shapes}}


def _insert_args(args, hole: str, park: bool = False, no_board_correction_holes=()) -> SimpleNamespace:
    return SimpleNamespace(hole=hole, attempts=None, height=None, tip_from=None, turn=0.0,
                           no_joint_correction=False, slide_dir=None, pattern=None, search=None, method=None,
                           touches=None,
                           no_board_correction=bool(getattr(args, "no_board_correction", False)
                                                    or hole in (no_board_correction_holes or ())),
                           no_centre=park, no_record=park, here=False, dry_run=False)


def _extract_args(hole: str) -> SimpleNamespace:
    return SimpleNamespace(hole=hole, no_joint_correction=False, dry_run=False)


def describe(state: dict) -> str:
    pegs = ", ".join(f"{s} {p['at']}" for s, p in state["pegs"].items())
    clearance = state["clearances"][state["step"]] if state["step"] < len(state["clearances"]) else "-"
    return (f"round {state['round']}/{state['rounds']}, clearance {state['step'] + 1}/{len(state['clearances'])} "
            f"({clearance}), shape {state['shape_index'] + 1}/{len(state['shapes'])}; pegs: {pegs}")


def _plan(state: dict) -> list[tuple[int, str, str]]:
    """The (round, shape, hole) inserts left, in row-wise order (each clearance across all shapes)."""
    out = []
    r, i, k = state["round"], state["shape_index"], state["step"]
    while r <= state["rounds"]:
        shape = state["shapes"][i]
        out.append((r, shape, f"{shape}_{state['clearances'][k]}"))
        i += 1
        if i == len(state["shapes"]):
            i, k = 0, k + 1
            if k == len(state["clearances"]):
                k, r = 0, r + 1
    return out


def _advance(state: dict) -> None:
    state["shape_index"] += 1
    if state["shape_index"] == len(state["shapes"]):
        state["shape_index"], state["step"] = 0, state["step"] + 1
        if state["step"] == len(state["clearances"]):
            state["step"], state["round"] = 0, state["round"] + 1


def _advance_and_save(session, state: dict) -> None:
    prev_round = state["round"]
    _advance(state)
    if state["round"] > prev_round:
        archived = _reset_found_holes(session, f"round_{prev_round}", save_for_extract=False)
        state["found_reset_round"] = state["round"]
        if archived is not None:
            print(f"campaign: archived round {prev_round} found_holes to {archived.name}; "
                  f"reset for round {state['round']}")
    save_state(session, state)


def _attempts(summary: dict) -> int:
    return len((summary or {}).get("attempts") or [])


def _insert(session, args, probe_target, state, shape, hole, role) -> dict:
    """One insert of the held peg. Updates the peg's place; returns insert's outcome."""
    cfg, s = session.cfg, insert.settings(session.cfg)
    if s.get("travel", "libfranka") != "libfranka":
        insert.start_launch(cfg, s)  # its probe approach is ROS (no-op when running)
    state["pending"] = {"action": role, "shape": shape, "hole": hole}
    save_state(session, state)
    started = time.monotonic()
    skip_corr = settings(cfg).get("no_board_correction_holes") or ()
    outcome = insert.run(session, _insert_args(args, hole, park=(role == "park"),
                                               no_board_correction_holes=skip_corr), probe_target,
                         restart_launch=False, raise_on_fail=False)
    summary = outcome["summary"]
    stopped = outcome["stopped"] or ""
    if outcome["inserted"] and not stopped:
        state["pegs"][shape] = {"at": hole, "record": summary.get("release")}
        state["over"] = hole if summary.get("release") else None
        if not summary.get("release"):
            print("warning: no release pose recorded (peg_search without lift --result?): the next extract aims at "
                  "the vision hole")
    elif not outcome["inserted"] and "lift" not in stopped and "reset" not in stopped:
        state["pegs"][shape] = {"at": "hand", "record": None}  # insert lifted it with the peg
        state["over"] = None
    else:
        state["pegs"][shape] = {"at": "unknown", "record": None}
        state["over"] = None
    state["pending"] = None
    save_state(session, state)
    run_name = outcome["out"].name
    log_attempts(session, state["round"], shape, hole, role, run_name, summary)
    log_result(session, {"time": datetime.datetime.now().isoformat(timespec="seconds"), "round": state["round"],
                         "shape": shape, "hole": hole, "role": role, "ok": int(bool(outcome["inserted"])),
                         "method_succeeded": _method_succeeded(summary, bool(outcome["inserted"])),
                         "attempts": _attempts(summary), "retries": _retries(summary),
                         "duration_s": f"{time.monotonic() - started:.0f}",
                         "run": run_name, "note": stopped})
    return outcome


def _extract(session, args, probe_target, state, shape) -> dict:
    cfg, s = session.cfg, insert.settings(session.cfg)
    peg = state["pegs"][shape]
    hole, record = peg["at"], peg.get("record")
    here = record is not None and state.get("over") == hole and not insert.launch_running(cfg, s)
    libfranka_travel = s.get("travel", "libfranka") == "libfranka"
    if not here and not libfranka_travel:
        insert.start_launch(cfg, s)  # its approach is ROS
    state["pending"] = {"action": "extract", "shape": shape, "hole": hole}
    save_state(session, state)
    started = time.monotonic()
    outcome = extract.run(session, _extract_args(hole), probe_target, record=record, here=here,
                          restart_launch=not libfranka_travel)
    state["over"] = None
    if outcome["ok"]:
        state["pegs"][shape] = {"at": "hand", "record": None}
    elif outcome["peg"] != "hole":
        state["pegs"][shape] = {"at": "unknown", "record": None}
    state["pending"] = None
    save_state(session, state)
    res = outcome.get("result") or {}
    note = outcome.get("error") or (f"contact {res['contact_above_board'] * 1e3:+.2f} mm, shift "
                                    f"{sum(v * v for v in res['seated_shift_xy']) ** 0.5 * 1e3:.2f} mm, turn "
                                    f"{res['seated_turn_deg']:+.2f} deg, width off {res['width_off'] * 1e3:+.2f} mm"
                                    if res.get("held") else "")
    log_result(session, {"time": datetime.datetime.now().isoformat(timespec="seconds"), "round": state["round"],
                         "shape": shape, "hole": hole, "role": "extract", "ok": int(bool(outcome["ok"])),
                         "duration_s": f"{time.monotonic() - started:.0f}",
                         "run": outcome["out"].name if outcome["out"] else "", "note": note})
    return outcome


def cmd_run(session, args, probe_target) -> None:
    state = load_state(session)
    if state is None:
        raise PipelineError("no campaign in this session: `pipeline campaign start` first")
    if state.get("pending"):
        p = state["pending"]
        raise PipelineError(f"the last run stopped during the {p['action']} of the {p['shape']} peg ({p['hole']}): "
                            f"where is it? Tell the campaign with `pipeline campaign peg {p['shape']} --at HOLE` "
                            f"(or --at hand), then run again")
    unknown = [shape for shape, peg in state["pegs"].items() if peg["at"] == "unknown"]
    if unknown:
        raise PipelineError(f"where the {', '.join(unknown)} peg(s) are is unknown: `pipeline campaign peg SHAPE --at "
                            "HOLE` after you have looked")
    c = settings(session.cfg)
    plan = _plan(state)
    if args.dry_run:
        print(describe(state))
        for r, shape, hole in plan[: args.max_inserts or 25]:
            print(f"  round {r}: {hole}")
        print(f"... {len(plan)} inserts left")
        return
    print(f"campaign: {describe(state)}; {len(plan)} inserts left")
    if state["round"] > state.get("found_reset_round", 0) and state["step"] == 0 and state["shape_index"] == 0:
        _reset_found_holes(
            session,
            f"before_round_{state['round']}",
            save_for_extract=(state["round"] == 1 and not (_dir(session) / EXTRACT_FOUND_FILE).exists()),
        )
        state["found_reset_round"] = state["round"]
        save_state(session, state)
    done = 0
    s = insert.settings(session.cfg)
    keep_stopped = (
        insert.launch_kept_stopped(session.cfg, s)
        if s.get("travel", "libfranka") == "libfranka"
        else contextlib.nullcontext()
    )
    try:
        with keep_stopped:
            while state["round"] <= state["rounds"]:
                if args.max_inserts and done >= args.max_inserts:
                    print(f"\ncampaign: stopping after {done} insert(s) (--max-inserts); `pipeline campaign run` goes on")
                    return
                shape = state["shapes"][state["shape_index"]]
                hole = f"{shape}_{state['clearances'][state['step']]}"
                holding = [sh for sh, peg in state["pegs"].items() if peg["at"] == "hand"]
                if shape in state["blocked"]:
                    print(f"\ncampaign: {hole} skipped: the {shape} peg is blocked ({state['blocked'][shape]})")
                    log_result(session, {"time": datetime.datetime.now().isoformat(timespec="seconds"),
                                         "round": state["round"], "shape": shape, "hole": hole, "role": "skipped",
                                         "ok": 0, "note": state["blocked"][shape]})
                    _advance_and_save(session, state)
                    continue
                if holding and holding != [shape]:
                    raise PipelineError(f"the hand holds the {holding[0]} peg, the next hole is {hole}: put it back "
                                        f"(`pipeline insert`), then `pipeline campaign peg {holding[0]} --at HOLE`")
                print(f"\n======== campaign round {state['round']}/{state['rounds']}: {hole} ========", flush=True)
                if state["pegs"][shape]["at"] != "hand":
                    out = _extract(session, args, probe_target, state, shape)
                    if not out["ok"]:
                        if out["peg"] != "hole" or c.get("on_extract_failure", "stop") == "stop":
                            raise PipelineError(f"extracting the {shape} peg failed ({out['error']}); results in {out['out']}")
                        state["blocked"][shape] = f"not extracted from {state['pegs'][shape]['at']}: {out['error']}"
                        save_state(session, state)
                        print(f"\ncampaign: the {shape} peg stays in {state['pegs'][shape]['at']}; its shape is skipped "
                              f"until `pipeline campaign peg {shape} --at HOLE`")
                        continue
                out = _insert(session, args, probe_target, state, shape, hole, "insert")
                done += 1
                at = state["pegs"][shape]["at"]
                if at == "unknown":
                    raise PipelineError(f"{hole}: {out['stopped']}; where the {shape} peg is is unknown")
                if at == "hand":
                    park = f"{shape}_{state['park_clearance']}"
                    print(f"\ncampaign: {hole} not inserted: parking the {shape} peg in {park}", flush=True)
                    out = _insert(session, args, probe_target, state, shape, park, "park")
                    if state["pegs"][shape]["at"] != park:
                        raise PipelineError(f"parking the {shape} peg in {park} failed ({out['stopped'] or 'not inserted'}); "
                                            f"it is {'in the hand' if state['pegs'][shape]['at'] == 'hand' else 'somewhere unknown'}")
                _advance_and_save(session, state)
        print(f"\ncampaign: all {state['rounds']} rounds done; `pipeline campaign status` sums them up")
    finally:
        insert.start_launch(session.cfg, insert.settings(session.cfg))


def cmd_start(session, args) -> None:
    if load_state(session) and not args.fresh:
        raise PipelineError("this session has a campaign already (`pipeline campaign status`); --fresh starts over "
                            "(existing results.csv / attempts.csv and found_holes.yaml are archived)")
    c = settings(session.cfg)
    holes = load_yaml(session.cfg.path(session.cfg["holes"]))["holes"]
    shapes = args.shapes or list(c["shapes"])
    state = new_state(holes, c, args.rounds or int(c["rounds"]), shapes)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if args.fresh:
        _archive_file(_dir(session) / "results.csv", _dir(session), f"results_{stamp}", ".csv")
        _archive_file(_dir(session) / "attempts.csv", _dir(session), f"attempts_{stamp}", ".csv")
    archived = _reset_found_holes(session, f"before_start_{stamp}", save_for_extract=True)
    save_state(session, state)
    arch_note = f" (archived {archived.name}, kept snapshot for extract)" if archived is not None else ""
    print(f"campaign: {state['rounds']} rounds, row-wise across clearances {' -> '.join(state['clearances'])} "
          f"and shapes {' -> '.join(shapes)}{arch_note}; the pegs must be in their {c['start_clearance']} holes "
          f"and the hand empty and open. `pipeline campaign run --dry-run` lists the inserts, "
          f"`pipeline campaign run` starts")


def cmd_peg(session, args) -> None:
    state = load_state(session)
    if state is None:
        raise PipelineError("no campaign in this session")
    if args.shape not in state["pegs"]:
        raise PipelineError(f"the campaign has no {args.shape} peg ({', '.join(state['pegs'])})")
    holes = load_yaml(session.cfg.path(session.cfg["holes"]))["holes"]
    if args.at not in ("hand", "unknown") and (args.at not in holes or holes[args.at]["shape"] != args.shape):
        raise PipelineError(f"{args.at} is not a {args.shape} hole (or hand / unknown)")
    record = None
    if args.from_run:
        record = extract.record_from_run(session.path / "insert" / args.from_run)
        if record.get("hole") != args.at:
            raise PipelineError(f"{args.from_run} inserted {record.get('hole')}, not {args.at}")
    state["pegs"][args.shape] = {"at": args.at, "record": record}
    state["blocked"].pop(args.shape, None)
    if state.get("pending") and state["pending"]["shape"] == args.shape:
        state["pending"] = None
    state["over"] = None
    if args.step:
        r, shape, clearance = args.step
        state["round"], state["shape_index"], state["step"] = int(r), state["shapes"].index(shape), \
            state["clearances"].index(clearance)
    save_state(session, state)
    print(f"campaign: the {args.shape} peg is at {args.at}"
          + (" (pose from the release of " + args.from_run + ")" if record else " (no release pose: vision aims)")
          + f"; {describe(state)}")


def _row_retries(row: dict) -> int:
    raw = row.get("retries")
    if raw not in (None, ""):
        return int(raw)
    att = row.get("attempts")
    return max(0, int(att) - 1) if att not in (None, "") else 0


def _row_method(row: dict) -> str:
    return (row.get("method_succeeded") or "").strip()


def _summarize_inserts(entries: list[dict]) -> dict:
    total = len(entries)
    ok_rows = [r for r in entries if int(r["ok"])]
    ok = len(ok_rows)
    plunge = sum(1 for r in ok_rows if _row_method(r) == "plunge")
    tilt = sum(1 for r in ok_rows if _row_method(r) == "tilt")
    att = [int(r["attempts"]) for r in ok_rows if r.get("attempts") not in (None, "")]
    ret = [_row_retries(r) for r in entries if r.get("retries") not in (None, "") or r.get("attempts") not in (None, "")]
    dur = [float(r["duration_s"]) for r in entries if r.get("duration_s") not in (None, "")]
    return {
        "total": total,
        "ok": ok,
        "pct": (100.0 * ok / total) if total else 0.0,
        "plunge": plunge,
        "tilt": tilt,
        "avg_att": (sum(att) / len(att)) if att else 0.0,
        "avg_retries": (sum(ret) / len(ret)) if ret else 0.0,
        "avg_dur": (sum(dur) / len(dur)) if dur else 0.0,
    }


def _print_insert_table(title: str, groups: list[tuple[str, list[dict]]]) -> None:
    if not groups:
        return
    print(f"\n{title:18s} {'ok':>13s} {'plunge':>6s} {'tilt':>5s} {'att':>5s} {'retries/ins':>11s} {'time s':>7s}")
    for label, entries in groups:
        s = _summarize_inserts(entries)
        ok_str = f"{s['ok']:3d}/{s['total']:<3d} ({s['pct']:3.0f}%)"
        print(f"{label:18s} {ok_str:>13s} {s['plunge']:6d} {s['tilt']:5d} "
              f"{s['avg_att']:5.1f} {s['avg_retries']:11.2f} {s['avg_dur']:7.0f}")


def cmd_status(session, args) -> None:
    state = load_state(session)
    if state is None:
        raise PipelineError("no campaign in this session")
    print(describe(state))
    if state.get("pending"):
        print(f"stopped during: {state['pending']}")
    for shape, reason in state["blocked"].items():
        print(f"blocked: {shape}: {reason}")
    path = _dir(session) / "results.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open()))
    inserts = [r for r in rows if r["role"] == "insert"]
    if inserts:
        by_hole: dict[str, list[dict]] = {}
        for r in inserts:
            by_hole.setdefault(r["hole"], []).append(r)
        ordered_holes = [f"{sh}_{cl}" for cl in state["clearances"] for sh in state["shapes"]
                         if f"{sh}_{cl}" in by_hole]
        extra_holes = sorted(h for h in by_hole if h not in ordered_holes)
        _print_insert_table("insert", [(h, by_hole[h]) for h in ordered_holes + extra_holes])

        by_clearance: dict[str, list[dict]] = {}
        for r in inserts:
            cl = r["hole"].split("_", 1)[1] if "_" in r["hole"] else r["hole"]
            if r.get("shape") and r["hole"].startswith(r["shape"] + "_"):
                cl = r["hole"][len(r["shape"]) + 1:]
            by_clearance.setdefault(cl, []).append(r)
        ordered_cls = [cl for cl in state["clearances"] if cl in by_clearance]
        extra_cls = sorted(cl for cl in by_clearance if cl not in ordered_cls)
        _print_insert_table("by clearance", [(cl, by_clearance[cl]) for cl in ordered_cls + extra_cls])

        overall = _summarize_inserts(inserts)
        print(f"\ninsert overall: {overall['ok']}/{overall['total']} ({overall['pct']:.0f}%), "
              f"plunge {overall['plunge']}, tilt {overall['tilt']}, "
              f"avg attempts {overall['avg_att']:.2f}, avg retries/insertion {overall['avg_retries']:.2f}, "
              f"avg time {overall['avg_dur']:.0f} s")

    extracts = [r for r in rows if r["role"] == "extract"]
    if extracts:
        by_ext: dict[str, list[dict]] = {}
        for r in extracts:
            by_ext.setdefault(r["hole"], []).append(r)
        print(f"\n{'extract':18s} {'ok':>7s} {'time s':>7s}")
        for hole, entries in sorted(by_ext.items()):
            ok = sum(int(r["ok"]) for r in entries)
            dur = [float(r["duration_s"]) for r in entries if r.get("duration_s")]
            print(f"{hole:18s} {ok:3d}/{len(entries):<3d} {(sum(dur) / len(dur)) if dur else 0:7.0f}")

    parks = [r for r in rows if r["role"] == "park"]
    if parks:
        print(f"\nparks: {sum(int(r['ok']) for r in parks)}/{len(parks)} into the {state['park_clearance']} holes")
