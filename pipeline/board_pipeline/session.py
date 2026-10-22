"""Session directories and the resumable step runner.

A session is one directory under sessions/ holding everything one setup
produces:

    sessions/<name>/
      config.yaml            settings snapshot (copied from pipeline/config)
      calibrate/1_look/ ...  one directory per step, each with a .done marker
      locate/1_look/ ...
      calibration.json       product of `calibrate`
      board_pose.json        product of `locate`
      targets/<hole>.json    per-hole robot target written by `check`/`probe`

sessions/current is a symlink to the session every command uses unless
--session is given; `pipeline new` and `pipeline use` move it.

A stage (calibrate, locate) is an ordered list of steps. A step whose .done
marker exists is skipped, so rerunning a stage after a failure continues at
the step that failed. Once one step runs, every later step of the stage runs
too, because its inputs changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

from board_pipeline.config import SETUPS, Config, PipelineError, load_yaml

DONE = ".done"
CALIBRATION = "calibration.json"
BOARD_POSE = "board_pose.json"
TARGETS = "targets"
REUSED = "REUSED_FROM"


@dataclass
class Step:
    name: str
    help: str
    run: Callable[["Context", Path], None]
    resumable: bool = False   # an unfinished attempt is continued, not wiped


@dataclass
class Context:
    session: "Session"
    cfg: Config
    args: object              # parsed CLI arguments (here, yes, ...)

    def step_dir(self, stage: str, name: str) -> Path:
        return self.session.step_dir(stage, name)


class Session:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.cfg = Config(load_yaml(path / "config.yaml"))

    # ------------------------------------------------------------ lookup
    @staticmethod
    def current_link(cfg: Config) -> Path:
        return cfg.sessions_dir / "current"

    @classmethod
    def open(cls, cfg: Config, name: str | None) -> "Session":
        if name is None:
            link = cls.current_link(cfg)
            if not link.is_symlink():
                raise PipelineError("no current session; create one with `pipeline new --setup ...`")
            path = link.resolve()
        else:
            path = cfg.sessions_dir / name
        if not (path / "config.yaml").exists():
            raise PipelineError(f"no session {path.name!r} in {cfg.sessions_dir}")
        return cls(path)

    @classmethod
    def create(cls, cfg: Config, setup: str, name: str | None, calib_from: "Session | None") -> "Session":
        if setup not in SETUPS:
            raise PipelineError(f"setup must be one of {SETUPS}")
        cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
        if name is None:
            base = f"{datetime.now():%Y-%m-%d}_{setup}"
            name, n = base, 2
            while (cfg.sessions_dir / name).exists():
                name, n = f"{base}_{n}", n + 1
        if name == "current" or "/" in name:
            raise PipelineError(f"invalid session name {name!r}")
        path = cfg.sessions_dir / name
        if path.exists():
            raise PipelineError(f"session {name!r} already exists; pick another name or `pipeline use {name}`")
        if calib_from is not None:
            if calib_from.cfg.setup != setup:
                raise PipelineError(f"{calib_from.name} is a {calib_from.cfg.setup} session, not {setup}")
            if not (calib_from.path / CALIBRATION).exists():
                raise PipelineError(f"{calib_from.name} has no {CALIBRATION}; run `pipeline calibrate` there first")
        path.mkdir()
        data = dict(cfg.data)
        data["setup"] = setup
        (path / "config.yaml").write_text(
            f"# Settings snapshot for session {name}, created {datetime.now():%Y-%m-%d %H:%M}.\n"
            + yaml.safe_dump(data, sort_keys=False))
        session = cls(path)
        if calib_from is not None:
            shutil.copyfile(calib_from.path / CALIBRATION, path / CALIBRATION)
            (path / REUSED).write_text(calib_from.name + "\n")
        session.make_current()
        return session

    def make_current(self) -> None:
        link = self.current_link(self.cfg)
        temporary = link.with_name(".current.tmp")
        if temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(self.name)
        os.replace(temporary, link)

    # ---------------------------------------------------------- products
    def product(self, name: str) -> Path:
        return self.path / name

    def require(self, name: str, hint: str) -> Path:
        path = self.path / name
        if not path.exists():
            raise PipelineError(f"{self.name} has no {name} yet; {hint}")
        return path

    def step_dir(self, stage: str, name: str) -> Path:
        matches = [d for d in (self.path / stage).glob("*_*")
                   if d.name.split("_", 1)[0].isdigit() and d.name.split("_", 1)[1] == name]
        if not matches or not (matches[0] / DONE).exists():
            raise PipelineError(f"step {stage}/{name} has not completed")
        return matches[0]

    def invalidate(self, *names: str) -> None:
        for name in names:
            path = self.path / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def step_status(stage_dir: Path, index: int, step: Step) -> str:
    path = stage_dir / f"{index}_{step.name}"
    if (path / DONE).exists():
        finished = json.loads((path / DONE).read_text())["finished"]
        return f"done {finished}"
    return "INCOMPLETE (rerun continues here)" if path.exists() else "-"


def remove_tree(cfg: Config, path: Path) -> None:
    """rmtree that also copes with root-owned files left by a killed container command."""
    try:
        shutil.rmtree(path)
    except PermissionError:
        from board_pipeline import ros
        ros.reclaim(cfg, path)
        shutil.rmtree(path)


def run_stage(ctx: Context, stage: str, steps: list[Step], fresh: bool,
              from_step: str | None, on_rerun: Callable[[], None],
              until: str | None = None) -> bool:
    """Run the stage; returns False when it stopped early at `until`."""
    names = [step.name for step in steps]
    for name in (from_step, until):
        if name is not None and name not in names:
            raise PipelineError(f"unknown step {name!r}; {stage} steps are: {', '.join(names)}")
    stage_dir = ctx.session.path / stage
    if fresh and stage_dir.exists():
        print(f"[{stage}] --fresh: deleting {stage_dir}")
        remove_tree(ctx.cfg, stage_dir)

    rerun = invalidated = False
    for index, step in enumerate(steps, 1):
        path = stage_dir / f"{index}_{step.name}"
        marker = path / DONE
        if step.name == from_step:
            rerun = True
        if marker.exists() and not rerun:
            print(f"[{stage}] {index}/{len(steps)} {step.name}: already done, skipped")
            if step.name == until:
                return False
            continue
        if not invalidated:
            on_rerun()        # downstream products are stale from here on
            invalidated = True
        rerun = True
        if path.exists():
            if step.resumable and not marker.exists():
                print(f"[{stage}] {index}/{len(steps)} {step.name}: continuing the unfinished attempt")
            else:
                remove_tree(ctx.cfg, path)
        # Every later step is stale now that this one reruns.
        for later in steps[index:]:
            stale = stage_dir / f"{steps.index(later) + 1}_{later.name}"
            if stale.exists():
                remove_tree(ctx.cfg, stale)
        path.mkdir(parents=True, exist_ok=True)
        print(f"\n[{stage}] {index}/{len(steps)} {step.name}: {step.help}")
        start = time.time()
        step.run(ctx, path)
        marker.write_text(json.dumps({"finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                                      "seconds": round(time.time() - start, 1)}) + "\n")
        if step.name == until and index < len(steps):
            print(f"\n[{stage}] stopped after {until} (--until); rerun without it to continue")
            return False
    return True
