"""No offline test may reach the robot. 2026-11-26: campaign tests ran the new gripper rehoming through the real
insert.lift / robot_program (the stand-ins replaced insert.run and extract.run only) and lifted the powered arm 9 x
60 mm. Every robot entry point raises here unless a test replaces it with its own stand-in (monkeypatch.setattr in the
test runs after this fixture and wins)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_pipeline import campaign, insert  # noqa: E402

REAL = {"robot_program": insert.robot_program}  # for the tests of robot_program itself (subprocess stubbed)


def _refuse(name):
    def refuse(*args, **kwargs):
        raise AssertionError(f"test reached the robot: insert.{name}{args[3:5] if len(args) > 4 else ''}; give it a "
                             "stand-in (monkeypatch)")
    return refuse


@pytest.fixture(autouse=True)
def no_robot(monkeypatch):
    for name in ("robot_program", "gripper_state", "ensure_robot_container"):
        monkeypatch.setattr(insert, name, _refuse(name))
    real_run = insert.subprocess.run

    def guarded_run(cmd, *args, **kwargs):
        text = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd)
        if "docker" in text and ("exec" in text or "compose" in text):
            raise AssertionError(f"test reached the robot container: {text[:120]}")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(insert.subprocess, "run", guarded_run)
    yield
