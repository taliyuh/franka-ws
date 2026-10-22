"""Pipeline configuration: the template in pipeline/config and per-session copies."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PIPELINE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = PIPELINE_DIR.parent
TEMPLATE = PIPELINE_DIR / "config" / "pipeline.yaml"
SETUPS = ("eye_in_hand", "stationary")


class PipelineError(Exception):
    """A failure with a message meant for the operator; no traceback is shown."""


def load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(Path(path).read_text()) or {}
    except OSError as error:
        raise PipelineError(f"cannot read {path}: {error}") from error


class Config:
    def __init__(self, data: dict):
        self.data = data

    @classmethod
    def template(cls) -> "Config":
        return cls(load_yaml(TEMPLATE))

    def __getitem__(self, key):
        return self.data[key]

    @property
    def setup(self) -> str:
        return self.data["setup"]

    @property
    def workspace(self) -> Path:
        """The folder containing pipeline/; never stored, so a workspace can be moved."""
        return WORKSPACE

    def path(self, value) -> Path:
        """Resolve a config path: ~ is expanded, relative means relative to the workspace."""
        path = Path(os.path.expanduser(str(value)))
        return path if path.is_absolute() else (self.workspace / path).resolve()

    @property
    def sessions_dir(self) -> Path:
        return self.path(self.data["paths"]["sessions"])

    @property
    def pose_refinement(self) -> Path:
        return self.path(os.environ.get("PIPELINE_POSE_REFINEMENT")
                         or self.data["paths"]["pose_refinement"])

    @property
    def setup_config(self) -> dict:
        return self.data[self.setup]
