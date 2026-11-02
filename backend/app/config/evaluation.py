"""
Evaluation-layer configuration (fault capture).

Environment:
    CORE_EVAL_CAPTURE_FAULTS  "true"/"1"/"yes" turns capture on. Default: off.
    CORE_EVAL_FAULTS_DIR      Where ``faults.jsonl`` is written. Default:
                              ``backend/data/evaluation/faults``. A relative
                              path is resolved against ``backend/``, not the
                              process cwd, so it never lands in the repo root
                              by accident.

Read on every call (no caching) so the flag can be flipped without a restart
and tests can use ``monkeypatch.setenv``.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

# app/config/evaluation.py -> app/ -> backend/  (parents[2])
BACKEND_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DATA_DIR = BACKEND_ROOT / "data"
DEFAULT_FAULTS_DIR = BACKEND_DATA_DIR / "evaluation" / "faults"

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def _faults_dir() -> Path:
    raw = os.getenv("CORE_EVAL_FAULTS_DIR", "").strip()
    if not raw:
        return DEFAULT_FAULTS_DIR
    path = Path(raw).expanduser()
    return path if path.is_absolute() else BACKEND_ROOT / path


class FaultCaptureConfig(BaseModel):
    """Settings for capturing failed pipeline runs to disk."""

    enabled: bool = Field(default_factory=lambda: _env_flag("CORE_EVAL_CAPTURE_FAULTS"))
    directory: Path = Field(default_factory=_faults_dir)


def get_fault_capture_config() -> FaultCaptureConfig:
    """Current fault-capture settings from the environment."""
    return FaultCaptureConfig()
