"""Fault capture hook.

Default-off: nothing is written unless ``CORE_EVAL_CAPTURE_FAULTS`` is truthy.
The hook never raises — evaluation tooling must not break a pipeline run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from app.evaluation.models import FaultReport

logger = logging.getLogger(__name__)

ENABLE_ENV = "CORE_EVAL_CAPTURE_FAULTS"
DIR_ENV = "CORE_EVAL_FAULT_DIR"
DEFAULT_DIR = "eval_faults"
FAULTS_FILENAME = "faults.jsonl"


def is_capture_enabled() -> bool:
    return os.getenv(ENABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def fault_path(directory: Optional[str | Path] = None) -> Path:
    base = Path(directory or os.getenv(DIR_ENV) or DEFAULT_DIR)
    return base / FAULTS_FILENAME


def capture_fault(
    report: FaultReport, directory: Optional[str | Path] = None
) -> Optional[Path]:
    """Append ``report`` to the faults JSONL. Returns the path, or None if
    capture is disabled or the write failed. Never raises."""
    if not is_capture_enabled():
        return None
    try:
        path = fault_path(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(report.model_dump_json() + "\n")
        return path
    except Exception:  # noqa: BLE001 - hook must be fail-safe
        logger.warning("fault capture failed", exc_info=True)
        return None


def load_faults(path: str | Path) -> list[FaultReport]:
    """Read faults back, skipping unparseable lines."""
    out: list[FaultReport] = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(FaultReport.model_validate_json(line))
        except ValueError:
            logger.warning("skipping corrupt fault line")
    return out
