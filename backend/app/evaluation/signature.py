"""Stable signatures used to de-duplicate golden cases."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

from app.evaluation.models import EvaluatorSpec

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and casefold so trivial variants collide."""
    return _WS.sub(" ", text).strip().casefold()


def case_signature(input_text: str, evaluators: Sequence[EvaluatorSpec]) -> str:
    """sha256 over the normalized input and the order-independent check set.

    Same prompt with different checks is a different case; same prompt with
    the same checks (any order, any whitespace/case in the prompt) is a dup.
    """
    checks = sorted(
        json.dumps(e.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        for e in evaluators
    )
    payload = json.dumps(
        {"input": normalize_text(input_text), "checks": checks},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
