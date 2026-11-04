# Evaluation data

* `golden/seed.jsonl` — seed Golden Set, one `GoldenCase` JSON object per line.
  Loaded and replayed by `backend/app/evaluation` (see `docs/evaluation_spec.md`).
* `gate/paired_twins.jsonl` — paired act/abstain twins for the comprehension
  gate (proceed / clarify / refuse). See [`gate/README.md`](gate/README.md);
  run with `uv run python -m app.evaluation.gate_suite` from `backend/`.

Evaluator types: `exact_match`, `contains`, `regex`, `max_latency_ms`,
`forbidden_substring`. A case passes only if all of its evaluators pass.

```python
from app.evaluation import GoldenSet, run_cases
cases = GoldenSet("docs/evaluation/golden/seed.jsonl").load().cases
print(run_cases(cases, my_callable).summary)   # my_callable(str) -> str | RunOutput
```

Add a fault: `GoldenSet(path).append_fault(fault_report)`; duplicates raise
`DuplicateCaseError`.
