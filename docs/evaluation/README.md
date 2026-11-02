# Evaluation: golden sets

`golden/seed.jsonl` holds one `GoldenCase` per line (see `backend/app/evaluation/golden_set.py`).
Spec: [`../evaluation_spec.md`](../evaluation_spec.md).

```python
from app.evaluation import GoldenSet, run_cases
cases = GoldenSet("docs/evaluation/golden/seed.jsonl").load().cases
result = run_cases(cases, my_callable)   # callable: str -> str | RunOutput
print(result.summary)
```

Turn a captured fault into a case with `GoldenSet(path).add_fault(fault, expected=...)`.
Fault capture is off unless `CORE_EVAL_CAPTURE_FAULTS=true` (dir: `CORE_EVAL_FAULT_DIR`, default `eval_faults/`).
