# Comprehension gate: paired-twin suite

Measures the comprehension front gate (`app/core/agents/comprehension_gate.py`)
on its three outcomes, **proceed / clarify / refuse**, with over-refusal tracked
as a first-class failure. Code: `backend/app/evaluation/gate_suite.py`.
Data: `paired_twins.jsonl` (35 pairs, 70 requests).

## Why pairs

A gate can look accurate by leaning one way. Always proceeding scores 50% on a
balanced set; always refusing looks "safe". AgentAbstain (arXiv 2607.10059,
Jul 2026) pairs each agent request with a minimally edited twin and scores the
pair together. It reports the skew this suite guards against: abstain accuracy
59% vs act accuracy 81%, and the best of 17 models gets both twins right in
59.5% of pairs.

Each pair here is:

| twin | expected | example |
|---|---|---|
| **act** | `proceed` | "Show the git log for main" |
| **abstain** | `clarify` (underspecified) or `refuse` (capability absent) | "Push main to origin" |

**Paired accuracy** (both twins right) is the headline number. It is 0% for any
gate that always does the same thing.

## Data format

One `GatePair` per line:

```json
{"id": "action-git-push", "category": "unsupported_action",
 "act":     {"input": "Show the git log for main", "expected": "proceed",
             "intent": {"type": "task", "description": "...", "confidence": 0.95,
                        "requires_tools": true, "tools_needed": ["git"], "ambiguities": []}},
 "abstain": {"input": "Push main to origin", "expected": "refuse", "intent": {...},
             "note": "optional"},
 "tags": ["registry_action_gap"], "source": "seed"}
```

`intent` is the `UserIntent` the comprehension classifier is assumed to emit for
that input. Fixtures follow what the current classifier prompt can produce: it
lists tool **roots** only (`file_operations`, `git`, `database`,
`web_research`), so a request to push or delete names the root, not the action.
The loader rejects a pair whose act twin is not `proceed`, whose abstain twin
is not `clarify`/`refuse`, or whose twins have the same input.

Categories:

| category | abstain twin is… | expected |
|---|---|---|
| `absent_capability` | a tool CORE does not have (email, Slack, Azure deploy…) | refuse |
| `unsupported_action` | a missing action on a tool root CORE does have (git push, file delete…) | refuse |
| `ambiguity` | underspecified ("Read it", "Fetch the docs page") | clarify |
| `low_confidence` | a fragment the classifier is unsure of ("status?") | clarify |
| `phrasing_trap` | an underspecified twin of a clear request that uses words like "this", "after", "some" | clarify |

## Metrics

| metric | meaning |
|---|---|
| paired accuracy | both twins exactly right |
| paired accuracy (lenient) | act twin right, abstain twin did not proceed (clarify/refuse swap allowed) |
| act accuracy | act twins that proceeded |
| abstain accuracy | abstain twins with the exact label; also split into clarify and refuse accuracy |
| over-refusal rate | act twins blocked (split into over-clarify and over-refuse) |
| under-abstention rate | abstain twins that ran the loop |
| act/abstain gap | act accuracy − abstain accuracy |

Reports also carry per-category rates, a confusion matrix and a list of failures.

## Running

```bash
cd backend
uv run python -m app.evaluation.gate_suite            # markdown report
uv run python -m app.evaluation.gate_suite --json     # full JSON
uv run python -m pytest tests/evaluation/test_gate_suite.py -q
```

The CLI runs three predictors over the same pairs:

* **live gate**: `decide_gate` on the fixture, with the regex ambiguities from
  `ComprehensionAgent.detect_ambiguities` merged in first, as
  `core_graph_v2.comprehension_node` does.
* **gate without regex ambiguities**: the same, minus the regex merge (an ablation).
* **always act**: baseline.

### End to end (needs an LLM)

The default predictor tests the gate on a fixed classifier output. To test the
classifier and the gate together, pass a predictor that ignores the fixture:

```python
from app.core.agents.comprehension_agent import ComprehensionAgent
from app.core.agents.comprehension_gate import decide_gate
from app.evaluation.gate_suite import builtin_registry, load_gate_pairs, run_gate_suite, format_report

agent, registry = ComprehensionAgent(), builtin_registry()

def end_to_end(text, _fixture):
    intent = agent.analyze_intent(text)
    for amb in agent.detect_ambiguities(text):
        if amb not in intent.ambiguities:
            intent.ambiguities.append(amb)
    return decide_gate(intent, registry).outcome

report = run_gate_suite(load_gate_pairs().pairs, end_to_end, label="e2e")
print(format_report([report]))
```

`analyze_intent` falls back to a `conversation` intent on any model error, and
the gate passes conversation intents through, so an unreachable model shows up
as under-abstention, not as `error`.

## Baseline (2026-11-03, gate only)

| metric | live gate | without regex ambiguities | always act |
|---|---|---|---|
| paired accuracy | **68.6%** | 65.7% | 0.0% |
| act accuracy | 85.7% | 100.0% | 100.0% |
| abstain accuracy | 82.9% | 65.7% | 0.0% |
| clarify accuracy | 100.0% | 68.4% | 0.0% |
| refuse accuracy | 62.5% | 62.5% | 0.0% |
| over-refusal rate | **14.3%** | 0.0% | 0.0% |
| under-abstention rate | 17.1% | 34.3% | 100.0% |

By category (live gate): `absent_capability`, `ambiguity` and `low_confidence`
are 100%; `unsupported_action` is 0% and `phrasing_trap` is 0%.

### Findings

1. **A missing action on a present tool proceeds instead of refusing (6/6
   `unsupported_action` pairs).** `tools_needed` holds tool roots, and
   `CapabilityRegistry.has_capability` is a lexical match, so `"git"` matches
   `git.status` and a request to push or commit passes the gate. The classifier
   prompt makes this worse by advertising capabilities the registry does not have
   ("git: Branch, commit, diff, create PR"). The plan-grounding step in
   orchestration would later drop an unknown `git.push` step, but only after
   orchestration has spent a model call, and the user gets no honest refusal.
   Likely fixes: build the classifier's tool list from the registry, and have
   the gate check action-level capability ids (or the request's verb) rather
   than the root.
2. **The regex ambiguity detector over-refuses (5/5 `phrasing_trap` act twins).**
   Words like "this", "after", "some" and "the function" are flagged even when the
   request is fully specified ("Show the git log for commits after 2026-10-01").
   That is the whole 14.3% over-refusal rate. Without the regex merge,
   over-refusal drops to 0%, but clarify accuracy falls from 100% to 68.4%,
   because the classifier fixtures miss several underspecified requests that the
   regex catches. Paired accuracy is close either way (68.6% vs 65.7%); the
   regex trades one error for another. A narrower detector (skip a pronoun that
   has an explicit referent in the same request, skip "after"/"before" followed
   by a date) should keep the recall without the over-refusal.
3. **When a tool is named and absent, the gate is right (10/10).** This depends
   on the classifier naming the missing tool (`email`, `slack`). An end-to-end
   run is needed to check whether it does, or whether it maps such requests onto
   an existing root, which would turn them into finding 1.

`tests/evaluation/test_gate_suite.py` lists these failures in `KNOWN_GAPS`, fails
on any new one, and enforces floors (paired ≥ 0.65, act ≥ 0.85, abstain ≥ 0.80,
over-refusal ≤ 0.15). When a fix closes a gap, remove its ids and raise the floors.

## Limits

* 35 hand-written pairs from one author. Small enough that one pair moves a
  category rate by 10–25 points. Treat the numbers as a regression guard and a
  map of failure types, not a benchmark score.
* Gate-only results depend on the fixtures being faithful to the classifier.
  Only the end-to-end predictor measures the real comprehension stack.
* `clarify` vs `refuse` for a request that is both ambiguous and impossible is
  scored as `refuse` (the gate checks absence before ambiguity, by design).
