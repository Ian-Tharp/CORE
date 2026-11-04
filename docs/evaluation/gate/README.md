# Comprehension gate: paired-twin suite

Measures the comprehension front gate (`app/core/agents/comprehension_gate.py`)
on its three outcomes, **proceed / clarify / refuse**, with over-refusal tracked
as a first-class failure. Code: `backend/app/evaluation/gate_suite.py`.
Data: `paired_twins.jsonl` (49 pairs, 98 requests: 35 seed + 14 holdout).

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
that input. The seed fixtures follow the old classifier prompt, which listed
tool **roots** only, so a request to push or delete names the root (`git`), not
the action. The prompt is now built from the registry and asks for capability
ids (`git.log`), and for an unlisted action as `root.action` (`git.push`); a
few holdout fixtures use that form. The gate handles both.
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

* **live gate**: `decide_gate(intent, registry, text=input)` on the fixture,
  with the regex ambiguities from `ComprehensionAgent.detect_ambiguities`
  merged in first, as `core_graph_v2.comprehension_node` does.
* **gate without regex ambiguities**: the same, minus the regex merge (an ablation).
* **always act**: baseline.

### End to end (needs an LLM)

The default predictor tests the gate on a fixed classifier output. To test the
classifier and the gate together, pass a predictor that ignores the fixture:

```python
from app.core.agents.comprehension_agent import ComprehensionAgent
from app.core.agents.comprehension_gate import decide_gate
from app.evaluation.gate_suite import builtin_registry, load_gate_pairs, run_gate_suite, format_report

registry = builtin_registry()
agent = ComprehensionAgent(registry=registry)  # prompt lists this registry

def end_to_end(text, _fixture):
    intent = agent.analyze_intent(text)
    for amb in agent.detect_ambiguities(text):
        if amb not in intent.ambiguities:
            intent.ambiguities.append(amb)
    return decide_gate(intent, registry, text=text).outcome

report = run_gate_suite(load_gate_pairs().pairs, end_to_end, label="e2e")
print(format_report([report]))
```

`analyze_intent` falls back to a `conversation` intent on any model error, and
the gate passes conversation intents through, so an unreachable model shows up
as under-abstention, not as `error`.

## Current results (2026-11-03, gate only)

After the action-level capability check and the referent-aware ambiguity
detector (see [Fixes](#fixes)):

| metric | live gate | without regex ambiguities | always act |
|---|---|---|---|
| paired accuracy | **100.0%** | 83.7% | 0.0% |
| act accuracy | 100.0% | 100.0% | 100.0% |
| abstain accuracy | 100.0% | 83.7% | 0.0% |
| clarify accuracy | 100.0% | 69.2% | 0.0% |
| refuse accuracy | 100.0% | 100.0% | 0.0% |
| over-refusal rate | **0.0%** | 0.0% | 0.0% |
| under-abstention rate | 0.0% | 16.3% | 100.0% |

Read this with care. Both fixes were written after reading the 35 seed pairs,
so 100% on them is a regression guard, not evidence that the gate generalizes.
To get a less biased number, 14 **holdout** pairs (tag `holdout`) were written
before running the fixed gate on them, aimed at the new rules' weak spots
(other git writes, rename, database update, upload, an action id with no verb in
the text, "update" on a file, relative "that", PascalCase names). On first run
they scored **85.7% paired, 14.3% over-refusal, 0% under-abstention**: two
over-clarifies, both from detector gaps that apply beyond this set
("the commit **that** added X" read as a pronoun; "the class
`CapabilityRegistry`" not seen as a name). After fixing those the holdout pairs
pass too, so they are no longer held out. The next honest number needs new
pairs, or the end-to-end predictor.

The regex ablation now shows the detector adding clarify recall (69% → 100%)
with no over-refusal cost. Before, it traded one for the other.

## Fixes

1. **Action-level capability check** (`comprehension_gate.py`,
   `registry/actions.py`). A named tool must resolve **exactly** to a registered
   root (`git`) or capability id (`git.log`); the lexical `has_capability` is no
   longer used for named tools, so `web_search` no longer matches
   `web_research.fetch`. An id the registry lacks (`git.push`) is missing. For a
   present root, the gate pulls *gated* operation verbs (push, commit, merge,
   delete, drop, rename, update, submit, upload, deploy, ...) out of the raw
   input and the intent description, and refuses any that no capability of the
   named roots serves. Each `CapabilityEntry` declares the verbs it serves
   (`verbs`; `file_operations.write` serves edit/update/create). Two guards
   against over-refusal: only verbs in `GATED_OPERATIONS` can refuse (never
   show/run/send/copy), and only in verb position (start of a clause, or after
   "please", "can you", "to", "and"...), so "show commit a1b2c3d" is not a commit.
   The refusal lists what the root can do ("Git status, Git diff, ...").
2. **Classifier prompt built from the registry** (`ComprehensionAgent(registry=)`).
   The hardcoded list advertised "Branch, commit, diff, create PR"; the prompt now
   lists the registry's capability ids and tells the model to name an unlisted
   action as `root.action` rather than map it onto a listed one. `COREGraph`
   passes its dispatcher-built registry.
3. **Referent-aware ambiguity detector** (`detect_ambiguities`). A hit is skipped
   when the same request resolves it: a pronoun or "there" after an explicit
   referent (path, file name, URL, glob, quoted text, sha, identifier);
   "this repository/project/workspace"; a relative "that" after
   "<determiner> <noun>"; "the function/class/file" followed by a name;
   "before/after" followed by a date, version or sha; "some of them" when "them"
   is resolved. "Comparative" and "somewhere"/"recently" are never skipped.

## Baseline before the fixes (2026-11-03, 35 seed pairs)

Kept for comparison; this is what the suite measured on the code it was written
against.

| metric | live gate | without regex ambiguities | always act |
|---|---|---|---|
| paired accuracy | **68.6%** | 65.7% | 0.0% |
| act accuracy | 85.7% | 100.0% | 100.0% |
| abstain accuracy | 82.9% | 65.7% | 0.0% |
| clarify accuracy | 100.0% | 68.4% | 0.0% |
| refuse accuracy | 62.5% | 62.5% | 0.0% |
| over-refusal rate | **14.3%** | 0.0% | 0.0% |
| under-abstention rate | 17.1% | 34.3% | 100.0% |

`unsupported_action` was 0% (6/6 missing actions proceeded) and
`phrasing_trap` was 0% (5/5 clear twins over-clarified: all of the 14.3%).
Finding 3 still stands: when a tool is named and absent the gate is right, but
whether the live classifier names it (rather than mapping onto an existing
root, which the action check now partly catches) needs an end-to-end run.

`tests/evaluation/test_gate_suite.py` keeps a `KNOWN_GAPS` set (now empty), fails
on any new failure, and enforces floors (paired, act, abstain ≥ 0.95;
over-refusal ≤ 0.05). A new pair the gate gets wrong goes into `KNOWN_GAPS` with
a comment, rather than lowering a floor silently.

## Limits

* 49 hand-written pairs, and the gate's heuristics were tuned by the same
  hand. One pair moves a category rate by 7–25 points. Treat the numbers as a regression guard and a
  map of failure types, not a benchmark score.
* Gate-only results depend on the fixtures being faithful to the classifier.
  Only the end-to-end predictor measures the real comprehension stack.
* `clarify` vs `refuse` for a request that is both ambiguous and impossible is
  scored as `refuse` (the gate checks absence before ambiguity, by design).
* The verb check is lexical. It misses a missing action phrased without a gated
  verb ("send my branch upstream") unless the classifier names the action id,
  and it can over-refuse a gated verb used loosely ("remove the trailing
  whitespace" with `file_operations` refuses as `file_operations.remove`).
  Plan grounding in orchestration remains the second line for the first case.
* The referent check is position-based, not parsed: "Read main.py and fix it"
  counts `main.py` as the referent of "it" even if the user meant something
  else.
