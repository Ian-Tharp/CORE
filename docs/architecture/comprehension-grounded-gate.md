# CORE as a self-grounded deliberation loop — design spec

**Status:** Draft for review (Ian) · **Date:** 2026-06-25 · **Branch:** `feat/comprehension-grounded-gate`
**Related:** the overnight ToT experiment (`atlas/overnight-2026-06-24`) — `app/reasoning/` harness +
`FINDINGS_tot_generalization_2026-06-25.md`; the reasoning-strategy proposal
`docs/research/reasoning-strategy-as-governed-policy.md`.

## 1. The idea in one paragraph

CORE is not a single forward pass; it is a **deliberation loop bracketed by two verifiers**.
Comprehension verifies *up front* whether the system understands the intent and actually has the
capabilities to attempt it. Evaluation verifies *after* whether the produced result achieves that
intent, looping back to retry a step or revise the plan when it doesn't. The loop generalizes exactly
to the degree those two verifiers are good — which is the same lesson the overnight ToT experiment
reached from the other side (*search is only worth its cost with a trustworthy verifier*). This spec
makes both verifiers real by giving the system a **self-model**: a queryable registry of what it can
do, plus a summary of what it knows about the user/session, both injected into comprehension.

## 2. The loop and the division of labor

```
                 ┌─────────────── revise plan (intent not achievable as planned) ──────────────┐
                 │                                                                              │
 user input → COMPREHENSION ──proceed──→ ORCHESTRATION ──→ REASONING ──→ EVALUATION ──finalize──→ response
                 │  (front verifier)        (sequencer)     (executor)   (back verifier) │
                 ├──clarify──→ ask user                                  └──retry step────┘
                 └──refuse────→ honest "I can't do that (here's what I can)"
```

Each node owns exactly one question. Keeping these separate is the core design discipline — it
prevents comprehension from quietly becoming a second evaluator (which causes over-conservative
refusals):

| Node | Question it answers | Verifies |
|------|--------------------|----------|
| **Comprehension** | "Do I understand this, and do the *primitives* + context exist to attempt it?" | capability **presence** |
| **Orchestration** | "Can I *wire* those primitives into an ordered plan?" | **composability** |
| **Reasoning** | "Execute each step." | — |
| **Evaluation** | "Did the result actually *satisfy the intent*?" | **achievement** |

Comprehension is cheap triage, not the final word: it *reduces* the intent-misalignment that reaches
Evaluation; it cannot eliminate it. A plan can have every tool present (comprehension OK) and still
fail to do what the user wanted (only Evaluation catches that).

## 3. Current state (what exists vs. what's missing)

Grounded in the code on `develop`:

- `app/models/core_state.py` — **contracts already support this.** `UserIntent` has `requires_tools`,
  `tools_needed`, `ambiguities`, `task_category`; `PlanStep` has `tool`, `params`, `dependencies`,
  `requires_hitl`, `retry_policy`; `EvaluationResult.next_action ∈ {finalize, retry_step, revise_plan,
  ask_user}`. The data model is ready; the *logic* isn't.
- `app/core/agents/comprehension_agent.py` — RAGs the KB and LLM-classifies intent. **But** its tool
  list is a **hardcoded prompt string** (`file_operations, git, database, web_research`) disconnected
  from the real dispatcher, and `tools_needed`/`ambiguities` are LLM-guessed, not checked against
  anything. It has a `detect_ambiguities()` regex method that **is not wired into** `analyze_intent`.
- `app/core/tools/dispatcher.py` — `ToolDispatcher.available_tools` returns **only names**. No
  descriptions, no parameter schemas, no MCP tool definitions, no capability metadata.
- `app/core/langgraph/core_graph_v2.py` — `route_from_comprehension` is binary (task/question →
  orchestration, else → conversation). There is **no clarify or refuse path** wired, even though the
  `clarification` intent type and `ask_user` action exist.
- Memory: there is no user/session memory summary feeding comprehension today.

**So the missing pieces are: (1) a real Capability Registry, (2) comprehension grounding + the
tri-state gate, (3) orchestration sequencing from the registry, (4) a memory summary input,
(5) optional per-step evaluation.** The data contracts mostly exist already.

## 4. Component specs

### 4.1 Capability Registry — the foundation (NEW)

A single source of truth describing every capability the system has. The dispatcher and MCP servers
**register into it**, so it can never drift from what actually runs (today's hardcoded-prompt-vs-
dispatcher drift is exactly the bug this kills).

Each entry (MCP-compatible so MCP tool descriptions map straight in):

```
CapabilityEntry:
  id:            str                  # "file_operations.read"
  name:          str
  description:   str                  # natural-language; what it does, when to use it
  params_schema: dict                 # JSON Schema (MCP inputSchema-compatible)
  side_effects:  "read" | "write" | "network" | "none"   # safety/governance class
  source:        "builtin" | "mcp:<server>"
  examples:      list[str]            # optional, for retrieval quality
```

API (sync, cheap, no LLM):
- `registry.search(intent_text, k) -> list[CapabilityEntry]` — **lightweight retrieval** (embedding
  similarity over `description`+`examples`, reusing the local embedder; falls back to keyword). This
  is the "capability presence" primitive comprehension calls.
- `registry.get(id) -> CapabilityEntry | None`
- `registry.all() -> list[CapabilityEntry]`

Source of truth: the dispatcher's handlers expose their own descriptors (a `describe()` per tool,
including action-level sub-capabilities like `file_operations.{read,write,list,search}`); MCP servers
contribute their advertised tools. The registry is built at startup and refreshed when MCP servers
change.

### 4.2 Comprehension — grounded tri-state gate

Two inputs feed comprehension (this is "understand intent from what it knows about the system"):

- **(a) self-model / capability presence** — `registry.search(user_input)` returns candidate
  capabilities; comprehension is told *what it can actually do*, replacing the hardcoded prompt list.
- **(b) memory summary** — a short, injected summary of the user/session: recent asks, this
  conversation's history, recent run/tool activity (§4.5).

Output: the existing `UserIntent`, with `tools_needed` now **validated against the registry** and a
new routing decision. Three outcomes:

| Outcome | When | Routes to |
|---|---|---|
| **proceed** | intent clear (confidence ≥ τ_high), required capabilities present | orchestration |
| **clarify** | intent plausible but ambiguous (`detect_ambiguities` hits, or confidence in [τ_low, τ_high)) | a clarify response: "do you mean X/Y/Z? here's what I can do — does that match?" |
| **refuse/redirect** | required capability **absent** from the registry | honest response: "I can't do that; here's what I *can* do" |

Concretely: wire `detect_ambiguities()` into `analyze_intent`; add a `registry` lookup; add a
`gate_decision ∈ {proceed, clarify, refuse}` field (or derive it in routing from the populated
fields). The clarify/refuse paths surface through the existing Conversation node as a base-model
reply — no O→R→E spend. This is the **highest-leverage verifier in the system**: it prevents the
entire expensive loop from running on requests that are ambiguous or impossible.

### 4.3 Orchestration — registry-driven sequencer

Orchestration becomes "given the intent + the candidate capabilities from the registry, produce the
**ordered** `ExecutionPlan` of `PlanStep`s (tool + params + dependencies) that accomplishes it" —
think of it as composing MCP tool calls into a sequence. It selects only from registry entries
(no invented tools) and validates each step's `params` against the entry's `params_schema` before
emitting the plan. A plan may have many steps; `dependencies` already expresses ordering/parallelism.

### 4.4 Evaluation — per-step and final achievement (the back verifier)

Largely exists. Two refinements:
- **Per-step review** when `PlanStep.requires_hitl` (or a step is flagged risky): evaluate after that
  step instead of only at the end, so a wrong direction is caught early (cheaper than running all 10
  steps first). Reuses `EvaluationResult` + the existing `retry_step`/`revise_plan` routing.
- Make explicit that `revise_plan` is the **intent-misalignment** path (plan executed but didn't
  achieve the goal), distinct from `retry_step` (execution flaked). The fields already support this;
  the rubric/prompt should name the distinction.

### 4.5 Memory summary input (reuse, don't reinvent)

Comprehension input (b) is the **memory layer already scoped** (see ATLAS eval-loop notes): LangGraph
`PostgresStore` over CORE pgvector + local embedder + a governed consolidation node. For this spec it
exposes one read: `memory.summarize(user_id, conversation_id, k) -> str` — a short context block
(recent asks, session history, recent tool activity) injected into the comprehension prompt. Phase 1
can stub this with the existing conversation history; the real memory layer lands in Phase 3.

## 5. How we measure it (reuse the overnight harness)

The `app/reasoning/` harness (cross-domain `Problem` set, cost-aware report) generalizes from "score a
reasoning node" to "score the whole loop." The viable-framework experiment: run full C→O→R→E on the
benchmark and **ablate each verifier** —

- comprehension grounding **on/off** (registry + memory vs. today's blind classifier),
- evaluation per-step **on/off**,

and measure accuracy + cost + how often the front gate correctly clarifies/refuses (precision/recall
of the gate itself). This directly answers "is the CORE framework viable for generalizable output,
and which verifier is carrying it." Depends on merging the overnight harness branch first.

## 6. Phased build (each phase shippable + measurable)

1. **Phase 1 — Registry + comprehension grounding (the foundation).** Build the Capability Registry
   (dispatcher `describe()` + registry + retrieval); wire it + `detect_ambiguities` into comprehension;
   add the tri-state gate + clarify/refuse routing in `core_graph_v2`. Memory input stubbed with
   conversation history. **This is the first slice and the one I'd build first** — everything else
   needs the registry, and the front gate is the highest-leverage verifier.
2. **Phase 2 — Orchestration from the registry.** Plans selected/validated against registry entries;
   param-schema validation before emit.
3. **Phase 3 — Memory layer + per-step evaluation.** Real `memory.summarize`; per-step eval on
   `requires_hitl` steps; name the retry-vs-revise distinction in the eval rubric.
4. **Phase 4 — Measure.** Wire the loop into the benchmark; ablate the two verifiers; report.

## 7. Open decisions for Ian

1. **Registry shape:** MCP-native schema as the internal format (recommended — MCP tools drop in with
   zero translation), or a CORE-internal schema MCP maps into?
2. **"Lightweight retrieval":** embedding similarity over registry descriptions (recommended, reuses
   local embedder), keyword match, or a cheap LLM pass? (The overnight finding argues against paying
   for an LLM where a cheap deterministic check works.)
3. **Refuse policy:** hard gate (don't enter the loop if a required capability is absent) vs. soft
   (warn + attempt). Recommended: hard for *absent* capability, clarify for *ambiguous* intent.
4. **Gate thresholds** τ_low / τ_high: start heuristic (e.g. 0.45 / 0.75) and tune on the Phase 4
   benchmark, or define up front?
5. **Scope of Phase 1 registry:** builtin dispatcher tools only first, or include MCP servers from
   day one?

## 8. Non-goals / out of scope

- Re-architecting orchestration's planning algorithm beyond "select+sequence from the registry."
- The reasoning-strategy switch (`linear|tot`) from the overnight proposal — orthogonal; can compose
  later (a node could deliberate, but that's downstream of this).
- Auto-tuning thresholds or learned routing — heuristics first, measure, then maybe.
- Changing live `core_graph_v2` default behavior without the Phase 4 measurement backing it.
