# CORE Evaluation Layer Technical Specification

## 1. Purpose
The CORE Evaluation layer serves as the primary diagnostic and quality-assurance engine for the core intelligence capabilities. It moves beyond simple output validation to evaluate **intent**, **reasoning accuracy**, and **logical coherence**.

## 2. Core Philosophy
*   **Failures are Data:** Every failed prompt or hallucination is automatically captured and transformed into a new training case in the "Golden Set."
*   **Iterative Refinement:** The system must be capable of self-correcting by identifying recurring patterns of error via the assessment loop.
*   **Decoupled Verification:** Evaluation logic is separated from inference to ensure that validation does not increase user-facing latency in production environments.

## 3. System Architecture

### A. Telemetry & Observation (The Evidence)
Leverage OpenTelemetry (OTEL) to capture every step of the "Chain of Thought" (CoT):
*   **Input State:** Original prompt, metadata, and system instructions.
*   **Internal Monologue:** The raw reasoning steps provided by the model before a tool call.
*   **Execution Context:** Tool names, parameters used, and full response payloads from external functions.
*   **Metadata:** Latency (ms), cost units, and token count per step.

### B. Dual-Track Evaluation Pipeline
1.  **Development Track (High Fidelity):**
    *   **Purpose:** Experimentation and refining complex logic.
    *   **Method:** Human-in-the-loop review against a "Golden Set" of peerless examples.
    *   **Mechanism:** High-latency, full-reasoning logs are analyzed to find points of divergence from the expected outcome.
2.  **Production Track (Low Latency):**
    *   **Purpose:** Real-time guardrails and consistency checks.
    *   **Method:** Automated code-based evaluators and heuristic "guardrail" filters.
    *   **Mechanism:** Rapid check against blacklisted behaviors, safety constraints, and critical logic failure patterns (e.g., infinite loops).

### C. The Training Loop (Automated Learning)
When an issue is identified in either track:
1.  **Capture:** A "Fault Report" is generated including the failed sequence.
2.  **Analyze:** An asynchronous evaluator compares the result to a target outcome.
3.  **Update:** The difference is codified into the **Golden Set**, ensuring future instances do not repeat the specific error signature.

## 4. Technical Requirements
*   **Trajectory Matching:** Use fuzzy matching or semantic distance to ensure the *path* taken by the model aligns with the desired logic flow.
*   **Confidence Scoring:** Every transition between "Thinking" and "Acting" must be assigned a confidence score; low scores trigger an internal re-evaluation pass before outputting to the user.
*   **Cost-Efficient Evaluation:** Use smaller, specialized models for production checks to maintain performance while keeping cost overhead minimal.

## 5. Roadmap & Milestones
*   [ ] **Phase I: Instrumentation.** Implement full OTEL logging across all core tool integrations in `backend/`.
*   [ ] **Phase II: Core Rule Set.** Establish the first 100 "Golden" edge cases for standard operations (e.g., data retrieval, scheduling).
*   [ ] **Phase III: Automated Audit.** Integrate an asynchronous monitoring service to flag and log failures in the `monitoring` module.

### Implementation status (Phase I/II slice)

No roadmap box above is ticked: the slice below is groundwork, not completion of either phase.

*   **Fault capture (§3.C "Capture") — implemented and wired, default off.** `backend/app/evaluation/` provides the `FaultReport` model (input, steps, tool calls, expected vs actual, timestamps, latency, token counts when available), `fault_report_from_state()` and `capture_fault()` (JSONL on disk). `COREGraph.conversation_node` in `core_graph_v2.py` calls `maybe_capture_fault()` next to `record_pipeline_run`. See "Live fault capture" below.
*   **Golden Set (§3.C "Update") — implemented.** JSONL store with dedupe by stable signature, corrupt-line tolerance, and `append_fault()`. 10 seed cases in `docs/evaluation/golden/seed.jsonl` (Phase II targets 100).
*   **Production-track evaluators (§3.B.2) — partially implemented.** Deterministic exact/contains/regex/max-latency/forbidden-substring checks and a replay runner that takes an injected callable. Heuristic guardrails for loops and safety constraints are not done.
*   **Abstention guard and claim verification (§3.B.2 guardrails) — implemented, wired, default on.** An unresolved refusal (comprehension gate clarify/refuse, or an LLM step that declines) blocks every later step and stops retry/re-plan; action claims in LLM step outputs are checked against the dispatched tool-call trace (`StepResult.dispatched_tool`), and unsupported ones lower evaluation confidence and are flagged in the response. The `claims_supported` evaluator applies the same check to golden cases. Fault reports no longer list never-dispatched steps as tool calls. See `docs/architecture/abstention-guard.md`.

Deviations from this spec:

*   **No OpenTelemetry.** Phase I asks for OTEL instrumentation; this slice captures faults without it. Phase I stays open. Token counts and latency are recorded only when the caller supplies them.
*   **Capture is synchronous.** §2 asks that evaluation not add user-facing latency. The hook does one small JSONL append, only when the flag is on and only for faulted runs; moving it to a background writer is a follow-up if that ever shows up in latency.
*   **Missing latency fails `max_latency_ms`** rather than passing silently.
*   **Not built:** trajectory matching, confidence scoring, the asynchronous analyzer and the `monitoring` service (§4, Phase III).
*   `app/eval` (verdict-drift harness for `evaluation_service`) is a separate, older package and is untouched.

### Live fault capture

| Setting | Default | Meaning |
|---|---|---|
| `CORE_EVAL_CAPTURE_FAULTS` | off | `true`/`1`/`yes`/`on` turns capture on. |
| `CORE_EVAL_FAULTS_DIR` | `backend/data/evaluation/faults` | Where `faults.jsonl` is appended. Relative paths resolve against `backend/`, never the cwd or repo root. In Docker this is `/app/data/...` on the `backend-data` volume. |

Read through `app/config/evaluation.py` (`get_fault_capture_config()`) on every run, so no restart is needed.

**Where a fault is detected.** Every path through the graph ends in `conversation_node` (task runs, plain chat, gate clarify/refuse, and every node's exception fallback), so the check runs once there, on the finished state, after the response is set. `fault_reasons()` returns:

*   `errors`: any node hit its exception path (`state.errors`). This includes an evaluation crash, which the graph otherwise hides behind a default "success".
*   `eval_failure`: the evaluator judged the run a failure.
*   `step_failure`: at least one plan step failed.
*   `unresolved_evaluation`: the run ended while the evaluator still asked to revise or retry, i.e. the plan-revision loop guard forced finalization (§3.B.2 "infinite loops").
*   `unsupported_claims`: an action claim in an LLM step output has no dispatched tool call behind it (`state.claim_report`, see "Abstention guard and claim verification" above).

A front-gate refusal or clarification on its own is not a fault: it is the gate doing its job. If one turns out to be wrong it shows up as a Golden Set case, not as an automatic capture.

**Fail-safe.** `maybe_capture_fault()` never raises. Any error (unwritable dir, disk full, a bug building the report) is logged at WARNING with the `run_id` and dropped; the request completes normally. The call site in the node has a second guard for the same reason. `capture_fault()` itself still raises, for callers that want to know.

Each captured report's `metadata` carries `source`, `fault_reasons`, `gate_outcome`, `plan_revisions`, `eval_status` and `eval_next_action`, plus `refusal`, `blocked_actions` and `unsupported_claims` when the run has them.
