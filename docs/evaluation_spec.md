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

*   **Fault capture (§3.C "Capture") — implemented, not wired.** `backend/app/evaluation/` provides the `FaultReport` model (input, steps, tool calls, expected vs actual, timestamps, latency, token counts when available), `fault_report_from_state()` and `capture_fault()` (JSONL on disk). Nothing in the live pipeline calls it yet.
*   **Golden Set (§3.C "Update") — implemented.** JSONL store with dedupe by stable signature, corrupt-line tolerance, and `append_fault()`. 10 seed cases in `docs/evaluation/golden/seed.jsonl` (Phase II targets 100).
*   **Production-track evaluators (§3.B.2) — partially implemented.** Deterministic exact/contains/regex/max-latency/forbidden-substring checks and a replay runner that takes an injected callable. Heuristic guardrails for loops and safety constraints are not done.
*   **Abstention guard and claim verification (§3.B.2 guardrails) — implemented, wired, default on.** An unresolved refusal (comprehension gate clarify/refuse, or an LLM step that declines) blocks every later step and stops retry/re-plan; action claims in LLM step outputs are checked against the dispatched tool-call trace (`StepResult.dispatched_tool`), and unsupported ones lower evaluation confidence and are flagged in the response. The `claims_supported` evaluator applies the same check to golden cases. Fault reports no longer list never-dispatched steps as tool calls. See `docs/architecture/abstention-guard.md`.

Deviations from this spec:

*   **No OpenTelemetry.** Phase I asks for OTEL instrumentation; this slice captures faults without it. Phase I stays open. Token counts and latency are recorded only when the caller supplies them.
*   **Not wired into the pipeline.** The task allowed an optional, default-off call site in `core/langgraph/core_graph.py`, but that file is an unused stub; the live graph is `core_graph_v2.py` (`COREGraph.conversation_node`, next to `record_pipeline_run`). A hook there is a follow-up.
*   **Missing latency fails `max_latency_ms`** rather than passing silently.
*   **Not built:** trajectory matching, confidence scoring, the asynchronous analyzer and the `monitoring` service (§4, Phase III).
*   `app/eval` (verdict-drift harness for `evaluation_service`) is a separate, older package and is untouched.
