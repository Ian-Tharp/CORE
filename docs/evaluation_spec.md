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
