# Abstention guard and claim verification

**Status:** Implemented (default on) · **Date:** 2026-11-03
**Related:** [`comprehension-grounded-gate.md`](comprehension-grounded-gate.md) (front verifier),
[`action-confirmation.md`](action-confirmation.md) (confirm tier for writes, dependency latch),
[`../evaluation_spec.md`](../evaluation_spec.md) (production-track guardrails).
**Motivation:** AgentAbstain (arXiv 2607.10059, 2026), on agents whose abstentions do not hold
once execution starts.

## Problem

Two failure modes the C→O→R→E loop allowed:

1. **Acting after a refusal.** A reasoning step's LLM output could decline ("I can't do that")
   and the pipeline kept going: later tool steps still ran, evaluation could send the run back
   to `retry_step` on the declined step, or `revise_plan` could plan around it.
2. **Claiming actions that never happened.** An LLM step could say "I wrote `config.yaml`" or
   "I committed the fix" with no tool call behind it. The fault-capture trace made this worse:
   it listed every *planned* tool step as a tool call, even when the LLM answered the step
   instead of the dispatcher.

## Part 1: block actions after an unresolved refusal

Code: `backend/app/core/agents/refusal_guard.py`, `ReasoningAgent._run_step`,
`COREGraph.reasoning_node` / `route_from_evaluation`.

- **What latches.** (a) The comprehension gate choosing `clarify` or `refuse`
  (`COREState.refusal.source == "gate"`). (b) An LLM-executed step whose output matches
  `detect_refusal` (`source == "reasoning"`). Dispatched tool output is never scanned: a fetched
  page saying "I can't" is data, not the agent declining.
- **What it blocks.** Every step that has not started yet: tool steps and LLM steps, dependent
  or not. Blocked steps are never handed to the dispatcher; they come back as failed
  `StepResult`s, the step is marked `skipped`, and each one is recorded in
  `COREState.blocked_actions` with `kind="refusal"`. (Steps whose dependency failed are recorded
  there too, with `kind="dependency_failed"`; see `action-confirmation.md`.) The refusing step itself is failed (`error="Refused: …"`) and keeps
  the model's words in its outputs.
- **No retry or re-plan around it.** `route_from_evaluation` sends `retry_step` and `revise_plan`
  to `conversation` while `state.refusal` is set.
- **What resolves it.** Only the user, by sending a new turn (a new run, so a fresh state).
  Nothing inside the run clears the latch.
- **Response.** The conversation node opens with "I stopped before finishing: a step declined
  (…)", lists the steps it did not run, then shows any partial output.
- **Switch.** `CORE_BLOCK_AFTER_REFUSAL=0` turns the guard off (for ablation runs).

The latch is created per run in `reasoning_node` and passed down; it never lives on the agent,
because the graph and its agents are shared between concurrent runs.

**Limit:** steps that started in the same parallel wave as the refusing step were already
running when the latch tripped and are not stopped. Set `CORE_MAX_PARALLEL_STEPS=1` if that
matters more than speed.

## Part 2: verify claims against the tool-call trace

Code: `backend/app/evaluation/claims.py`, `COREGraph._verify_claims` (runs at the end of
`evaluation_node`).

- **Trace.** `StepResult.dispatched_tool` is set only when the dispatcher actually ran a tool.
  `trace_from_state` builds the trace from those results alone (as `ToolCallRecord`s), so a
  planned tool is not evidence that it ran.
- **Claims.** `extract_claims` finds first-person, completed-action statements in LLM step
  outputs: file write, file read/list/search, git read, git write, web fetch, database, and
  "ran the tests / a command". Plans ("I will…"), conditionals ("if I wrote…") and
  objects that are not files ("I created a plan") are skipped.
- **Support.** A claim is supported only by a successful trace entry of the right tool and
  action, and, when the claim names a path or URL, a matching target. Git writes and shell
  commands are never supported today: the builtin git tool is read-only and there is no shell
  tool.
- **Effect.** Unsupported claims go into `COREState.claim_report`, add a warning, set
  `meets_requirements=False`, cap evaluation confidence at 0.5 and append a note to the
  feedback. `next_action` is left alone, so this cannot cause a loop. The response ends with
  "⚠️ Not verified: no tool call backs these statements:" and the list.
- **Evaluation layer.** `state_failed` treats unsupported claims as a fault; the fault report
  records `refusal`, `blocked_actions` and `unsupported_claims` in `metadata` and no longer lists
  never-dispatched steps as tool calls. The `claims_supported` Golden Set evaluator applies the
  same check to `RunOutput.output` against `RunOutput.tool_calls`.

**Limits.** Extraction is regex-based with deliberately modest recall: a clean report means no
*recognised* claim is unsupported, not that every sentence is true. Refusal detection is also
pattern-based and errs toward latching (e.g. "I cannot complete this step because the file is
missing" latches). Both are cheap deterministic first passes in the spirit of the production
track; an LLM judge can sit behind the same interfaces later.
