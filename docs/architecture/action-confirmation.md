# Side-effect tiers: confirm writes, latch on failed dependencies

**Status:** Implemented (default on) · **Date:** 2026-11-04
**Related:** [`abstention-guard.md`](abstention-guard.md) (refusal latch, claim verification),
[`comprehension-grounded-gate.md`](comprehension-grounded-gate.md) (capability registry),
[`../evaluation_spec.md`](../evaluation_spec.md).

## Problem

The capability registry classifies every capability as `none | read | network | write`, but
nothing read the class, and `PlanStep.requires_hitl` was only counted by the cost estimate. A plan
step that wrote a file ran exactly like one that read it.

The refusal latch stops the run when the agent declines, but a plain failure did not stop
anything that depended on it:

- A retry (`retry_step`) ran the retried step with no dependency check, so a write whose
  prerequisite had failed could run on the retry.
- An LLM step whose call errored (or came back empty) was reported as a *success* with the error
  text as its output, so its dependents, writes included, ran on an answer that never came.
- Steps caught in a dependency cycle all ran together in one wave.
- Steps skipped for a failed dependency were not recorded anywhere the response or the fault
  report could see.

## Part 1: confirm tier

Code: `backend/app/core/agents/action_guard.py` (`ActionGuard`), `side_effect_for_step` in
`backend/app/core/registry/builtin.py`, `ReasoningAgent._run_step`.

- **Classification.** Each step the dispatcher would run is resolved to its capability id
  (`capability_id_for_step`) and the registry's `side_effects`. Params can raise the class: a
  `database` call with a method other than GET is a `write`, except the three knowledge-base
  search endpoints. A step that maps to no registry entry counts as a `write` (fail closed).
- **What is held.** A `write` step, and any step the plan marked `requires_hitl`. Reads and network
  fetches run as before. LLM steps and dry runs are never held: they dispatch nothing. (An LLM
  step that *says* it wrote something is caught by the claim verifier instead.)
- **What holding means.** The step is not handed to the dispatcher. It comes back as a failed
  `StepResult` (`error="Awaiting confirmation: <capability id>"`, `dispatched_tool=None`), the
  step is marked `skipped`, and a `PendingConfirmation` (capability id, side effect, reason,
  target path/url/endpoint) goes into `COREState.pending_confirmations`. Its dependents wait too
  (`BlockedAction.kind="awaiting_confirmation"`). Independent steps keep running.
- **No retry or re-plan around it.** `evaluation_node` turns `retry_step`, `revise_plan` and
  `finalize` into `ask_user` while a confirmation is pending, and `route_from_evaluation` sends
  retry/revise to `conversation` as a second check. Retrying would only hold the step again;
  re-planning could route around the user.
- **Response.** "I need your go-ahead before changing anything", the held steps with their
  targets, the steps that waited on them, and which capability ids to approve.
- **How the user approves.** By sending the request again (a new run) with
  `config["approved_capabilities"]`, a list of capability ids such as
  `["file_operations.write"]`, or `["*"]` for everything. `POST /engine/run` passes `config`
  through. Approval is per capability id, not per step: step ids are regenerated with each plan.
- **Plan reporting.** After grounding, `orchestration_node` sets `requires_hitl=True` on every
  write step (`mark_confirmation_steps`), so the plan and its cost estimate show the checkpoints.
  The guard holds write steps whether or not the flag is set.
- **Switch.** `CORE_CONFIRM_WRITES=0` turns the confirm tier off, including the
  `requires_hitl` holds (for ablation runs).

**Trust boundary.** `approved_capabilities` comes from the caller of the run API. Anyone who can
start a run can pre-approve writes, so the UI should only send it after the user has said yes.

## Part 2: dependency latch

Code: `ReasoningAgent._execute_with_parallelism` / `_dependency_block`, `execute_plan`.

- **Rule.** A step runs only if every dependency that is in the plan has a *successful* result.
  Anything else (failed, refused, blocked, held, or no result at all) keeps it from running.
  Dependency ids that are not in the plan are ignored as before: planners emit placeholders like
  `"step_1_id"`, and grounding already cascade-drops dependents of steps it removes.
- **Retries are checked too.** `reasoning_node` passes the run's earlier results as
  `prior_results`; the retried step's dependencies are checked against them.
- **Retries resume dependents.** When the retried step succeeds, its transitive dependents whose
  last result was a dependency block are re-run (and checked again). Other failures are left to
  the evaluator. The node replaces the results and block records of every re-run step and keeps
  `step_results` in plan order.
- **Cycles do not run.** Steps in a dependency cycle are blocked, since none of them can have its
  prerequisite done first.
- **LLM errors fail the step.** `_execute_with_llm` raises on an API error or an empty answer, so
  the step fails and its dependents are blocked.
- **Recording.** Each block is a `BlockedAction` with `kind="dependency_failed"` (or
  `"awaiting_confirmation"`) and `cause_step_id`. The response lists them under "⚠️ Not run". The
  refusal notice lists only `kind="refusal"` blocks.

There is no switch: running a step without its prerequisite is never the intended behaviour.

## Evaluation layer

- `fault_reasons` does not count a held step, or a step waiting on one, as `step_failure`: holding
  a write is the guard working. Any other failure in the same run still counts.
- Fault reports add `pending_confirmations` to `metadata`; `blocked_actions` entries now carry
  `kind` and `cause_step_id`.

## Limits

- Classification is only as good as the registry. MCP capabilities will need honest
  `side_effects` when they register; unregistered ones are held as writes.
- `requires_hitl` comes from the planner LLM. A planner that sets it on every step holds every
  dispatchable step; the user can approve with `["*"]`.
- Approval lasts one run. Nothing remembers it across turns, so each request that writes needs
  its own `approved_capabilities`.
- The streaming endpoint does not emit a dedicated "confirmation needed" event yet; the UI sees
  it in the final response and in `pending_confirmations` on the run state.
