# CORE Cognitive Engine — Architecture (Draft)

This document encodes the first-draft cognitive engine from the diagrams into actionable implementation guidance.

## Pipeline
Start → Input → Comprehension → Orchestration → Reasoning → Evaluation → Output

- Dynamic planning: Orchestration updates plan based on Reasoning/Evaluation feedback
- Max Retry per stage; HITL checkpoints
- Step outputs logged to an Intelligence Layer for ranking/vectorization

## Agent Responsibilities
- Comprehension: consult KB; decide tool usage; clarify capability gaps
- Orchestration: synthesize plan (steps, deps, retry policy, HITL flags)
- Reasoning: execute steps, call tools, produce intermediate artifacts
- Evaluation: assess outputs; trigger retries/plan updates or finalize

## Knowledge Base (KB)
- RAG over system docs, tool inventories, OpenAPI specs
- Recent-query cache: canonicalize routes/tool-calls (including request bodies)

## Interfaces (initial)
- POST `/engine/run` { input }
- GET `/engine/runs/{id}` (status)
- GET `/engine/runs/{id}/stream` (SSE)

## Data Schemas (sketch)
- Plan: id, steps[{id, name, tool, params, hitl, retry{max,backoff}}]
- Step: id, inputs, logs, outputs, status
- Eval: policy id, rubric, pass/fail, notes

## Intelligence Layer
- Log every step result; normalize tool outputs → function-call records
- Rank answers; store embeddings for accelerated recall

## Implementation Phases
1) Contracts & skeleton (routes, SSE, types)
2) KB + Tool registry
3) Planner + dynamic updates
4) Executor + Evaluation policies
5) Intelligence Layer logging and ranking
6) UI run viewer + HITL
