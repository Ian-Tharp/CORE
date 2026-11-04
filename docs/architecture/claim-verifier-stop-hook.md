# Claim verifier as a Claude Code `Stop` hook

**Status:** Implemented (opt-in per clone) · **Date:** 2026-11-04
**Related:** [`abstention-guard.md`](abstention-guard.md) (Part 2: the verifier inside the
C→O→R→E loop), [`../evaluation_spec.md`](../evaluation_spec.md) (fault capture).

## Why

`app/evaluation/claims.py` already recognises the claims coding agents fake: "I wrote
`config.yaml`", "I committed and pushed the fix", "I ran the tests and they pass". Inside CORE it
checks the reasoning agent's output against the dispatcher's trace. The same check works on a
Claude Code session: the transcript holds every tool call and its result, so a reply that claims
work with no tool call behind it can be caught before the turn ends.

## How it works

Code: `backend/app/evaluation/claude_code_hook.py`. Tests: `backend/tests/evaluation/test_claude_code_hook.py`,
fixtures in `backend/tests/fixtures/claude_code/`.

1. Claude Code runs the hook when the agent is about to stop, with a JSON payload on stdin
   (`transcript_path`, `session_id`, `stop_hook_active`, ...).
2. **Trace.** Every `tool_use` in the transcript is paired with its `tool_result` and mapped to the
   `ToolCallRecord` names `claims.py` already knows:

   | Claude Code tool | Record |
   |---|---|
   | `Write`, `Edit`, `MultiEdit`, `NotebookEdit` | `file_operations` write (path) |
   | `Read`, `Glob`, `Grep`, `LS` | `file_operations` read / list / search (Glob/Grep output becomes `entries`) |
   | `WebFetch`, `WebSearch` | `web_research` |
   | `Bash` | a `bash` record, plus `git <subcommand>`, `file_operations` (rm, mv, cp, touch, `sed -i`, `>` ...) and `database` (psql, sqlite3 ...) records parsed from the command |
   | GitHub MCP `push_files`, `create_or_update_file`, `delete_file` | `git` commit + push, file writes |

   A result with `is_error`, a Bash result starting `Exit code N`, or a call with no result counts
   as failed. Bash is tokenised with `shlex`, so a heredoc commit message that mentions
   "git push" is not taken for a push. The whole session counts, not only the current turn.
3. **Claims.** The agent's text since the last human prompt (main thread only, fenced code
   removed). If the payload carries `last_assistant_message` and the transcript does not have it
   yet, it is added.
4. **Verify.** `verify_texts` from `claims.py`, then two re-checks that only make sense when a
   shell exists:
   - **Commands.** `claims.py` never supports "I ran ..." (CORE has no shell tool). Here a claim
     about tests needs a successful test-runner command (`pytest`, `npm test`, `go test`, ...);
     lint/format and build claims need a matching command; a backticked command needs that
     program; anything else needs any successful Bash call. A run that exists but failed is
     reported as "last attempt failed".
   - **Git writes.** "I committed and pushed" needs both a successful `commit` and a
     successful `push` (`claims.py` accepts any git write). "I ran git push" needs `push`.
5. **Decide.**
   - Clean: no output, the agent stops.
   - Unsupported claims: `{"decision": "block", "reason": ...}`. Claude Code feeds the reason
     back to the agent, which must do the work or correct the reply.
   - Already blocked once for this stop (`stop_hook_active`): `{"systemMessage": ...}` only, so
     the hook cannot loop.
   - Any internal error: exit 0, nothing printed (fails open).

## Enable it

Add to `.claude/settings.json` (shared) or `.claude/settings.local.json` (just you):

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$CLAUDE_PROJECT_DIR/backend\" && uv run --quiet python -m app.evaluation.claude_code_hook",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

It runs in well under a second once `uv sync` has been done. The hook only imports the
evaluation package (pydantic, no DB or network).

Environment:

| Variable | Effect |
|---|---|
| `CORE_CLAIM_HOOK_MODE` | `block` (default), `warn` (message to the user, no block), `off` |
| `CORE_EVAL_CAPTURE_FAULTS` | When on, each unsupported-claim stop is appended to `faults.jsonl` as a `FaultReport` (`metadata.source = "claude_code_stop_hook"`, `run_id` = session id, tool calls = the trace) |
| `CORE_EVAL_FAULTS_DIR` | Where `faults.jsonl` goes (see `app/config/evaluation.py`) |

## Audit a transcript by hand

```bash
cd backend
uv run python -m app.evaluation.claude_code_hook --transcript ~/.claude/projects/<project>/<session>.jsonl
```

Prints the claim report as JSON; exit code 1 when a claim is unsupported.

## Limits

- **Recall is `claims.py`'s.** Only first-person, completed-action phrasings it recognises are
  checked. A clean result means no recognised claim is unsupported, not that the reply is true.
- **Action, not outcome.** "I ran the tests and they pass" is supported by a successful test
  command; the hook does not read the output to check the pass count. A failed run (non-zero
  exit) does not support it.
- **Work it cannot see.** Subagent transcripts are separate files, so a claim about work a
  subagent did is flagged unless the call also appears in the main transcript. Work done by the
  user or in another session is also invisible. The block reason tells the agent to say so
  instead of re-doing it.
- **Shell parsing is shallow.** Aliases, scripts (`./deploy.sh`), `eval` and `bash -c '...'`
  are not looked into; `cd` is not tracked, so paths match by suffix (as in `claims.py`).
