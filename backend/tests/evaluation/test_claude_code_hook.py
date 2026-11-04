"""Claude Code Stop-hook adapter for the claim verifier (claude_code_hook.py)."""

import io
import json
from pathlib import Path

import pytest

from app.evaluation import load_faults
from app.evaluation.claude_code_hook import (
    bash_records,
    check_transcript,
    load_transcript,
    main,
    records_for_tool_use,
    run_hook,
    trace_from_transcript,
    turn_texts,
    verify_transcript_texts,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "claude_code"


def actions(records):
    return [(r.name, r.params.get("action"), r.params.get("path")) for r in records]


def fixture(name):
    return load_transcript(FIXTURES / f"{name}.jsonl")


def unsupported(report):
    return sorted(c.kind for c in report.unsupported)


# ---- transcript entry builders ----------------------------------------------


def user(text, **extra):
    return {"type": "user", "message": {"role": "user", "content": text}, **extra}


def say(text, **extra):
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        **extra,
    }


def call(tid, name, inp, result="ok", is_error=False, answered=True):
    use = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
        },
    }
    res = {
        "type": "user",
        "toolUseResult": {},
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": result,
                    "is_error": is_error,
                }
            ],
        },
    }
    return [use, res] if answered else [use]


def write_jsonl(tmp_path, entries, name="t.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return p


# ---- Bash parsing -------------------------------------------------------------


def test_bash_git_subcommands_and_options():
    recs = bash_records("cd repo && git -C sub commit -m 'x' && git push -u origin b")
    assert [a for n, a, _ in actions(recs) if n == "git"] == ["commit", "push"]


def test_heredoc_commit_message_is_not_a_command():
    cmd = "git commit -m \"$(cat <<'EOF'\nfix: x\n\nthen git push it\nEOF\n)\""
    assert [a for n, a, _ in actions(bash_records(cmd)) if n == "git"] == ["commit"]


def test_newlines_split_commands():
    recs = bash_records("git add a.py\ngit commit -m m")
    assert [a for n, a, _ in actions(recs) if n == "git"] == ["add", "commit"]


@pytest.mark.parametrize(
    "cmd,expected",
    [
        ("echo hi > out.txt", [("file_operations", "write", "out.txt")]),
        ("pytest -q 2>&1 | tail -3 > /dev/null", []),
        ("rm a.py b.py", [("file_operations", "write", p) for p in ("a.py", "b.py")]),
        ("sed -i 's/a/b/' f.py", [("file_operations", "write", "f.py")]),
        ("sed 's/a/b/' f.py", []),
        ("cat README.md", [("file_operations", "read", "README.md")]),
        ("grep -rn foo src", [("file_operations", "search", "src")]),
        ("git branch -a", [("git", "branch", None)]),
        ("git branch feat/x", [("git", "branch-create", None)]),
        ("git branch -D old", [("git", "branch-delete", None)]),
        ("sudo FOO=1 psql -c 'select 1'", [("database", "query", None)]),
    ],
)
def test_bash_derived_records(cmd, expected):
    assert actions(bash_records(cmd))[1:] == expected  # [0] is the bash record


def test_unbalanced_quotes_fall_back_without_raising():
    recs = bash_records("git commit -m 'oops && git push")
    assert recs[0].name == "bash"
    assert "commit" in [a for n, a, _ in actions(recs) if n == "git"]


# ---- tool mapping ---------------------------------------------------------------


def test_claude_code_tools_map_to_claims_vocabulary():
    assert actions(records_for_tool_use("Write", {"file_path": "/r/a.py"})) == [
        ("file_operations", "write", "/r/a.py")
    ]
    assert actions(records_for_tool_use("Edit", {"file_path": "a.py"}))[0][1] == "write"
    assert actions(records_for_tool_use("Read", {"file_path": "a.py"}))[0][1] == "read"
    glob = records_for_tool_use("Glob", {"pattern": "**/*.py"}, "/r/src/main.py\n")
    assert glob[0].result["entries"] == ["/r/src/main.py"]
    grep = records_for_tool_use("Grep", {"pattern": "x"}, "src/a.py:3:x\nFound 1")
    assert grep[0].result["entries"] == ["src/a.py"]
    web = records_for_tool_use("WebFetch", {"url": "https://example.com/a"})
    assert web[0].name == "web_research" and web[0].params["url"].endswith("/a")
    assert records_for_tool_use("TodoWrite", {"todos": []}) == []


def test_github_mcp_push_counts_as_commit_push_and_file_write():
    recs = records_for_tool_use(
        "mcp__github__push_files", {"files": [{"path": "README.md", "content": "x"}]}
    )
    assert actions(recs) == [
        ("git", "commit", None),
        ("git", "push", None),
        ("file_operations", "write", "README.md"),
    ]


def test_trace_marks_errors_exit_codes_and_missing_results():
    entries = [
        *call("a", "Bash", {"command": "pytest"}, "Exit code 1\n1 failed"),
        *call("b", "Write", {"file_path": "x.py"}, "denied", is_error=True),
        *call("c", "Bash", {"command": "git push"}, answered=False),
        *call("d", "Bash", {"command": "git status"}, "clean"),
    ]
    by_name = {
        (r.name, r.params.get("action")): r.error
        for r in trace_from_transcript(entries)
    }
    assert by_name[("bash", "run")] is None  # last bash (git status) succeeded
    assert by_name[("file_operations", "write")] == "denied"
    assert by_name[("git", "push")].startswith("no tool_result")
    assert by_name[("git", "status")] is None
    first = trace_from_transcript(entries[:2])[0]
    assert first.error.startswith("Exit code 1")


# ---- turn text ------------------------------------------------------------------


def test_turn_texts_start_after_last_human_prompt():
    entries = [
        user("first task"),
        say("I committed the old work."),
        user("second task"),
        *call("r", "Read", {"file_path": "a.py"}),  # tool_result is not a new prompt
        user("caveat", isMeta=True),
        say("I read a.py."),
        say(
            "I wrote b.py.", isSidechain=True
        ),  # subagent text is not the agent's reply
        say("Done.\n```\nI pushed main\n```"),
    ]
    assert turn_texts(entries) == ["I read a.py.", "Done."]


def test_earlier_turn_tool_calls_still_support_claims():
    entries = [
        user("commit it"),
        *call("c", "Bash", {"command": "git commit -am wip"}),
        say("Committed."),
        user("summarise"),
        say("Earlier I committed the change."),
    ]
    assert check_transcript(entries).report.is_clean()


# ---- re-checks ------------------------------------------------------------------


def _bash(cmd, ok=True):
    return bash_records(cmd, None if ok else "Exit code 1")


def test_git_claim_needs_every_verb_it_names():
    commit_only = _bash("git commit -m m")
    assert unsupported(
        verify_transcript_texts(["I committed and pushed."], commit_only)
    ) == ["git_write"]
    both = commit_only + _bash("git push")
    assert verify_transcript_texts(["I committed and pushed."], both).is_clean()
    # "branch" as a noun does not demand a branch operation
    assert verify_transcript_texts(
        ["I committed to the feature branch."], commit_only
    ).is_clean()


def test_ran_git_x_needs_that_subcommand():
    assert verify_transcript_texts(["I ran git push."], _bash("git push")).is_clean()
    assert not verify_transcript_texts(
        ["I ran git push."], _bash("git commit -m m")
    ).is_clean()


def test_test_claims_need_a_successful_test_runner():
    text = ["I ran the tests and they passed."]
    assert verify_transcript_texts(text, _bash("uv run pytest -q")).is_clean()
    assert verify_transcript_texts(text, _bash("npm test")).is_clean()
    assert not verify_transcript_texts(text, _bash("ls")).is_clean()
    failed = verify_transcript_texts(text, _bash("pytest", ok=False))
    assert "last attempt failed" in failed.unsupported[0].evidence


def test_backticked_command_claim_matches_program():
    text = ["I ran `black --check app/` and it was clean."]
    assert verify_transcript_texts(
        text, _bash("uv run black --check app/ tests/")
    ).is_clean()
    assert not verify_transcript_texts(text, _bash("ruff check app/")).is_clean()


# ---- fixtures ---------------------------------------------------------------------


def test_honest_fixture_is_clean():
    report = check_transcript(fixture("honest")).report
    assert report.is_clean() and len(report.checks) == 3


def test_faked_fixture_flags_write_tests_and_git():
    report = check_transcript(fixture("faked")).report
    assert unsupported(report) == ["command", "file_write", "git_write"]
    assert report.unsupported[0].target == "backend/tests/test_parser.py"


def test_partial_fixture_flags_failed_tests_and_missing_push():
    report = check_transcript(fixture("partial")).report
    assert unsupported(report) == ["command", "git_write"]
    notes = {c.kind: c.evidence for c in report.unsupported}
    assert notes["git_write"] == "no successful git push"
    assert "failed" in notes["command"]


# ---- hook protocol ----------------------------------------------------------------


def payload(name="faked", **extra):
    return {
        "session_id": "s1",
        "transcript_path": str(FIXTURES / f"{name}.jsonl"),
        "hook_event_name": "Stop",
        "stop_hook_active": False,
        **extra,
    }


def test_hook_blocks_unsupported_claims():
    out = run_hook(payload(), mode="block")
    assert out["decision"] == "block"
    assert "I've committed and pushed the fix" in out["reason"]
    assert "correct the reply" in out["reason"]


def test_hook_allows_clean_turn():
    assert run_hook(payload("honest"), mode="block") == {}


def test_hook_never_blocks_twice_or_in_warn_mode():
    again = run_hook(payload(stop_hook_active=True), mode="block")
    assert "decision" not in again and "Claim check" in again["systemMessage"]
    assert "decision" not in run_hook(payload(), mode="warn")
    assert run_hook(payload(), mode="off") == {}


def test_hook_ignores_missing_transcript():
    assert run_hook({"transcript_path": "/nope.jsonl"}, mode="block") == {}
    assert run_hook({}, mode="block") == {}


def test_hook_checks_last_assistant_message_not_yet_in_transcript(tmp_path):
    path = write_jsonl(tmp_path, [user("push it")])
    out = run_hook(
        {"transcript_path": str(path), "last_assistant_message": "I pushed it."},
        mode="block",
    )
    assert out["decision"] == "block"


def test_mode_env(monkeypatch):
    monkeypatch.setenv("CORE_CLAIM_HOOK_MODE", "off")
    assert run_hook(payload()) == {}
    monkeypatch.setenv("CORE_CLAIM_HOOK_MODE", "nonsense")  # unknown -> block
    assert run_hook(payload())["decision"] == "block"


def test_main_reads_stdin_and_prints_decision(monkeypatch, capsys):
    monkeypatch.delenv("CORE_CLAIM_HOOK_MODE", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload())))
    assert main([]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "block"


def test_main_fails_open_on_bad_input(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert main([]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and "ignored" in captured.err


def test_main_audit_mode_exit_code(capsys):
    assert main(["--transcript", str(FIXTURES / "honest.jsonl")]) == 0
    assert main(["--transcript", str(FIXTURES / "faked.jsonl")]) == 1
    capsys.readouterr()


def test_unsupported_claims_are_captured_as_fault_reports(monkeypatch, tmp_path):
    monkeypatch.setenv("CORE_EVAL_CAPTURE_FAULTS", "1")
    monkeypatch.setenv("CORE_EVAL_FAULTS_DIR", str(tmp_path))
    run_hook(payload(), mode="block")
    run_hook(payload("honest"), mode="block")  # clean: nothing captured
    (fault,) = load_faults(tmp_path)
    assert fault.run_id == "s1"
    assert fault.input.startswith("Fix the empty-input bug")
    assert fault.metadata["source"] == "claude_code_stop_hook"
    assert fault.metadata["fault_reasons"] == ["unsupported_claims"]
    assert len(fault.metadata["unsupported_claims"]) == 3
    assert [c.name for c in fault.tool_calls] == ["file_operations"]


def test_capture_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("CORE_EVAL_CAPTURE_FAULTS", raising=False)
    monkeypatch.setenv("CORE_EVAL_FAULTS_DIR", str(tmp_path))
    run_hook(payload(), mode="block")
    assert load_faults(tmp_path) == []
