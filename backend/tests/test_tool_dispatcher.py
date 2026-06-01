"""
Tests for ToolDispatcher and individual tool handlers.

Sentinel-value contract:
- Remove FileOperationsTool._read → test_read_returns_file_content fails
- Remove GitTool.execute → test_git_status_calls_subprocess fails
- Remove WebResearchTool.execute → test_web_research_calls_httpx fails
- Remove ToolDispatcher dispatch → test_dispatcher_calls_handler fails
- Remove safety path check → test_path_traversal_blocked fails
- Remove git allowlist → test_git_disallowed_action_returns_error fails
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.tools.dispatcher import (
    FileOperationsTool,
    GitTool,
    WebResearchTool,
    DatabaseTool,
    ToolDispatcher,
    ToolSafetyError,
    _GIT_SAFE_ACTIONS,
)


# ---------------------------------------------------------------------------
# FileOperationsTool
# ---------------------------------------------------------------------------


class TestFileOperationsTool:
    @pytest.fixture
    def workspace(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def tool(self, workspace):
        return FileOperationsTool(workspace_dir=str(workspace))

    # --- read ---

    def test_read_returns_file_content(self, tool, workspace):
        """
        SENTINEL TEST — read action must return the actual file contents.
        Return hardcoded string → 'SENTINEL_CONTENT' not in result → fails.
        """
        (workspace / "hello.txt").write_text("SENTINEL_CONTENT")
        result = tool.execute({"action": "read", "path": "hello.txt"})
        assert result["status"] == "success"
        assert "SENTINEL_CONTENT" in result["result"]

    def test_read_missing_file_returns_error(self, tool):
        result = tool.execute({"action": "read", "path": "nonexistent.txt"})
        assert result["status"] == "error"

    def test_read_reports_path_in_output(self, tool, workspace):
        (workspace / "report.md").write_text("# Title")
        result = tool.execute({"action": "read", "path": "report.md"})
        assert "report.md" in result.get("path", "")

    # --- write ---

    def test_write_creates_file(self, tool, workspace):
        """
        SENTINEL TEST — write action must create the file with the given content.
        Skip write → file does not exist → assertion fails.
        """
        result = tool.execute(
            {"action": "write", "path": "output.txt", "content": "WRITTEN_DATA"}
        )
        assert result["status"] == "success"
        assert (workspace / "output.txt").read_text() == "WRITTEN_DATA"

    def test_write_reports_bytes_written(self, tool):
        result = tool.execute({"action": "write", "path": "x.txt", "content": "hello"})
        assert result["bytes_written"] == 5

    def test_write_creates_parent_dirs(self, tool, workspace):
        result = tool.execute(
            {"action": "write", "path": "a/b/c.txt", "content": "deep"}
        )
        assert result["status"] == "success"
        assert (workspace / "a" / "b" / "c.txt").exists()

    # --- list ---

    def test_list_returns_entries(self, tool, workspace):
        """
        SENTINEL TEST — list action must enumerate actual workspace contents.
        Return empty list → SENTINEL_FILE not found → fails.
        """
        (workspace / "SENTINEL_FILE.py").write_text("")
        (workspace / "other.txt").write_text("")
        result = tool.execute({"action": "list", "path": "."})
        assert result["status"] == "success"
        assert any("SENTINEL_FILE.py" in e for e in result["entries"])

    def test_list_missing_dir_returns_error(self, tool):
        result = tool.execute({"action": "list", "path": "no_such_dir"})
        assert result["status"] == "error"

    # --- search ---

    def test_search_finds_matching_files(self, tool, workspace):
        """
        SENTINEL TEST — search by pattern must find the file with that name.
        Return all files → pattern not checked → SENTINEL_UNIQUE fails.
        """
        (workspace / "SENTINEL_UNIQUE_MODULE.py").write_text("")
        (workspace / "other.ts").write_text("")
        result = tool.execute(
            {"action": "search", "pattern": "SENTINEL_UNIQUE", "path": "."}
        )
        assert result["status"] == "success"
        assert result["count"] >= 1
        assert any("SENTINEL_UNIQUE_MODULE" in m for m in result["matches"])

    def test_search_returns_empty_for_no_matches(self, tool):
        result = tool.execute(
            {"action": "search", "pattern": "ZZZNOMATCH_XYZ", "path": "."}
        )
        assert result["count"] == 0

    # --- safety ---

    def test_path_traversal_blocked(self, tool):
        """
        SENTINEL TEST — '../../../etc/passwd' must be rejected.
        Remove safe_path check → real file read → test fails.
        """
        result = tool.execute({"action": "read", "path": "../../../etc/passwd"})
        assert result["status"] == "error"

    def test_unknown_action_returns_error(self, tool):
        result = tool.execute({"action": "delete_everything"})
        assert result["status"] == "error"

    # --- artifacts ---

    def test_get_artifacts_returns_path(self, tool):
        result = tool.execute({"action": "write", "path": "artifact.py", "content": ""})
        artifacts = tool.get_artifacts(result)
        assert "artifact.py" in artifacts


# ---------------------------------------------------------------------------
# GitTool
# ---------------------------------------------------------------------------


class TestGitTool:
    @pytest.fixture
    def git_repo(self, tmp_path):
        """Initialise a minimal bare git repo for testing."""
        import subprocess

        subprocess.run(["git", "init", str(tmp_path)], capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(tmp_path),
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(tmp_path),
            capture_output=True,
        )
        (tmp_path / "README.md").write_text("# Test")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=str(tmp_path), capture_output=True
        )
        return tmp_path

    @pytest.fixture
    def tool(self, git_repo):
        return GitTool(workspace_dir=str(git_repo))

    def test_git_status_calls_subprocess(self, tool):
        """
        SENTINEL TEST — git status must call subprocess.run with git commands.
        Mock subprocess to raise → result is error → NOT what a real call returns.
        Remove the subprocess.run call → returncode never set → test fails.
        """
        result = tool.execute({"action": "status"})
        assert result["status"] == "success"
        assert "returncode" in result
        assert result["returncode"] == 0

    def test_git_log_returns_commits(self, tool):
        result = tool.execute({"action": "log", "--oneline": True})
        # log action just runs: git log []
        assert result["status"] == "success"

    def test_git_disallowed_action_returns_error(self):
        """
        SENTINEL TEST — actions like 'commit', 'push', 'reset' must be rejected.
        Remove allowlist → arbitrary git command runs → test fails.
        """
        tool = GitTool(workspace_dir=os.getcwd())
        result = tool.execute({"action": "commit"})
        assert result["status"] == "error"
        assert (
            "allow-list" in result["result"].lower()
            or "not in" in result["result"].lower()
        )

    def test_git_push_blocked(self):
        tool = GitTool(workspace_dir=os.getcwd())
        result = tool.execute({"action": "push"})
        assert result["status"] == "error"

    def test_git_reset_blocked(self):
        tool = GitTool(workspace_dir=os.getcwd())
        result = tool.execute({"action": "reset"})
        assert result["status"] == "error"

    def test_allowlist_contains_safe_commands(self):
        """All expected safe commands are in the allow-list."""
        for cmd in ("status", "diff", "log", "show", "branch", "ls-files"):
            assert cmd in _GIT_SAFE_ACTIONS

    def test_git_not_found_returns_error(self):
        """If git is missing, return error dict (not exception)."""
        tool = GitTool(workspace_dir=os.getcwd())
        with patch(
            "app.core.tools.dispatcher.subprocess.run",
            side_effect=FileNotFoundError("git not found"),
        ):
            result = tool.execute({"action": "status"})
        assert result["status"] == "error"
        assert "git" in result["result"].lower()


# ---------------------------------------------------------------------------
# WebResearchTool
# ---------------------------------------------------------------------------


class TestWebResearchTool:
    @pytest.fixture
    def tool(self):
        return WebResearchTool()

    def test_web_research_calls_httpx(self, tool):
        """
        SENTINEL TEST — web_research must call httpx.Client.get with the URL.
        Replace with hardcoded string → URL not in response → test fails.
        """
        mock_resp = MagicMock()
        mock_resp.content = b"SENTINEL_RESPONSE_BODY"
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com"
        mock_resp.headers = {"content-type": "text/html"}

        with patch("app.core.tools.dispatcher.httpx.Client") as mock_client_cls:
            mock_ctx = MagicMock()
            mock_ctx.__enter__.return_value.get.return_value = mock_resp
            mock_ctx.__exit__.return_value = False
            mock_client_cls.return_value = mock_ctx

            result = tool.execute({"url": "https://example.com"})

        assert result["status"] == "success"
        assert "SENTINEL_RESPONSE_BODY" in result["result"]

    def test_missing_url_returns_error(self, tool):
        result = tool.execute({})
        assert result["status"] == "error"

    def test_non_http_url_blocked(self, tool):
        result = tool.execute({"url": "ftp://evil.com/payload"})
        assert result["status"] == "error"

    def test_timeout_returns_error(self, tool):
        import httpx as _httpx

        with patch("app.core.tools.dispatcher.httpx.Client") as mock_cls:
            ctx = MagicMock()
            ctx.__enter__.return_value.get.side_effect = _httpx.TimeoutException(
                "timeout"
            )
            ctx.__exit__.return_value = False
            mock_cls.return_value = ctx
            result = tool.execute({"url": "https://slow.example.com"})
        assert result["status"] == "error"
        assert "timed out" in result["result"].lower()

    def test_http_error_response_sets_status_error(self, tool):
        mock_resp = MagicMock()
        mock_resp.content = b"Not Found"
        mock_resp.status_code = 404
        mock_resp.url = "https://example.com/missing"
        mock_resp.headers = {}

        with patch("app.core.tools.dispatcher.httpx.Client") as mock_cls:
            ctx = MagicMock()
            ctx.__enter__.return_value.get.return_value = mock_resp
            ctx.__exit__.return_value = False
            mock_cls.return_value = ctx
            result = tool.execute({"url": "https://example.com/missing"})

        assert result["status"] == "error"
        assert result["status_code"] == 404


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------


class TestToolDispatcher:
    def test_dispatcher_calls_handler(self, tmp_path):
        """
        SENTINEL TEST — dispatcher.dispatch('file_operations', ...) must call
        the FileOperationsTool handler, not a simulation.
        Swap handler for stub → SENTINEL_VALUE not in result → test fails.
        """
        (tmp_path / "sentinel.txt").write_text("SENTINEL_VALUE_IN_FILE")
        dispatcher = ToolDispatcher(workspace_dir=str(tmp_path))
        result = dispatcher.dispatch(
            "file_operations", {"action": "read", "path": "sentinel.txt"}
        )
        assert "SENTINEL_VALUE_IN_FILE" in result["result"]

    def test_unknown_tool_returns_error(self, tmp_path):
        """
        SENTINEL TEST — unknown tool names must return error dict, not raise.
        Remove unknown-tool guard → KeyError propagates → test fails differently.
        """
        dispatcher = ToolDispatcher(workspace_dir=str(tmp_path))
        result = dispatcher.dispatch("nuclear_option", {})
        assert result["status"] == "error"
        assert "nuclear_option" in result["result"]

    def test_available_tools_contains_four_handlers(self, tmp_path):
        dispatcher = ToolDispatcher(workspace_dir=str(tmp_path))
        tools = dispatcher.available_tools
        for expected in ("file_operations", "git", "web_research", "database"):
            assert expected in tools

    def test_get_artifacts_delegates_to_handler(self, tmp_path):
        """get_artifacts must return paths from the handler, not an empty list."""
        dispatcher = ToolDispatcher(workspace_dir=str(tmp_path))
        # write a file then check artifacts
        write_out = dispatcher.dispatch(
            "file_operations", {"action": "write", "path": "art.py", "content": "x"}
        )
        arts = dispatcher.get_artifacts("file_operations", write_out)
        assert "art.py" in arts

    def test_git_dispatch_calls_subprocess(self, tmp_path):
        """
        SENTINEL TEST — dispatching 'git' tool must call subprocess.run.
        Remove subprocess call → mock never called → test fails.
        """
        dispatcher = ToolDispatcher(workspace_dir=str(tmp_path))
        with patch("app.core.tools.dispatcher.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="on branch main\n", stderr=""
            )
            result = dispatcher.dispatch("git", {"action": "status"})
        mock_run.assert_called_once()
        call_cmd = mock_run.call_args[0][0]
        assert call_cmd[0] == "git"
        assert call_cmd[1] == "status"

    def test_web_research_dispatch_calls_httpx(self, tmp_path):
        """Dispatching 'web_research' must trigger an HTTP request."""
        dispatcher = ToolDispatcher(workspace_dir=str(tmp_path))
        mock_resp = MagicMock(
            content=b"page content", status_code=200, url="https://ex.com", headers={}
        )
        with patch("app.core.tools.dispatcher.httpx.Client") as mock_cls:
            ctx = MagicMock()
            ctx.__enter__.return_value.get.return_value = mock_resp
            ctx.__exit__.return_value = False
            mock_cls.return_value = ctx
            result = dispatcher.dispatch("web_research", {"url": "https://ex.com"})
        assert "page content" in result["result"]


# ---------------------------------------------------------------------------
# ReasoningAgent integration — dispatcher is used (not simulation)
# ---------------------------------------------------------------------------


class TestReasoningAgentUsesDispatcher:
    def _make_step(self, tool: str, params: dict = None):
        from app.models.core_state import PlanStep

        return PlanStep(
            name="test-step", description="Test", tool=tool, params=params or {}
        )

    def _make_plan(self, *steps):
        from app.models.core_state import ExecutionPlan

        return ExecutionPlan(goal="test", steps=list(steps))

    def test_dispatcher_called_for_file_operations(self, tmp_path):
        """
        SENTINEL TEST — when a step has tool='file_operations', the plan
        executor must call tool_dispatcher.dispatch(), NOT _simulate_tool_call.
        If simulation is still called → sentinel not returned → test fails.
        """
        from app.core.agents.reasoning_agent import ReasoningAgent

        (tmp_path / "target.txt").write_text("SENTINEL_DISPATCH_RESULT")

        agent = ReasoningAgent()
        agent.tool_dispatcher = ToolDispatcher(workspace_dir=str(tmp_path))

        step = self._make_step(
            "file_operations", {"action": "read", "path": "target.txt"}
        )
        plan = self._make_plan(step)

        results = agent.execute_plan(plan)

        assert len(results) == 1
        assert results[0].status == "success"
        assert "SENTINEL_DISPATCH_RESULT" in results[0].outputs.get("result", "")

    def test_dispatcher_called_for_git(self, tmp_path):
        """Reasoning agent must route git steps through the dispatcher."""
        from app.core.agents.reasoning_agent import ReasoningAgent

        agent = ReasoningAgent()
        mock_dispatcher = MagicMock()
        mock_dispatcher.available_tools = ["git"]
        mock_dispatcher.dispatch.return_value = {
            "result": "SENTINEL_GIT_OUT",
            "status": "success",
        }
        mock_dispatcher.get_artifacts.return_value = []
        agent.tool_dispatcher = mock_dispatcher

        step = self._make_step("git", {"action": "status"})
        plan = self._make_plan(step)
        results = agent.execute_plan(plan)

        mock_dispatcher.dispatch.assert_called_once_with("git", {"action": "status"})
        assert "SENTINEL_GIT_OUT" in results[0].outputs.get("result", "")

    def test_tool_error_produces_failure_result(self, tmp_path):
        """When a tool returns status=error, the step result must be failure."""
        from app.core.agents.reasoning_agent import ReasoningAgent

        agent = ReasoningAgent()
        mock_dispatcher = MagicMock()
        mock_dispatcher.available_tools = ["file_operations"]
        mock_dispatcher.dispatch.return_value = {
            "result": "Permission denied",
            "status": "error",
        }
        mock_dispatcher.get_artifacts.return_value = []
        agent.tool_dispatcher = mock_dispatcher

        step = self._make_step(
            "file_operations", {"action": "read", "path": "secret.txt"}
        )
        plan = self._make_plan(step)
        results = agent.execute_plan(plan)

        assert results[0].status == "failure"
