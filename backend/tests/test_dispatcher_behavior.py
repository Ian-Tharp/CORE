"""
Behavior tests for app.core.tools.dispatcher.

Focus: error/edge branches that orchestration + reasoning agents rely on —
every failure must come back as a structured ``{"status": "error"}`` dict,
never as a raised exception. All HTTP and subprocess boundaries are mocked;
filesystem access is confined to pytest's ``tmp_path``.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.core.tools.dispatcher import (
    DatabaseTool,
    FileOperationsTool,
    GitTool,
    ToolDispatcher,
    WebResearchTool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_httpx_client(mock_cls: MagicMock) -> MagicMock:
    """Wire ``httpx.Client(...)`` used as a context manager to a mock client."""
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    mock_cls.return_value.__exit__.return_value = False
    return client


def _resp(status_code: int = 200, text: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# FileOperationsTool
# ---------------------------------------------------------------------------


class TestFileOperationsBehavior:
    def test_write_then_read_roundtrip(self, tmp_path):
        tool = FileOperationsTool(str(tmp_path))
        w = tool.execute({"action": "write", "path": "sub/a.txt", "content": "héllo"})
        assert w["status"] == "success"
        assert w["bytes_written"] == len("héllo".encode("utf-8"))

        r = tool.execute({"action": "read", "path": "sub/a.txt"})
        assert r["status"] == "success"
        assert r["result"] == "héllo"
        assert r["truncated"] is False

    def test_search_without_pattern_is_structured_error(self, tmp_path):
        tool = FileOperationsTool(str(tmp_path))
        out = tool.execute({"action": "search", "pattern": ""})
        assert out == {"result": "No search pattern provided", "status": "error"}

    def test_search_is_case_insensitive_and_reports_no_match(self, tmp_path):
        (tmp_path / "Report.MD").write_text("x")
        tool = FileOperationsTool(str(tmp_path))
        hit = tool.execute({"action": "search", "pattern": "report"})
        assert hit["status"] == "success"
        assert hit["matches"] == ["Report.MD"]

        miss = tool.execute({"action": "search", "pattern": "nope"})
        assert miss["status"] == "success"
        assert miss["count"] == 0
        assert "No files matching 'nope'" in miss["result"]

    def test_unknown_action_is_error(self, tmp_path):
        out = FileOperationsTool(str(tmp_path)).execute({"action": "delete"})
        assert out == {"result": "Unknown file action: 'delete'", "status": "error"}

    def test_read_missing_file_is_error(self, tmp_path):
        out = FileOperationsTool(str(tmp_path)).execute(
            {"action": "read", "path": "ghost.txt"}
        )
        assert out["status"] == "error"
        assert "File not found: ghost.txt" in out["result"]

    def test_list_marks_dirs_and_handles_empty_and_missing(self, tmp_path):
        (tmp_path / "d").mkdir()
        (tmp_path / "f.txt").write_text("x")
        tool = FileOperationsTool(str(tmp_path))
        out = tool.execute({"action": "list"})
        assert out["entries"] == ["d/", "f.txt"]
        assert out["count"] == 2
        assert tool.execute({"action": "list", "path": "d"})["result"] == (
            "(empty directory)"
        )
        missing = tool.execute({"action": "list", "path": "nope"})
        assert missing["status"] == "error"

    def test_get_artifacts_empty_when_no_path(self, tmp_path):
        tool = FileOperationsTool(str(tmp_path))
        assert tool.get_artifacts({"result": "x"}) == []
        assert tool.get_artifacts({"path": "a.txt"}) == ["a.txt"]

    def test_parent_traversal_is_blocked(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (tmp_path / "secret.txt").write_text("top secret")
        tool = FileOperationsTool(str(ws))
        out = tool.execute({"action": "read", "path": "../secret.txt"})
        assert out["status"] == "error"
        assert "escapes workspace" in out["result"]

    def test_sibling_dir_sharing_name_prefix_is_blocked(self, tmp_path):
        """
        BUG REPRO — a sibling directory whose name starts with the workspace
        name (``ws`` vs ``ws_evil``) passed the old string-prefix check, so
        ``../ws_evil/secret.txt`` escaped the sandbox.
        """
        ws = tmp_path / "ws"
        ws.mkdir()
        evil = tmp_path / "ws_evil"
        evil.mkdir()
        (evil / "secret.txt").write_text("top secret")
        tool = FileOperationsTool(str(ws))

        read = tool.execute({"action": "read", "path": "../ws_evil/secret.txt"})
        assert read["status"] == "error"
        assert "top secret" not in read["result"]

        write = tool.execute(
            {"action": "write", "path": "../ws_evil/pwned.txt", "content": "x"}
        )
        assert write["status"] == "error"
        assert not (evil / "pwned.txt").exists()


# ---------------------------------------------------------------------------
# GitTool
# ---------------------------------------------------------------------------


class TestGitToolBehavior:
    def test_write_action_is_blocked_without_running_git(self, tmp_path):
        with patch("app.core.tools.dispatcher.subprocess.run") as run:
            out = GitTool(str(tmp_path)).execute({"action": "push"})
        assert out["status"] == "error"
        assert "not in the safe allow-list" in out["result"]
        run.assert_not_called()

    def test_missing_git_binary_is_captured(self, tmp_path):
        with patch(
            "app.core.tools.dispatcher.subprocess.run", side_effect=FileNotFoundError
        ):
            out = GitTool(str(tmp_path)).execute({"action": "status"})
        assert out == {"result": "git not found in PATH", "status": "error"}

    def test_nonzero_exit_without_output_reports_exit_code(self, tmp_path):
        proc = MagicMock(returncode=128, stdout="", stderr="")
        with patch(
            "app.core.tools.dispatcher.subprocess.run", return_value=proc
        ) as run:
            out = GitTool(str(tmp_path)).execute({"action": "log"})
        assert out["status"] == "error"
        assert out["returncode"] == 128
        assert out["result"] == "git log exited with code 128"
        assert run.call_args[0][0] == ["git", "log"]

    def test_nonzero_exit_prefers_stderr(self, tmp_path):
        proc = MagicMock(returncode=1, stdout="out", stderr="fatal: bad")
        with patch("app.core.tools.dispatcher.subprocess.run", return_value=proc):
            out = GitTool(str(tmp_path)).execute({"action": "status"})
        assert out["result"] == "fatal: bad"

    def test_timeout_is_captured_not_propagated(self, tmp_path):
        with patch(
            "app.core.tools.dispatcher.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=15),
        ):
            out = GitTool(str(tmp_path)).execute({"action": "diff"})
        assert out["status"] == "error"
        assert "timed out after 15s" in out["result"]

    def test_shell_operator_args_are_dropped(self, tmp_path):
        proc = MagicMock(returncode=0, stdout="clean", stderr="")
        with patch(
            "app.core.tools.dispatcher.subprocess.run", return_value=proc
        ) as run:
            out = GitTool(str(tmp_path)).execute(
                {"action": "log", "args": ["-n", 1, "&& rm -rf /", "; ls", "| cat"]}
            )
        assert out["status"] == "success"
        assert run.call_args[0][0] == ["git", "log", "-n", "1"]
        assert run.call_args.kwargs.get("shell") in (None, False)

    def test_get_artifacts_is_empty(self, tmp_path):
        assert GitTool(str(tmp_path)).get_artifacts({"result": "x"}) == []


# ---------------------------------------------------------------------------
# WebResearchTool
# ---------------------------------------------------------------------------


class TestWebResearchBehavior:
    def test_missing_url_is_error(self):
        assert WebResearchTool().execute({})["status"] == "error"

    def test_non_http_scheme_rejected_without_request(self):
        with patch("app.core.tools.dispatcher.httpx.Client") as cls:
            out = WebResearchTool().execute({"url": "file:///etc/passwd"})
        assert out["status"] == "error"
        assert "Unsupported URL scheme" in out["result"]
        cls.assert_not_called()

    def test_http_error_status_marks_error_and_truncates(self):
        with patch("app.core.tools.dispatcher.httpx.Client") as cls:
            client = _mock_httpx_client(cls)
            resp = MagicMock()
            resp.content = b"a" * (WebResearchTool.MAX_RESPONSE_BYTES + 5)
            resp.url = "https://example.test/x"
            resp.status_code = 503
            resp.headers = {"content-type": "text/plain"}
            client.get.return_value = resp
            out = WebResearchTool().execute({"url": "  https://example.test/x  "})
        assert out["status"] == "error"
        assert out["status_code"] == 503
        assert out["truncated"] is True
        assert len(out["result"]) == WebResearchTool.MAX_RESPONSE_BYTES
        assert client.get.call_args[0][0] == "https://example.test/x"

    def test_timeout_is_captured(self):
        with patch("app.core.tools.dispatcher.httpx.Client") as cls:
            client = _mock_httpx_client(cls)
            client.get.side_effect = httpx.ReadTimeout("slow")
            out = WebResearchTool().execute({"url": "https://example.test"})
        assert out == {
            "result": "Request to https://example.test timed out",
            "status": "error",
        }

    def test_generic_failure_is_captured(self):
        with patch("app.core.tools.dispatcher.httpx.Client") as cls:
            client = _mock_httpx_client(cls)
            client.get.side_effect = httpx.ConnectError("refused")
            out = WebResearchTool().execute({"url": "http://example.test"})
        assert out["status"] == "error"
        assert out["result"].startswith("HTTP request failed: ")
        assert "refused" in out["result"]

    def test_get_artifacts_is_empty(self):
        assert WebResearchTool().get_artifacts({"url": "x"}) == []


# ---------------------------------------------------------------------------
# DatabaseTool
# ---------------------------------------------------------------------------


class TestDatabaseToolBehavior:
    def test_missing_endpoint_is_error_without_request(self):
        with patch("app.core.tools.dispatcher.httpx.Client") as cls:
            out = DatabaseTool(api_key="k").execute({"endpoint": "/"})
        assert out["status"] == "error"
        assert "No endpoint" in out["result"]
        cls.assert_not_called()

    def test_get_sends_api_key_and_query_params(self):
        with patch("app.core.tools.dispatcher.httpx.Client") as cls:
            client = _mock_httpx_client(cls)
            client.get.return_value = _resp(200, "[1,2]")
            tool = DatabaseTool(base_url="http://core.test/", api_key="secret")
            out = tool.execute(
                {"endpoint": "/knowledgebase", "payload": {"q": "x"}, "method": "get"}
            )
        assert out == {"result": "[1,2]", "status_code": 200, "status": "success"}
        client.get.assert_called_once_with(
            "http://core.test/knowledgebase",
            headers={"X-API-Key": "secret"},
            params={"q": "x"},
        )

    def test_post_sends_json_and_4xx_is_error(self):
        with patch("app.core.tools.dispatcher.httpx.Client") as cls:
            client = _mock_httpx_client(cls)
            client.post.return_value = _resp(404, "x" * 20_000)
            tool = DatabaseTool(base_url="http://core.test", api_key="secret")
            out = tool.execute(
                {"endpoint": "conversations", "method": "POST", "payload": {"a": 1}}
            )
        assert out["status"] == "error"
        assert out["status_code"] == 404
        assert len(out["result"]) == 10_000
        client.post.assert_called_once_with(
            "http://core.test/conversations",
            headers={"X-API-Key": "secret"},
            json={"a": 1},
        )

    def test_no_api_key_means_no_auth_header(self, monkeypatch):
        monkeypatch.delenv("CORE_API_KEY", raising=False)
        with patch("app.core.tools.dispatcher.httpx.Client") as cls:
            client = _mock_httpx_client(cls)
            client.get.return_value = _resp()
            DatabaseTool(base_url="http://core.test").execute({"endpoint": "x"})
        assert client.get.call_args.kwargs["headers"] == {}

    def test_api_key_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("CORE_API_KEY", "from-env")
        assert DatabaseTool().api_key == "from-env"

    def test_unsupported_method_is_error(self):
        with patch("app.core.tools.dispatcher.httpx.Client") as cls:
            client = _mock_httpx_client(cls)
            out = DatabaseTool(api_key="k").execute(
                {"endpoint": "x", "method": "DELETE"}
            )
        assert out == {"result": "Unsupported HTTP method: DELETE", "status": "error"}
        client.get.assert_not_called()
        client.post.assert_not_called()

    def test_transport_failure_is_captured(self):
        with patch("app.core.tools.dispatcher.httpx.Client") as cls:
            client = _mock_httpx_client(cls)
            client.get.side_effect = httpx.ConnectError("down")
            out = DatabaseTool(api_key="k").execute({"endpoint": "x"})
        assert out["status"] == "error"
        assert out["result"].startswith("Database tool request failed: ")

    def test_get_artifacts_is_empty(self):
        assert DatabaseTool(api_key="k").get_artifacts({"result": "x"}) == []


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------


class TestToolDispatcherBehavior:
    def test_unknown_tool_returns_structured_error(self, tmp_path):
        d = ToolDispatcher(workspace_dir=str(tmp_path))
        out = d.dispatch("rm_rf", {})
        assert out["status"] == "error"
        assert "Unknown tool: 'rm_rf'" in out["result"]
        for name in d.available_tools:
            assert name in out["result"]

    def test_get_artifacts_unknown_tool_is_empty(self, tmp_path):
        d = ToolDispatcher(workspace_dir=str(tmp_path))
        assert d.get_artifacts("nope", {"path": "a.txt"}) == []

    def test_get_artifacts_handler_without_hook_is_empty(self, tmp_path):
        d = ToolDispatcher(workspace_dir=str(tmp_path))
        d._handlers["bare"] = object()  # instance-local; no global mutation
        assert d.get_artifacts("bare", {"path": "a.txt"}) == []

    def test_dispatch_routes_to_file_tool_and_reports_artifact(self, tmp_path):
        d = ToolDispatcher(workspace_dir=str(tmp_path))
        out = d.dispatch(
            "file_operations", {"action": "write", "path": "o.txt", "content": "z"}
        )
        assert out["status"] == "success"
        assert d.get_artifacts("file_operations", out) == ["o.txt"]
        assert (tmp_path / "o.txt").read_text() == "z"

    def test_default_workspace_is_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = ToolDispatcher()
        assert d.workspace_dir == str(tmp_path)

    def test_dispatcher_wires_database_config(self):
        d = ToolDispatcher(
            workspace_dir="/tmp", core_base_url="http://x.test/", core_api_key="k"
        )
        db = d._handlers["database"]
        assert db.base_url == "http://x.test"
        assert db.api_key == "k"

    @pytest.mark.parametrize(
        "tool", ["file_operations", "git", "web_research", "database"]
    )
    def test_registered_tools(self, tool, tmp_path):
        assert tool in ToolDispatcher(workspace_dir=str(tmp_path)).available_tools
