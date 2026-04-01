"""
CORE Tool Dispatcher — routes plan-step tool calls to real handlers.

Supported tools:
  - file_operations: read / write / list / search within workspace
  - git:             status / diff / log / show (read-only safe subset)
  - web_research:    HTTP GET for external content retrieval
  - database:        forwarded to CORE REST API (no direct DB socket needed)

All handlers return a dict with at minimum:
    {"result": <str>, "status": "success"|"error"}
Additional keys are tool-specific (see each handler's docstring).

Safety contracts:
  FileOperationsTool  — all paths resolved relative to workspace_dir; reads
                        outside the workspace raise ToolSafetyError.
  GitTool             — only a safe allow-listed set of git sub-commands are
                        executed; arbitrary shell injection is prevented by
                        never using shell=True and validating action names.
  WebResearchTool     — HTTP only, 10-second timeout, response body capped at
                        20 KB to prevent memory abuse.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class ToolSafetyError(ValueError):
    """Raised when a tool call would violate a safety constraint."""


# ---------------------------------------------------------------------------
# FileOperationsTool
# ---------------------------------------------------------------------------

class FileOperationsTool:
    """
    File read/write/list/search within a sandboxed workspace directory.

    All paths are resolved relative to workspace_dir.  Any attempt to access
    a path outside the workspace raises ToolSafetyError.
    """

    MAX_READ_BYTES = 100_000  # 100 KB

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()

    def _safe_path(self, rel_path: str) -> Path:
        resolved = (self.workspace_dir / rel_path).resolve()
        if not str(resolved).startswith(str(self.workspace_dir)):
            raise ToolSafetyError(
                f"Path '{rel_path}' escapes workspace '{self.workspace_dir}'"
            )
        return resolved

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "list")
        dispatch = {
            "read": self._read,
            "write": self._write,
            "list": self._list,
            "search": self._search,
        }
        handler = dispatch.get(action)
        if not handler:
            return {"result": f"Unknown file action: '{action}'", "status": "error"}
        try:
            return handler(params)
        except ToolSafetyError as exc:
            logger.warning("File safety check failed: %s", exc)
            return {"result": str(exc), "status": "error"}

    def _read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = self._safe_path(params.get("path", ""))
        if not path.is_file():
            return {"result": f"File not found: {params.get('path')}", "status": "error"}
        content = path.read_bytes()[: self.MAX_READ_BYTES].decode("utf-8", errors="replace")
        return {
            "result": content,
            "path": str(path.relative_to(self.workspace_dir)),
            "size_bytes": path.stat().st_size,
            "truncated": path.stat().st_size > self.MAX_READ_BYTES,
            "status": "success",
        }

    def _write(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = self._safe_path(params.get("path", ""))
        content = params.get("content", "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "result": f"Wrote {len(content)} chars to {params.get('path')}",
            "path": str(path.relative_to(self.workspace_dir)),
            "bytes_written": len(content.encode("utf-8")),
            "status": "success",
        }

    def _list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        base = self._safe_path(params.get("path", "."))
        if not base.exists():
            return {"result": f"Directory not found: {params.get('path', '.')}", "status": "error"}
        entries = sorted(p.relative_to(self.workspace_dir) for p in base.iterdir())
        names = [str(e) + ("/" if (self.workspace_dir / e).is_dir() else "") for e in entries]
        return {
            "result": "\n".join(names) or "(empty directory)",
            "entries": names,
            "count": len(names),
            "status": "success",
        }

    def _search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pattern = params.get("pattern", "")
        base = self._safe_path(params.get("path", "."))
        if not pattern:
            return {"result": "No search pattern provided", "status": "error"}
        matches: List[str] = []
        for p in base.rglob("*"):
            if p.is_file() and pattern.lower() in p.name.lower():
                matches.append(str(p.relative_to(self.workspace_dir)))
        return {
            "result": "\n".join(matches) or f"No files matching '{pattern}'",
            "matches": matches,
            "count": len(matches),
            "status": "success",
        }

    def get_artifacts(self, outputs: Dict[str, Any]) -> List[str]:
        if outputs.get("path"):
            return [outputs["path"]]
        return []


# ---------------------------------------------------------------------------
# GitTool
# ---------------------------------------------------------------------------

_GIT_SAFE_ACTIONS = {"status", "diff", "log", "show", "branch", "ls-files"}


class GitTool:
    """
    Run a safe, allow-listed subset of git commands in the workspace.

    Only read-only commands are permitted (status, diff, log, show, branch,
    ls-files).  Writes (commit, push, reset, …) are blocked to prevent
    unintended repository mutations during autonomous execution.
    """

    TIMEOUT_SECONDS = 15

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "status")
        if action not in _GIT_SAFE_ACTIONS:
            return {
                "result": (
                    f"Git action '{action}' is not in the safe allow-list "
                    f"({', '.join(sorted(_GIT_SAFE_ACTIONS))})"
                ),
                "status": "error",
            }

        extra_args: List[str] = params.get("args", [])
        # Ensure all extra args are strings and do not start with shell operators
        sanitized = [str(a) for a in extra_args if not str(a).startswith(("&&", "||", ";", "|"))]
        cmd = ["git", action] + sanitized

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
            )
            stdout = proc.stdout[:10_000]  # cap output
            stderr = proc.stderr[:2_000]
            if proc.returncode != 0:
                return {
                    "result": stderr or stdout or f"git {action} exited with code {proc.returncode}",
                    "returncode": proc.returncode,
                    "status": "error",
                }
            return {
                "result": stdout or "(no output)",
                "returncode": 0,
                "command": " ".join(cmd),
                "status": "success",
            }
        except subprocess.TimeoutExpired:
            return {"result": f"git {action} timed out after {self.TIMEOUT_SECONDS}s", "status": "error"}
        except FileNotFoundError:
            return {"result": "git not found in PATH", "status": "error"}

    def get_artifacts(self, outputs: Dict[str, Any]) -> List[str]:
        return []


# ---------------------------------------------------------------------------
# WebResearchTool
# ---------------------------------------------------------------------------

class WebResearchTool:
    """
    Fetch external content via HTTP GET.

    Response bodies are capped at 20 KB.  Follows up to 5 redirects.
    Times out after 10 seconds.
    """

    MAX_RESPONSE_BYTES = 20_000
    TIMEOUT_SECONDS = 10.0

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = params.get("url", "").strip()
        if not url:
            return {"result": "No URL provided", "status": "error"}
        if not url.startswith(("http://", "https://")):
            return {"result": f"Unsupported URL scheme: {url}", "status": "error"}

        try:
            with httpx.Client(
                timeout=self.TIMEOUT_SECONDS,
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                resp = client.get(url, headers={"User-Agent": "CORE/1.0 (research-bot)"})

            body = resp.content[: self.MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")
            return {
                "result": body,
                "url": str(resp.url),
                "status_code": resp.status_code,
                "content_type": resp.headers.get("content-type", ""),
                "truncated": len(resp.content) > self.MAX_RESPONSE_BYTES,
                "status": "success" if resp.status_code < 400 else "error",
            }
        except httpx.TimeoutException:
            return {"result": f"Request to {url} timed out", "status": "error"}
        except Exception as exc:
            return {"result": f"HTTP request failed: {exc}", "status": "error"}

    def get_artifacts(self, outputs: Dict[str, Any]) -> List[str]:
        return []


# ---------------------------------------------------------------------------
# DatabaseTool  (CORE REST API bridge)
# ---------------------------------------------------------------------------

class DatabaseTool:
    """
    Database access routed through the CORE REST API.

    Rather than opening a direct DB socket from a sync thread, this tool
    calls the appropriate CORE endpoint (e.g. /knowledgebase, /conversations)
    so it can share the existing auth + connection pool.
    """

    TIMEOUT_SECONDS = 10.0

    def __init__(self, base_url: str = "http://localhost:8001", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("CORE_API_KEY", "")

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = params.get("endpoint", "").lstrip("/")
        method = params.get("method", "GET").upper()
        payload = params.get("payload", {})

        if not endpoint:
            return {"result": "No endpoint specified for database tool", "status": "error"}

        url = f"{self.base_url}/{endpoint}"
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            with httpx.Client(timeout=self.TIMEOUT_SECONDS) as client:
                if method == "GET":
                    resp = client.get(url, headers=headers, params=payload)
                elif method == "POST":
                    resp = client.post(url, headers=headers, json=payload)
                else:
                    return {"result": f"Unsupported HTTP method: {method}", "status": "error"}

            return {
                "result": resp.text[:10_000],
                "status_code": resp.status_code,
                "status": "success" if resp.status_code < 400 else "error",
            }
        except Exception as exc:
            return {"result": f"Database tool request failed: {exc}", "status": "error"}

    def get_artifacts(self, outputs: Dict[str, Any]) -> List[str]:
        return []


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """
    Dispatches CORE plan-step tool calls to the appropriate handler.

    Registered tools: file_operations, git, web_research, database.

    Usage:
        dispatcher = ToolDispatcher(workspace_dir="/path/to/workspace")
        outputs    = dispatcher.dispatch("file_operations", {"action": "read", "path": "main.py"})
        artifacts  = dispatcher.get_artifacts("file_operations", outputs)
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        core_base_url: str = "http://localhost:8001",
        core_api_key: Optional[str] = None,
    ):
        self.workspace_dir = workspace_dir or os.getcwd()
        self._handlers: Dict[str, Any] = {
            "file_operations": FileOperationsTool(workspace_dir=self.workspace_dir),
            "git": GitTool(workspace_dir=self.workspace_dir),
            "web_research": WebResearchTool(),
            "database": DatabaseTool(base_url=core_base_url, api_key=core_api_key),
        }

    def dispatch(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch a tool call and return its output dict.

        Returns an error dict (not an exception) if the tool is unknown.
        """
        handler = self._handlers.get(tool_name)
        if not handler:
            logger.warning("ToolDispatcher: unknown tool '%s'", tool_name)
            return {
                "result": f"Unknown tool: '{tool_name}'. Available: {list(self._handlers)}",
                "status": "error",
            }
        logger.info("ToolDispatcher: dispatching tool='%s' params=%s", tool_name, params)
        return handler.execute(params)

    def get_artifacts(self, tool_name: str, outputs: Dict[str, Any]) -> List[str]:
        """Extract artifact paths (files, diffs) from tool output."""
        handler = self._handlers.get(tool_name)
        if handler and hasattr(handler, "get_artifacts"):
            return handler.get_artifacts(outputs)
        return []

    @property
    def available_tools(self) -> List[str]:
        return list(self._handlers)
