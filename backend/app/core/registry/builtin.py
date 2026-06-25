"""
Builtin capability descriptors for the in-process tool dispatcher.

The registry is built *from* the live dispatcher: a descriptor is only registered
when the dispatcher actually exposes its tool (`available_tools`). That is the
anti-drift contract — the system can never advertise a capability it can't run.
(MCP-server capabilities register through the same `CapabilityRegistry` in a later
phase; this module covers the builtin tools.)

Descriptions are written for retrieval quality, since comprehension's
capability-presence check is lexical search over them.
"""

from __future__ import annotations

from typing import Dict, List

from app.core.registry.capability import CapabilityEntry, CapabilityRegistry

_PATH_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
}

# Descriptors grouped by dispatcher tool root (matches ToolDispatcher handlers).
_BUILTIN: Dict[str, List[CapabilityEntry]] = {
    "file_operations": [
        CapabilityEntry(
            id="file_operations.read",
            name="Read file",
            description="Read the contents of a file in the workspace",
            params_schema=_PATH_SCHEMA,
            side_effects="read",
            examples=["read main.py", "open and read a file from disk"],
        ),
        CapabilityEntry(
            id="file_operations.write",
            name="Write file",
            description="Write or create a file in the workspace, replacing its contents",
            params_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            side_effects="write",
            examples=["create a new file", "save changes to a file"],
        ),
        CapabilityEntry(
            id="file_operations.list",
            name="List files",
            description="List the files and directories in a workspace folder",
            params_schema=_PATH_SCHEMA,
            side_effects="read",
            examples=["list the files in a directory"],
        ),
        CapabilityEntry(
            id="file_operations.search",
            name="Search files",
            description="Search for files by name pattern within the workspace",
            params_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
            side_effects="read",
            examples=["find files matching a name"],
        ),
    ],
    "git": [
        CapabilityEntry(
            id="git.status",
            name="Git status",
            description="Show the git working tree status: changed, staged and untracked files",
            side_effects="read",
        ),
        CapabilityEntry(
            id="git.diff",
            name="Git diff",
            description="Show the git diff of changes in the repository",
            side_effects="read",
            examples=["show me the diff", "what changed in the repo"],
        ),
        CapabilityEntry(
            id="git.log",
            name="Git log",
            description="Show the git commit history log for the repository",
            side_effects="read",
        ),
        CapabilityEntry(
            id="git.show",
            name="Git show",
            description="Show the contents of a git object such as a specific commit",
            side_effects="read",
        ),
    ],
    "web_research": [
        CapabilityEntry(
            id="web_research.fetch",
            name="Fetch web content",
            description="Fetch external web content over HTTP from a URL",
            params_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            side_effects="network",
            examples=["fetch a documentation page", "download a web page over http"],
        ),
    ],
    "database": [
        CapabilityEntry(
            id="database.query",
            name="Query database",
            description="Query the CORE database through the REST API (knowledge base, conversations)",
            params_schema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string"},
                    "method": {"type": "string"},
                },
                "required": ["endpoint"],
            },
            side_effects="network",
            examples=["look up a record in the database"],
        ),
    ],
}


def build_registry_from_dispatcher(dispatcher) -> CapabilityRegistry:
    """Build a registry containing only capabilities the dispatcher can run.

    `dispatcher` only needs an `available_tools` iterable, so this stays
    decoupled from the concrete `ToolDispatcher` class.
    """
    registry = CapabilityRegistry()
    available = set(getattr(dispatcher, "available_tools", []))
    for root, entries in _BUILTIN.items():
        if root in available:
            for entry in entries:
                registry.register(entry)
    return registry
