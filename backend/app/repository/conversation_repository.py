from __future__ import annotations

"""Lightweight JSON-file backed storage for chat conversations.

This is **not** intended for production use. It exists purely so that the
application can persist chat history between restarts without requiring an
external database.  For production workloads consider PostgreSQL, MongoDB or
similar managed stores.
"""

from pathlib import Path
import json
import uuid
import asyncio
from typing import Dict, List, TypedDict, Any


class _Message(TypedDict):
    role: str
    content: str


class _Conversation(TypedDict):
    id: str
    title: str
    messages: List[_Message]


# ---------------------------------------------------------------------------
# File-system paths / global runtime state
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_CONVERSATIONS_FILE = _DATA_DIR / "conversations.json"

# An async lock to ensure we do not perform concurrent read/write access to
# the underlying JSON file which may lead to corruption.
_LOCK = asyncio.Lock()


async def _load_raw() -> Dict[str, Any]:
    """Load the raw JSON payload from disk."""
    if not _CONVERSATIONS_FILE.exists():
        return {"conversations": []}

    async with _LOCK:
        loop = asyncio.get_running_loop()
        data_str = await loop.run_in_executor(None, _CONVERSATIONS_FILE.read_text)
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            # If the file is corrupted we fall back to an empty structure so as
            # to not crash the entire application.
            return {"conversations": []}


async def _flush_raw(payload: Dict[str, Any]) -> None:
    """Persist the given JSON payload to disk atomically."""
    async with _LOCK:
        loop = asyncio.get_running_loop()
        data_str = json.dumps(payload, indent=2)
        await loop.run_in_executor(None, _CONVERSATIONS_FILE.write_text, data_str)


# ---------------------------------------------------------------------------
# Public repository API
# ---------------------------------------------------------------------------

async def list_conversations() -> List[_Conversation]:
    """Return a **copy** of all stored conversations (without mutating state)."""
    raw = await _load_raw()
    return list(raw.get("conversations", []))


async def get_conversation(conv_id: str) -> _Conversation | None:
    """Return a single conversation by id or *None* if it cannot be found."""
    raw = await _load_raw()
    for conv in raw.get("conversations", []):
        if conv.get("id") == conv_id:
            return conv
    return None


async def create_conversation(
    initial_messages: List[_Message] | None = None,
    title: str | None = None,
) -> str:
    """Create a new conversation and return its *id*."""
    conv_id = str(uuid.uuid4())
    new_conv: _Conversation = {
        "id": conv_id,
        "title": title or "New Conversation",
        "messages": initial_messages or [],
    }

    raw = await _load_raw()
    raw.setdefault("conversations", []).append(new_conv)
    await _flush_raw(raw)
    return conv_id


async def append_message(conv_id: str, message: _Message) -> None:
    """Append *message* to the conversation identified by *conv_id*."""
    raw = await _load_raw()
    for conv in raw.setdefault("conversations", []):
        if conv.get("id") == conv_id:
            conv.setdefault("messages", []).append(message)
            await _flush_raw(raw)
            return

    # If we reached this point the conversation did not exist – create it so we
    # do not silently drop data. This should never happen in normal flows but
    # guards against corrupt state.
    new_conv: _Conversation = {"id": conv_id, "title": "Recovered", "messages": [message]}
    raw["conversations"].append(new_conv)
    await _flush_raw(raw)


async def update_title(conv_id: str, title: str) -> None:
    """Update the title of a conversation."""
    raw = await _load_raw()
    for conv in raw.setdefault("conversations", []):
        if conv.get("id") == conv_id:
            conv["title"] = title
            await _flush_raw(raw)
            return 