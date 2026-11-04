"""
Operation verbs in a request, for the front gate's action-level check.

The classifier names tool *roots* (``git``, ``file_operations``). A root being
present does not mean the requested action is: CORE's git is read-only, so
"push main to origin" names a present root and an absent action. This module
pulls the operation verbs out of a request so the gate can compare them with
the verbs the root's capabilities serve (``CapabilityEntry.verbs``).

Two rules keep it from over-refusing:

* Only verbs in :data:`GATED_OPERATIONS` can cause a refusal. These are
  operations with real side effects or a distinct meaning (push, commit,
  delete, drop, submit...). Common verbs like "show", "run", "send" or "copy"
  are never gated: they are either served, composable from what is served, or
  too often used loosely ("send me the diff").
* A word counts as a verb only in verb position: first word of the request or of
  a clause, or after a request word ("please", "can you", "to"). That keeps the
  noun in "show commit a1b2c3d" from reading as the verb in "commit my changes".

Pure, no model, no IO.
"""

from __future__ import annotations

import re
from typing import List

# Verbs that refuse when no capability of the named root serves them.
GATED_OPERATIONS = frozenset(
    {
        # destructive
        "delete",
        "remove",
        "rm",
        "erase",
        "unlink",
        "destroy",
        "wipe",
        "purge",
        "drop",
        "truncate",
        # moving things the file tool cannot delete afterwards
        "move",
        "rename",
        # version control writes
        "commit",
        "push",
        "pull",
        "merge",
        "rebase",
        "checkout",
        "clone",
        "reset",
        "revert",
        "stash",
        "amend",
        "cherry-pick",
        "tag",
        # creating / changing records
        "create",
        "insert",
        "update",
        "alter",
        "edit",
        "modify",
        # outbound actions
        "submit",
        "post",
        "upload",
        "publish",
        "deploy",
    }
)

# A gated verb right after one of these is in verb position:
# "please delete", "can you push", "I want to commit", "help me drop".
_VERB_LEADERS = frozenset(
    {
        "and",
        "then",
        "also",
        "to",
        "please",
        "pls",
        "you",
        "can",
        "could",
        "would",
        "will",
        "should",
        "must",
        "now",
        "just",
        "me",
        "lets",
        "let's",
    }
)

# Words; sentence punctuation (a "." only when it ends a sentence, so the dot in
# "README.md" or "example.com" is not a break); any other single character.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|[.!?](?=\s|$)|[,;:\n]|\S")
_CLAUSE_BREAKS = frozenset({",", ";", ":", ".", "!", "?", "\n"})


def requested_operations(text: str) -> List[str]:
    """Gated operation verbs in ``text``, in order, de-duplicated.

    >>> requested_operations("Push main to origin")
    ['push']
    >>> requested_operations("Show commit a1b2c3d with git show")
    []
    >>> requested_operations("Fill in and submit the signup form")
    ['submit']
    """
    if not text:
        return []
    found: List[str] = []
    prev: str | None = None  # None = start of text or of a clause
    for raw in _WORD_RE.findall(text):
        tok = raw.lower()
        if tok in _CLAUSE_BREAKS:
            prev = None
            continue
        if not tok[0].isalpha():  # digits, quotes, slashes: keep scanning
            prev = tok
            continue
        in_verb_position = prev is None or prev in _VERB_LEADERS
        if in_verb_position and tok in GATED_OPERATIONS and tok not in found:
            found.append(tok)
        prev = tok
    return found
