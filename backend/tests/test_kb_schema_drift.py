"""Schema-drift guard for the knowledgebase repository.

Regression test for POST /knowledgebase/upload returning 500 with
`column "world_id" of relation "kb_documents" does not exist`: the repository
wrote a column that no migration created.

How it works (no Postgres needed):
1. Replay app/migrations/sql/*.sql in the runner's order into a simple
   {table: {column: type}} model (CREATE TABLE, ALTER TABLE ADD/DROP/RENAME).
2. Call every public function in knowledgebase_repository against a fake
   asyncpg pool that records the SQL it is sent.
3. Pull the column references out of that SQL and check each one exists.

The extractor is deliberately small and tuned to this repository's SQL style.
If you add SQL it cannot read, extend KEYWORDS or the call table below rather
than weakening the assertions.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

import app.dependencies as dependencies
from app.migrations.runner import MIGRATIONS_DIR
from app.repository import knowledgebase_repository as kb_repo

KB_TABLES = {"kb_documents", "kb_chunks", "kb_activity"}

KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "null", "is", "as", "order",
    "by", "asc", "desc", "limit", "offset", "insert", "into", "values", "update",
    "set", "delete", "group", "having", "join", "left", "right", "inner", "outer",
    "on", "in", "any", "all", "exists", "distinct", "count", "coalesce", "lower",
    "upper", "min", "max", "sum", "avg", "like", "ilike", "true", "false",
    "current_timestamp", "now", "case", "when", "then", "else", "end",
    "returning", "conflict", "do", "nothing", "with", "union", "between",
}  # fmt: skip

TABLE_CONSTRAINT_WORDS = {
    "constraint",
    "primary",
    "foreign",
    "unique",
    "check",
    "exclude",
}


# ---------------------------------------------------------------------------
# Migration replay
# ---------------------------------------------------------------------------


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


def _split_top_level(body: str) -> List[str]:
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _base_type(type_sql: str) -> str:
    """'VARCHAR(512) NOT NULL' -> 'varchar'; 'vector(768)' -> 'vector'."""
    m = re.match(r"\s*([a-z_]+(?:\s+precision)?)", type_sql.lower())
    return m.group(1) if m else type_sql.lower().strip()


def _events(sql: str) -> List[Tuple[int, str, Any]]:
    """Ordered schema events in one SQL text."""
    events: List[Tuple[int, str, Any]] = []
    name = r"(?:public\.)?(\w+)"

    for m in re.finditer(
        rf"\bcreate\s+table\s+(?:if\s+not\s+exists\s+)?{name}\s*\(", sql
    ):
        start = m.end()
        depth, i = 1, start
        while depth and i < len(sql):
            depth += {"(": 1, ")": -1}.get(sql[i], 0)
            i += 1
        columns: Dict[str, str] = {}
        for item in _split_top_level(sql[start : i - 1]):
            first, _, rest = item.partition(" ")
            if first.lower() not in TABLE_CONSTRAINT_WORDS:
                columns[first.lower()] = _base_type(rest)
        events.append((m.start(), "create", (m.group(1).lower(), columns)))

    for m in re.finditer(rf"\bdrop\s+table\s+(?:if\s+exists\s+)?{name}", sql):
        events.append((m.start(), "drop_table", m.group(1).lower()))

    alter = rf"\balter\s+table\s+(?:if\s+exists\s+)?(?:only\s+)?{name}\s+(.*?);"
    for m in re.finditer(alter, sql, flags=re.S):
        table = m.group(1).lower()
        for action in _split_top_level(m.group(2)):
            a = action.lower()
            add = re.match(
                r"add\s+(?:column\s+)?(?:if\s+not\s+exists\s+)?(\w+)\s+(.*)", a, re.S
            )
            drop = re.match(r"drop\s+column\s+(?:if\s+exists\s+)?(\w+)", a)
            rename = re.match(r"rename\s+column\s+(\w+)\s+to\s+(\w+)", a)
            if add and add.group(1) not in TABLE_CONSTRAINT_WORDS:
                events.append(
                    (m.start(), "add", (table, add.group(1), _base_type(add.group(2))))
                )
            elif drop:
                events.append((m.start(), "drop", (table, drop.group(1))))
            elif rename:
                events.append(
                    (m.start(), "rename", (table, rename.group(1), rename.group(2)))
                )
            else:
                # ALTER COLUMN ... TYPE/DROP NOT NULL etc: column set unchanged.
                events.append((m.start(), "touch", table))
    return sorted(events, key=lambda e: e[0])


def replay_migrations(
    sql_texts: List[Tuple[str, str]],
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Apply (name, sql) pairs in order. Returns (schema, ordering_errors).

    ADD COLUMN IF NOT EXISTS keeps the first type, matching Postgres.
    """
    schema: Dict[str, Dict[str, str]] = {}
    errors: List[str] = []
    for fname, raw in sql_texts:
        for _, kind, data in _events(_strip_sql_comments(raw).lower()):
            if kind == "create":
                table, cols = data
                schema.setdefault(table, dict(cols))
                continue
            if kind == "drop_table":
                schema.pop(data, None)
                continue
            table = data if kind == "touch" else data[0]
            if table not in schema:
                if table in KB_TABLES:
                    errors.append(
                        f"{fname}: ALTERs {table} before any migration creates it"
                    )
                continue
            if kind == "add":
                schema[table].setdefault(data[1], data[2])
            elif kind == "drop":
                schema[table].pop(data[1], None)
            elif kind == "rename" and data[1] in schema[table]:
                schema[table][data[2]] = schema[table].pop(data[1])
    return schema, errors


def _runner_migrations() -> List[Tuple[str, str]]:
    # Same file selection, order and decoding as app/migrations/runner.py.
    return [
        (f.name, f.read_text(encoding="utf-8-sig"))
        for f in sorted(Path(MIGRATIONS_DIR).glob("*.sql"))
    ]


# ---------------------------------------------------------------------------
# Repository SQL capture
# ---------------------------------------------------------------------------


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def start(self):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass


class _FakeConn:
    def __init__(self, log: List[str]):
        self.log = log

    async def execute(self, sql, *args):
        self.log.append(sql)
        return "OK 0"

    async def executemany(self, sql, *args):
        self.log.append(sql)

    async def fetch(self, sql, *args):
        self.log.append(sql)
        return []

    async def fetchrow(self, sql, *args):
        self.log.append(sql)
        return None

    async def fetchval(self, sql, *args):
        self.log.append(sql)
        return 1

    def transaction(self):
        return _FakeTransaction()


class _FakePool:
    def __init__(self):
        self.log: List[str] = []

    def acquire(self):
        conn = _FakeConn(self.log)

        class _Ctx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


VEC = [0.0] * 768
DOC = "00000000-0000-0000-0000-000000000001"

# Every public repository coroutine, with calls that hit each optional branch.
REPO_CALLS: Dict[str, List[Dict[str, Any]]] = {
    "create_document": [
        dict(
            filename="f", original_name="f", size=1, mime_type="text/plain",
            storage_path="/x", world_id="w",
        )
    ],
    "list_documents_by_world": [dict(world_id="w")],
    "update_document_embedding": [
        dict(document_id=DOC, embedding=VEC, model="m", dimensions=768)
    ],
    "insert_chunk_embeddings": [
        dict(document_id=DOC, items=[(0, "t", VEC)], model="m", dimensions=768)
    ],
    "update_chunk_embeddings": [
        dict(document_id=DOC, chunks=[(0, VEC)], model="m", dimensions=768)
    ],
    "list_documents": [dict(), dict(q="x", is_global=True)],
    "get_document": [dict(document_id=DOC)],
    "delete_document": [dict(document_id=DOC)],
    "list_chunks_for_document": [dict(document_id=DOC)],
    "list_chunks_for_documents": [dict(document_ids=[DOC])],
    "search_chunks_by_vector": [
        dict(query_vec=VEC),
        dict(query_vec=VEC, document_filter=[DOC], model="m"),
    ],
    "update_document_description": [dict(document_id=DOC, description="d")],
    "update_document_title_and_description": [
        dict(document_id=DOC, title="t", description="d")
    ],
    "get_document_by_hash": [dict(file_hash="h")],
    "insert_activity": [dict(action="a", document_id=DOC, file_name="f")],
    "list_recent_activity": [dict()],
    "search_chunks_by_instance": [
        dict(query_vec=VEC, instance_name="i"),
        dict(query_vec=VEC, instance_name="i", document_filter=[DOC], model="m"),
    ],
    "list_instances": [dict()],
    # Attribution placeholders: no SQL today, listed so new SQL gets checked.
    "get_chunk_attributions": [dict(chunk_ids=[DOC])],
    "get_document_attribution": [dict(document_id=DOC)],
    "create_chunk_attribution": [dict(chunk_id=DOC, source_instance_id="i")],
    "create_document_attribution": [dict(document_id=DOC, source_instance_id="i")],
    "list_attributions_for_instance": [dict(instance_id="i")],
    "delete_attributions_for_instance": [dict(instance_id="i")],
}  # fmt: skip


def _public_repo_coroutines() -> Set[str]:
    return {
        name
        for name, obj in inspect.getmembers(kb_repo, inspect.iscoroutinefunction)
        if not name.startswith("_") and obj.__module__ == kb_repo.__name__
    }


async def _capture_repository_sql(monkeypatch) -> Dict[str, List[str]]:
    captured: Dict[str, List[str]] = {}
    for fname, calls in REPO_CALLS.items():
        pool = _FakePool()

        async def _get_pool(pool=pool):
            return pool

        monkeypatch.setattr(kb_repo, "get_db_pool", _get_pool)
        for kwargs in calls:
            await getattr(kb_repo, fname)(**kwargs)
        captured[fname] = list(pool.log)
    return captured


# ---------------------------------------------------------------------------
# Column extraction
# ---------------------------------------------------------------------------


def extract_column_refs(sql: str) -> Tuple[Set[Tuple[str, str]], Set[str], Set[str]]:
    """Return (qualified {(table, col)}, unqualified {col}, tables in statement)."""
    s = _strip_sql_comments(sql).lower()
    s = re.sub(r"'[^']*'", "''", s)
    s = re.sub(r"\$\d+", " ", s)
    s = re.sub(r"::\s*\w+(?:\s*\(\s*\d+\s*\))?(?:\[\])?", " ", s)

    aliases: Dict[str, str] = {}
    tables: Set[str] = set()
    for m in re.finditer(
        r"\b(?:from|join|into|update)\s+(\w+)(?:\s+(?:as\s+)?(\w+))?", s
    ):
        if m.group(1) in KB_TABLES:
            tables.add(m.group(1))
            alias = m.group(2)
            if alias and alias not in KEYWORDS:
                aliases[alias] = m.group(1)

    qualified: Set[Tuple[str, str]] = set()
    for m in re.finditer(r"\binsert\s+into\s+(\w+)\s*\(([^)]*)\)", s):
        for col in m.group(2).split(","):
            qualified.add((m.group(1), col.strip()))
    s = re.sub(r"\binsert\s+into\s+\w+\s*\([^)]*\)", " ", s)

    def _qualified(m):
        alias, col = m.group(1), m.group(2)
        qualified.add((aliases.get(alias, f"<unknown alias {alias}>"), col))
        return " "

    s = re.sub(r"\b([a-z_]\w*)\.([a-z_]\w*)\b", _qualified, s)
    # Output aliases (`... AS chunk_count`) are not columns, and ORDER BY may
    # refer back to them.
    output_aliases = set(re.findall(r"\bas\s+(\w+)", s))
    s = re.sub(r"\bas\s+\w+", " ", s)

    skip = KEYWORDS | tables | set(aliases) | output_aliases
    unqualified = {
        tok for tok in re.findall(r"\b[a-z_][a-z0-9_]*\b", s) if tok not in skip
    }
    return qualified, unqualified, tables


def _missing_columns(schema, captured) -> List[str]:
    problems: List[str] = []
    for fname, statements in captured.items():
        for sql in statements:
            qualified, unqualified, tables = extract_column_refs(sql)
            for table, col in sorted(qualified):
                if col not in schema.get(table, {}):
                    problems.append(f"{fname}: {table}.{col}")
            for col in sorted(unqualified):
                if not any(col in schema.get(t, {}) for t in tables):
                    problems.append(f"{fname}: {col} (in {sorted(tables)})")
    return problems


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_every_repository_function_is_exercised():
    missing = _public_repo_coroutines() - set(REPO_CALLS)
    assert not missing, (
        "Add these knowledgebase_repository functions to REPO_CALLS so their "
        f"SQL is checked against the migrations: {sorted(missing)}"
    )


async def test_migrations_define_every_kb_column_the_repository_uses(monkeypatch):
    schema, _ = replay_migrations(_runner_migrations())
    captured = await _capture_repository_sql(monkeypatch)

    problems = _missing_columns(schema, captured)
    assert not problems, (
        "knowledgebase_repository references columns no migration in "
        f"{MIGRATIONS_DIR} creates. Add an idempotent migration:\n  "
        + "\n  ".join(problems)
    )


async def test_extractor_sees_the_world_id_column(monkeypatch):
    """Guards the guard: the extractor must actually find the columns."""
    captured = await _capture_repository_sql(monkeypatch)
    insert_q, _, _ = extract_column_refs(captured["create_document"][0])
    _, where_u, _ = extract_column_refs(captured["list_documents_by_world"][0])
    assert ("kb_documents", "world_id") in insert_q
    assert "world_id" in where_u
    assert ("kb_documents", "instance_name") in insert_q


async def test_check_fails_without_the_world_id_migration(monkeypatch):
    """Dropping migration 005 must make the drift check report world_id."""
    files = [f for f in _runner_migrations() if "world_id" not in f[0]]
    schema, _ = replay_migrations(files)
    captured = await _capture_repository_sql(monkeypatch)
    problems = _missing_columns(schema, captured)
    assert "create_document: kb_documents.world_id" in problems
    assert any(p.startswith("list_documents_by_world: world_id") for p in problems)


def test_kb_tables_exist_before_migrations_alter_them():
    """Migrations run before setup_db_schema(), so a migration that ALTERs a KB
    table must come after one that creates it (fresh-DB first boot)."""
    _, errors = replay_migrations(_runner_migrations())
    assert not errors, "\n".join(errors)


async def test_setup_db_schema_kb_types_match_migrations(monkeypatch):
    """setup_db_schema() also adds KB columns. If it disagrees with the
    migrations on a type, the result depends on which ran first."""
    pool = _FakePool()

    async def _get_pool():
        return pool

    monkeypatch.setattr(dependencies, "get_db_pool", _get_pool)
    await dependencies.setup_db_schema()

    migrated, _ = replay_migrations(_runner_migrations())
    from_setup, _ = replay_migrations([("setup_db_schema", ";\n".join(pool.log) + ";")])
    mismatches = [
        f"{t}.{c}: setup={from_setup[t][c]} migrations={migrated[t][c]}"
        for t in KB_TABLES
        for c in from_setup.get(t, {})
        if c in migrated.get(t, {}) and from_setup[t][c] != migrated[t][c]
    ]
    assert not mismatches, "\n".join(mismatches)


async def test_setup_db_schema_guards_legacy_embedding_alter(monkeypatch):
    """An unguarded `ALTER COLUMN embedding DROP NOT NULL` fails on DBs that
    already dropped the column, rolling back all of setup_db_schema() (which
    is how world_id went missing in the first place)."""
    pool = _FakePool()

    async def _get_pool():
        return pool

    monkeypatch.setattr(dependencies, "get_db_pool", _get_pool)
    await dependencies.setup_db_schema()
    for sql in pool.log:
        if re.search(r"alter\s+column\s+embedding\s+drop\s+not\s+null", sql, re.I):
            assert "information_schema.columns" in sql, sql


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("ALTER TABLE t ADD COLUMN IF NOT EXISTS c TEXT;", {"a": "text", "c": "text"}),
        (
            "ALTER TABLE t ADD COLUMN c INT, ADD COLUMN d UUID;",
            {"a": "text", "c": "int", "d": "uuid"},
        ),
        ("ALTER TABLE t ADD COLUMN IF NOT EXISTS a UUID;", {"a": "text"}),
        ("ALTER TABLE t DROP COLUMN IF EXISTS a;", {}),
        ("ALTER TABLE t RENAME COLUMN a TO z;", {"z": "text"}),
        (
            "DO $$ BEGIN ALTER TABLE t ADD COLUMN b TEXT; END $$;",
            {"a": "text", "b": "text"},
        ),
    ],
)
def test_replay_handles_alter_forms(sql, expected):
    schema, _ = replay_migrations([("base", "CREATE TABLE t (a TEXT);"), ("m", sql)])
    assert schema["t"] == expected
