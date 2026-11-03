"""
Behavior tests for the CORE engine controller (``app/controllers/engine.py``).

Boundaries mocked:
- ``get_core_graph`` (the LangGraph / COREGraph singleton) -> MagicMock
- ``run_repository`` (Postgres persistence) -> MagicMock with AsyncMocks
- ``get_webhook_service`` -> MagicMock whose ``fire`` is an AsyncMock
- ``app.dependencies.get_db_pool`` (used directly by DELETE /runs/{id})

Auth is exercised for real: ``require_role`` resolves the master key from
``CORE_API_KEY`` and ``require_api_key`` checks ``app.auth.VALID_API_KEYS``.
All module globals / env vars are restored via monkeypatch.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth as legacy_auth
from app.controllers import engine
from app.core import security
from app.models.core_state import (
    COREState,
    EvaluationResult,
    ExecutionPlan,
    PlanStep,
    StepResult,
    UserIntent,
)
from app.services.webhook_service import WebhookEvent

TEST_KEY = "engine-test-master-key"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def active_runs(monkeypatch):
    runs: dict = {}
    monkeypatch.setattr(engine, "_active_runs", runs)
    return runs


@pytest.fixture
def repo(monkeypatch):
    mock = MagicMock()
    mock.save_run = AsyncMock(return_value=None)
    mock.get_run = AsyncMock(return_value=None)
    mock.list_runs = AsyncMock(return_value=[])
    mock.get_run_stats = AsyncMock(return_value={"total_runs": 0})
    monkeypatch.setattr(engine, "run_repository", mock)
    return mock


@pytest.fixture
def webhooks(monkeypatch):
    svc = MagicMock()
    svc.fire = AsyncMock(return_value=None)
    monkeypatch.setattr(engine, "get_webhook_service", lambda: svc)
    return svc


@pytest.fixture
def graph(monkeypatch):
    """A fake COREGraph. Node methods mutate and return the state like the real one."""
    g = MagicMock()
    monkeypatch.setattr(engine, "get_core_graph", lambda: g)
    return g


@pytest.fixture
def client(monkeypatch, active_runs, repo, webhooks, graph):
    monkeypatch.delenv("CORE_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("CORE_API_KEY", TEST_KEY)
    added = TEST_KEY not in legacy_auth.VALID_API_KEYS
    legacy_auth.VALID_API_KEYS.add(TEST_KEY)

    app = FastAPI()
    app.include_router(engine.router)
    with TestClient(app) as c:
        c.headers.update({"X-API-Key": TEST_KEY})
        yield c

    if added:
        legacy_auth.VALID_API_KEYS.discard(TEST_KEY)


def _sse_events(text: str) -> list:
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def _completing_invoke(response="Hello there"):
    """Graph.invoke stand-in returning a dict, as LangGraph does."""

    def _invoke(state):
        data = state.model_dump()
        data.update(
            response=response,
            current_node="END",
            execution_history=["comprehension", "conversation"],
            intent={
                "type": "conversation",
                "description": "greeting",
                "confidence": 0.9,
            },
        )
        return data

    return _invoke


def _wire_task_nodes(graph, *, response="final answer"):
    """Configure node methods to simulate a full task pipeline."""
    step = PlanStep(id="s1", name="Search docs", description="look things up")

    def comprehension(state):
        state.intent = UserIntent(type="task", description="do it", confidence=0.8)
        return state

    def orchestration(state):
        state.plan = ExecutionPlan(goal="Answer question", steps=[step])
        return state

    def reasoning(state):
        state.step_results = [
            StepResult(step_id="s1", status="success", duration_seconds=1.5),
            StepResult(step_id="ghost", status="failure", error="boom"),
        ]
        return state

    def evaluation(state):
        state.eval_result = EvaluationResult(
            overall_status="success",
            confidence=0.9,
            meets_requirements=True,
            quality_score=0.75,
        )
        return state

    def conversation(state):
        state.response = response
        return state

    graph.comprehension_node.side_effect = comprehension
    graph.orchestration_node.side_effect = orchestration
    graph.reasoning_node.side_effect = reasoning
    graph.evaluation_node.side_effect = evaluation
    graph.conversation_node.side_effect = conversation


# ---------------------------------------------------------------------------
# POST /engine/run
# ---------------------------------------------------------------------------


class TestRunCore:
    def test_success_returns_completed_and_persists_and_fires_webhooks(
        self, client, graph, repo, webhooks, active_runs
    ):
        graph.get_graph.return_value.invoke.side_effect = _completing_invoke("Hi!")

        resp = client.post(
            "/engine/run",
            json={
                "input": "hello",
                "config": {"temperature": 0.1},
                "conversation_id": "conv-1",
                "user_id": "u-1",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["message"] == "Hi!"
        run_id = body["run_id"]

        # Final state stored in memory under the original run_id
        stored = active_runs[run_id]
        assert stored.response == "Hi!"
        assert stored.config["temperature"] == 0.1
        assert stored.user_id == "u-1"

        # Invoked with a state carrying the request data
        invoked_state = graph.get_graph.return_value.invoke.call_args.args[0]
        assert invoked_state.user_input == "hello"
        assert invoked_state.conversation_id == "conv-1"

        # Persisted twice: initial + final
        assert repo.save_run.await_count == 2
        final_saved = repo.save_run.await_args_list[-1].args[0]
        assert final_saved["run_id"] == run_id
        assert final_saved["response"] == "Hi!"

        events = [c.kwargs["event"] for c in webhooks.fire.await_args_list]
        assert events == [WebhookEvent.RUN_STARTED, WebhookEvent.RUN_COMPLETED]
        completed = webhooks.fire.await_args_list[1].kwargs
        assert completed["run_id"] == run_id
        assert completed["payload"]["intent_type"] == "conversation"
        assert completed["payload"]["response"] == "Hi!"

    def test_non_dict_result_and_incomplete_state_reports_running(
        self, client, graph, active_runs
    ):
        def _invoke(state):
            # Returns a COREState (not dict) that has not reached END
            return state

        graph.get_graph.return_value.invoke.side_effect = _invoke

        resp = client.post("/engine/run", json={"input": "partial"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["message"] == "Execution in progress"
        assert active_runs[body["run_id"]].user_input == "partial"

    def test_graph_failure_returns_500_records_error_and_fires_failed(
        self, client, graph, repo, webhooks, active_runs
    ):
        graph.get_graph.return_value.invoke.side_effect = RuntimeError("LLM down")

        resp = client.post("/engine/run", json={"input": "hello"})

        assert resp.status_code == 500
        assert "LLM down" in resp.json()["detail"]

        assert len(active_runs) == 1
        (state,) = active_runs.values()
        assert state.errors == ["Execution failed: LLM down"]

        # Initial + error persistence
        assert repo.save_run.await_count == 2
        assert repo.save_run.await_args_list[-1].args[0]["errors"] == [
            "Execution failed: LLM down"
        ]

        events = [c.kwargs["event"] for c in webhooks.fire.await_args_list]
        assert events == [WebhookEvent.RUN_STARTED, WebhookEvent.RUN_FAILED]
        failed = webhooks.fire.await_args_list[1].kwargs
        assert failed["payload"]["error"] == "LLM down"
        assert failed["run_id"] == state.run_id

    def test_persistence_and_webhook_failures_are_tolerated_on_success(
        self, client, graph, repo, webhooks
    ):
        repo.save_run.side_effect = ConnectionError("db gone")
        webhooks.fire.side_effect = RuntimeError("webhook queue full")
        graph.get_graph.return_value.invoke.side_effect = _completing_invoke("ok")

        resp = client.post("/engine/run", json={"input": "hello"})

        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        assert repo.save_run.await_count == 2
        assert webhooks.fire.await_count == 2

    def test_persistence_and_webhook_failures_tolerated_on_error_path(
        self, client, graph, repo, webhooks
    ):
        repo.save_run.side_effect = ConnectionError("db gone")
        webhooks.fire.side_effect = RuntimeError("webhook queue full")
        graph.get_graph.return_value.invoke.side_effect = ValueError("bad")

        resp = client.post("/engine/run", json={"input": "hello"})

        # Original graph error surfaces, not the secondary failures
        assert resp.status_code == 500
        assert "bad" in resp.json()["detail"]
        assert webhooks.fire.await_count == 2

    def test_missing_input_is_422_and_graph_not_called(self, client, graph, repo):
        resp = client.post("/engine/run", json={"config": {}})

        assert resp.status_code == 422
        graph.get_graph.assert_not_called()
        repo.save_run.assert_not_awaited()

    def test_missing_api_key_is_401(self, client, graph):
        resp = client.post(
            "/engine/run", json={"input": "hi"}, headers={"X-API-Key": ""}
        )
        assert resp.status_code == 401
        graph.get_graph.assert_not_called()

    def test_invalid_api_key_is_401(self, client, graph):
        resp = client.post(
            "/engine/run", json={"input": "hi"}, headers={"X-API-Key": "nope"}
        )
        assert resp.status_code == 401
        graph.get_graph.assert_not_called()

    def test_viewer_role_is_forbidden(self, client, graph, monkeypatch):
        monkeypatch.setattr(
            "app.repository.api_key_repository.update_last_used",
            AsyncMock(return_value=None),
        )
        viewer_key = "viewer-only-key"
        monkeypatch.setitem(
            security._API_KEY_CACHE,
            security._hash_key(viewer_key),
            {
                "name": "viewer",
                "permissions": ["viewer"],
                "last_used": None,
                "request_count": 0,
            },
        )

        resp = client.post(
            "/engine/run", json={"input": "hi"}, headers={"X-API-Key": viewer_key}
        )

        assert resp.status_code == 403
        graph.get_graph.assert_not_called()


# ---------------------------------------------------------------------------
# POST /engine/run/stream
# ---------------------------------------------------------------------------


class TestRunCoreStream:
    def test_task_intent_streams_full_pipeline(self, client, graph, repo, active_runs):
        _wire_task_nodes(graph, response="final answer")

        resp = client.post(
            "/engine/run/stream",
            json={"input": "find docs", "config": {"model": "x"}},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-cache"
        events = _sse_events(resp.text)
        kinds = [e["event"] for e in events]
        assert kinds == [
            "start",
            "thinking",  # comprehension
            "intent",
            "thinking",  # orchestration
            "plan",
            "thinking",  # reasoning
            "step",
            "step",
            "thinking",  # evaluation
            "thinking",  # conversation
            "result",
        ]
        assert all("timestamp" in e for e in events)

        intent = events[2]
        assert intent["type"] == "task"
        assert intent["message"] == "Intent: task (80% confident)"

        plan = events[4]
        assert plan["goal"] == "Answer question"
        assert plan["steps_count"] == 1
        assert plan["steps_preview"] == ["Search docs"]

        # Known step resolved by name, unknown step falls back to its id
        assert events[6]["name"] == "Search docs"
        assert events[6]["message"] == "✓ Search docs"
        assert events[6]["duration"] == 1.5
        assert events[7]["name"] == "ghost"
        assert events[7]["message"] == "✗ ghost"

        result = events[-1]
        assert result["response"] == "final answer"
        assert result["quality"] == 0.75
        run_id = events[0]["run_id"]
        assert result["run_id"] == run_id
        assert active_runs[run_id].config["model"] == "x"
        assert repo.save_run.await_count == 2

    def test_conversation_intent_skips_planning_nodes(self, client, graph, repo):
        def comprehension(state):
            state.intent = UserIntent(
                type="conversation", description="chit-chat", confidence=0.5
            )
            return state

        def conversation(state):
            state.response = "hey"
            return state

        graph.comprehension_node.side_effect = comprehension
        graph.conversation_node.side_effect = conversation
        repo.save_run.side_effect = RuntimeError("db down")  # tolerated

        resp = client.post("/engine/run/stream", json={"input": "hi"})

        events = _sse_events(resp.text)
        nodes = [e.get("node") for e in events if e["event"] == "thinking"]
        assert nodes == ["comprehension", "conversation"]
        graph.orchestration_node.assert_not_called()
        graph.reasoning_node.assert_not_called()
        graph.evaluation_node.assert_not_called()
        assert events[-1]["event"] == "result"
        assert events[-1]["response"] == "hey"
        assert events[-1]["quality"] is None

    def test_node_exception_yields_error_event(self, client, graph):
        graph.comprehension_node.side_effect = RuntimeError("classifier exploded")

        resp = client.post("/engine/run/stream", json={"input": "hi"})

        assert resp.status_code == 200  # stream already started
        events = _sse_events(resp.text)
        assert [e["event"] for e in events] == ["start", "thinking", "error"]
        assert events[-1]["error"] == "classifier exploded"
        graph.conversation_node.assert_not_called()

    def test_stream_requires_auth(self, client, graph):
        resp = client.post(
            "/engine/run/stream", json={"input": "hi"}, headers={"X-API-Key": ""}
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /engine/runs/stats
# ---------------------------------------------------------------------------


class TestRunStats:
    def test_stats_include_in_memory_count(self, client, repo, active_runs):
        active_runs["a"] = COREState(user_input="x")
        repo.get_run_stats.return_value = {"total_runs": 7, "completed_runs": 5}

        resp = client.get("/engine/runs/stats")

        assert resp.status_code == 200
        assert resp.json() == {
            "total_runs": 7,
            "completed_runs": 5,
            "active_in_memory": 1,
        }

    def test_stats_db_failure_falls_back_to_memory(self, client, repo, active_runs):
        active_runs["a"] = COREState(user_input="x")
        active_runs["b"] = COREState(user_input="y")
        repo.get_run_stats.side_effect = RuntimeError("pool closed")

        resp = client.get("/engine/runs/stats")

        assert resp.status_code == 200
        assert resp.json() == {
            "total_runs": 2,
            "active_in_memory": 2,
            "error": "pool closed",
        }

    def test_stats_requires_legacy_api_key(self, client):
        resp = client.get("/engine/runs/stats", headers={"X-API-Key": "bogus"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /engine/runs/{run_id}
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_in_memory_run_returned_without_db(self, client, repo, active_runs):
        state = COREState(user_input="cached", response="r")
        active_runs[state.run_id] = state

        resp = client.get(f"/engine/runs/{state.run_id}")

        assert resp.status_code == 200
        assert resp.json()["user_input"] == "cached"
        repo.get_run.assert_not_awaited()

    def test_db_run_with_dict_state(self, client, repo):
        snapshot = COREState(user_input="from db", response="answer").model_dump(
            mode="json"
        )
        repo.get_run.return_value = {"run_id": snapshot["run_id"], "state": snapshot}

        resp = client.get(f"/engine/runs/{snapshot['run_id']}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["user_input"] == "from db"
        assert body["response"] == "answer"
        repo.get_run.assert_awaited_once_with(snapshot["run_id"])

    def test_db_run_with_json_string_state(self, client, repo):
        snapshot = COREState(user_input="stringly", response="s").model_dump(
            mode="json"
        )
        repo.get_run.return_value = {
            "run_id": snapshot["run_id"],
            "state": json.dumps(snapshot),
        }

        resp = client.get(f"/engine/runs/{snapshot['run_id']}")

        assert resp.status_code == 200
        assert resp.json()["user_input"] == "stringly"

    def test_db_run_without_state_built_from_repository_columns(self, client, repo):
        """Rows from run_repository use ``input_text`` and have no current_node.

        Regression: the fallback read ``user_input`` (silently empty) and passed
        ``current_node=None`` which failed validation and turned into a 404.
        """
        run_id = "11111111-1111-1111-1111-111111111111"
        repo.get_run.return_value = {
            "run_id": run_id,
            "user_id": "u-9",
            "input_text": "persisted question",
            "status": "running",
            "state": None,
        }

        resp = client.get(f"/engine/runs/{run_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == run_id
        assert body["user_input"] == "persisted question"
        assert body["user_id"] == "u-9"
        assert body["current_node"] == "START"

    def test_unknown_run_is_404(self, client, repo):
        repo.get_run.return_value = None

        resp = client.get("/engine/runs/does-not-exist")

        assert resp.status_code == 404
        assert "does-not-exist" in resp.json()["detail"]

    def test_db_error_is_404(self, client, repo):
        repo.get_run.side_effect = RuntimeError("timeout")

        resp = client.get("/engine/runs/abc")

        assert resp.status_code == 404

    def test_requires_auth(self, client):
        resp = client.get("/engine/runs/abc", headers={"X-API-Key": ""})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /engine/runs/{run_id}/stream
# ---------------------------------------------------------------------------


class TestStreamRun:
    def test_new_run_streams_and_completes(self, client, graph, active_runs):
        def conversation(state):
            state.response = "streamed reply"
            return state

        graph.comprehension_node.side_effect = lambda s: s  # no intent
        graph.conversation_node.side_effect = conversation

        resp = client.get(
            "/engine/runs/run-xyz/stream", params={"user_input": "tell me"}
        )

        assert resp.status_code == 200
        events = _sse_events(resp.text)
        assert events[0] == {
            "event": "start",
            "run_id": "run-xyz",
            "timestamp": events[0]["timestamp"],
        }
        assert events[-1]["event"] == "complete"
        assert events[-1]["status"] == "success"
        assert events[-1]["response"] == "streamed reply"
        assert active_runs["run-xyz"].user_input == "tell me"

    def test_existing_run_is_reused(self, client, graph, active_runs):
        existing = COREState(user_input="already here")
        active_runs["known"] = existing
        graph.comprehension_node.side_effect = lambda s: s
        graph.conversation_node.side_effect = lambda s: s

        client.get("/engine/runs/known/stream")

        assert graph.comprehension_node.call_args.args[0] is existing

    def test_graph_lookup_failure_yields_error_event(
        self, client, monkeypatch, active_runs
    ):
        def boom():
            raise RuntimeError("graph init failed")

        monkeypatch.setattr(engine, "get_core_graph", boom)

        resp = client.get("/engine/runs/r1/stream")

        events = _sse_events(resp.text)
        assert [e["event"] for e in events] == ["error"]
        assert events[0]["error"] == "graph init failed"
        # New state defaults the input when none supplied
        assert active_runs["r1"].user_input == "Hello, CORE!"


# ---------------------------------------------------------------------------
# _stream_graph_execution (helper, exercised directly to avoid SSE sleeps)
# ---------------------------------------------------------------------------


async def _collect(agen):
    return [e async for e in agen]


class TestStreamGraphExecution:
    async def test_task_pipeline_events(self, graph):
        _wire_task_nodes(graph)
        state = COREState(user_input="do it")

        events = await _collect(engine._stream_graph_execution(None, state))

        kinds = [(e["event"], e.get("node")) for e in events]
        assert kinds == [
            ("node_start", "comprehension"),
            ("node_complete", "comprehension"),
            ("intent_classified", None),
            ("node_start", "orchestration"),
            ("node_complete", "orchestration"),
            ("plan_created", None),
            ("node_start", "reasoning"),
            ("node_complete", "reasoning"),
            ("step_executed", None),
            ("step_executed", None),
            ("node_start", "evaluation"),
            ("node_complete", "evaluation"),
            ("evaluation_complete", None),
            ("node_start", "conversation"),
            ("node_complete", "conversation"),
        ]
        plan = events[5]
        assert plan["steps"] == [
            {"name": "Search docs", "description": "look things up"}
        ]
        failed_step = events[9]
        assert failed_step["status"] == "failure"
        assert failed_step["error"] == "boom"
        assert failed_step["artifacts"] == [] and failed_step["logs"] == []
        evaluation = events[12]
        assert evaluation["overall_status"] == "success"
        assert evaluation["quality_score"] == 0.75
        assert state.response == "final answer"

    async def test_task_pipeline_without_plan_steps_or_eval(self, graph):
        def comprehension(state):
            state.intent = UserIntent(type="question", description="q", confidence=1)
            return state

        graph.comprehension_node.side_effect = comprehension
        for node in ("orchestration", "reasoning", "evaluation", "conversation"):
            getattr(graph, f"{node}_node").side_effect = lambda s: s

        events = await _collect(
            engine._stream_graph_execution(None, COREState(user_input="q"))
        )

        kinds = {e["event"] for e in events}
        assert "plan_created" not in kinds
        assert "step_executed" not in kinds
        assert "evaluation_complete" not in kinds
        assert events[-1] == {**events[-1], "event": "node_complete"}
        assert events[-1]["node"] == "conversation"

    async def test_exception_reports_nodes_executed(self, graph):
        _wire_task_nodes(graph)
        graph.reasoning_node.side_effect = RuntimeError("tool crash")

        events = await _collect(
            engine._stream_graph_execution(None, COREState(user_input="x"))
        )

        err = events[-1]
        assert err["event"] == "error"
        assert err["error"] == "Node execution failed: tool crash"
        assert err["nodes_executed"] == ["comprehension", "orchestration"]


# ---------------------------------------------------------------------------
# DELETE /engine/runs/{run_id}
# ---------------------------------------------------------------------------


def _pool_with(execute_result=None, execute_side_effect=None):
    conn = MagicMock()
    conn.execute = AsyncMock(
        return_value=execute_result, side_effect=execute_side_effect
    )
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


class TestDeleteRun:
    def test_deletes_from_memory_and_database(self, client, monkeypatch, active_runs):
        active_runs["r1"] = COREState(user_input="x")
        pool, conn = _pool_with("DELETE 1")
        monkeypatch.setattr(
            "app.dependencies.get_db_pool", AsyncMock(return_value=pool)
        )

        resp = client.delete("/engine/runs/r1")

        assert resp.status_code == 200
        assert resp.json() == {"message": "Run r1 deleted from: memory, database"}
        assert "r1" not in active_runs
        conn.execute.assert_awaited_once_with(
            "DELETE FROM core_runs WHERE run_id = $1", "r1"
        )

    def test_database_only(self, client, monkeypatch):
        pool, _ = _pool_with("DELETE 1")
        monkeypatch.setattr(
            "app.dependencies.get_db_pool", AsyncMock(return_value=pool)
        )

        resp = client.delete("/engine/runs/r2")

        assert resp.status_code == 200
        assert resp.json()["message"] == "Run r2 deleted from: database"

    def test_memory_only_when_db_fails(self, client, monkeypatch, active_runs):
        active_runs["r3"] = COREState(user_input="x")
        monkeypatch.setattr(
            "app.dependencies.get_db_pool",
            AsyncMock(side_effect=RuntimeError("no pool")),
        )

        resp = client.delete("/engine/runs/r3")

        assert resp.status_code == 200
        assert resp.json()["message"] == "Run r3 deleted from: memory"

    def test_not_found_anywhere_is_404(self, client, monkeypatch):
        pool, _ = _pool_with("DELETE 0")
        monkeypatch.setattr(
            "app.dependencies.get_db_pool", AsyncMock(return_value=pool)
        )

        resp = client.delete("/engine/runs/missing")

        assert resp.status_code == 404

    def test_requires_auth(self, client, active_runs):
        active_runs["r4"] = COREState(user_input="x")
        resp = client.delete("/engine/runs/r4", headers={"X-API-Key": ""})
        assert resp.status_code == 401
        assert "r4" in active_runs


# ---------------------------------------------------------------------------
# GET /engine/runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_memory_runs_with_filters_and_truncation(self, client, repo, active_runs):
        long_input = "x" * 150
        done = COREState(user_input=long_input, user_id="alice", current_node="END")
        done.completed_at = datetime(2026, 1, 2, 3, 4, 5)
        running = COREState(user_input="short", user_id="alice")
        other = COREState(user_input="bob's", user_id="bob")
        for s in (done, running, other):
            active_runs[s.run_id] = s

        resp = client.get(
            "/engine/runs",
            params={"user_id": "alice", "status_filter": "completed", "limit": 5},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_runs"] == 1
        entry = body["runs"][done.run_id]
        assert entry["status"] == "completed"
        assert entry["user_input"] == "x" * 100 + "..."
        assert entry["completed_at"] == "2026-01-02T03:04:05"
        assert entry["source"] == "memory"
        repo.list_runs.assert_awaited_once_with(
            user_id="alice", status="completed", limit=5
        )

    def test_merges_repository_rows_without_duplicates(self, client, repo, active_runs):
        """Repository rows carry ``input_text`` and ISO-string timestamps.

        Regression: the controller called ``.isoformat()`` on those strings and
        read ``user_input``; the AttributeError was swallowed so database runs
        never appeared in the listing.
        """
        mem = COREState(user_input="in memory")
        active_runs[mem.run_id] = mem
        repo.list_runs.return_value = [
            {
                "run_id": mem.run_id,  # duplicate -> memory entry wins
                "input_text": "dup",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00",
                "completed_at": None,
            },
            {
                "run_id": "db-1",
                "input_text": "y" * 120,
                "status": "completed",
                "created_at": "2026-01-01T00:00:00",
                "completed_at": "2026-01-01T00:05:00",
            },
            {
                "run_id": "db-2",
                "input_text": "short",
                "status": "running",
                "created_at": datetime(2026, 2, 1),
                "completed_at": None,
            },
        ]

        resp = client.get("/engine/runs")

        body = resp.json()
        assert body["total_runs"] == 3
        assert body["runs"][mem.run_id]["source"] == "memory"
        assert body["runs"][mem.run_id]["user_input"] == "in memory"
        db1 = body["runs"]["db-1"]
        assert db1 == {
            "status": "completed",
            "current_node": None,
            "user_input": "y" * 100 + "...",
            "started_at": "2026-01-01T00:00:00",
            "completed_at": "2026-01-01T00:05:00",
            "source": "database",
        }
        db2 = body["runs"]["db-2"]
        assert db2["user_input"] == "short"
        assert db2["started_at"] == "2026-02-01T00:00:00"
        assert db2["completed_at"] is None
        repo.list_runs.assert_awaited_once_with(user_id=None, status=None, limit=50)

    def test_include_db_false_skips_repository(self, client, repo):
        resp = client.get("/engine/runs", params={"include_db": "false"})

        assert resp.json() == {"total_runs": 0, "runs": {}}
        repo.list_runs.assert_not_awaited()

    def test_repository_failure_returns_memory_runs(self, client, repo, active_runs):
        s = COREState(user_input="hi")
        active_runs[s.run_id] = s
        repo.list_runs.side_effect = RuntimeError("db down")

        resp = client.get("/engine/runs")

        assert resp.status_code == 200
        assert list(resp.json()["runs"]) == [s.run_id]

    def test_running_filter_excludes_completed(self, client, repo, active_runs):
        done = COREState(user_input="a", current_node="END")
        live = COREState(user_input="b")
        active_runs[done.run_id] = done
        active_runs[live.run_id] = live

        resp = client.get(
            "/engine/runs", params={"status_filter": "running", "include_db": False}
        )

        assert list(resp.json()["runs"]) == [live.run_id]

    def test_invalid_limit_is_422(self, client):
        resp = client.get("/engine/runs", params={"limit": "lots"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_sse_event_format(self):
        out = engine._sse_event({"event": "x"})
        assert out.startswith("data: ") and out.endswith("\n\n")
        payload = json.loads(out[len("data: ") :])
        assert payload["event"] == "x"
        datetime.fromisoformat(payload["timestamp"])

    def test_iso_helper(self):
        assert engine._iso(None) is None
        assert engine._iso("2026-01-01T00:00:00") == "2026-01-01T00:00:00"
        assert engine._iso(datetime(2026, 1, 1)) == "2026-01-01T00:00:00"


def test_engine_only_calls_repository_functions_that_exist():
    """Contract: every run_repository function the controller awaits must exist.

    Regression: ``save_run`` was missing, so every persistence attempt raised
    AttributeError, was swallowed as a warning, and runs were never stored.
    """
    from app.repository import run_repository as real_repo

    for name in ("save_run", "get_run", "list_runs", "get_run_stats"):
        assert callable(getattr(real_repo, name, None)), name
