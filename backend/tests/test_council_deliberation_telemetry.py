"""
Tests for council deliberation -> cognition telemetry emission.

The deliberation flow publishes WebSocket cognition events (council state plus
stage-mapped agent_activity / task_progress) so the command deck lights up. These
tests exercise the emit helpers in isolation with a mocked event_publisher.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.ws_events import CouncilEventType, TaskStage
from app.services.council import deliberation_service as ds
from app.services.council.deliberation_service import CouncilService


def _make_service() -> CouncilService:
    """Construct a CouncilService without touching real model/bridge deps."""
    return CouncilService(
        model_router=MagicMock(),
        default_model="test-model",
        consciousness_bridge=MagicMock(),
    )


@pytest.fixture
def mock_publisher():
    """Patch the module-level event_publisher with an AsyncMock."""
    pub = AsyncMock()
    with patch.object(ds, "event_publisher", pub):
        yield pub


class TestEmitSafe:
    async def test_awaits_factory(self):
        svc = _make_service()
        called = {}

        async def coro():
            called["yes"] = True

        await svc._emit_safe(lambda: coro())
        assert called.get("yes") is True

    async def test_swallows_async_error(self):
        svc = _make_service()

        async def boom():
            raise RuntimeError("publish failed")

        # Telemetry failure must never propagate.
        await svc._emit_safe(lambda: boom())

    async def test_swallows_sync_construction_error(self):
        svc = _make_service()

        def factory():
            raise ValueError("bad event payload")

        await svc._emit_safe(factory)  # must not raise


class TestRoundStart:
    async def test_emits_debate_round_and_core_thinking(self, mock_publisher):
        svc = _make_service()

        await svc._emit_round_start(
            "sess-1",
            round_num=2,
            rounds=3,
            summoned_ids=["core_c", "core_o", "core_r", "core_e", "oracle"],
        )

        published = [c.args[0] for c in mock_publisher.publish.call_args_list]
        council = [e for e in published if e.__class__.__name__ == "CouncilEvent"]
        tasks = [e for e in published if e.__class__.__name__ == "TaskProgressEvent"]

        assert len(council) == 1
        assert council[0].event == CouncilEventType.DEBATE_ROUND
        assert council[0].round_number == 2
        assert len(tasks) == 1
        assert tasks[0].stage == TaskStage.PROCESSING

        thinking_ids = {
            c.kwargs["agent_id"] for c in mock_publisher.agent_thinking.call_args_list
        }
        assert thinking_ids == {
            "council-comprehension",
            "council-orchestration",
            "council-reasoning",
            "council-evaluation",
        }

    async def test_skips_unsummoned_core_voices(self, mock_publisher):
        svc = _make_service()

        await svc._emit_round_start(
            "sess-1", round_num=1, rounds=1, summoned_ids=["core_c"]
        )

        thinking_ids = {
            c.kwargs["agent_id"] for c in mock_publisher.agent_thinking.call_args_list
        }
        assert thinking_ids == {"council-comprehension"}


class TestRoundPerspectives:
    async def test_emits_perspective_and_completes_core_stage(self, mock_publisher):
        svc = _make_service()
        perspectives = [
            {
                "voice_id": "core_r",
                "voice_name": "CORE-R",
                "position": "Do X",
                "confidence": 0.8,
            },
            {
                "voice_id": "oracle",
                "voice_name": "Oracle",
                "position": "A vision",
                "confidence": 0.6,
            },
        ]

        await svc._emit_round_perspectives("sess-1", perspectives)

        # One council perspective per voice.
        assert mock_publisher.council_perspective.call_count == 2
        first = mock_publisher.council_perspective.call_args_list[0].kwargs
        assert first["agent_id"] == "CORE-R"
        assert first["content"] == "Do X"
        assert first["confidence"] == 0.8

        # agent_complete only for the CORE voice, mapped to its pipeline stage.
        assert mock_publisher.agent_complete.call_count == 1
        ac = mock_publisher.agent_complete.call_args_list[0].kwargs
        assert ac["agent_id"] == "council-reasoning"
        assert ac["action"] == "reasoning"

    async def test_non_core_perspectives_do_not_complete_a_stage(self, mock_publisher):
        svc = _make_service()
        await svc._emit_round_perspectives(
            "sess-1",
            [{"voice_id": "oracle", "voice_name": "Oracle", "position": "x"}],
        )
        assert mock_publisher.council_perspective.call_count == 1
        assert mock_publisher.agent_complete.call_count == 0
