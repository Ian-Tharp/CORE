"""
Tests for council_models.py — enum values, bounds, and defaults.

Sentinel-value contract:
- Remove CouncilSession.max_rounds le=10 → 100-round sessions stored → loop never ends → test fails
- Remove CouncilSession.current_round ge=1 → round=0 stored → 1-indexed display breaks → test fails
- Remove CouncilPerspective.confidence bounds → confidence=2.0 stored → consensus math breaks → test fails
- Remove CouncilVote.weight le=2.0 → weight=100 stored → single voter controls all decisions → test fails
- Remove SessionStatus.GATHERING default → new sessions in unknown state → lifecycle crashes → test fails
"""

from __future__ import annotations

import pytest
from uuid import UUID
from pydantic import ValidationError

from app.models.council_models import (
    CouncilPerspective,
    CouncilSession,
    CouncilSessionFull,
    CouncilVote,
    SessionStatus,
    SessionSummary,
    VoiceType,
    VoteType,
)
from uuid import uuid4
from datetime import datetime


# ===========================================================================
# Helpers
# ===========================================================================


def _min_perspective(**overrides) -> dict:
    base = {
        "session_id": uuid4(),
        "voice_type": VoiceType.STRATEGIC,
        "voice_name": "Ethicist",
        "position": "SENTINEL_POSITION",
        "reasoning": "SENTINEL_REASONING",
    }
    base.update(overrides)
    return base


def _min_vote(**overrides) -> dict:
    base = {
        "session_id": uuid4(),
        "perspective_id": uuid4(),
        "voter_voice_type": VoiceType.META,
        "voter_voice_name": "Synthesizer",
        "vote_type": VoteType.AGREE,
    }
    base.update(overrides)
    return base


# ===========================================================================
# SessionStatus enum
# ===========================================================================


class TestSessionStatus:
    def test_gathering_value(self):
        """
        SENTINEL — GATHERING must equal 'gathering'.
        Change string → API filter on gathering sessions returns nothing → test fails.
        """
        assert SessionStatus.GATHERING == "gathering"

    def test_deliberating_value(self):
        """DELIBERATING must equal 'deliberating'."""
        assert SessionStatus.DELIBERATING == "deliberating"

    def test_voting_value(self):
        """VOTING must equal 'voting'."""
        assert SessionStatus.VOTING == "voting"

    def test_synthesizing_value(self):
        """SYNTHESIZING must equal 'synthesizing'."""
        assert SessionStatus.SYNTHESIZING == "synthesizing"

    def test_complete_value(self):
        """COMPLETE must equal 'complete'."""
        assert SessionStatus.COMPLETE == "complete"

    def test_cancelled_value(self):
        """CANCELLED must equal 'cancelled'."""
        assert SessionStatus.CANCELLED == "cancelled"


# ===========================================================================
# VoteType enum
# ===========================================================================


class TestVoteType:
    def test_agree_value(self):
        """
        SENTINEL — AGREE must equal 'agree'.
        Change → vote counts by type are wrong → consensus calculation breaks → test fails.
        """
        assert VoteType.AGREE == "agree"

    def test_disagree_value(self):
        """DISAGREE must equal 'disagree'."""
        assert VoteType.DISAGREE == "disagree"

    def test_abstain_value(self):
        """ABSTAIN must equal 'abstain'."""
        assert VoteType.ABSTAIN == "abstain"

    def test_amend_value(self):
        """AMEND must equal 'amend'."""
        assert VoteType.AMEND == "amend"


# ===========================================================================
# CouncilSession — bounds and defaults
# ===========================================================================


class TestCouncilSession:
    def test_valid_session_created(self):
        """Minimal valid session must be accepted."""
        session = CouncilSession(topic="SENTINEL_TOPIC")
        assert session.topic == "SENTINEL_TOPIC"

    def test_session_id_auto_generated(self):
        """session_id must be auto-generated UUID."""
        session = CouncilSession(topic="test")
        assert isinstance(session.session_id, UUID)

    def test_default_status_is_gathering(self):
        """
        SENTINEL — default status must be GATHERING.
        Default COMPLETE → new sessions immediately closed → no deliberation happens → test fails.
        """
        session = CouncilSession(topic="test")
        assert session.status == SessionStatus.GATHERING

    def test_default_max_rounds_is_3(self):
        """
        SENTINEL — default max_rounds must be 3.
        Default 1 → session forces synthesis after first round → deliberation skipped → test fails.
        """
        session = CouncilSession(topic="test")
        assert session.max_rounds == 3

    def test_default_current_round_is_1(self):
        """default current_round must be 1 (first round)."""
        session = CouncilSession(topic="test")
        assert session.current_round == 1

    def test_current_round_below_1_rejected(self):
        """
        SENTINEL — current_round < 1 must raise ValidationError.
        Remove ge=1 → round=0 stored → 1-indexed display shows round 0 → test fails.
        """
        with pytest.raises(ValidationError):
            CouncilSession(topic="test", current_round=0)

    def test_max_rounds_above_10_rejected(self):
        """
        SENTINEL — max_rounds > 10 must raise ValidationError.
        Remove le=10 → 100-round session stored → deliberation never terminates → test fails.
        """
        with pytest.raises(ValidationError):
            CouncilSession(topic="test", max_rounds=11)

    def test_max_rounds_below_1_rejected(self):
        """max_rounds < 1 must raise ValidationError."""
        with pytest.raises(ValidationError):
            CouncilSession(topic="test", max_rounds=0)

    def test_max_rounds_boundary_10_accepted(self):
        """max_rounds=10 must be accepted (boundary)."""
        session = CouncilSession(topic="test", max_rounds=10)
        assert session.max_rounds == 10

    def test_synthesis_defaults_none(self):
        """synthesis defaults to None (not yet synthesized)."""
        session = CouncilSession(topic="test")
        assert session.synthesis is None

    def test_summoned_voices_defaults_empty(self):
        """summoned_voices defaults to empty list."""
        session = CouncilSession(topic="test")
        assert session.summoned_voices == []


# ===========================================================================
# CouncilPerspective — confidence bounds
# ===========================================================================


class TestCouncilPerspective:
    def test_valid_perspective_created(self):
        """Minimal valid perspective must be accepted."""
        p = CouncilPerspective(**_min_perspective())
        assert p.position == "SENTINEL_POSITION"
        assert isinstance(p.perspective_id, UUID)

    def test_confidence_above_1_rejected(self):
        """
        SENTINEL — confidence > 1.0 must raise ValidationError.
        Remove le=1.0 → confidence=2.0 stored → weighted consensus math breaks → test fails.
        """
        with pytest.raises(ValidationError):
            CouncilPerspective(**_min_perspective(confidence=1.1))

    def test_confidence_below_0_rejected(self):
        """confidence < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            CouncilPerspective(**_min_perspective(confidence=-0.1))

    def test_default_confidence_is_05(self):
        """
        SENTINEL — default confidence must be 0.5.
        Default 1.0 → all unscored perspectives treated as fully confident → test fails.
        """
        p = CouncilPerspective(**_min_perspective())
        assert p.confidence == pytest.approx(0.5)

    def test_round_below_1_rejected(self):
        """
        SENTINEL — round < 1 must raise ValidationError.
        Remove ge=1 → round=0 stored → display shows 'Round 0' → test fails.
        """
        with pytest.raises(ValidationError):
            CouncilPerspective(**_min_perspective(round=0))

    def test_default_round_is_1(self):
        """default round must be 1."""
        p = CouncilPerspective(**_min_perspective())
        assert p.round == 1

    def test_references_perspectives_defaults_empty(self):
        """references_perspectives defaults to empty list."""
        p = CouncilPerspective(**_min_perspective())
        assert p.references_perspectives == []


# ===========================================================================
# CouncilVote — weight bounds
# ===========================================================================


class TestCouncilVote:
    def test_valid_vote_created(self):
        """Minimal valid vote must be accepted."""
        vote = CouncilVote(**_min_vote())
        assert vote.vote_type == VoteType.AGREE
        assert isinstance(vote.vote_id, UUID)

    def test_weight_above_2_rejected(self):
        """
        SENTINEL — weight > 2.0 must raise ValidationError.
        Remove le=2.0 → single voter with weight=100 overrides all others → test fails.
        """
        with pytest.raises(ValidationError):
            CouncilVote(**_min_vote(weight=2.1))

    def test_weight_below_0_rejected(self):
        """weight < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            CouncilVote(**_min_vote(weight=-0.1))

    def test_default_weight_is_1(self):
        """
        SENTINEL — default weight must be 1.0.
        Default 0.0 → all unweighted votes count as nothing → consensus never reached → test fails.
        """
        vote = CouncilVote(**_min_vote())
        assert vote.weight == pytest.approx(1.0)

    def test_weight_boundary_2_accepted(self):
        """weight=2.0 must be accepted (boundary — e.g., senior voice)."""
        vote = CouncilVote(**_min_vote(weight=2.0))
        assert vote.weight == pytest.approx(2.0)

    def test_comment_defaults_none(self):
        """comment defaults to None."""
        vote = CouncilVote(**_min_vote())
        assert vote.comment is None

    def test_amendment_defaults_none(self):
        """amendment defaults to None."""
        vote = CouncilVote(**_min_vote())
        assert vote.amendment is None


# ===========================================================================
# CouncilSessionFull — composite
# ===========================================================================


class TestCouncilSessionFull:
    def test_contains_session_perspectives_votes(self):
        """
        SENTINEL — CouncilSessionFull must include session + perspectives + votes lists.
        Remove votes list → deliberation summary incomplete → test fails.
        """
        session = CouncilSession(topic="SENTINEL_FULL")
        full = CouncilSessionFull(session=session)
        assert full.session.topic == "SENTINEL_FULL"
        assert full.perspectives == []
        assert full.votes == []

    def test_session_with_perspectives_and_votes(self):
        """Full session with data must store everything."""
        session = CouncilSession(topic="test")
        p = CouncilPerspective(**_min_perspective(session_id=session.session_id))
        v = CouncilVote(**_min_vote(session_id=session.session_id))
        full = CouncilSessionFull(session=session, perspectives=[p], votes=[v])
        assert len(full.perspectives) == 1
        assert len(full.votes) == 1
