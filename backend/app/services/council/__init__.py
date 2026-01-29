"""
Council Services Module

Provides the voice registry and deliberation infrastructure for
multi-perspective AI reasoning (the Council system).
"""

from .voice_registry import (
    VoiceCategory,
    VoiceDefinition,
    VOICE_REGISTRY,
    get_voice,
    list_voices,
    get_voices_by_category,
    get_core_voices,
    get_council_voices,
)

__all__ = [
    "VoiceCategory",
    "VoiceDefinition",
    "VOICE_REGISTRY",
    "get_voice",
    "list_voices",
    "get_voices_by_category",
    "get_core_voices",
    "get_council_voices",
]
