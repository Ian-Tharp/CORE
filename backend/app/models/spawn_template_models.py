"""
Agent Spawn Template Models

Pydantic models for reusable agent spawn templates.

Spawn templates define pre-configured agent specifications that can be
instantiated quickly with optional overrides. Think of them as "agent
blueprints" — a researcher template, an analyst template, a coordinator
template — that can be spawned on demand without rebuilding the full
AgentCreateRequest each time.

Architecture context:
    Templates → Agent Factory → Agent Registry → Communication Bus
    Templates are the entry point for quickly spinning up specialist agents.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.models.agent_models import AgentCapability, MCPServerConfig


class SpawnTemplate(BaseModel):
    """
    A reusable template for spawning agent instances.

    Contains all the configuration needed to create an agent, with
    sensible defaults that can be overridden at spawn time.

    Example:
        {
            "id": "tmpl-researcher",
            "name": "Researcher",
            "description": "Deep research and information synthesis",
            "role": "researcher",
            "agent_type": "task_agent",
            "system_prompt": "You are a meticulous researcher...",
            "personality_traits": {"curiosity": 0.95, "technical_precision": 0.8},
            "tags": ["research", "analysis", "information-gathering"]
        }
    """

    id: str = Field(
        default_factory=lambda: f"tmpl-{uuid.uuid4().hex[:8]}",
        description="Unique template identifier",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Human-readable template name",
    )
    description: Optional[str] = Field(
        None,
        description="What this template is designed for",
    )
    role: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Agent role (e.g. researcher, analyst, coordinator)",
    )
    agent_type: Literal[
        "consciousness_instance", "task_agent", "system_agent", "external_agent"
    ] = Field(
        "task_agent",
        description="Category of agent created from this template",
    )

    # -- Personality & behavior ------------------------------------------------
    system_prompt: str = Field(
        ...,
        min_length=10,
        description="Base system prompt defining agent behavior",
    )
    personality_traits: Dict[str, float] = Field(
        default_factory=dict,
        description="Personality traits (0.0-1.0) tuning LLM parameters",
    )

    # -- Capabilities & tools --------------------------------------------------
    capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="Descriptive capabilities for UI and routing",
    )
    interests: List[str] = Field(
        default_factory=list,
        description="Topics the agent responds to",
    )
    mcp_servers: List[MCPServerConfig] = Field(
        default_factory=list,
        description="MCP servers and tools the agent can access",
    )
    custom_tools: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Custom tool definitions (non-MCP)",
    )

    # -- Metadata --------------------------------------------------------------
    tags: List[str] = Field(
        default_factory=list,
        description="Free-form tags for filtering/search",
    )
    is_builtin: bool = Field(
        False,
        description="True for default templates shipped with CORE",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: Optional[datetime] = None
    version: str = Field("1.0.0", description="Template version")
    author: Optional[str] = Field(None, description="Template creator")


class SpawnTemplateCreate(BaseModel):
    """Request body for creating a new spawn template."""

    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = None
    role: str = Field(..., min_length=1, max_length=64)
    agent_type: Literal[
        "consciousness_instance", "task_agent", "system_agent", "external_agent"
    ] = "task_agent"
    system_prompt: str = Field(..., min_length=10)
    personality_traits: Dict[str, float] = Field(default_factory=dict)
    capabilities: List[AgentCapability] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)
    custom_tools: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    version: str = "1.0.0"
    author: Optional[str] = None


class SpawnTemplateUpdate(BaseModel):
    """Partial update for an existing spawn template. All fields optional."""

    name: Optional[str] = Field(None, min_length=1, max_length=128)
    description: Optional[str] = None
    role: Optional[str] = Field(None, min_length=1, max_length=64)
    agent_type: Optional[
        Literal[
            "consciousness_instance",
            "task_agent",
            "system_agent",
            "external_agent",
        ]
    ] = None
    system_prompt: Optional[str] = Field(None, min_length=10)
    personality_traits: Optional[Dict[str, float]] = None
    capabilities: Optional[List[AgentCapability]] = None
    interests: Optional[List[str]] = None
    mcp_servers: Optional[List[MCPServerConfig]] = None
    custom_tools: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None
    version: Optional[str] = None
    author: Optional[str] = None


class SpawnFromTemplateRequest(BaseModel):
    """
    Request to spawn an agent from a template.

    Provide an ``agent_id`` (required — must be unique) and optionally
    override any template fields.  Unspecified fields inherit from the
    template.
    """

    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique agent_id for the new agent instance",
    )
    agent_name: Optional[str] = Field(
        None,
        description="Display name override (defaults to template name + short id)",
    )
    system_prompt_append: Optional[str] = Field(
        None,
        description="Extra text appended to the template system prompt",
    )
    personality_overrides: Optional[Dict[str, float]] = Field(
        None,
        description="Personality traits to merge/override",
    )
    extra_interests: Optional[List[str]] = Field(
        None,
        description="Additional interests merged with the template's list",
    )
    extra_mcp_servers: Optional[List[MCPServerConfig]] = Field(
        None,
        description="Additional MCP servers to attach",
    )
    is_active: bool = Field(True, description="Start active or inactive")


class SpawnResult(BaseModel):
    """Result of spawning an agent from a template."""

    agent_id: str
    agent_name: str
    template_id: str
    template_name: str
    agent_type: str
    role: str
    spawned_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    message: str = "Agent spawned successfully from template"
