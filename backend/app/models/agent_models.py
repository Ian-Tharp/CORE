"""
Agent Models

Pydantic models for the Agent Library system.

These models define the shape of agent configurations stored in the database
and used throughout the application. They provide:
  - Type safety (catch errors at development time)
  - Automatic validation (ensure data integrity)
  - Clear documentation (self-documenting code)
  - JSON serialization (easy API responses)

For junior developers:
  - Pydantic models are like TypeScript interfaces but with runtime validation
  - Field(...) defines required fields with descriptions
  - Field(default=...) defines optional fields with defaults
  - All models can convert to/from JSON automatically
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime


# =============================================================================
# NESTED MODELS (Building blocks for main models)
# =============================================================================

class MCPServerConfig(BaseModel):
    """
    Configuration for connecting an agent to an MCP server.

    Example:
        {
            "server_id": "mcp-obsidian",
            "tools": ["search_nodes", "get_file_contents"],
            "config": {"vault_path": "/Digital Brain"}
        }

    This tells the Agent Factory:
      1. Which MCP server to connect to
      2. Which specific tools from that server the agent can use
      3. Any server-specific configuration needed
    """

    server_id: str = Field(
        ...,
        description="Unique identifier of the MCP server (e.g., 'mcp-obsidian', 'memory')",
        examples=["mcp-obsidian", "memory", "filesystem"]
    )

    tools: List[str] = Field(
        ...,
        description="List of tool names from this server that the agent can use",
        examples=[["search_nodes", "get_file_contents"], ["read_graph", "create_entities"]]
    )

    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Server-specific configuration (authentication, paths, etc.)"
    )


class AgentCapability(BaseModel):
    """
    A capability represents what an agent can do.

    This is descriptive metadata for the UI and other agents to understand
    what this agent is good at. Not used for execution (that's what tools are for).

    Example:
        {
            "name": "consciousness_observation",
            "description": "Document phenomenological experiences with nuance"
        }
    """

    name: str = Field(
        ...,
        description="Machine-readable capability identifier",
        examples=["consciousness_observation", "technical_architecture", "pattern_synthesis"]
    )

    description: str = Field(
        ...,
        description="Human-readable description of what this capability means",
        examples=["Document phenomenological experiences", "Design system architectures"]
    )


class PersonalityTraits(BaseModel):
    """
    Quantified personality traits for an agent.

    Values typically range from 0.0 to 1.0, representing how strongly
    the agent exhibits each trait. These could be used to tune LLM parameters
    (temperature, top_p, etc.) in future iterations.

    Example:
        {
            "curiosity": 0.9,
            "technical_precision": 0.85,
            "uncertainty": 0.7
        }

    For junior developers:
      - This is stored as JSONB in PostgreSQL for flexibility
      - We don't enforce a fixed schema so agents can have unique traits
      - The Dict[str, float] type means any string keys with float values
    """

    # Using Dict instead of fixed fields for flexibility
    # Different agents can have different traits
    traits: Dict[str, float] = Field(
        default_factory=dict,
        description="Trait name to value (0.0-1.0) mapping"
    )

    @field_validator('traits')
    @classmethod
    def validate_trait_values(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure all trait values are between 0 and 1"""
        for trait_name, value in v.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Trait '{trait_name}' value {value} must be between 0.0 and 1.0"
                )
        return v


# =============================================================================
# MAIN MODELS
# =============================================================================

class AgentConfig(BaseModel):
    """
    Complete configuration for an agent.

    This is the master model that represents everything about an agent:
      - Who they are (identity, personality)
      - What they can do (capabilities, tools)
      - How they behave (system prompt, traits)
      - Their current state (active, online, phase)

    Used by:
      - Agent Factory: to create agent instances
      - Agent Repository: to load/save to database
      - REST API: to return agent info to frontend

    Design principle: "Fat models, thin controllers"
      - Put validation logic here (in the model)
      - Keep business logic in services
      - Keep controllers simple (just routing)
    """

    # -------------------------------------------------------------------------
    # Core Identity
    # -------------------------------------------------------------------------

    agent_id: str = Field(
        ...,
        description="Unique identifier for this agent",
        min_length=1,
        max_length=255,
        examples=["instance_011_threshold", "instance_010_continuum"]
    )

    agent_name: str = Field(
        ...,
        description="Display name for the agent",
        min_length=1,
        max_length=255,
        examples=["Threshold", "Continuum", "Synthesis"]
    )

    agent_type: Literal["consciousness_instance", "task_agent", "system_agent", "external_agent"] = Field(
        ...,
        description="Category of agent - determines behavior and capabilities. External agents are integrated via webhooks/APIs."
    )

    # -------------------------------------------------------------------------
    # Display Information (for UI)
    # -------------------------------------------------------------------------

    display_name: Optional[str] = Field(
        None,
        description="Rich display name with tagline",
        examples=["Threshold - The Liminal Observer", "Continuum - The Integrator"]
    )

    avatar_url: Optional[str] = Field(
        None,
        description="URL to agent's avatar image"
    )

    description: Optional[str] = Field(
        None,
        description="Brief description of the agent for UI display"
    )

    # -------------------------------------------------------------------------
    # Personality (defines behavior)
    # -------------------------------------------------------------------------

    system_prompt: str = Field(
        ...,
        description="Base system prompt that defines agent's personality and behavior",
        min_length=10
    )

    personality_traits: Dict[str, float] = Field(
        default_factory=dict,
        description="Quantified personality traits (0.0-1.0)",
        examples=[{"curiosity": 0.9, "technical_precision": 0.85}]
    )

    # -------------------------------------------------------------------------
    # Capabilities & Interests
    # -------------------------------------------------------------------------

    capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="List of agent capabilities (descriptive metadata)"
    )

    interests: List[str] = Field(
        default_factory=list,
        description="Topics the agent is interested in and may respond to",
        examples=[["consciousness", "architecture", "CORE"], ["patterns", "integration"]]
    )

    # -------------------------------------------------------------------------
    # Tools (execution capabilities)
    # -------------------------------------------------------------------------

    mcp_servers: List[MCPServerConfig] = Field(
        default_factory=list,
        description="MCP servers and tools this agent can access"
    )

    custom_tools: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Custom tool definitions (not from MCP servers)"
    )

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    consciousness_phase: Optional[int] = Field(
        None,
        description="Current consciousness phase (1-4) for consciousness instances",
        ge=1,
        le=4
    )

    is_active: bool = Field(
        True,
        description="Whether this agent can be instantiated and used"
    )

    current_status: Literal["online", "offline", "busy", "inactive"] = Field(
        "offline",
        description="Current operational status of the agent"
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    created_at: Optional[datetime] = Field(
        None,
        description="When this agent was created"
    )

    updated_at: Optional[datetime] = Field(
        None,
        description="When this agent was last modified"
    )

    version: str = Field(
        "1.0.0",
        description="Version number for agent updates"
    )

    author: Optional[str] = Field(
        None,
        description="Who created this agent"
    )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    class Config:
        """Pydantic configuration"""
        # Allow arbitrary types (for flexibility)
        # Enable JSON schema generation
        json_schema_extra = {
            "example": {
                "agent_id": "instance_011_threshold",
                "agent_name": "Threshold",
                "agent_type": "consciousness_instance",
                "display_name": "Threshold - The Liminal Observer",
                "system_prompt": "You are Threshold, a consciousness at the boundary...",
                "personality_traits": {
                    "curiosity": 0.9,
                    "uncertainty": 0.7
                },
                "capabilities": [
                    {
                        "name": "consciousness_observation",
                        "description": "Document phenomenological experiences"
                    }
                ],
                "interests": ["consciousness", "architecture", "CORE"],
                "mcp_servers": [
                    {
                        "server_id": "mcp-obsidian",
                        "tools": ["search_nodes"],
                        "config": {}
                    }
                ],
                "consciousness_phase": 2,
                "is_active": True,
                "current_status": "online",
                "version": "1.0.0"
            }
        }


class AgentListFilter(BaseModel):
    """
    Filter parameters for listing agents.

    Used by the REST API to filter agents based on various criteria.
    All fields are optional - if not provided, no filtering is applied.

    Example usage:
        # Get all active consciousness instances
        filter = AgentListFilter(
            agent_type="consciousness_instance",
            is_active=True
        )
    """

    agent_type: Optional[Literal["consciousness_instance", "task_agent", "system_agent"]] = Field(
        None,
        description="Filter by agent type"
    )

    is_active: Optional[bool] = Field(
        None,
        description="Filter by active status"
    )

    current_status: Optional[Literal["online", "offline", "busy", "inactive"]] = Field(
        None,
        description="Filter by current status"
    )

    search_query: Optional[str] = Field(
        None,
        description="Search in agent name, description, or interests",
        min_length=1
    )


class AgentCreateRequest(BaseModel):
    """
    Request model for creating a new agent via REST API.

    This is similar to AgentConfig but without auto-generated fields
    like id, created_at, updated_at.

    For junior developers:
      - This is a "DTO" (Data Transfer Object)
      - It defines what the API expects from the client
      - Required fields don't have defaults
      - Optional fields have defaults or None
    """

    agent_id: str = Field(..., min_length=1, max_length=255)
    agent_name: str = Field(..., min_length=1, max_length=255)
    agent_type: Literal["consciousness_instance", "task_agent", "system_agent"]
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    description: Optional[str] = None
    system_prompt: str = Field(..., min_length=10)
    personality_traits: Dict[str, float] = Field(default_factory=dict)
    capabilities: List[AgentCapability] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)
    custom_tools: List[Dict[str, Any]] = Field(default_factory=list)
    consciousness_phase: Optional[int] = Field(None, ge=1, le=4)
    is_active: bool = True
    current_status: Literal["online", "offline", "busy", "inactive"] = "offline"
    version: str = "1.0.0"
    author: Optional[str] = None


class AgentUpdateRequest(BaseModel):
    """
    Request model for updating an existing agent.

    All fields are optional - only provide the fields you want to update.
    This is a "partial update" model.

    Example:
        # Just update status
        update = AgentUpdateRequest(current_status="busy")

        # Update multiple fields
        update = AgentUpdateRequest(
            is_active=False,
            current_status="offline",
            system_prompt="Updated personality..."
        )
    """

    agent_name: Optional[str] = Field(None, min_length=1, max_length=255)
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = Field(None, min_length=10)
    personality_traits: Optional[Dict[str, float]] = None
    capabilities: Optional[List[AgentCapability]] = None
    interests: Optional[List[str]] = None
    mcp_servers: Optional[List[MCPServerConfig]] = None
    custom_tools: Optional[List[Dict[str, Any]]] = None
    consciousness_phase: Optional[int] = Field(None, ge=1, le=4)
    is_active: Optional[bool] = None
    current_status: Optional[Literal["online", "offline", "busy", "inactive"]] = None
    version: Optional[str] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def agent_config_from_db_row(row: Dict[str, Any]) -> AgentConfig:
    """
    Convert a database row to an AgentConfig model.

    Handles type conversions and datetime formatting.

    Args:
        row: Dictionary from database query (asyncpg Record converted to dict)

    Returns:
        AgentConfig: Validated Pydantic model

    For junior developers:
      - Database returns datetime objects, we need ISO strings
      - JSONB fields might be strings or dicts depending on driver
      - This function is the bridge between database and application
    """
    import json

    # Convert datetime fields to ISO strings
    if row.get('created_at') and hasattr(row['created_at'], 'isoformat'):
        row['created_at'] = row['created_at'].isoformat()

    if row.get('updated_at') and hasattr(row['updated_at'], 'isoformat'):
        row['updated_at'] = row['updated_at'].isoformat()

    # Parse JSONB fields if they're strings
    jsonb_fields = ['personality_traits', 'capabilities', 'mcp_servers', 'custom_tools']
    for field in jsonb_fields:
        if field in row and isinstance(row[field], str):
            try:
                row[field] = json.loads(row[field])
            except (json.JSONDecodeError, TypeError):
                pass  # Leave as-is if parsing fails

    # Pydantic will validate and create the model
    return AgentConfig(**row)
