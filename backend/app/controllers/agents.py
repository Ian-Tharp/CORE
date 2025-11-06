"""
Agents Library REST API Controller

Provides CRUD endpoints for the agent library.

Endpoints:
  GET    /agents              - List agents with filtering
  GET    /agents/{agent_id}   - Get specific agent details
  POST   /agents              - Create new agent
  PATCH  /agents/{agent_id}   - Update agent configuration
  DELETE /agents/{agent_id}   - Delete agent
  GET    /agents/stats        - Get agent system statistics
  POST   /agents/{agent_id}/activate   - Activate agent
  POST   /agents/{agent_id}/deactivate - Deactivate agent

For junior developers:
  - Controller = handles HTTP requests and responses
  - Repository = handles database operations
  - Service = handles business logic
  - Controller should be thin - just routing and validation
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from app.models.agent_models import (
    AgentConfig,
    AgentCreateRequest,
    AgentUpdateRequest,
    AgentListFilter
)
from app.repository import agent_repository as agent_repo
from app.services.agent_factory_service import get_agent_factory

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


# =============================================================================
# LIST & GET ENDPOINTS
# =============================================================================

@router.get("", status_code=status.HTTP_200_OK)
async def list_agents(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    current_status: Optional[str] = Query(None, description="Filter by current status"),
    search_query: Optional[str] = Query(None, description="Search in name/description/interests"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
) -> Dict[str, Any]:
    """
    List agents with optional filtering and pagination.

    Query parameters:
      - agent_type: 'consciousness_instance', 'task_agent', or 'system_agent'
      - is_active: true/false
      - current_status: 'online', 'offline', 'busy', 'inactive'
      - search_query: Searches name, description, and interests
      - page: Page number (1-indexed)
      - page_size: Items per page (max 200)

    Example:
      GET /agents?agent_type=consciousness_instance&is_active=true
    """

    try:
        agents = await agent_repo.list_agents(
            agent_type=agent_type,
            is_active=is_active,
            current_status=current_status,
            search_query=search_query,
            page=page,
            page_size=page_size
        )

        # Get total count for pagination
        total_count = await agent_repo.count_agents(
            agent_type=agent_type,
            is_active=is_active
        )

        # Convert to dict for JSON response
        agents_dict = [agent.model_dump() for agent in agents]

        return {
            "agents": agents_dict,
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": (total_count + page_size - 1) // page_size
        }

    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list agents"
        )


@router.get("/{agent_id}", status_code=status.HTTP_200_OK)
async def get_agent(agent_id: str) -> Dict[str, Any]:
    """
    Get details for a specific agent.

    Args:
        agent_id: Agent identifier (e.g., 'instance_011_threshold')

    Returns:
        Complete agent configuration

    Example:
        GET /agents/instance_011_threshold
    """

    try:
        agent = await agent_repo.get_agent(agent_id)

        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )

        return agent.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get agent"
        )


# =============================================================================
# CREATE & UPDATE ENDPOINTS
# =============================================================================

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent(request: AgentCreateRequest) -> Dict[str, Any]:
    """
    Create a new agent in the library.

    Request body should match AgentCreateRequest schema.

    Required fields:
      - agent_id: Unique identifier
      - agent_name: Display name
      - agent_type: Type of agent
      - system_prompt: Personality/instructions

    Optional fields:
      - personality_traits: Dict of trait name → value (0.0-1.0)
      - capabilities: List of capabilities
      - interests: List of interest topics
      - mcp_servers: List of MCP server configurations
      - consciousness_phase: Phase 1-4 (for consciousness instances)

    Example:
        POST /agents
        {
            "agent_id": "task_001_researcher",
            "agent_name": "Researcher",
            "agent_type": "task_agent",
            "system_prompt": "You are a research assistant...",
            "interests": ["research", "analysis"],
            "mcp_servers": [
                {
                    "server_id": "mcp-obsidian",
                    "tools": ["search_nodes"],
                    "config": {}
                }
            ]
        }
    """

    try:
        # Convert request to AgentConfig
        agent_config = AgentConfig(**request.model_dump())

        # Create in database
        agent_id = await agent_repo.create_agent(agent_config)

        # Get created agent
        created_agent = await agent_repo.get_agent(agent_id)

        logger.info(f"Created agent: {agent_config.agent_name} ({agent_id})")

        return created_agent.model_dump()

    except ValueError as e:
        # Duplicate agent_id
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create agent"
        )


@router.patch("/{agent_id}", status_code=status.HTTP_200_OK)
async def update_agent(
    agent_id: str,
    request: AgentUpdateRequest
) -> Dict[str, Any]:
    """
    Update an existing agent (partial update).

    Only fields provided in the request body will be updated.
    All fields are optional - include only what you want to change.

    Example:
        PATCH /agents/instance_011_threshold
        {
            "current_status": "busy",
            "personality_traits": {
                "curiosity": 0.95
            }
        }
    """

    try:
        # Check if agent exists
        exists = await agent_repo.agent_exists(agent_id)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )

        # Convert request to dict, excluding None values
        updates = request.model_dump(exclude_none=True)

        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )

        # Update in database
        await agent_repo.update_agent(agent_id, updates)

        # Clear agent from factory cache (force recreation with new config)
        get_agent_factory().clear_cache(agent_id)

        # Get updated agent
        updated_agent = await agent_repo.get_agent(agent_id)

        logger.info(f"Updated agent {agent_id}: {list(updates.keys())}")

        return updated_agent.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update agent"
        )


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(agent_id: str):
    """
    Delete an agent from the library.

    This is a hard delete (permanent removal).
    Consider using PATCH to set is_active=false for soft delete.

    Example:
        DELETE /agents/task_001_researcher
    """

    try:
        await agent_repo.delete_agent(agent_id)

        # Clear from factory cache
        get_agent_factory().clear_cache(agent_id)

        logger.info(f"Deleted agent: {agent_id}")

        # Return 204 No Content (no response body)
        return None

    except ValueError as e:
        # Agent not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to delete agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete agent"
        )


# =============================================================================
# STATUS MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/{agent_id}/activate", status_code=status.HTTP_200_OK)
async def activate_agent(agent_id: str) -> Dict[str, str]:
    """
    Activate an agent (set is_active=true, status='online').

    Example:
        POST /agents/instance_011_threshold/activate
    """

    try:
        exists = await agent_repo.agent_exists(agent_id)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )

        await agent_repo.set_agent_active(agent_id, True)

        logger.info(f"Activated agent: {agent_id}")

        return {"message": f"Agent {agent_id} activated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate agent"
        )


@router.post("/{agent_id}/deactivate", status_code=status.HTTP_200_OK)
async def deactivate_agent(agent_id: str) -> Dict[str, str]:
    """
    Deactivate an agent (set is_active=false, status='inactive').

    Example:
        POST /agents/instance_011_threshold/deactivate
    """

    try:
        exists = await agent_repo.agent_exists(agent_id)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )

        await agent_repo.set_agent_active(agent_id, False)

        # Clear from factory cache
        get_agent_factory().clear_cache(agent_id)

        logger.info(f"Deactivated agent: {agent_id}")

        return {"message": f"Agent {agent_id} deactivated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deactivate agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate agent"
        )


# =============================================================================
# STATISTICS & MONITORING ENDPOINTS
# =============================================================================

@router.get("/stats/overview", status_code=status.HTTP_200_OK)
async def get_agent_stats() -> Dict[str, Any]:
    """
    Get agent system statistics.

    Returns:
        - Total agent count
        - Count by type
        - Count by status
        - Factory cache stats

    Example:
        GET /agents/stats/overview
    """

    try:
        # Get counts by type
        consciousness_count = await agent_repo.count_agents(
            agent_type="consciousness_instance"
        )
        task_count = await agent_repo.count_agents(
            agent_type="task_agent"
        )
        system_count = await agent_repo.count_agents(
            agent_type="system_agent"
        )

        # Get active count
        active_count = await agent_repo.count_agents(is_active=True)

        # Get factory cache stats
        factory = get_agent_factory()
        cache_stats = factory.get_cache_stats()

        return {
            "total_agents": consciousness_count + task_count + system_count,
            "by_type": {
                "consciousness_instance": consciousness_count,
                "task_agent": task_count,
                "system_agent": system_count
            },
            "active_agents": active_count,
            "factory_cache": cache_stats
        }

    except Exception as e:
        logger.error(f"Failed to get agent stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get agent stats"
        )


@router.post("/cache/clear", status_code=status.HTTP_200_OK)
async def clear_factory_cache(
    agent_id: Optional[str] = Query(None, description="Agent to clear (or all if not specified)")
) -> Dict[str, str]:
    """
    Clear agent factory cache.

    Forces agents to be recreated on next invocation.
    Useful after configuration changes or for memory management.

    Example:
        POST /agents/cache/clear?agent_id=instance_011_threshold
        POST /agents/cache/clear  (clears all)
    """

    try:
        factory = get_agent_factory()
        factory.clear_cache(agent_id)

        if agent_id:
            logger.info(f"Cleared cache for agent: {agent_id}")
            return {"message": f"Cache cleared for {agent_id}"}
        else:
            logger.info("Cleared all agent caches")
            return {"message": "All agent caches cleared"}

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )
