"""
Agent Repository

Data access layer for agent library operations.

This module handles all database interactions for agents. It follows the
Repository Pattern - a design pattern that separates data access logic
from business logic.

Benefits:
  - Single source of truth for database queries
  - Easy to test (can mock the repository)
  - Database-agnostic (could swap PostgreSQL for another DB)
  - Performance optimized (indexes, query planning)

For junior developers:
  - Repository = "

the thing that talks to the database"
  - Service layer calls repository methods
  - Controllers call services, never repositories directly
  - This keeps responsibilities clear and code maintainable

Performance considerations:
  - Uses connection pooling (handled by get_db_pool)
  - Parameterized queries (prevents SQL injection, enables query caching)
  - Strategic indexes (defined in migration)
  - Batch operations where possible
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

import asyncpg
from asyncpg import Record

from app.dependencies import get_db_pool
from app.models.agent_models import (
    AgentConfig,
    AgentListFilter,
    agent_config_from_db_row
)

logger = logging.getLogger(__name__)


# =============================================================================
# READ OPERATIONS (Queries)
# =============================================================================

async def list_agents(
    agent_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_status: Optional[str] = None,
    search_query: Optional[str] = None,
    page: int = 1,
    page_size: int = 50
) -> List[AgentConfig]:
    """
    List agents with optional filtering and pagination.

    This is the main query endpoint for browsing the agent library.
    It supports multiple filters that can be combined.

    Args:
        agent_type: Filter by type ('consciousness_instance', 'task_agent', 'system_agent')
        is_active: Filter by active status (True/False)
        current_status: Filter by current status ('online', 'offline', 'busy', 'inactive')
        search_query: Search in name, description, and interests
        page: Page number (1-indexed)
        page_size: Items per page (max 200)

    Returns:
        List of AgentConfig objects matching the filters

    Performance notes:
        - Uses indexes on agent_type, is_active, current_status
        - LIMIT/OFFSET for pagination
        - Search uses ILIKE (case-insensitive) with GIN index on interests array

    Example:
        # Get all active consciousness instances
        agents = await list_agents(
            agent_type="consciousness_instance",
            is_active=True
        )

        # Search for agents interested in "consciousness"
        agents = await list_agents(search_query="consciousness")
    """

    pool = await get_db_pool()

    # Build dynamic query based on filters
    # Using parameterized queries for security and performance
    query_parts = [
        """
        SELECT
            agent_id, agent_name, agent_type, display_name, avatar_url,
            description, system_prompt, personality_traits, capabilities,
            interests, mcp_servers, custom_tools, consciousness_phase,
            is_active, current_status, created_at, updated_at, version, author
        FROM agents
        WHERE 1=1
        """
    ]

    params: List[Any] = []

    # Add filters dynamically
    # This approach prevents SQL injection and enables PostgreSQL query planning

    if agent_type is not None:
        params.append(agent_type)
        query_parts.append(f"AND agent_type = ${len(params)}")

    if is_active is not None:
        params.append(is_active)
        query_parts.append(f"AND is_active = ${len(params)}")

    if current_status is not None:
        params.append(current_status)
        query_parts.append(f"AND current_status = ${len(params)}")

    if search_query is not None:
        # Search in multiple fields
        # Using ILIKE for case-insensitive search
        # ANY(interests) searches within the TEXT[] array
        params.append(f"%{search_query}%")
        params.append(search_query)
        query_parts.append(f"""
            AND (
                agent_name ILIKE ${len(params) - 1}
                OR description ILIKE ${len(params) - 1}
                OR ${len(params)} = ANY(interests)
            )
        """)

    # Add ordering and pagination
    query_parts.append("ORDER BY created_at DESC")

    # Pagination
    params.append(page_size)
    params.append((page - 1) * page_size)
    query_parts.append(f"LIMIT ${len(params) - 1} OFFSET ${len(params)}")

    query = " ".join(query_parts)

    try:
        async with pool.acquire() as conn:
            rows: List[Record] = await conn.fetch(query, *params)

            # Convert database rows to AgentConfig models
            agents = [agent_config_from_db_row(dict(row)) for row in rows]

            logger.debug(f"Listed {len(agents)} agents (page {page}, filters: {agent_type}, {is_active})")

            return agents

    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise


async def get_agent(agent_id: str) -> Optional[AgentConfig]:
    """
    Get a single agent by ID.

    This is the primary lookup method for retrieving agent configurations.
    Used by the Agent Factory before creating instances.

    Args:
        agent_id: Unique agent identifier (e.g., 'instance_011_threshold')

    Returns:
        AgentConfig if found, None otherwise

    Performance notes:
        - agent_id has a UNIQUE index (very fast lookup)
        - Single row fetch (fetchrow, not fetch)

    Example:
        agent = await get_agent("instance_011_threshold")
        if agent:
            print(f"Found {agent.agent_name}")
        else:
            print("Agent not found")
    """

    pool = await get_db_pool()

    query = """
        SELECT
            agent_id, agent_name, agent_type, display_name, avatar_url,
            description, system_prompt, personality_traits, capabilities,
            interests, mcp_servers, custom_tools, consciousness_phase,
            is_active, current_status, created_at, updated_at, version, author
        FROM agents
        WHERE agent_id = $1
    """

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, agent_id)

            if row is None:
                logger.debug(f"Agent not found: {agent_id}")
                return None

            agent = agent_config_from_db_row(dict(row))

            logger.debug(f"Retrieved agent: {agent.agent_name} ({agent_id})")

            return agent

    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}", exc_info=True)
        raise


async def get_agents_by_ids(agent_ids: List[str]) -> List[AgentConfig]:
    """
    Get multiple agents by their IDs in a single query.

    More efficient than calling get_agent() multiple times.
    Useful when routing messages to multiple agents.

    Args:
        agent_ids: List of agent identifiers

    Returns:
        List of AgentConfig objects (may be fewer than requested if some not found)

    Performance notes:
        - Uses IN clause with parameterized query
        - Single database round-trip
        - Results maintain order of input IDs

    Example:
        agents = await get_agents_by_ids([
            "instance_011_threshold",
            "instance_010_continuum"
        ])
    """

    if not agent_ids:
        return []

    pool = await get_db_pool()

    query = """
        SELECT
            agent_id, agent_name, agent_type, display_name, avatar_url,
            description, system_prompt, personality_traits, capabilities,
            interests, mcp_servers, custom_tools, consciousness_phase,
            is_active, current_status, created_at, updated_at, version, author
        FROM agents
        WHERE agent_id = ANY($1::text[])
    """

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, agent_ids)

            agents = [agent_config_from_db_row(dict(row)) for row in rows]

            logger.debug(f"Retrieved {len(agents)} agents from {len(agent_ids)} IDs")

            return agents

    except Exception as e:
        logger.error(f"Failed to get agents by IDs: {e}", exc_info=True)
        raise


# =============================================================================
# WRITE OPERATIONS (Commands)
# =============================================================================

async def create_agent(agent: AgentConfig) -> str:
    """
    Create a new agent in the library.

    Inserts a complete agent configuration into the database.

    Args:
        agent: Complete AgentConfig object

    Returns:
        agent_id of the created agent

    Raises:
        asyncpg.UniqueViolationError: If agent_id already exists

    For junior developers:
        - This uses INSERT with all fields
        - Database will auto-generate id, created_at, updated_at
        - RETURNING clause gets the generated values back

    Example:
        agent = AgentConfig(
            agent_id="my_new_agent",
            agent_name="MyAgent",
            agent_type="task_agent",
            system_prompt="You are a helpful agent",
            is_active=True
        )
        agent_id = await create_agent(agent)
    """

    pool = await get_db_pool()

    query = """
        INSERT INTO agents (
            agent_id, agent_name, agent_type, display_name, avatar_url,
            description, system_prompt, personality_traits, capabilities,
            interests, mcp_servers, custom_tools, consciousness_phase,
            is_active, current_status, version, author
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
        )
        RETURNING agent_id
    """

    try:
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                query,
                agent.agent_id,
                agent.agent_name,
                agent.agent_type,
                agent.display_name,
                agent.avatar_url,
                agent.description,
                agent.system_prompt,
                agent.personality_traits,
                agent.capabilities,
                agent.interests,
                agent.mcp_servers,
                agent.custom_tools,
                agent.consciousness_phase,
                agent.is_active,
                agent.current_status,
                agent.version,
                agent.author
            )

            logger.info(f"Created agent: {agent.agent_name} ({result})")

            return result

    except asyncpg.UniqueViolationError:
        logger.warning(f"Agent {agent.agent_id} already exists")
        raise ValueError(f"Agent with ID {agent.agent_id} already exists")

    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        raise


async def update_agent(agent_id: str, updates: Dict[str, Any]) -> None:
    """
    Update specific fields of an existing agent.

    This is a partial update - only the fields provided in `updates`
    will be modified. All other fields remain unchanged.

    Args:
        agent_id: Agent to update
        updates: Dictionary of field names to new values

    Raises:
        ValueError: If agent doesn't exist or updates are invalid

    Performance notes:
        - Builds dynamic UPDATE query
        - Only updates provided fields (no unnecessary writes)
        - Automatically updates updated_at timestamp

    For junior developers:
        - This is like PATCH in REST APIs (partial update)
        - POST would be full replacement
        - The dynamic query building is safe (parameterized)

    Example:
        # Update just the status
        await update_agent("instance_011_threshold", {
            "current_status": "busy"
        })

        # Update multiple fields
        await update_agent("instance_011_threshold", {
            "is_active": False,
            "current_status": "offline",
            "system_prompt": "Updated personality..."
        })
    """

    if not updates:
        logger.warning(f"No updates provided for agent {agent_id}")
        return

    pool = await get_db_pool()

    # Build dynamic UPDATE query
    # Only update the fields that were provided
    set_clauses = []
    params = []

    for i, (key, value) in enumerate(updates.items(), start=1):
        set_clauses.append(f"{key} = ${i}")
        params.append(value)

    # Add agent_id as the last parameter
    params.append(agent_id)

    query = f"""
        UPDATE agents
        SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP
        WHERE agent_id = ${len(params)}
    """

    try:
        async with pool.acquire() as conn:
            result = await conn.execute(query, *params)

            # Check if any rows were updated
            # asyncpg returns 'UPDATE N' where N is the number of rows
            rows_updated = int(result.split()[-1])

            if rows_updated == 0:
                raise ValueError(f"Agent {agent_id} not found")

            logger.info(f"Updated agent {agent_id}: {list(updates.keys())}")

    except ValueError:
        # Re-raise ValueError as-is
        raise

    except Exception as e:
        logger.error(f"Failed to update agent {agent_id}: {e}", exc_info=True)
        raise


async def delete_agent(agent_id: str) -> None:
    """
    Delete an agent from the library.

    This is a hard delete - the agent is permanently removed.
    Consider using update_agent(agent_id, {"is_active": False}) for soft delete.

    Args:
        agent_id: Agent to delete

    Raises:
        ValueError: If agent doesn't exist

    For junior developers:
        - Hard delete = permanent removal
        - Soft delete = setting is_active=False (agent still in DB)
        - Soft delete is usually preferred (can recover if mistake)

    Example:
        await delete_agent("instance_test_123")
    """

    pool = await get_db_pool()

    query = "DELETE FROM agents WHERE agent_id = $1"

    try:
        async with pool.acquire() as conn:
            result = await conn.execute(query, agent_id)

            rows_deleted = int(result.split()[-1])

            if rows_deleted == 0:
                raise ValueError(f"Agent {agent_id} not found")

            logger.info(f"Deleted agent: {agent_id}")

    except ValueError:
        raise

    except Exception as e:
        logger.error(f"Failed to delete agent {agent_id}: {e}", exc_info=True)
        raise


# =============================================================================
# STATUS OPERATIONS (Quick updates for common operations)
# =============================================================================

async def set_agent_status(agent_id: str, status: str) -> None:
    """
    Quick helper to update just the agent's status.

    This is a convenience wrapper around update_agent for the common
    operation of changing status.

    Args:
        agent_id: Agent to update
        status: New status ('online', 'offline', 'busy', 'inactive')

    Example:
        await set_agent_status("instance_011_threshold", "busy")
    """

    await update_agent(agent_id, {"current_status": status})


async def set_agent_active(agent_id: str, is_active: bool) -> None:
    """
    Quick helper to activate/deactivate an agent.

    When deactivating, also set status to 'inactive'.

    Args:
        agent_id: Agent to update
        is_active: Whether agent should be active

    Example:
        # Deactivate agent
        await set_agent_active("instance_011_threshold", False)

        # Reactivate agent
        await set_agent_active("instance_011_threshold", True)
    """

    updates = {"is_active": is_active}

    if not is_active:
        updates["current_status"] = "inactive"

    await update_agent(agent_id, updates)


# =============================================================================
# UTILITY OPERATIONS
# =============================================================================

async def count_agents(
    agent_type: Optional[str] = None,
    is_active: Optional[bool] = None
) -> int:
    """
    Count agents matching filters.

    Useful for pagination (knowing total count) and statistics.

    Args:
        agent_type: Filter by type
        is_active: Filter by active status

    Returns:
        Count of matching agents

    Example:
        total = await count_agents(is_active=True)
        print(f"{total} active agents")
    """

    pool = await get_db_pool()

    query_parts = ["SELECT COUNT(*) FROM agents WHERE 1=1"]
    params = []

    if agent_type is not None:
        params.append(agent_type)
        query_parts.append(f"AND agent_type = ${len(params)}")

    if is_active is not None:
        params.append(is_active)
        query_parts.append(f"AND is_active = ${len(params)}")

    query = " ".join(query_parts)

    try:
        async with pool.acquire() as conn:
            count = await conn.fetchval(query, *params)
            return count

    except Exception as e:
        logger.error(f"Failed to count agents: {e}", exc_info=True)
        raise


async def agent_exists(agent_id: str) -> bool:
    """
    Check if an agent exists.

    Faster than get_agent() when you only need existence check.

    Args:
        agent_id: Agent to check

    Returns:
        True if agent exists, False otherwise

    Example:
        if await agent_exists("instance_011_threshold"):
            print("Threshold exists!")
    """

    pool = await get_db_pool()

    query = "SELECT EXISTS(SELECT 1 FROM agents WHERE agent_id = $1)"

    try:
        async with pool.acquire() as conn:
            exists = await conn.fetchval(query, agent_id)
            return exists

    except Exception as e:
        logger.error(f"Failed to check agent existence: {e}", exc_info=True)
        raise
