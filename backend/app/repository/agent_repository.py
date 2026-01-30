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
    tags: Optional[List[str]] = None,
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
        search_query: Full-text search across name, description, and interests (case-insensitive)
        tags: Filter by interest tags (agents must have ALL specified tags)
        page: Page number (1-indexed)
        page_size: Items per page (max 200)

    Returns:
        List of AgentConfig objects matching the filters

    Performance notes:
        - Uses indexes on agent_type, is_active, current_status
        - LIMIT/OFFSET for pagination
        - Search uses ILIKE (case-insensitive) across name, description, interests
        - Tag filtering uses PostgreSQL array contains operator (@>)

    Example:
        # Get all active consciousness instances
        agents = await list_agents(
            agent_type="consciousness_instance",
            is_active=True
        )

        # Search for agents interested in "consciousness"
        agents = await list_agents(search_query="consciousness")

        # Filter by tags (must have ALL specified tags)
        agents = await list_agents(tags=["consciousness", "architecture"])
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
        # Full-text search across name, description, and interests
        # Using ILIKE for case-insensitive partial matching on all fields
        # array_to_string converts the interests TEXT[] to a searchable string
        search_pattern = f"%{search_query}%"
        params.append(search_pattern)
        param_idx = len(params)
        query_parts.append(f"""
            AND (
                agent_name ILIKE ${param_idx}
                OR description ILIKE ${param_idx}
                OR array_to_string(interests, ' ') ILIKE ${param_idx}
            )
        """)

    if tags is not None and len(tags) > 0:
        # Tag filtering: agent must have ALL specified tags
        # Uses PostgreSQL array contains operator (@>)
        # e.g., interests @> ARRAY['consciousness', 'architecture']
        params.append(tags)
        query_parts.append(f"AND interests @> ${len(params)}::text[]")

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
        import json
        
        # Serialize complex types to JSON strings for JSONB columns
        personality_json = json.dumps(agent.personality_traits) if agent.personality_traits else '{}'
        capabilities_json = json.dumps([c.model_dump() if hasattr(c, 'model_dump') else c for c in agent.capabilities]) if agent.capabilities else '[]'
        interests_list = agent.interests if agent.interests else []
        mcp_servers_json = json.dumps([s.model_dump() if hasattr(s, 'model_dump') else s for s in agent.mcp_servers]) if agent.mcp_servers else '[]'
        custom_tools_json = json.dumps(agent.custom_tools) if agent.custom_tools else '[]'
        
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
                personality_json,
                capabilities_json,
                interests_list,
                mcp_servers_json,
                custom_tools_json,
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


# =============================================================================
# SEARCH & DISCOVERY OPERATIONS
# =============================================================================

async def get_all_tags(
    agent_type: Optional[str] = None,
    is_active: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Get all unique tags (interests) with their usage counts.

    Useful for building tag filter UIs — shows which tags exist
    and how many agents have each tag.

    Args:
        agent_type: Only count tags from agents of this type
        is_active: Only count tags from active/inactive agents

    Returns:
        List of dicts with 'tag' and 'count' keys, sorted by count descending

    Example:
        tags = await get_all_tags(is_active=True)
        # [{"tag": "consciousness", "count": 5}, {"tag": "architecture", "count": 3}, ...]
    """

    pool = await get_db_pool()

    # unnest(interests) expands the TEXT[] array into individual rows
    # so we can GROUP BY and COUNT each unique tag
    query_parts = [
        """
        SELECT tag, COUNT(*) as count
        FROM agents, unnest(interests) AS tag
        WHERE 1=1
        """
    ]

    params: List[Any] = []

    if agent_type is not None:
        params.append(agent_type)
        query_parts.append(f"AND agent_type = ${len(params)}")

    if is_active is not None:
        params.append(is_active)
        query_parts.append(f"AND is_active = ${len(params)}")

    query_parts.append("GROUP BY tag ORDER BY count DESC, tag ASC")

    query = " ".join(query_parts)

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            tags = [{"tag": row["tag"], "count": row["count"]} for row in rows]

            logger.debug(f"Found {len(tags)} unique tags")

            return tags

    except Exception as e:
        logger.error(f"Failed to get tags: {e}", exc_info=True)
        raise


async def search_agents_fulltext(
    query_text: str,
    agent_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    tags: Optional[List[str]] = None,
    page: int = 1,
    page_size: int = 50
) -> List[AgentConfig]:
    """
    Full-text search across agent fields with relevance ranking.

    Searches agent_name, description, system_prompt, and interests.
    Results are ranked by relevance: name matches rank highest,
    then description, then interests, then system_prompt.

    Args:
        query_text: Search string
        agent_type: Optional type filter
        is_active: Optional active filter
        tags: Optional tag filter (must have ALL specified tags)
        page: Page number (1-indexed)
        page_size: Items per page

    Returns:
        List of AgentConfig objects sorted by relevance

    Example:
        results = await search_agents_fulltext("consciousness exploration")
    """

    pool = await get_db_pool()

    search_pattern = f"%{query_text}%"
    params: List[Any] = [search_pattern]

    # Build query with relevance scoring
    # Higher weight for name matches, lower for system_prompt
    query_parts = [
        f"""
        SELECT
            agent_id, agent_name, agent_type, display_name, avatar_url,
            description, system_prompt, personality_traits, capabilities,
            interests, mcp_servers, custom_tools, consciousness_phase,
            is_active, current_status, created_at, updated_at, version, author,
            (
                CASE WHEN agent_name ILIKE $1 THEN 4 ELSE 0 END +
                CASE WHEN description ILIKE $1 THEN 3 ELSE 0 END +
                CASE WHEN array_to_string(interests, ' ') ILIKE $1 THEN 2 ELSE 0 END +
                CASE WHEN system_prompt ILIKE $1 THEN 1 ELSE 0 END
            ) AS relevance
        FROM agents
        WHERE (
            agent_name ILIKE $1
            OR description ILIKE $1
            OR array_to_string(interests, ' ') ILIKE $1
            OR system_prompt ILIKE $1
        )
        """
    ]

    if agent_type is not None:
        params.append(agent_type)
        query_parts.append(f"AND agent_type = ${len(params)}")

    if is_active is not None:
        params.append(is_active)
        query_parts.append(f"AND is_active = ${len(params)}")

    if tags is not None and len(tags) > 0:
        params.append(tags)
        query_parts.append(f"AND interests @> ${len(params)}::text[]")

    # Order by relevance, then by name for ties
    query_parts.append("ORDER BY relevance DESC, agent_name ASC")

    # Pagination
    params.append(page_size)
    params.append((page - 1) * page_size)
    query_parts.append(f"LIMIT ${len(params) - 1} OFFSET ${len(params)}")

    query = " ".join(query_parts)

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            # Convert rows (drop the relevance column for AgentConfig)
            agents = []
            for row in rows:
                row_dict = dict(row)
                row_dict.pop("relevance", None)
                agents.append(agent_config_from_db_row(row_dict))

            logger.debug(f"Full-text search '{query_text}' returned {len(agents)} results")

            return agents

    except Exception as e:
        logger.error(f"Failed full-text search: {e}", exc_info=True)
        raise
