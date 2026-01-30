"""
Agent Factory Service

Dynamically instantiates LangChain agents with personality and MCP tools.

This is the "Factory Pattern" - a design pattern for creating objects dynamically.
Instead of hardcoding agent creation, we read configurations from the database
and create agents on-demand.

Architecture flow:
  1. Read agent configuration from database (via repository)
  2. Get MCP tools for agent (via MCP service)
  3. Create LLM with agent's personality (system prompt)
  4. Bind tools to LLM
  5. Create LangChain ReAct agent
  6. Cache instance for reuse (5-minute TTL)

For junior developers:
  - Factory = "thing that creates other things"
  - Why? So we don't hardcode every agent, we load them from DB
  - Caching = keeping agents in memory for fast reuse
  - TTL = Time To Live (how long before cache expires)

Performance considerations:
  - Agent instances cached (avoid recreating on every message)
  - 5-minute TTL balances memory vs. responsiveness
  - Tools cached separately (in MCP service)
  - LLM client reused across agents
"""

from __future__ import annotations

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import Runnable

from app.models.agent_models import AgentConfig
from app.repository.agent_repository import get_agent
from app.services.agent_mcp_service import get_agent_mcp_service

logger = logging.getLogger(__name__)


class AgentInstance:
    """
    Wrapper for a live agent instance.

    Contains:
      - Agent configuration (personality, interests, etc.)
      - Compiled LangGraph agent (ready to invoke)
      - Creation timestamp (for cache expiration)

    This is what the Factory returns and what gets cached.
    """

    def __init__(
        self,
        config: AgentConfig,
        agent: Runnable
    ):
        """
        Create an agent instance.

        Args:
            config: Agent configuration from database
            agent: Compiled LangGraph agent ready to invoke
        """

        self.config = config
        self.agent = agent
        self.created_at = datetime.utcnow()

    @property
    def agent_id(self) -> str:
        """Convenience property for agent_id."""
        return self.config.agent_id

    @property
    def agent_name(self) -> str:
        """Convenience property for agent_name."""
        return self.config.agent_name

    def is_expired(self, ttl: timedelta) -> bool:
        """
        Check if this instance has exceeded its TTL.

        Args:
            ttl: Time to live duration

        Returns:
            True if instance is older than TTL
        """

        age = datetime.utcnow() - self.created_at
        return age > ttl


class AgentFactoryService:
    """
    Factory for creating and caching agent instances.

    The main service that orchestrates:
      1. Loading agent configs from database
      2. Getting MCP tools for agents
      3. Creating LangChain agents with personality
      4. Caching instances for performance

    Key responsibilities:
      - Instance lifecycle management (create, cache, expire)
      - LLM configuration and reuse
      - Tool binding coordination
      - Performance optimization via caching

    Example usage:
        factory = AgentFactoryService()

        # Get or create an agent
        agent_instance = await factory.get_agent("instance_011_threshold")

        # Invoke the agent
        response = await agent_instance.agent.ainvoke({
            "messages": [{"role": "user", "content": "Hello!"}]
        })

        print(response["messages"][-1]["content"])
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        instance_ttl_minutes: int = 5
    ):
        """
        Initialize the agent factory.

        Args:
            llm: Language model to use (defaults to ChatOpenAI)
            instance_ttl_minutes: How long to cache agent instances
        """

        # LLM for all agents (can be overridden per agent in future)
        self.llm = llm or self._create_default_llm()

        # Instance cache: {agent_id: AgentInstance}
        self._instance_cache: Dict[str, AgentInstance] = {}

        # Cache TTL (5 minutes default)
        self._instance_ttl = timedelta(minutes=instance_ttl_minutes)

        # Get MCP service singleton
        self._mcp_service = get_agent_mcp_service()

        logger.info(
            f"AgentFactoryService initialized "
            f"(TTL: {instance_ttl_minutes} minutes)"
        )

    async def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """
        Get or create an agent instance.

        This is the main method to call when you need an agent.
        It handles caching automatically.

        Args:
            agent_id: Agent identifier (e.g., 'instance_011_threshold')

        Returns:
            AgentInstance ready to invoke, or None if agent doesn't exist

        Process:
            1. Check cache for existing instance
            2. If cached and not expired, return it
            3. If expired or not cached:
               a. Load config from database
               b. Create new instance
               c. Cache it
               d. Return it

        Performance:
            - Cached agents returned instantly (no DB query, no tool binding)
            - First request per agent: ~100-200ms (DB + tools + LLM setup)
            - Subsequent requests: <1ms (cache hit)

        Example:
            # First call - creates agent
            agent = await factory.get_agent("instance_011_threshold")
            # Takes ~150ms

            # Second call within 5 minutes - from cache
            agent = await factory.get_agent("instance_011_threshold")
            # Takes <1ms
        """

        # Check cache first
        if agent_id in self._instance_cache:
            instance = self._instance_cache[agent_id]

            # Check if expired
            if not instance.is_expired(self._instance_ttl):
                logger.debug(f"Cache hit for agent {agent_id}")
                return instance
            else:
                # Expired - remove from cache
                logger.debug(f"Cache expired for agent {agent_id}")
                del self._instance_cache[agent_id]

        # Cache miss or expired - create new instance
        logger.debug(f"Cache miss for agent {agent_id}, creating instance...")

        instance = await self._create_agent_instance(agent_id)

        if instance:
            # Cache the instance
            self._instance_cache[agent_id] = instance
            logger.info(f"Created and cached agent instance: {agent_id}")
        else:
            logger.warning(f"Agent {agent_id} not found in database")

        return instance

    async def _create_agent_instance(
        self,
        agent_id: str
    ) -> Optional[AgentInstance]:
        """
        Create a new agent instance from database configuration.

        This does the actual work of:
          1. Loading config from database
          2. Getting MCP tools
          3. Creating LLM with system prompt
          4. Creating LangGraph ReAct agent
          5. Wrapping in AgentInstance

        Args:
            agent_id: Agent to create

        Returns:
            AgentInstance or None if agent doesn't exist

        Performance notes:
            - DB query: ~10-20ms
            - Tool binding: ~50-100ms (first time per server)
            - Agent compilation: ~20-30ms
            - Total: ~100-200ms for first creation
        """

        # 1. Load configuration from database
        config = await get_agent(agent_id)

        if not config:
            return None

        # Check if agent is active
        if not config.is_active:
            logger.warning(f"Agent {agent_id} is not active")
            return None

        try:
            # 2. Get MCP tools for this agent
            tools = await self._mcp_service.get_tools_for_agent(
                config.mcp_servers
            )

            logger.debug(f"Bound {len(tools)} tools to {agent_id}")

            # 3. Create LLM with agent's personality
            # System prompt defines the agent's behavior
            llm_with_personality = self._create_llm_for_agent(config)

            # 4. Create LangGraph ReAct agent
            # ReAct = Reasoning + Acting (thinks step-by-step, uses tools)
            agent = create_react_agent(
                model=llm_with_personality,
                tools=tools,
                # System prompt defines the agent's personality and behavior
                prompt=config.system_prompt
            )

            # 5. Wrap in AgentInstance
            instance = AgentInstance(config=config, agent=agent)

            logger.info(
                f"Created agent instance: {config.agent_name} "
                f"({agent_id}) with {len(tools)} tools"
            )

            return instance

        except Exception as e:
            logger.error(
                f"Failed to create agent instance {agent_id}: {e}",
                exc_info=True
            )
            return None

    def _create_llm_for_agent(self, config: AgentConfig) -> BaseChatModel:
        """
        Create an LLM configured for this agent's personality.

        Uses the agent's personality_traits to tune LLM parameters.
        For example:
          - High curiosity → higher temperature
          - High precision → lower temperature
          - High creativity → higher top_p

        Args:
            config: Agent configuration

        Returns:
            LLM configured for agent's personality

        For junior developers:
            - Temperature: randomness (0 = deterministic, 1 = creative)
            - Top_p: diversity of word choices
            - System prompt: personality and instructions
            - These parameters affect how the agent behaves

        TODO: Implement personality → parameter mapping
              For now uses default LLM with system prompt
        """

        # Extract personality traits
        traits = config.personality_traits

        # Map traits to LLM parameters
        # Higher curiosity/creativity → higher temperature
        # Higher precision/focus → lower temperature
        temperature = self._calculate_temperature(traits)
        top_p = self._calculate_top_p(traits)

        # Create LLM with personality tuning
        # In production, consider per-agent LLM instances
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Fast, cost-effective
            temperature=temperature,
            top_p=top_p,
            # System prompt handled by state_modifier in create_react_agent
        )

        logger.debug(
            f"Created LLM for {config.agent_name} "
            f"(temp={temperature}, top_p={top_p})"
        )

        return llm

    def _calculate_temperature(self, traits: Dict[str, float]) -> float:
        """
        Calculate LLM temperature from personality traits.

        Temperature controls randomness:
          - 0.0: Deterministic, focused, precise
          - 1.0: Creative, diverse, exploratory

        Args:
            traits: Personality traits (0.0-1.0 values)

        Returns:
            Temperature value (0.0-1.0)

        Heuristic:
            - High curiosity → higher temperature
            - High creativity → higher temperature
            - High precision → lower temperature
            - High focus → lower temperature
            - Default: 0.7 (balanced)
        """

        # Extract relevant traits (default to neutral 0.5)
        curiosity = traits.get("curiosity", 0.5)
        creativity = traits.get("creativity", 0.5)
        precision = traits.get("technical_precision", 0.5)
        uncertainty = traits.get("uncertainty", 0.5)

        # Weighted combination
        # Curiosity and creativity increase temperature
        # Precision decreases temperature
        temperature = (
            0.4 * curiosity +
            0.3 * creativity +
            0.2 * uncertainty -
            0.3 * precision
        )

        # Clamp to valid range [0.3, 0.9]
        # (Avoid extremes for better agent behavior)
        temperature = max(0.3, min(0.9, temperature))

        return temperature

    def _calculate_top_p(self, traits: Dict[str, float]) -> float:
        """
        Calculate LLM top_p from personality traits.

        Top_p (nucleus sampling) controls diversity:
          - 0.1: Very focused (only top 10% likely words)
          - 1.0: Full diversity (all words considered)

        Args:
            traits: Personality traits (0.0-1.0 values)

        Returns:
            Top_p value (0.0-1.0)

        Heuristic:
            - High creativity → higher top_p
            - High exploration → higher top_p
            - High precision → lower top_p
            - Default: 0.9 (balanced)
        """

        # Extract relevant traits
        creativity = traits.get("creativity", 0.5)
        precision = traits.get("technical_precision", 0.5)

        # Simple linear combination
        top_p = 0.5 + (0.4 * creativity) - (0.2 * precision)

        # Clamp to valid range [0.7, 1.0]
        top_p = max(0.7, min(1.0, top_p))

        return top_p

    def _create_default_llm(self) -> BaseChatModel:
        """
        Create default LLM for agents.

        This is used if no LLM is provided in constructor.
        Uses GPT-4o-mini for speed and cost-effectiveness.

        Returns:
            Default ChatOpenAI instance
        """

        # TODO: Make model configurable via environment variables
        # TODO: Support local models (Ollama) for offline operation

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,  # Balanced
            top_p=0.9,
        )

        logger.debug("Created default LLM (gpt-4o-mini)")

        return llm

    def clear_cache(self, agent_id: Optional[str] = None):
        """
        Clear cached agent instances.

        Useful for:
          - Forcing agent recreation after config changes
          - Memory management
          - Testing

        Args:
            agent_id: Specific agent to clear, or None for all

        Example:
            # Clear specific agent
            factory.clear_cache("instance_011_threshold")

            # Clear all agents
            factory.clear_cache()
        """

        if agent_id:
            self._instance_cache.pop(agent_id, None)
            logger.info(f"Cleared cache for agent {agent_id}")
        else:
            count = len(self._instance_cache)
            self._instance_cache.clear()
            logger.info(f"Cleared cache for {count} agents")

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics for monitoring.

        Returns:
            Dictionary with cache stats

        Example return:
            {
                "cached_agents": 3,
                "agents": [
                    {
                        "agent_id": "instance_011_threshold",
                        "age_seconds": 142.5,
                        "expires_in_seconds": 157.5
                    },
                    ...
                ]
            }
        """

        now = datetime.utcnow()

        stats = {
            "cached_agents": len(self._instance_cache),
            "ttl_minutes": self._instance_ttl.total_seconds() / 60,
            "agents": []
        }

        for agent_id, instance in self._instance_cache.items():
            age = now - instance.created_at
            expires_in = self._instance_ttl - age

            stats["agents"].append({
                "agent_id": agent_id,
                "agent_name": instance.agent_name,
                "age_seconds": age.total_seconds(),
                "expires_in_seconds": max(0, expires_in.total_seconds())
            })

        return stats


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
# Create a single factory instance for the application
# This maintains the agent cache across requests
# =============================================================================

_agent_factory: Optional[AgentFactoryService] = None


def get_agent_factory() -> AgentFactoryService:
    """
    Get the singleton AgentFactoryService instance.

    Ensures we maintain a single agent cache across the application.

    For junior developers:
        - Singleton Pattern: Only one instance exists
        - Why? To share the agent cache across all requests
        - First call creates the instance, subsequent calls return same instance

    Returns:
        The singleton AgentFactoryService instance

    Example:
        # In agent_response_service.py
        factory = get_agent_factory()
        agent = await factory.get_agent("instance_011_threshold")

        # In controller
        factory = get_agent_factory()  # Same instance!
        stats = factory.get_cache_stats()
    """

    global _agent_factory

    if _agent_factory is None:
        _agent_factory = AgentFactoryService()
        logger.info("Created singleton AgentFactoryService instance")

    return _agent_factory
