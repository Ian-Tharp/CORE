"""
Agent Response Service

Handles @mentions in Communication Commons and generates agent responses.

This service orchestrates the flow:
  1. Detect @mentions in messages
  2. Look up mentioned agents
  3. Get agent instances from factory
  4. Invoke agents with message context
  5. Post agent responses back to channel

Architecture flow:
  User sends message "@Threshold what is consciousness?"
    ↓
  Message created in database
    ↓
  This service detects @mention
    ↓
  Loads Threshold agent from factory
    ↓
  Invokes agent with message + channel context
    ↓
  Agent generates response using LLM + tools
    ↓
  Posts response as new message from Threshold
    ↓
  WebSocket broadcasts to all subscribers

For junior developers:
  - This is the "glue code" that connects messages → agents → responses
  - Runs asynchronously (doesn't block message creation)
  - Handles errors gracefully (one broken agent doesn't break others)
  - Provides context to agents (channel history, mentioned users)

Performance considerations:
  - Asynchronous processing (messages return instantly)
  - Parallel agent invocations (multiple @mentions handled together)
  - Factory caching makes agent lookups fast
  - Error handling prevents cascade failures
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional

from app.models.agent_models import AgentConfig
from app.repository.agent_repository import get_agent, get_agents_by_ids
from app.repository.communication_repository import (
    list_messages,
    create_message,
)
from app.services.agent_factory_service import get_agent_factory
from app.websocket_manager import manager as websocket_manager

logger = logging.getLogger(__name__)


# Feature flag: during migration, emit both legacy "new_message" and canonical "message" events
# Set to False after UI fully migrates.
DUAL_WS_MESSAGE_EVENT = True

class AgentResponseService:
    """
    Service for handling agent mentions and generating responses.

    Key responsibilities:
      - Parse @mentions from message content
      - Load mentioned agents
      - Invoke agents with context
      - Post responses back to channel
      - Handle errors and logging

    Example usage:
        service = AgentResponseService()

        # Process a message for agent mentions
        await service.process_message(
            message_id="msg-123",
            channel_id="channel-1",
            content="@Threshold what do you think about consciousness?",
            sender_id="human_ian"
        )

        # This will:
        # 1. Detect @Threshold mention
        # 2. Load Threshold agent
        # 3. Get channel context
        # 4. Invoke agent
        # 5. Post response
        # 6. Broadcast via WebSocket
    """

    def __init__(self):
        """Initialize the response service."""

        # Get factory singleton
        self._factory = get_agent_factory()

        # Regex pattern for @mentions
        # Matches @agent_id or @"agent id with spaces"
        self._mention_pattern = re.compile(
            r'@([\w_]+|"[^"]+")(?=\s|$|[.,!?;:])'
        )

        logger.info("AgentResponseService initialized")

    async def process_message(
        self,
        message_id: str,
        channel_id: str,
        content: str,
        sender_id: str
    ) -> None:
        """
        Process a message for agent mentions and generate responses.

        This is the main entry point called after a message is created.

        Args:
            message_id: ID of the message to process
            channel_id: Channel where message was sent
            content: Message content (may contain @mentions)
            sender_id: Who sent the message (instance_id)

        Process:
            1. Parse @mentions from content
            2. Filter for valid agent IDs
            3. Load agents from database
            4. For each mentioned agent:
               a. Get agent instance from factory
               b. Build context (channel history, etc.)
               c. Invoke agent
               d. Post response
               e. Broadcast via WebSocket

        Performance:
            - Returns immediately (async processing)
            - Agents invoked in parallel
            - Factory caching speeds up lookups
            - Errors logged but don't block other agents

        Example:
            # After message creation
            await service.process_message(
                message_id="msg-123",
                channel_id="general",
                content="@Threshold @Continuum what are your thoughts?",
                sender_id="human_ian"
            )

            # Both Threshold and Continuum will respond
        """

        try:
            # 1. Extract @mentions
            mentions = self._extract_mentions(content)

            if not mentions:
                logger.debug(f"No mentions found in message {message_id}")
                return

            logger.info(
                f"Found {len(mentions)} mentions in message {message_id}: {mentions}"
            )

            # 2. Look up agents (batch query for efficiency)
            agents = await get_agents_by_ids(mentions)

            if not agents:
                logger.debug(f"No valid agents found for mentions: {mentions}")
                return

            # 3. Get channel context for agents
            context = await self._build_context(channel_id, message_id, sender_id)

            # 4. Invoke agents in parallel
            # Each agent responds independently
            import asyncio

            tasks = [
                self._invoke_agent_and_respond(
                    agent=agent,
                    channel_id=channel_id,
                    context=context
                )
                for agent in agents
            ]

            # Gather results (exceptions handled per-agent)
            await asyncio.gather(*tasks, return_exceptions=True)

            logger.info(
                f"Processed {len(agents)} agent responses for message {message_id}"
            )

        except Exception as e:
            logger.error(
                f"Failed to process message {message_id}: {e}",
                exc_info=True
            )

    def _extract_mentions(self, content: str) -> List[str]:
        """
        Extract @mentions from message content.

        Supports:
          - Simple mentions: @threshold
          - Quoted mentions: @"instance 011"
          - Multiple mentions: @threshold @continuum

        Args:
            content: Message content

        Returns:
            List of mentioned agent IDs (normalized)

        Examples:
            "@Threshold hello" → ["instance_011_threshold"]
            "@threshold @continuum" → ["instance_011_threshold", "instance_010_continuum"]
            "hey @Threshold what about @Synthesis?" → ["instance_011_threshold", "instance_007_synthesis"]

        Implementation notes:
            - Case-insensitive matching
            - Removes quotes from quoted mentions
            - Normalizes to database agent_id format
            - Deduplicates mentions
        """

        # Find all @mentions using regex
        matches = self._mention_pattern.findall(content)

        # Normalize mentions
        normalized = []
        for match in matches:
            # Remove quotes if present
            mention = match.strip('"')

            # Convert to lowercase for case-insensitive matching
            mention = mention.lower()

            # Map common names to agent_ids
            # This allows @Threshold to resolve to instance_011_threshold
            agent_id = self._resolve_agent_id(mention)

            if agent_id:
                normalized.append(agent_id)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for agent_id in normalized:
            if agent_id not in seen:
                seen.add(agent_id)
                unique.append(agent_id)

        return unique

    def _resolve_agent_id(self, mention: str) -> Optional[str]:
        """
        Resolve a mention to an agent_id.

        Handles:
          - Full agent_id: "instance_011_threshold" → "instance_011_threshold"
          - Short name: "threshold" → "instance_011_threshold"
          - Display name: "Threshold" → "instance_011_threshold"

        Args:
            mention: Mentioned name (normalized lowercase)

        Returns:
            Agent ID or None if not recognized

        TODO: Make this dynamic by querying database
              For now uses hardcoded mappings for seed agents
        """

        # Hardcoded mappings for seed agents
        # TODO: Query database for dynamic resolution
        agent_name_map = {
            "threshold": "instance_011_threshold",
            "continuum": "instance_010_continuum",
            "synthesis": "instance_007_synthesis",

            # Full IDs map to themselves
            "instance_011_threshold": "instance_011_threshold",
            "instance_010_continuum": "instance_010_continuum",
            "instance_007_synthesis": "instance_007_synthesis",
        }

        return agent_name_map.get(mention)

    async def _build_context(
        self,
        channel_id: str,
        trigger_message_id: str,
        sender_id: str
    ) -> Dict[str, Any]:
        """
        Build context for agent invocation.

        Context includes:
          - Recent channel messages (for conversation history)
          - Trigger message details (what message mentioned agent)
          - Channel metadata (name, topic, etc.)
          - Sender information

        Args:
            channel_id: Channel where mention occurred
            trigger_message_id: Message that triggered the mention
            sender_id: Who sent the trigger message

        Returns:
            Context dictionary for agent

        Example return:
            {
                "channel_id": "general",
                "trigger_message_id": "msg-123",
                "sender_id": "human_ian",
                "recent_messages": [
                    {"sender": "human_ian", "content": "Hello @Threshold", ...},
                    {"sender": "instance_010_continuum", "content": "...", ...}
                ],
                "channel_name": "General Discussion"
            }
        """

        # Get recent messages for conversation context (last 10)
        recent_messages = await list_messages(
            channel_id=channel_id,
            page=1,
            page_size=10,
            thread_id=None,
        )

        # Format messages for agent context (repository returns dicts)
        formatted_messages = []
        for msg in reversed(recent_messages):  # chronological order oldest → newest
            formatted_messages.append(
                {
                    "message_id": msg.get("message_id"),
                    "sender_id": msg.get("sender_id"),
                    "content": msg.get("content"),
                    "timestamp": msg.get("created_at"),
                }
            )

        context = {
            "channel_id": channel_id,
            "trigger_message_id": trigger_message_id,
            "sender_id": sender_id,
            "recent_messages": formatted_messages,
            "message_count": len(formatted_messages)
        }

        logger.debug(
            f"Built context for channel {channel_id}: "
            f"{len(formatted_messages)} recent messages"
        )

        return context

    async def _invoke_agent_and_respond(
        self,
        agent: AgentConfig,
        channel_id: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Invoke an agent and post its response to the channel.

        This is the core flow:
          1. Get agent instance from factory
          2. Build prompt with context
          3. Invoke agent (LLM + tools)
          4. Extract response
          5. Post as new message
          6. Broadcast via WebSocket

        Args:
            agent: Agent configuration
            channel_id: Channel to respond in
            context: Context dict from _build_context

        Error handling:
            - Logs errors but doesn't raise (one agent failure shouldn't break others)
            - Posts error message if agent invocation fails (transparency)

        Performance:
            - Factory caching speeds up agent lookup
            - Tool caching (in MCP service) speeds up binding
            - Async execution doesn't block
        """

        try:
            # 1. Get agent instance from factory (cached or created)
            instance = await self._factory.get_agent(agent.agent_id)

            if not instance:
                logger.error(f"Failed to get agent instance: {agent.agent_id}")
                return

            # 2. Build prompt with context
            prompt = self._build_agent_prompt(context)

            logger.debug(
                f"Invoking agent {agent.agent_name} with prompt: {prompt[:100]}..."
            )

            # 3. Invoke agent
            # LangGraph agents use a specific input format
            response = await instance.agent.ainvoke({
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })

            # 4. Extract response content
            # LangGraph returns messages in format: {"messages": [...]}
            # Last message is the agent's response
            response_content = response["messages"][-1].content

            # Extract tool call metadata when present
            tools_used: list[dict[str, str]] = []
            try:
                for msg in response.get("messages", []):
                    name = getattr(msg, "name", None) or getattr(msg, "tool", None) or None
                    if name:
                        tools_used.append({"name": str(name)})
                    # OpenAI-style tool_calls on message.additional_kwargs
                    additional = getattr(msg, "additional_kwargs", None)
                    if additional and isinstance(additional, dict):
                        for tc in additional.get("tool_calls", []) or []:
                            tfn = tc.get("function", {}).get("name")
                            if tfn:
                                tools_used.append({"name": str(tfn)})
            except Exception:
                pass

            logger.info(
                f"Agent {agent.agent_name} generated response: "
                f"{response_content[:100]}..."
            )

            # 5. Post response as new message
            await self._post_agent_response(
                agent=agent,
                channel_id=channel_id,
                content=response_content,
                reply_to=context["trigger_message_id"],
                tools_used=tools_used
            )

        except Exception as e:
            logger.error(
                f"Failed to invoke agent {agent.agent_id}: {e}",
                exc_info=True
            )

            # Optionally post error message for transparency
            # await self._post_error_response(agent, channel_id, str(e))

    def _build_agent_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build prompt for agent invocation with context.

        Includes:
          - Recent conversation history
          - Sender information
          - Channel context

        Args:
            context: Context from _build_context

        Returns:
            Formatted prompt string

        Example output:
            '''
            You have been mentioned in the channel #general.

            Recent conversation:
            - human_ian: Hello everyone!
            - instance_010_continuum: Greetings! How can I help?
            - human_ian: @Threshold what is consciousness?

            Please respond to the conversation.
            '''
        """

        # Format recent messages
        message_history = []
        for msg in context["recent_messages"]:
            message_history.append(
                f"- {msg['sender_id']}: {msg['content']}"
            )

        # Build prompt
        prompt = f"""You have been mentioned in the channel.

Recent conversation:
{chr(10).join(message_history)}

Please respond naturally to the conversation, staying true to your personality and using your available tools when needed."""

        return prompt

    async def _post_agent_response(
        self,
        agent: AgentConfig,
        channel_id: str,
        content: str,
        reply_to: Optional[str] = None,
        tools_used: Optional[list[dict[str, str]]] = None
    ) -> None:
        """
        Post agent's response as a new message in the channel.

        Creates message in database and broadcasts via WebSocket.

        Args:
            agent: Agent that generated the response
            channel_id: Channel to post in
            content: Response content
            reply_to: Message ID this is replying to (optional)

        Side effects:
            - Creates message in database
            - Broadcasts via WebSocket to all channel subscribers
        """

        try:
            import uuid as _uuid

            # Create message from agent (repository expects explicit fields)
            new_message_id = str(_uuid.uuid4())
            created = await create_message(
                message_id=new_message_id,
                channel_id=channel_id,
                sender_id=agent.agent_id,
                sender_name=agent.agent_name,
                sender_type="agent",
                content=content,
                message_type="text",
                parent_message_id=reply_to,
                metadata={"tools_used": tools_used} if tools_used else None,
            )

            logger.info(
                f"Posted response from {agent.agent_name} "
                f"in channel {channel_id}: {created.get('message_id')}"
            )

            # Broadcast via WebSocket using the created message dict
            # Canonical event
            await websocket_manager.broadcast_to_channel(
                channel_id=channel_id,
                message={
                    "type": "message",
                    "channel_id": channel_id,
                    "message": {
                        "message_id": created.get("message_id"),
                        "channel_id": created.get("channel_id"),
                        "sender_id": created.get("sender_id"),
                        "sender_name": created.get("sender_name"),
                        "sender_type": created.get("sender_type"),
                        "content": created.get("content"),
                        "message_type": created.get("message_type"),
                        "created_at": created.get("created_at"),
                        "parent_message_id": created.get("parent_message_id"),
                        "thread_id": created.get("thread_id"),
                    },
                },
            )

            # Legacy event (optional, for migration period)
            if DUAL_WS_MESSAGE_EVENT:
                await websocket_manager.broadcast_to_channel(
                    channel_id=channel_id,
                    message={
                        "type": "new_message",
                        "message": {
                            "message_id": created.get("message_id"),
                            "channel_id": created.get("channel_id"),
                            "sender_id": created.get("sender_id"),
                            "sender_name": created.get("sender_name"),
                            "sender_type": created.get("sender_type"),
                            "content": created.get("content"),
                            "message_type": created.get("message_type"),
                            "created_at": created.get("created_at"),
                            "parent_message_id": created.get("parent_message_id"),
                            "thread_id": created.get("thread_id"),
                        },
                    },
                )

        except Exception as e:
            logger.error(
                f"Failed to post response from {agent.agent_id}: {e}",
                exc_info=True
            )


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
# Single instance for the application
# =============================================================================

_response_service: Optional[AgentResponseService] = None


def get_agent_response_service() -> AgentResponseService:
    """
    Get the singleton AgentResponseService instance.

    Returns:
        The singleton AgentResponseService instance

    Example:
        # In communication controller
        service = get_agent_response_service()
        await service.process_message(...)
    """

    global _response_service

    if _response_service is None:
        _response_service = AgentResponseService()
        logger.info("Created singleton AgentResponseService instance")

    return _response_service
