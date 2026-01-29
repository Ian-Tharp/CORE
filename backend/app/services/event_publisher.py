"""
WebSocket Event Publisher Service.

Provides a centralized interface for publishing events to connected clients
via the WebSocket connection manager. Supports broadcasting to all clients,
specific channels, or individual connections.

Usage:
    from app.services.event_publisher import event_publisher
    from app.models.ws_events import AgentActivityEvent, AgentStatus
    
    # Publish an agent activity event to all clients
    await event_publisher.publish(AgentActivityEvent(
        agent_id="comprehension-agent",
        action="analyzing",
        status=AgentStatus.ACTIVE,
        message="Processing input..."
    ))
    
    # Publish to a specific channel
    await event_publisher.publish_to_channel(
        channel_id="session-123",
        event=TaskProgressEvent(task_id="task-1", progress_pct=50, stage=TaskStage.PROCESSING)
    )
"""

from __future__ import annotations

import logging
from typing import Optional, List

from app.models.ws_events import (
    BaseEvent,
    WSEvent,
    AgentActivityEvent,
    TaskProgressEvent,
    CouncilEvent,
    SystemEvent,
    NotificationEvent,
    EventType,
    AgentStatus,
    TaskStage,
    CouncilEventType,
    SystemLevel,
    NotificationPriority,
)
from app.websocket_manager import manager as ws_manager

logger = logging.getLogger(__name__)


class EventPublisher:
    """
    Service for publishing WebSocket events.
    
    Wraps the WebSocket connection manager and provides a clean API
    for publishing typed events. Handles serialization and error logging.
    """
    
    def __init__(self):
        self._manager = ws_manager
    
    # =========================================================================
    # Core Publishing Methods
    # =========================================================================
    
    async def publish(self, event: BaseEvent) -> bool:
        """
        Publish an event to all connected clients.
        
        Args:
            event: The event to broadcast
            
        Returns:
            True if broadcast was attempted, False on error
        """
        try:
            message = event.to_ws_message()
            await self._manager.broadcast_to_all(message)
            logger.debug(f"Published {event.event_type} event: {event.event_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
            return False
    
    async def publish_to_channel(self, channel_id: str, event: BaseEvent) -> bool:
        """
        Publish an event to all subscribers of a specific channel.
        
        Args:
            channel_id: The channel to broadcast to
            event: The event to broadcast
            
        Returns:
            True if broadcast was attempted, False on error
        """
        try:
            message = event.to_ws_message()
            await self._manager.broadcast_to_channel(channel_id, message)
            logger.debug(f"Published {event.event_type} to channel {channel_id}: {event.event_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel_id}: {e}")
            return False
    
    async def publish_to_instance(self, instance_id: str, event: BaseEvent) -> bool:
        """
        Publish an event to a specific client instance.
        
        Args:
            instance_id: The target instance ID
            event: The event to send
            
        Returns:
            True if sent successfully, False on error
        """
        try:
            message = event.to_ws_message()
            await self._manager.send_personal_message(instance_id, message)
            logger.debug(f"Published {event.event_type} to {instance_id}: {event.event_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to instance {instance_id}: {e}")
            return False
    
    # =========================================================================
    # Convenience Methods for Common Events
    # =========================================================================
    
    async def agent_started(
        self,
        agent_id: str,
        action: str,
        message: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit event when an agent starts an action."""
        return await self.publish(AgentActivityEvent(
            agent_id=agent_id,
            action=action,
            status=AgentStatus.ACTIVE,
            message=message or f"Agent {agent_id} started {action}",
            session_id=session_id,
        ))
    
    async def agent_thinking(
        self,
        agent_id: str,
        message: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit event when an agent is thinking/processing."""
        return await self.publish(AgentActivityEvent(
            agent_id=agent_id,
            action="thinking",
            status=AgentStatus.THINKING,
            message=message or f"Agent {agent_id} is thinking...",
            session_id=session_id,
        ))
    
    async def agent_complete(
        self,
        agent_id: str,
        action: str,
        message: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit event when an agent completes an action."""
        return await self.publish(AgentActivityEvent(
            agent_id=agent_id,
            action=action,
            status=AgentStatus.COMPLETE,
            message=message or f"Agent {agent_id} completed {action}",
            session_id=session_id,
        ))
    
    async def agent_error(
        self,
        agent_id: str,
        action: str,
        error_message: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit event when an agent encounters an error."""
        return await self.publish(AgentActivityEvent(
            agent_id=agent_id,
            action=action,
            status=AgentStatus.ERROR,
            message=error_message,
            session_id=session_id,
        ))
    
    async def task_started(
        self,
        task_id: str,
        message: Optional[str] = None,
        total_steps: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit event when a task starts."""
        return await self.publish(TaskProgressEvent(
            task_id=task_id,
            progress_pct=0,
            stage=TaskStage.STARTING,
            message=message or f"Task {task_id} started",
            total_steps=total_steps,
            current_step_num=1 if total_steps else None,
            session_id=session_id,
        ))
    
    async def task_progress(
        self,
        task_id: str,
        progress_pct: int,
        message: Optional[str] = None,
        eta_seconds: Optional[int] = None,
        current_step: Optional[str] = None,
        current_step_num: Optional[int] = None,
        total_steps: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit task progress update."""
        return await self.publish(TaskProgressEvent(
            task_id=task_id,
            progress_pct=progress_pct,
            stage=TaskStage.PROCESSING,
            message=message,
            eta_seconds=eta_seconds,
            current_step=current_step,
            current_step_num=current_step_num,
            total_steps=total_steps,
            session_id=session_id,
        ))
    
    async def task_complete(
        self,
        task_id: str,
        message: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit event when a task completes."""
        return await self.publish(TaskProgressEvent(
            task_id=task_id,
            progress_pct=100,
            stage=TaskStage.COMPLETE,
            message=message or f"Task {task_id} completed",
            session_id=session_id,
        ))
    
    async def task_failed(
        self,
        task_id: str,
        error_message: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit event when a task fails."""
        return await self.publish(TaskProgressEvent(
            task_id=task_id,
            progress_pct=0,
            stage=TaskStage.FAILED,
            message=error_message,
            session_id=session_id,
        ))
    
    async def council_perspective(
        self,
        council_session_id: str,
        agent_id: str,
        content: str,
        confidence: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit event when a council member adds a perspective."""
        return await self.publish(CouncilEvent(
            council_session_id=council_session_id,
            event=CouncilEventType.PERSPECTIVE_ADDED,
            agent_id=agent_id,
            content=content,
            confidence=confidence,
            session_id=session_id,
        ))
    
    async def council_vote(
        self,
        council_session_id: str,
        agent_id: str,
        vote: str,
        confidence: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit event when a council member casts a vote."""
        return await self.publish(CouncilEvent(
            council_session_id=council_session_id,
            event=CouncilEventType.VOTE_CAST,
            agent_id=agent_id,
            vote=vote,
            confidence=confidence,
            session_id=session_id,
        ))
    
    async def council_synthesis(
        self,
        council_session_id: str,
        content: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Emit event when council synthesis is ready."""
        return await self.publish(CouncilEvent(
            council_session_id=council_session_id,
            event=CouncilEventType.SYNTHESIS_READY,
            content=content,
            session_id=session_id,
        ))
    
    async def system_info(
        self,
        message: str,
        source: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> bool:
        """Emit info-level system event."""
        return await self.publish(SystemEvent(
            level=SystemLevel.INFO,
            message=message,
            source=source,
            details=details,
        ))
    
    async def system_warning(
        self,
        message: str,
        source: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> bool:
        """Emit warning-level system event."""
        return await self.publish(SystemEvent(
            level=SystemLevel.WARNING,
            message=message,
            source=source,
            details=details,
        ))
    
    async def system_error(
        self,
        message: str,
        source: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> bool:
        """Emit error-level system event."""
        return await self.publish(SystemEvent(
            level=SystemLevel.ERROR,
            message=message,
            source=source,
            error_code=error_code,
            details=details,
        ))
    
    async def notify(
        self,
        title: str,
        body: str,
        action_url: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        auto_dismiss_ms: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Send a notification to all connected clients."""
        return await self.publish(NotificationEvent(
            title=title,
            body=body,
            action_url=action_url,
            priority=priority,
            auto_dismiss_ms=auto_dismiss_ms,
            session_id=session_id,
        ))
    
    async def notify_instance(
        self,
        instance_id: str,
        title: str,
        body: str,
        action_url: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> bool:
        """Send a notification to a specific client."""
        return await self.publish_to_instance(
            instance_id,
            NotificationEvent(
                title=title,
                body=body,
                action_url=action_url,
                priority=priority,
            )
        )


# =============================================================================
# Global Singleton Instance
# =============================================================================

event_publisher = EventPublisher()
