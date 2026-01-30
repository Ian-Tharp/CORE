"""
WebSocket Event Schema Definitions for CORE.

Pydantic models defining the structure of real-time events broadcast
over WebSocket connections. These events enable reactive UI updates
for agent activities, task progress, council deliberations, and system notifications.

Usage:
    from app.models.ws_events import AgentActivityEvent, TaskProgressEvent
    
    event = AgentActivityEvent(
        agent_id="agent-001",
        action="thinking",
        status="active",
        message="Processing user request..."
    )
    await event_publisher.publish(event)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Any, Dict
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class EventType(str, Enum):
    """All supported WebSocket event types."""
    # Agent events
    AGENT_ACTIVITY = "agent_activity"
    
    # Task events
    TASK_PROGRESS = "task_progress"
    
    # Council events
    COUNCIL = "council"
    
    # System events
    SYSTEM = "system"
    
    # Notification events
    NOTIFICATION = "notification"


class AgentStatus(str, Enum):
    """Status of an agent activity."""
    IDLE = "idle"
    ACTIVE = "active"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETE = "complete"


class TaskStage(str, Enum):
    """Stage of a task's lifecycle."""
    QUEUED = "queued"
    STARTING = "starting"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CouncilEventType(str, Enum):
    """Types of council deliberation events."""
    SESSION_STARTED = "session_started"
    PERSPECTIVE_ADDED = "perspective_added"
    VOTE_CAST = "vote_cast"
    DEBATE_ROUND = "debate_round"
    SYNTHESIS_READY = "synthesis_ready"
    SESSION_COMPLETE = "session_complete"


class SystemLevel(str, Enum):
    """Severity level for system events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationPriority(str, Enum):
    """Priority level for notifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# =============================================================================
# Base Event
# =============================================================================

class BaseEvent(BaseModel):
    """
    Base class for all WebSocket events.
    
    Every event includes:
    - event_type: Discriminator for event routing
    - event_id: Unique identifier for this event instance
    - timestamp: When the event occurred (ISO 8601)
    - session_id: Optional session context
    """
    event_type: EventType
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_ws_message(self) -> Dict[str, Any]:
        """Convert event to WebSocket-ready dict format."""
        return self.model_dump(mode="json")


# =============================================================================
# Agent Activity Events
# =============================================================================

class AgentActivityEvent(BaseEvent):
    """
    Event for agent activity updates.
    
    Fired when an agent starts, updates, or completes an action.
    Useful for showing real-time agent status in the UI.
    
    Example:
        {
            "event_type": "agent_activity",
            "agent_id": "comprehension-agent",
            "action": "analyzing_input",
            "status": "active",
            "message": "Parsing user intent..."
        }
    """
    event_type: EventType = EventType.AGENT_ACTIVITY
    agent_id: str = Field(..., description="Unique identifier for the agent")
    action: str = Field(..., description="Current action being performed")
    status: AgentStatus = Field(..., description="Current status of the agent")
    message: Optional[str] = Field(None, description="Human-readable status message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional action-specific data")


# =============================================================================
# Task Progress Events
# =============================================================================

class TaskProgressEvent(BaseEvent):
    """
    Event for task progress updates.
    
    Tracks the progress of long-running tasks through various stages.
    Includes percentage completion and estimated time remaining.
    
    Example:
        {
            "event_type": "task_progress",
            "task_id": "task-abc123",
            "progress_pct": 45,
            "stage": "processing",
            "eta_seconds": 30,
            "message": "Processing chunk 9 of 20..."
        }
    """
    event_type: EventType = EventType.TASK_PROGRESS
    task_id: str = Field(..., description="Unique identifier for the task")
    progress_pct: int = Field(..., ge=0, le=100, description="Progress percentage (0-100)")
    stage: TaskStage = Field(..., description="Current stage of the task")
    eta_seconds: Optional[int] = Field(None, ge=0, description="Estimated seconds remaining")
    message: Optional[str] = Field(None, description="Human-readable progress message")
    current_step: Optional[str] = Field(None, description="Name of the current step")
    total_steps: Optional[int] = Field(None, ge=1, description="Total number of steps")
    current_step_num: Optional[int] = Field(None, ge=1, description="Current step number")


# =============================================================================
# Council Events
# =============================================================================

class CouncilEvent(BaseEvent):
    """
    Event for council deliberation updates.
    
    Tracks the progress of multi-agent council sessions, including
    perspective submissions, voting, and synthesis generation.
    
    Example:
        {
            "event_type": "council",
            "council_session_id": "council-xyz789",
            "event": "perspective_added",
            "agent_id": "ethics-agent",
            "content": "From an ethical standpoint..."
        }
    """
    event_type: EventType = EventType.COUNCIL
    council_session_id: str = Field(..., description="Unique identifier for the council session")
    event: CouncilEventType = Field(..., description="Type of council event")
    agent_id: Optional[str] = Field(None, description="Agent involved in this event")
    content: Optional[str] = Field(None, description="Content of the perspective/vote/synthesis")
    vote: Optional[str] = Field(None, description="Vote value if event is vote_cast")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score (0-1)")
    round_number: Optional[int] = Field(None, ge=1, description="Debate round number")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional event-specific data")


# =============================================================================
# System Events
# =============================================================================

class SystemEvent(BaseEvent):
    """
    Event for system-level notifications.
    
    Used for operational messages like service health,
    configuration changes, or error conditions.
    
    Example:
        {
            "event_type": "system",
            "level": "warning",
            "message": "High memory usage detected",
            "source": "health_monitor",
            "details": {"memory_pct": 85}
        }
    """
    event_type: EventType = EventType.SYSTEM
    level: SystemLevel = Field(..., description="Severity level of the event")
    message: str = Field(..., description="Human-readable message")
    source: Optional[str] = Field(None, description="Component that generated the event")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional diagnostic data")


# =============================================================================
# Notification Events
# =============================================================================

class NotificationEvent(BaseEvent):
    """
    Event for user-facing notifications.
    
    Used to display alerts, updates, or action prompts to the user.
    Can include optional action URLs for interactive notifications.
    
    Example:
        {
            "event_type": "notification",
            "title": "Task Complete",
            "body": "Your analysis has finished processing.",
            "action_url": "/tasks/abc123/results",
            "priority": "normal"
        }
    """
    event_type: EventType = EventType.NOTIFICATION
    title: str = Field(..., description="Notification title")
    body: str = Field(..., description="Notification body text")
    action_url: Optional[str] = Field(None, description="URL to navigate to on click")
    priority: NotificationPriority = Field(
        default=NotificationPriority.NORMAL, 
        description="Notification priority"
    )
    icon: Optional[str] = Field(None, description="Icon name or URL")
    category: Optional[str] = Field(None, description="Notification category for grouping")
    dismissible: bool = Field(default=True, description="Whether notification can be dismissed")
    auto_dismiss_ms: Optional[int] = Field(None, ge=0, description="Auto-dismiss after milliseconds")


# =============================================================================
# Type Union for Event Handling
# =============================================================================

# Union type for type-safe event handling
WSEvent = AgentActivityEvent | TaskProgressEvent | CouncilEvent | SystemEvent | NotificationEvent
