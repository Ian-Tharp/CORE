# WebSocket Events API

This document describes the WebSocket event schema for real-time CORE updates. Events are broadcast to connected clients for reactive UI updates.

## Connection

Connect to the WebSocket endpoint:

```
ws://localhost:8000/ws/{instance_id}
```

Where `instance_id` is a unique identifier for your client session.

## Event Structure

All events share a common base structure:

```json
{
  "event_type": "string",
  "event_id": "uuid",
  "timestamp": "ISO 8601 datetime",
  "session_id": "optional string"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | string | Discriminator for event routing |
| `event_id` | string | Unique identifier for this event instance |
| `timestamp` | string | ISO 8601 timestamp when event occurred |
| `session_id` | string? | Optional session context |

---

## Event Types

### 1. Agent Activity Events

**Type:** `agent_activity`

Fired when an agent starts, updates, or completes an action.

```json
{
  "event_type": "agent_activity",
  "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2025-01-29T14:30:00.000Z",
  "session_id": "session-123",
  "agent_id": "comprehension-agent",
  "action": "analyzing_input",
  "status": "active",
  "message": "Parsing user intent...",
  "metadata": {
    "input_length": 150,
    "detected_language": "en"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | string | Unique identifier for the agent |
| `action` | string | Current action being performed |
| `status` | enum | Agent status (see below) |
| `message` | string? | Human-readable status message |
| `metadata` | object? | Additional action-specific data |

**Status Values:**
- `idle` - Agent is not currently active
- `active` - Agent is processing
- `thinking` - Agent is reasoning/planning
- `executing` - Agent is executing an action
- `waiting` - Agent is waiting for input/response
- `error` - Agent encountered an error
- `complete` - Agent finished the action

---

### 2. Task Progress Events

**Type:** `task_progress`

Tracks progress of long-running tasks.

```json
{
  "event_type": "task_progress",
  "event_id": "b2c3d4e5-f6a7-8901-bcde-f23456789012",
  "timestamp": "2025-01-29T14:30:05.000Z",
  "session_id": "session-123",
  "task_id": "task-abc123",
  "progress_pct": 45,
  "stage": "processing",
  "eta_seconds": 30,
  "message": "Processing chunk 9 of 20...",
  "current_step": "data_extraction",
  "total_steps": 5,
  "current_step_num": 2
}
```

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Unique identifier for the task |
| `progress_pct` | int | Progress percentage (0-100) |
| `stage` | enum | Current stage (see below) |
| `eta_seconds` | int? | Estimated seconds remaining |
| `message` | string? | Human-readable progress message |
| `current_step` | string? | Name of current step |
| `total_steps` | int? | Total number of steps |
| `current_step_num` | int? | Current step number |

**Stage Values:**
- `queued` - Task is waiting to start
- `starting` - Task is initializing
- `processing` - Task is actively running
- `finalizing` - Task is wrapping up
- `complete` - Task finished successfully
- `failed` - Task failed
- `cancelled` - Task was cancelled

---

### 3. Council Events

**Type:** `council`

Tracks multi-agent council deliberation sessions.

```json
{
  "event_type": "council",
  "event_id": "c3d4e5f6-a7b8-9012-cdef-345678901234",
  "timestamp": "2025-01-29T14:30:10.000Z",
  "session_id": "session-123",
  "council_session_id": "council-xyz789",
  "event": "perspective_added",
  "agent_id": "ethics-agent",
  "content": "From an ethical standpoint, we should consider...",
  "confidence": 0.85,
  "round_number": 1,
  "metadata": {
    "perspective_type": "ethical",
    "key_points": ["privacy", "consent", "transparency"]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `council_session_id` | string | Unique identifier for the council session |
| `event` | enum | Type of council event (see below) |
| `agent_id` | string? | Agent involved in this event |
| `content` | string? | Content of perspective/vote/synthesis |
| `vote` | string? | Vote value (if vote_cast event) |
| `confidence` | float? | Confidence score (0-1) |
| `round_number` | int? | Debate round number |
| `metadata` | object? | Additional event-specific data |

**Event Values:**
- `session_started` - Council session began
- `perspective_added` - Agent added a perspective
- `vote_cast` - Agent cast a vote
- `debate_round` - New debate round started
- `synthesis_ready` - Final synthesis is available
- `session_complete` - Council session ended

---

### 4. System Events

**Type:** `system`

System-level operational notifications.

```json
{
  "event_type": "system",
  "event_id": "d4e5f6a7-b8c9-0123-def0-456789012345",
  "timestamp": "2025-01-29T14:30:15.000Z",
  "level": "warning",
  "message": "High memory usage detected",
  "source": "health_monitor",
  "error_code": "MEM_HIGH",
  "details": {
    "memory_pct": 85,
    "threshold": 80
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `level` | enum | Severity level (see below) |
| `message` | string | Human-readable message |
| `source` | string? | Component that generated the event |
| `error_code` | string? | Machine-readable error code |
| `details` | object? | Additional diagnostic data |

**Level Values:**
- `debug` - Debugging information
- `info` - Informational message
- `warning` - Warning condition
- `error` - Error condition
- `critical` - Critical failure

---

### 5. Notification Events

**Type:** `notification`

User-facing notifications for alerts and updates.

```json
{
  "event_type": "notification",
  "event_id": "e5f6a7b8-c9d0-1234-ef01-567890123456",
  "timestamp": "2025-01-29T14:30:20.000Z",
  "title": "Task Complete",
  "body": "Your analysis has finished processing.",
  "action_url": "/tasks/abc123/results",
  "priority": "normal",
  "icon": "check-circle",
  "category": "task",
  "dismissible": true,
  "auto_dismiss_ms": 5000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Notification title |
| `body` | string | Notification body text |
| `action_url` | string? | URL to navigate on click |
| `priority` | enum | Priority level (see below) |
| `icon` | string? | Icon name or URL |
| `category` | string? | Category for grouping |
| `dismissible` | bool | Whether user can dismiss |
| `auto_dismiss_ms` | int? | Auto-dismiss after milliseconds |

**Priority Values:**
- `low` - Low priority, non-urgent
- `normal` - Normal priority (default)
- `high` - High priority, should be noticed
- `urgent` - Urgent, requires immediate attention

---

## Usage Examples

### Python (Backend)

```python
from app.services.event_publisher import event_publisher
from app.models.ws_events import AgentActivityEvent, AgentStatus

# Using convenience methods
await event_publisher.agent_started(
    agent_id="comprehension-agent",
    action="analyzing",
    message="Processing user input..."
)

await event_publisher.task_progress(
    task_id="task-123",
    progress_pct=50,
    message="Halfway done!"
)

await event_publisher.notify(
    title="Complete",
    body="Your request has been processed.",
    action_url="/results/123"
)

# Using raw events
await event_publisher.publish(AgentActivityEvent(
    agent_id="reasoning-agent",
    action="planning",
    status=AgentStatus.THINKING,
    message="Developing response strategy..."
))

# Target specific channel
await event_publisher.publish_to_channel(
    channel_id="session-abc",
    event=TaskProgressEvent(...)
)
```

### TypeScript (Frontend)

```typescript
interface BaseEvent {
  event_type: string;
  event_id: string;
  timestamp: string;
  session_id?: string;
}

interface AgentActivityEvent extends BaseEvent {
  event_type: 'agent_activity';
  agent_id: string;
  action: string;
  status: 'idle' | 'active' | 'thinking' | 'executing' | 'waiting' | 'error' | 'complete';
  message?: string;
  metadata?: Record<string, unknown>;
}

// Subscribe to events
websocket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.event_type) {
    case 'agent_activity':
      handleAgentActivity(data as AgentActivityEvent);
      break;
    case 'task_progress':
      handleTaskProgress(data as TaskProgressEvent);
      break;
    case 'notification':
      showNotification(data as NotificationEvent);
      break;
    // ...
  }
};
```

---

## Channel Subscriptions

Clients can subscribe to specific channels to receive targeted events:

```json
// Send to server
{
  "type": "subscribe",
  "channel_id": "session-123"
}

// Unsubscribe
{
  "type": "unsubscribe", 
  "channel_id": "session-123"
}
```

Events published to a channel will only be received by subscribed clients.

---

## Best Practices

1. **Handle Unknown Events**: Always have a default case for unknown event types to ensure forward compatibility.

2. **Use Event IDs**: Track `event_id` for deduplication if your connection has reliability issues.

3. **Respect Priorities**: Notification priorities should influence UI behavior (e.g., urgent notifications may interrupt the user).

4. **Progress UX**: Use `eta_seconds` when available to show meaningful time estimates.

5. **Session Context**: Use `session_id` to correlate events with specific user sessions or conversations.
