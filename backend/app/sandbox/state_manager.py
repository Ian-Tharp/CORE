"""
State Manager for Cross-Container Communication

Provides state persistence and sharing between sandboxed agent containers.
Uses Redis for coordination/signaling and SQLite for structured state.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from uuid import uuid4
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StateScope(str, Enum):
    """Scope of state data."""
    TASK = "task"           # Single task execution
    SESSION = "session"     # User session
    AGENT = "agent"         # Agent-specific persistent state
    GLOBAL = "global"       # Shared across all agents


class AgentState(BaseModel):
    """State data for an agent execution."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    task_id: str
    scope: StateScope = StateScope.TASK
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class EventType(str, Enum):
    """Types of state events."""
    STATE_UPDATED = "state_updated"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_READY = "agent_ready"
    AGENT_BUSY = "agent_busy"
    ARTIFACT_CREATED = "artifact_created"


class StateEvent(BaseModel):
    """Event for state changes and coordination."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    source_agent: str
    target_agent: Optional[str] = None  # None = broadcast
    task_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class StateManager:
    """
    Manages state persistence and coordination for sandboxed agents.
    
    Architecture:
    - In-memory store for fast access (with optional Redis backend)
    - SQLite for persistent structured state
    - Event-based coordination between agents
    
    Note: This is a simplified implementation. For production, use actual
    Redis for pub/sub and distributed locking.
    """
    
    def __init__(
        self,
        data_dir: str = "./state_data",
        redis_url: Optional[str] = None,
        state_ttl_minutes: int = 60
    ):
        """
        Initialize the state manager.
        
        Args:
            data_dir: Directory for SQLite and artifact storage
            redis_url: Redis URL for coordination (optional)
            state_ttl_minutes: Default TTL for task-scoped state
        """
        self.data_dir = Path(data_dir)
        self.redis_url = redis_url
        self.state_ttl = timedelta(minutes=state_ttl_minutes)
        
        # In-memory state store (replace with Redis for production)
        self._states: Dict[str, AgentState] = {}
        
        # Event queues by agent (replace with Redis pub/sub for production)
        self._event_queues: Dict[str, asyncio.Queue] = {}
        
        # Artifacts directory
        self.artifacts_dir = self.data_dir / "artifacts"
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        # Database connection (lazily initialized)
        self._db_initialized = False
    
    async def initialize(self):
        """Initialize the state manager."""
        logger.info("Initializing StateManager...")
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database
        await self._init_database()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
        
        logger.info("StateManager initialized")
    
    async def _init_database(self):
        """Initialize SQLite database for persistent state."""
        import aiosqlite
        
        db_path = self.data_dir / "state.db"
        
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_states_agent_id 
                ON agent_states(agent_id)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_states_task_id 
                ON agent_states(task_id)
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    content_type TEXT,
                    size_bytes INTEGER,
                    path TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            await db.commit()
        
        self._db_initialized = True
        logger.debug("SQLite database initialized")
    
    # State Operations
    
    async def set_state(
        self,
        agent_id: str,
        task_id: str,
        data: Dict[str, Any],
        scope: StateScope = StateScope.TASK,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_minutes: Optional[int] = None
    ) -> AgentState:
        """
        Set state for an agent.
        
        Args:
            agent_id: ID of the agent
            task_id: ID of the current task
            data: State data to store
            scope: Scope of the state
            metadata: Optional metadata
            ttl_minutes: Time to live in minutes (None for default)
            
        Returns:
            Created AgentState
        """
        async with self._lock:
            state_key = self._make_state_key(agent_id, task_id, scope)
            
            # Check for existing state
            existing = self._states.get(state_key)
            
            if existing:
                # Update existing state
                existing.data.update(data)
                existing.updated_at = datetime.utcnow()
                if metadata:
                    existing.metadata.update(metadata)
                state = existing
            else:
                # Create new state
                expires_at = None
                if ttl_minutes or (scope == StateScope.TASK and not ttl_minutes):
                    ttl = timedelta(minutes=ttl_minutes) if ttl_minutes else self.state_ttl
                    expires_at = datetime.utcnow() + ttl
                
                state = AgentState(
                    agent_id=agent_id,
                    task_id=task_id,
                    scope=scope,
                    data=data,
                    metadata=metadata or {},
                    expires_at=expires_at
                )
                self._states[state_key] = state
            
            # Persist to database for non-task scopes
            if scope != StateScope.TASK:
                await self._persist_state(state)
            
            # Publish state update event
            await self._publish_event(StateEvent(
                event_type=EventType.STATE_UPDATED,
                source_agent=agent_id,
                task_id=task_id,
                payload={"state_key": state_key, "scope": scope}
            ))
            
            return state
    
    async def get_state(
        self,
        agent_id: str,
        task_id: str,
        scope: StateScope = StateScope.TASK
    ) -> Optional[AgentState]:
        """
        Get state for an agent.
        
        Args:
            agent_id: ID of the agent
            task_id: ID of the current task
            scope: Scope of the state
            
        Returns:
            AgentState if found, None otherwise
        """
        state_key = self._make_state_key(agent_id, task_id, scope)
        
        # Check in-memory first
        state = self._states.get(state_key)
        
        if state:
            # Check expiration
            if state.expires_at and datetime.utcnow() > state.expires_at:
                del self._states[state_key]
                return None
            return state
        
        # Try to load from database for persistent scopes
        if scope != StateScope.TASK:
            state = await self._load_state(agent_id, task_id, scope)
            if state:
                self._states[state_key] = state
            return state
        
        return None
    
    async def delete_state(
        self,
        agent_id: str,
        task_id: str,
        scope: StateScope = StateScope.TASK
    ) -> bool:
        """
        Delete state for an agent.
        
        Args:
            agent_id: ID of the agent
            task_id: ID of the current task
            scope: Scope of the state
            
        Returns:
            True if state was deleted
        """
        async with self._lock:
            state_key = self._make_state_key(agent_id, task_id, scope)
            
            if state_key in self._states:
                del self._states[state_key]
                
                if scope != StateScope.TASK:
                    await self._delete_persisted_state(agent_id, task_id, scope)
                
                return True
            return False
    
    # Event Operations
    
    async def subscribe_events(self, agent_id: str) -> asyncio.Queue:
        """
        Subscribe to events for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Queue to receive events
        """
        if agent_id not in self._event_queues:
            self._event_queues[agent_id] = asyncio.Queue()
        return self._event_queues[agent_id]
    
    async def unsubscribe_events(self, agent_id: str):
        """Unsubscribe from events for an agent."""
        if agent_id in self._event_queues:
            del self._event_queues[agent_id]
    
    async def publish_event(
        self,
        event_type: EventType,
        source_agent: str,
        target_agent: Optional[str] = None,
        task_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None
    ):
        """
        Publish an event.
        
        Args:
            event_type: Type of event
            source_agent: ID of the source agent
            target_agent: ID of target agent (None for broadcast)
            task_id: Associated task ID
            payload: Event payload
        """
        event = StateEvent(
            event_type=event_type,
            source_agent=source_agent,
            target_agent=target_agent,
            task_id=task_id,
            payload=payload or {}
        )
        await self._publish_event(event)
    
    async def _publish_event(self, event: StateEvent):
        """Internal event publishing."""
        if event.target_agent:
            # Targeted event
            if event.target_agent in self._event_queues:
                await self._event_queues[event.target_agent].put(event)
        else:
            # Broadcast event
            for queue in self._event_queues.values():
                await queue.put(event)
    
    # Artifact Operations
    
    async def save_artifact(
        self,
        agent_id: str,
        task_id: str,
        name: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save an artifact from agent execution.
        
        Args:
            agent_id: ID of the agent
            task_id: ID of the current task
            name: Name of the artifact
            content: Artifact content
            content_type: MIME type
            metadata: Optional metadata
            
        Returns:
            Artifact ID
        """
        artifact_id = str(uuid4())
        
        # Create artifact directory for task
        task_dir = self.artifacts_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = task_dir / f"{artifact_id}_{name}"
        file_path.write_bytes(content)
        
        # Save to database
        import aiosqlite
        
        async with aiosqlite.connect(self.data_dir / "state.db") as db:
            await db.execute("""
                INSERT INTO artifacts (id, task_id, agent_id, name, content_type, 
                                       size_bytes, path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                artifact_id,
                task_id,
                agent_id,
                name,
                content_type,
                len(content),
                str(file_path),
                json.dumps(metadata or {}),
                datetime.utcnow().isoformat()
            ))
            await db.commit()
        
        # Publish event
        await self.publish_event(
            EventType.ARTIFACT_CREATED,
            source_agent=agent_id,
            task_id=task_id,
            payload={"artifact_id": artifact_id, "name": name}
        )
        
        return artifact_id
    
    async def get_artifact(self, artifact_id: str) -> Optional[tuple]:
        """
        Get an artifact by ID.
        
        Returns:
            Tuple of (content, content_type, metadata) or None
        """
        import aiosqlite
        
        async with aiosqlite.connect(self.data_dir / "state.db") as db:
            async with db.execute(
                "SELECT path, content_type, metadata FROM artifacts WHERE id = ?",
                (artifact_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                path, content_type, metadata = row
                file_path = Path(path)
                
                if not file_path.exists():
                    return None
                
                content = file_path.read_bytes()
                return (content, content_type, json.loads(metadata))
    
    async def list_artifacts(self, task_id: str) -> List[Dict[str, Any]]:
        """List artifacts for a task."""
        import aiosqlite
        
        async with aiosqlite.connect(self.data_dir / "state.db") as db:
            async with db.execute(
                """SELECT id, name, content_type, size_bytes, created_at, metadata 
                   FROM artifacts WHERE task_id = ?""",
                (task_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "content_type": row[2],
                        "size_bytes": row[3],
                        "created_at": row[4],
                        "metadata": json.loads(row[5])
                    }
                    for row in rows
                ]
    
    # Helper Methods
    
    def _make_state_key(self, agent_id: str, task_id: str, scope: StateScope) -> str:
        """Create a unique key for state lookup."""
        if scope == StateScope.GLOBAL:
            return f"global:{scope.value}"
        elif scope == StateScope.AGENT:
            return f"agent:{agent_id}:{scope.value}"
        elif scope == StateScope.SESSION:
            return f"session:{task_id}:{scope.value}"
        else:  # TASK
            return f"task:{agent_id}:{task_id}:{scope.value}"
    
    async def _persist_state(self, state: AgentState):
        """Persist state to SQLite."""
        import aiosqlite
        
        async with aiosqlite.connect(self.data_dir / "state.db") as db:
            await db.execute("""
                INSERT OR REPLACE INTO agent_states 
                (id, agent_id, task_id, scope, data, metadata, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.id,
                state.agent_id,
                state.task_id,
                state.scope,
                json.dumps(state.data),
                json.dumps(state.metadata),
                state.created_at.isoformat(),
                state.updated_at.isoformat(),
                state.expires_at.isoformat() if state.expires_at else None
            ))
            await db.commit()
    
    async def _load_state(
        self,
        agent_id: str,
        task_id: str,
        scope: StateScope
    ) -> Optional[AgentState]:
        """Load state from SQLite."""
        import aiosqlite
        
        async with aiosqlite.connect(self.data_dir / "state.db") as db:
            async with db.execute(
                """SELECT id, data, metadata, created_at, updated_at, expires_at 
                   FROM agent_states 
                   WHERE agent_id = ? AND task_id = ? AND scope = ?""",
                (agent_id, task_id, scope.value)
            ) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return AgentState(
                    id=row[0],
                    agent_id=agent_id,
                    task_id=task_id,
                    scope=scope,
                    data=json.loads(row[1]),
                    metadata=json.loads(row[2]) if row[2] else {},
                    created_at=datetime.fromisoformat(row[3]),
                    updated_at=datetime.fromisoformat(row[4]),
                    expires_at=datetime.fromisoformat(row[5]) if row[5] else None
                )
    
    async def _delete_persisted_state(
        self,
        agent_id: str,
        task_id: str,
        scope: StateScope
    ):
        """Delete state from SQLite."""
        import aiosqlite
        
        async with aiosqlite.connect(self.data_dir / "state.db") as db:
            await db.execute(
                "DELETE FROM agent_states WHERE agent_id = ? AND task_id = ? AND scope = ?",
                (agent_id, task_id, scope.value)
            )
            await db.commit()
    
    async def _cleanup_loop(self):
        """Background task to cleanup expired state."""
        while True:
            try:
                await asyncio.sleep(60)
                await self._cleanup_expired_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in state cleanup loop: {e}")
    
    async def _cleanup_expired_state(self):
        """Cleanup expired state entries."""
        async with self._lock:
            now = datetime.utcnow()
            
            # Cleanup in-memory state
            expired_keys = [
                key for key, state in self._states.items()
                if state.expires_at and now > state.expires_at
            ]
            
            for key in expired_keys:
                del self._states[key]
            
            # Cleanup database
            import aiosqlite
            
            async with aiosqlite.connect(self.data_dir / "state.db") as db:
                await db.execute(
                    "DELETE FROM agent_states WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now.isoformat(),)
                )
                await db.commit()
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired state entries")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the state manager."""
        return {
            "state_entries": len(self._states),
            "subscribed_agents": len(self._event_queues),
            "data_dir": str(self.data_dir),
            "state_ttl_minutes": self.state_ttl.total_seconds() / 60
        }
