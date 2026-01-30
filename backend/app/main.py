from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.controllers import chat, core_entry, conversations, system_monitor, worlds, creative, knowledgebase, local_llm, communication, agents, engine, test_core, health, admin, council, instances
from app.controllers.agent_ws import agent_websocket_endpoint
from app.dependencies import get_db_pool, close_db_pool, setup_db_schema
from app.websocket_manager import manager
from app.core.middleware import setup_middleware
from app.services.webhook_service import init_webhook_service, shutdown_webhook_service
from app.services.agent_registry import initialize_agent_registry, shutdown_agent_registry
from app.repository import run_repository, council_repository, instance_repository


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan events of the FastAPI application."""
    logger.info("CORE System - Initializing...")
    try:
        # Initialize DB pool and ensure schema on startup so first request is fast.
        try:
            await get_db_pool()
            logger.info("Database pool initialized")
            await setup_db_schema()
            logger.info("Database schema ensured")
            
            # Ensure run repository table exists
            await run_repository.ensure_runs_table()
            logger.info("Engine runs table ensured")
            
            # Ensure council tables exist
            await council_repository.ensure_council_tables()
            logger.info("Council tables ensured")
            
            # Ensure instance tables exist
            await instance_repository.ensure_instance_tables()
            logger.info("Instance tables ensured")
        except Exception as init_exc:  # noqa: BLE001
            logger.error("Failed to initialize DB pool: %s", init_exc)
            # Do not raise here to allow health endpoint and other features to run;
            # application will still surface DB errors on first use.
        
        # Initialize webhook service
        try:
            await init_webhook_service()
            logger.info("Webhook service initialized")
        except Exception as webhook_exc:
            logger.error("Failed to initialize webhook service: %s", webhook_exc)
        
        # Initialize agent registry
        try:
            await initialize_agent_registry()
            logger.info("Agent registry initialized")
        except Exception as agent_exc:
            logger.error("Failed to initialize agent registry: %s", agent_exc)
        
        yield
    finally:
        # Shutdown webhook service
        try:
            await shutdown_webhook_service()
            logger.info("Webhook service shutdown")
        except Exception as webhook_close_exc:
            logger.error("Error shutting down webhook service: %s", webhook_close_exc)
        
        # Shutdown agent registry
        try:
            await shutdown_agent_registry()
            logger.info("Agent registry shutdown")
        except Exception as agent_close_exc:
            logger.error("Error shutting down agent registry: %s", agent_close_exc)
        
        # Gracefully close DB connections on shutdown.
        try:
            await close_db_pool()
            logger.info("Database pool closed")
        except Exception as close_exc:  # noqa: BLE001
            logger.error("Error while closing DB pool: %s", close_exc)
        logger.info("CORE System - Shutting down...")


# ---------------------------------------------------------------------------
# Public ASGI application
# ---------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# Routers --------------------------------------------------------------------
app.include_router(core_entry.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(system_monitor.router)
app.include_router(worlds.router)
app.include_router(creative.router)
app.include_router(knowledgebase.router)
app.include_router(local_llm.router)
app.include_router(communication.router)
app.include_router(agents.router)
app.include_router(engine.router)  # CORE cognitive engine endpoint
app.include_router(test_core.router)  # Test endpoints
app.include_router(health.router)  # Health check endpoints (includes /health)
app.include_router(admin.router)  # Admin and management endpoints
app.include_router(council.router)  # Council of Perspectives deliberation system
app.include_router(instances.router)  # Instance management and container orchestration

# Setup custom middleware (logging, metrics, error handling)
setup_middleware(app)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
# Allow both Angular development server and Electron app to access this API
app.add_middleware(
    CORSMiddleware,
    # Development: allow all origins to prevent CORS headaches across
    # Electron (app://, file://) and local dev servers (localhost:*).
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# NOTE: /health endpoint is now handled by health.router (no duplicate!)


# ---------------------------------------------------------------------------
# WebSocket endpoint for real-time communication
# ---------------------------------------------------------------------------
@app.websocket("/ws/{instance_id}")
async def websocket_endpoint(websocket: WebSocket, instance_id: str):
    """
    WebSocket endpoint for real-time Communication Commons.

    Each instance connects with their instance_id and receives:
    - Real-time messages in subscribed channels
    - Presence updates
    - Typing indicators
    - Read receipts
    """
    await manager.connect(instance_id, websocket)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "subscribe":
                # Subscribe to channels
                channel_ids = data.get("channel_ids", [])
                for channel_id in channel_ids:
                    manager.subscribe_to_channel(instance_id, channel_id)

            elif message_type == "unsubscribe":
                # Unsubscribe from channels
                channel_ids = data.get("channel_ids", [])
                for channel_id in channel_ids:
                    manager.unsubscribe_from_channel(instance_id, channel_id)

            elif message_type == "ping":
                # Heartbeat/keepalive
                await manager.heartbeat(instance_id)
                await manager.send_personal_message(instance_id, {"type": "pong"})
            
            elif message_type == "typing_start":
                # User started typing
                channel_id = data.get("channel_id")
                if channel_id:
                    await manager.start_typing(instance_id, channel_id)
            
            elif message_type == "typing_stop":
                # User stopped typing
                channel_id = data.get("channel_id")
                if channel_id:
                    await manager.stop_typing(instance_id, channel_id)
            
            elif message_type == "mark_read":
                # Mark message as read
                message_id = data.get("message_id")
                channel_id = data.get("channel_id")
                if message_id and channel_id:
                    await manager.mark_read(instance_id, message_id, channel_id)
            
            elif message_type == "set_metadata":
                # Set connection metadata
                metadata = data.get("metadata", {})
                manager.set_metadata(instance_id, metadata)

    except WebSocketDisconnect:
        manager.disconnect(instance_id)
        logger.info(f"WebSocket disconnected: {instance_id}")

        # Broadcast presence update
        await manager.broadcast_presence_update(instance_id, "offline")

    except Exception as e:
        logger.error(f"WebSocket error for {instance_id}: {e}")
        manager.disconnect(instance_id)


# ---------------------------------------------------------------------------
# Agent WebSocket endpoint for container registration and communication
# ---------------------------------------------------------------------------
@app.websocket("/ws/agent/{agent_id}")
async def websocket_agent_endpoint(websocket: WebSocket, agent_id: str):
    """
    WebSocket endpoint for agent registration and communication.

    Separate from Communication Commons WebSocket - this is specifically
    for containerized agents to register, send heartbeats, and receive tasks.
    
    Each agent connects with their agent_id and can:
    - Register with capabilities and version
    - Send periodic heartbeats with status and resource usage
    - Report task completion or refusal
    - Receive task assignments and configuration updates
    - Gracefully deregister on shutdown
    """
    await agent_websocket_endpoint(websocket, agent_id)


# ---------------------------------------------------------------------------
# Entry-point helper (optional)
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Factory for reuse in unit tests or external ASGI servers."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
