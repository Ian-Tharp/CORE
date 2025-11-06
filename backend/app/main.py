from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.controllers import chat, core_entry, conversations, system_monitor, worlds, creative, knowledgebase, local_llm, communication, agents
from app.dependencies import get_db_pool, close_db_pool, setup_db_schema
from app.websocket_manager import manager


logging.basicConfig(level=logging.INFO)
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
        except Exception as init_exc:  # noqa: BLE001
            logger.error("Failed to initialize DB pool: %s", init_exc)
            # Do not raise here to allow health endpoint and other features to run;
            # application will still surface DB errors on first use.
        yield
    finally:
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


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring."""
    return {"status": "healthy", "service": "core-backend"}


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
    - Reaction updates
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
                await manager.send_personal_message(instance_id, {"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(instance_id)
        logger.info(f"WebSocket disconnected: {instance_id}")

        # Broadcast presence update
        await manager.broadcast_presence_update(instance_id, "offline")

    except Exception as e:
        logger.error(f"WebSocket error for {instance_id}: {e}")
        manager.disconnect(instance_id)


# ---------------------------------------------------------------------------
# Entry-point helper (optional)
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Factory for reuse in unit tests or external ASGI servers."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
