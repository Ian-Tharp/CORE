from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controllers import chat, core_entry


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan events of the FastAPI application."""
    logger.info("CORE System- Initializing...")
    try:
        yield
    finally:
        logger.info("CORE System - Shutting down...")


# ---------------------------------------------------------------------------
# Public ASGI application
# ---------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# Routers --------------------------------------------------------------------
app.include_router(core_entry.router)
app.include_router(chat.router)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
# Allow the Angular development server (usually running on http://localhost:4200)
# to access this API during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Entry-point helper (optional)
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:  # pragma: no cover
    """Factory for reuse in unit tests or external ASGI servers."""
    return app


if __name__ == "__main__":  # pragma: no cover
    # For direct execution:  `python -m backend.app.main` or `python backend/app/main.py`
    # Pass the application instance directly so `uvicorn` doesn't have to
    # resolve the module path – this works no matter which directory you
    # launch the script from (project root or backend/).
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
