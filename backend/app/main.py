from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from controllers import core_entry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize and run the FastAPI application."""
    app = FastAPI(lifespan=lifespan)
    app.include_router(core_entry.router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan events of the FastAPI application."""
    # Initialization (Startup)
    logger.info("Cognitive CORE - Initializing...")
    try:
        yield
    finally:
        # Shutdown
        logger.info("Cognitive CORE - Shutting down...")


if __name__ == "__main__":
    main()
