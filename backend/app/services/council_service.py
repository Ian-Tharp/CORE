"""
Council Service — convenience re-export.

The implementation lives in app.services.council.deliberation_service.
This module re-exports the public API at the services root for callers
who prefer ``from app.services.council_service import ...``.
"""

from app.services.council.deliberation_service import (  # noqa: F401
    CouncilService,
    get_council_service,
)

__all__ = ["CouncilService", "get_council_service"]
