"""
Circuit Breaker pattern for CORE provider calls.

State machine:
  CLOSED     — normal operation; failures are counted
  OPEN       — provider is down; calls rejected immediately
  HALF_OPEN  — recovery probe; one request allowed through

Transitions:
  CLOSED  → OPEN      : consecutive failures >= failure_threshold
  OPEN    → HALF_OPEN : recovery_timeout seconds have elapsed
  HALF_OPEN → CLOSED  : probe request succeeds
  HALF_OPEN → OPEN    : probe request fails

Usage:
    breaker = CircuitBreaker(name="openai", failure_threshold=3)
    if not breaker.allow_request():
        raise ProviderUnavailableError("Circuit open for openai")
    try:
        result = await call_provider()
        breaker.record_success()
    except Exception:
        breaker.record_failure()
        raise
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from threading import Lock
from typing import Dict

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ProviderUnavailableError(RuntimeError):
    """Raised when a circuit breaker is open and rejects the request."""


class CircuitBreaker:
    """
    Thread-safe circuit breaker for a single named provider.

    Parameters
    ----------
    name : str
        Human-readable provider name (used in log messages).
    failure_threshold : int
        Number of consecutive failures before opening the circuit.
    recovery_timeout : float
        Seconds to wait in OPEN state before moving to HALF_OPEN.
    success_threshold : int
        Number of consecutive successes in HALF_OPEN needed to close.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._opened_at: float | None = None
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow_request(self) -> bool:
        """
        Return True if the request should be allowed through.

        In OPEN state, checks whether the recovery timeout has elapsed;
        if so, transitions to HALF_OPEN and allows one probe request.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0)
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._consecutive_successes = 0
                    logger.info(
                        "Circuit '%s' → HALF_OPEN after %.1fs recovery", self.name, elapsed
                    )
                    return True
                return False

            # HALF_OPEN: allow exactly one probe request
            return True

    def record_success(self) -> None:
        """Record a successful call; may close the circuit if in HALF_OPEN."""
        with self._lock:
            self._consecutive_failures = 0
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._opened_at = None
                    logger.info("Circuit '%s' → CLOSED (recovered)", self.name)

    def record_failure(self) -> None:
        """Record a failed call; may open the circuit."""
        with self._lock:
            self._consecutive_failures += 1
            self._consecutive_successes = 0

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "Circuit '%s' → OPEN (probe failed; will retry in %.0fs)",
                    self.name, self.recovery_timeout,
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._consecutive_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "Circuit '%s' → OPEN after %d consecutive failures",
                    self.name, self._consecutive_failures,
                )

    def reset(self) -> None:
        """Force the circuit back to CLOSED (for testing / admin use)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._opened_at = None

    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "opened_at": self._opened_at,
            }


# ---------------------------------------------------------------------------
# Per-provider circuit breaker registry
# ---------------------------------------------------------------------------

_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = Lock()


def get_circuit_breaker(
    provider: str,
    failure_threshold: int = 3,
    recovery_timeout: float = 60.0,
) -> CircuitBreaker:
    """Return (or lazily create) the circuit breaker for a provider."""
    with _registry_lock:
        if provider not in _breakers:
            _breakers[provider] = CircuitBreaker(
                name=provider,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return _breakers[provider]


def reset_all_breakers() -> None:
    """Reset every registered circuit breaker. Useful for test isolation."""
    with _registry_lock:
        for breaker in _breakers.values():
            breaker.reset()
