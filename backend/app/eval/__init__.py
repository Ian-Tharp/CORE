"""
Offline evaluation layer for CORE.

This package defines the *contracts* for the offline eval harness — the
``Verifier`` structural protocol and its ``VerifierResult`` value object —
and nothing heavy. Importing ``app.eval`` must stay cheap and IO-free: it
pulls in no models, no DB drivers, and no service modules. Concrete
verifiers, the benchmark cases, and the runner live in sibling modules and
import their dependencies lazily.

Re-exported here are only the two leaf contracts. Do not re-export the
runner / cases / verifier modules from this file, or ``import app.eval``
would transitively load the evaluation service (and its repository → DB
driver chain).
"""

from __future__ import annotations

from app.eval.protocol import Verifier, VerifierResult

__all__ = ["Verifier", "VerifierResult"]
