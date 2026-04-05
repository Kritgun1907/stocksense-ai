"""
StockSense AI — api/deps.py
============================
FastAPI dependency functions (Depends).

WHY THIS FILE EXISTS — The Circular Import Problem
─────────────────────────────────────────────────────
main.py imports from api.routes.predict
api.routes.predict used to import from api.main  ← circular!

The fix: route files NEVER import from main.py.
Instead they declare a Depends() on a function here.
FastAPI calls the function at request time and injects the result.

Usage in any route file:
    from api.deps import get_app_state, require_model, get_redis_client

    @router.get("/")
    async def my_endpoint(
        state: dict = Depends(require_model),
        redis_client = Depends(get_redis_client),
    ):
        pipeline = state["pipeline"]    # guaranteed to be non-None
        ...
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  STATE ACCESSOR
#  Returns the live app_state dict without importing main at module level.
#  FastAPI calls this function at request time — after main.py is fully loaded.
# ══════════════════════════════════════════════════════════════════════════════

def get_app_state() -> dict:
    """
    Dependency: returns the global app_state dict from main.py.

    Imported lazily (inside the function body) so the module-level
    circular import is completely avoided.
    """
    from api.main import app_state
    return app_state


def require_model(state: dict = Depends(get_app_state)) -> dict:
    """
    Dependency: returns app_state only if the ML pipeline is loaded.

    Raises HTTP 503 Service Unavailable if trainer.py has not been run.
    Use this in any endpoint that needs the XGBoost pipeline.

    Example
    -------
        @router.get("/")
        async def predict(state: dict = Depends(require_model)):
            pipeline = state["pipeline"]
    """
    if not state.get("model_loaded"):
        raise HTTPException(
            status_code=503,
            detail={
                "error":  "Model not loaded",
                "detail": (
                    "The ML pipeline is not ready. "
                    "Run models/trainer.py to train and save a model first.\n"
                    "Command: cd backend && python3 -m models.pipeline"
                ),
            },
        )
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  REDIS CLIENT
#  Returns the Redis client from app_state, or None if Redis is not available.
#  Prediction pipeline handles None gracefully — it just skips caching.
# ══════════════════════════════════════════════════════════════════════════════

def get_redis_client(state: dict = Depends(get_app_state)) -> Optional[object]:
    """
    Dependency: returns the async Redis client if available, else None.

    Why Optional?
    ─────────────────────────────────────────────────────────────
    Redis is a PERFORMANCE OPTIMISATION, not a requirement.
    If Redis is not running, the prediction endpoint still works —
    it just computes every request from scratch instead of caching.
    Returning None instead of raising an error keeps the server usable.
    """
    return state.get("redis_client", None)


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE SESSION
#  Re-exports get_db from database.py so route files only need to import
#  from api.deps — one consistent place for all dependencies.
# ══════════════════════════════════════════════════════════════════════════════

from api.database import get_db  # noqa: F401, E402
# Usage in any route file:
#   from api.deps import get_db
#   from sqlalchemy.ext.asyncio import AsyncSession
#
#   @router.get("/items")
#   async def list_items(db: AsyncSession = Depends(get_db)):
#       ...
