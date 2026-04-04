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
    from api.deps import get_app_state, require_model

    @router.get("/")
    def my_endpoint(state: dict = Depends(get_app_state)):
        pipeline = state["pipeline"]
        ...

    # Or use the guard directly to auto-raise 503 if model not loaded:
    @router.get("/")
    def my_endpoint(state: dict = Depends(require_model)):
        pipeline = state["pipeline"]    # guaranteed to be non-None
        feature_cols = state["feature_cols"]
        ...
"""

from fastapi import Depends, HTTPException


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
    # Late import — safe because main.py is fully initialised by the time
    # any request arrives. This is the standard FastAPI pattern.
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
        def predict(state: dict = Depends(require_model)):
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
