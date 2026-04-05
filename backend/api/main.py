"""
StockSense AI — main.py
========================
FastAPI application entry point.

This file owns:
  - App creation and configuration
  - Startup/shutdown lifecycle (model loading)
  - Router registration (connecting route files)
  - Global exception handlers
  - Health check endpoint

It does NOT own:
  - Individual endpoint logic → api/routes/*.py
  - Request/response schemas → api/schemas.py
  - ML model code            → models/trainer.py etc.
"""

import json
import os
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import joblib

# ── FastAPI imports ───────────────────────────────────────────────────────────
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

warnings.filterwarnings("ignore")

# ── Ensure backend/ is importable ────────────────────────────────────────────
# This lets all your existing code (models/, backtest/, nlp/) import normally
# __file__ is backend/api/main.py → .parent = api/ → .parent = backend/
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION STATE
#  This dict lives in memory for the entire server lifetime.
#  Load expensive objects once here; reuse them in every endpoint.
# ══════════════════════════════════════════════════════════════════════════════

# app.state is FastAPI's built-in place to store shared objects
# We'll attach our loaded pipeline here during startup
app_state = {
    "pipeline":      None,   # fitted sklearn Pipeline from trainer.py
    "feature_cols":  None,   # list of feature column names
    "model_loaded":  False,  # flag for health checks
    "redis_client":  None,   # async Redis client (None = caching disabled)
}


# ══════════════════════════════════════════════════════════════════════════════
#  LIFESPAN — Startup and Shutdown
# ══════════════════════════════════════════════════════════════════════════════

# In main.py — replace the load section inside lifespan:

@asynccontextmanager
async def lifespan(app: FastAPI):

    # ── STARTUP ────────────────────────────────────────────────────────────
    print("StockSense AI — Starting up...")

    try:
        # _BACKEND is already defined at module level as backend/
        # Using absolute paths so the server works from any working directory
        latest_pkl  = _BACKEND / "models" / "saved" / "stocksense_latest.pkl"
        latest_json = _BACKEND / "models" / "saved" / "metadata" / "stocksense_latest.json"

        if not latest_pkl.exists():
            print("⚠️  No trained model found.")
            print("   Run: python scripts/train_and_save.py")
            print("   Server starts but /predict won't work until model exists.")
        else:
            print(f"Loading model from {latest_pkl}...")
            bundle = joblib.load(latest_pkl)

            app_state["pipeline"]     = bundle["pipeline"]
            app_state["feature_cols"] = bundle["feature_cols"]
            app_state["model_loaded"] = True

            print(f"✅ Model loaded: {len(app_state['feature_cols'])} features")

            # Load and print metadata if it exists
            if latest_json.exists():
                with open(latest_json) as f:
                    meta = json.load(f)
                print(f"✅ Model trained on: {meta['training']['tickers']}")
                print(f"✅ Test accuracy: "
                      f"{meta['performance']['test_accuracy']*100:.1f}%")

    except Exception as e:
        print(f"❌ Model load failed: {e}")

    # FinBERT warmup — non-fatal: server works without it
    # /predict (price-based ML) does NOT need FinBERT
    # /sentiment endpoints will fail gracefully if FinBERT is unavailable
    try:
        from nlp.finbert import warmup
        warmup()
        print("✅ FinBERT ready")
    except Exception as e:
        print(f"⚠️  FinBERT unavailable (sentiment endpoints will be disabled): {e}")
        print("   /predict and /backtest endpoints are unaffected.")

    # ── Redis connection — non-fatal: server works without it ──────────────
    # If Redis is not installed or not running, predictions still work
    # but every request recomputes from scratch (no caching).
    # See docs/redis_guide.py for full explanation.
    # ── PostgreSQL / database tables ───────────────────────────────────────
    # Creates all tables defined in api/database.py if they don't exist yet.
    # On subsequent startups it's a no-op (tables already exist).
    # See docs/sqlalchemy_alembic_guide.py for migration workflow.
    try:
        from api.database import init_db
        await init_db()
        print("✅ Database tables ready")
    except Exception as e:
        print(f"⚠️  Database init failed (DB endpoints disabled): {e}")
        print("   Is PostgreSQL running? See docs/sqlalchemy_alembic_guide.py")

    try:
        import redis.asyncio as aioredis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = aioredis.from_url(
            redis_url,
            decode_responses=True,       # return strings, not bytes
            socket_connect_timeout=2,    # fail fast if Redis not running
        )
        # Test connection
        await redis_client.ping()
        app_state["redis_client"] = redis_client
        print(f"✅ Redis connected: {redis_url}")
    except Exception as e:
        print(f"⚠️  Redis unavailable (caching disabled): {e}")
        print("   Predictions will work but won't be cached.")
        app_state["redis_client"] = None

    print("✅ Server ready")

    yield

    # ── SHUTDOWN ───────────────────────────────────────────────────────────
    # Close Redis connection gracefully
    if app_state.get("redis_client"):
        try:
            await app_state["redis_client"].close()
            print("Redis connection closed")
        except Exception:
            pass

    app_state["pipeline"]     = None
    app_state["model_loaded"] = False
    app_state["redis_client"] = None
    print("Server shut down cleanly")


# ══════════════════════════════════════════════════════════════════════════════
#  CREATE APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="StockSense AI",
    description="""
    ML-powered stock prediction and analysis API.
    
    Features:
    - UP/DOWN predictions with SHAP explanations
    - Full backtesting with performance metrics
    - FinBERT sentiment analysis from financial news
    - Multi-stock screener
    """,
    version="1.0.0",
    lifespan=lifespan,  # connect our startup/shutdown logic
)


# ══════════════════════════════════════════════════════════════════════════════
#  CORS MIDDLEWARE
#  This is REQUIRED for your React frontend to talk to this API.
# ══════════════════════════════════════════════════════════════════════════════

"""
CORS = Cross-Origin Resource Sharing.

The browser has a security rule: JavaScript on website A cannot
make requests to website B unless website B explicitly allows it.

Your React app runs on  → http://localhost:3000 (development)
Your FastAPI runs on    → http://localhost:8000

Without CORS config, the browser BLOCKS the request with:
"Access to fetch at 'http://localhost:8000' from origin 
 'http://localhost:3000' has been blocked by CORS policy"

The middleware below tells the browser: "Yes, localhost:3000 is allowed."
"""

app.add_middleware(
    CORSMiddleware,
    # Which frontend origins are allowed to call this API
    allow_origins=[
        "http://localhost:3000",   # React dev server
        "http://localhost:5173",   # Vite dev server (alternative)
        "https://stocksense.ai",   # Production frontend (when you deploy)
    ],
    allow_credentials=True,         # Allow cookies (for auth later)
    allow_methods=["*"],            # Allow GET, POST, etc.
    allow_headers=["*"],            # Allow any headers
)


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL EXCEPTION HANDLER
#  Catches any unhandled Python exception and returns clean JSON
#  instead of crashing the server with a 500 stack trace.
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Without this, an unhandled exception returns HTML to your JSON API.
    With this, it returns a structured JSON error the frontend can display.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error":   "Internal server error",
            "detail":  str(exc),
            "path":    str(request.url),
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK ENDPOINT
#  Always build this first — it lets you verify the server is running
#  before you add any ML logic.
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    """
    Returns server status and model loading state.
    
    Used by:
    - Frontend: shows a "Model Ready" badge in the UI
    - DevOps: monitoring services ping this to confirm server is up
    - You: sanity check during development
    """
    from nlp.finbert import get_model_info
    
    return {
        "status":         "ok",
        "version":        "1.0.0",
        "model_loaded":   app_state["model_loaded"],
        "n_features":     len(app_state["feature_cols"]) 
                          if app_state["feature_cols"] else 0,
        "finbert_info":   get_model_info(),
    }


@app.get("/", tags=["System"])
def root():
    """Root endpoint — confirms the server is alive."""
    return {
        "message": "StockSense AI API",
        "docs":    "/docs",
        "health":  "/health",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTER ROUTERS
#  Each route file handles one feature area.
#  We'll build these files in Chapter 6.2 onwards.
# ══════════════════════════════════════════════════════════════════════════════

# ── Routers are registered AFTER app is created to avoid circular imports ─────
# Each router reads app_state via FastAPI's dependency injection (Depends),
# NOT via "from api.main import app_state" which causes circular import errors.
from api.routes.predict import router as predict_router

app.include_router(predict_router,   prefix="/predict",   tags=["Predictions"])
# app.include_router(backtest_router,  prefix="/backtest",  tags=["Backtesting"])
# app.include_router(sentiment_router, prefix="/sentiment", tags=["Sentiment"])
# app.include_router(screener_router,  prefix="/screener",  tags=["Screener"])


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — Run with: uvicorn main:app --reload
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",     # full module path relative to backend/
        host="0.0.0.0",    # listen on all network interfaces
        port=8000,
        reload=True,        # auto-restart when you edit files (dev only)
        log_level="info",
    )