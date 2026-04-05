"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    REDIS — COMPLETE LEARNING GUIDE                          ║
║                    For StockSense AI Prediction API                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

This is a fully annotated learning file. Every concept has:
  - WHAT it is
  - WHY we need it
  - HOW to use it (syntax)
  - WHERE it fits in our StockSense architecture
  - RUNNABLE code examples you can execute directly

Table of Contents:
─────────────────────────────────────────────────────────────
  1. What is Redis?
  2. Why Use Redis for StockSense?
  3. Installation and Setup
  4. Redis Data Types (with examples)
  5. Python Redis Client (sync)
  6. Python Redis Async Client (for FastAPI)
  7. TTL — Time-To-Live (Expiry)
  8. Cache Key Design
  9. The Cache-Aside Pattern (our approach)
  10. Integration with FastAPI Lifespan
  11. Our Full Implementation Walkthrough
  12. Redis CLI — Debugging and Monitoring
  13. Production Considerations

Prerequisites:
─────────────────────────────────────────────────────────────
  pip install redis[hiredis]   (already in requirements.txt)
  brew install redis           (macOS)
  brew services start redis    (start Redis server)
"""


# ══════════════════════════════════════════════════════════════════════════════
#  1. WHAT IS REDIS?
# ══════════════════════════════════════════════════════════════════════════════
"""
Redis = Remote Dictionary Server

Think of it as a giant Python dict that lives OUTSIDE your application:
  - Multiple processes can read/write the same keys
  - Data persists across server restarts (optional)
  - It's incredibly fast: 100,000+ operations per second
  - It runs as a separate service (like PostgreSQL)

Regular Python dict:
    cache = {}
    cache["prediction:AAPL"] = {"prediction": "UP", "prob": 0.73}
    result = cache.get("prediction:AAPL")    # → dict or None

Redis (conceptually identical):
    redis.set("prediction:AAPL", '{"prediction": "UP", "prob": 0.73}')
    result = redis.get("prediction:AAPL")    # → string or None

The difference:
─────────────────────────────────────────────────────────────
  Python dict          │  Redis
  ─────────────────────┼──────────────────────────────
  Lives in process RAM │  Lives in separate server
  Lost when app stops  │  Persists across restarts
  One process only     │  Shared across processes
  No TTL (manual del)  │  Built-in TTL auto-expiry
  O(1) lookup          │  O(1) lookup + network RTT

Architecture:
  ┌─────────────┐     TCP:6379     ┌─────────────┐
  │  FastAPI     │ ◄─────────────► │   Redis      │
  │  (port 8000) │                 │  (port 6379) │
  └─────────────┘                  └─────────────┘
        │                                │
     Handles HTTP                  Stores cached
     requests from                 predictions as
     React frontend               JSON strings
"""


# ══════════════════════════════════════════════════════════════════════════════
#  2. WHY USE REDIS FOR STOCKSENSE?
# ══════════════════════════════════════════════════════════════════════════════
"""
Our prediction pipeline takes 3-15 seconds per stock:
  ┌───────────────────────────────────┐
  │  yfinance download    → 1-3 sec   │  (network I/O)
  │  Feature engineering  → 0.5 sec   │  (CPU)
  │  XGBoost predict      → 0.1 sec   │  (CPU)
  │  SHAP explanation     → 1-5 sec   │  (CPU)
  │  Sentiment (FinBERT)  → 2-5 sec   │  (CPU/GPU)
  │  ─────────────────────────────────│
  │  TOTAL                → 5-14 sec  │
  └───────────────────────────────────┘

Without caching: Every request for AAPL runs the full pipeline.
With Redis cache: Second request for AAPL returns in <5ms.

When is caching appropriate?
─────────────────────────────────────────────────────────────
  ✅ Same stock predicted multiple times within an hour
  ✅ Screener endpoint scanning 500 stocks (many cache hits)
  ✅ Multiple users viewing the same stock
  ✅ Frontend making redundant requests (component re-renders)

  ❌ First prediction for a stock (always a cache miss)
  ❌ After market close (new data, cache should expire)

Key insight:
  Our ML features are computed from daily OHLCV bars.
  Within the same trading day, the features are IDENTICAL.
  So caching for 1 hour is perfectly safe — the answer won't change
  until the market produces new daily data.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  3. INSTALLATION AND SETUP
# ══════════════════════════════════════════════════════════════════════════════
"""
Step 1: Install the Redis SERVER (the database that runs separately)
─────────────────────────────────────────────────────────────

  macOS (Homebrew):
    brew install redis
    brew services start redis       # start as background service
    brew services stop redis        # stop the service

  Ubuntu/Debian:
    sudo apt install redis-server
    sudo systemctl start redis
    sudo systemctl enable redis     # auto-start on boot

  Docker (recommended for production):
    docker run -d --name redis \
      -p 6379:6379 \
      --restart unless-stopped \
      redis:7-alpine

  Verify it's running:
    redis-cli ping                  # should print: PONG


Step 2: Install the Python CLIENT (library to talk to Redis)
─────────────────────────────────────────────────────────────

  pip install redis[hiredis]

  What's in the [hiredis] extra?
  - redis      = Pure Python Redis client
  - hiredis    = C library that parses Redis protocol 10× faster
  - Together   = Full speed, zero effort

  The redis package includes BOTH sync and async clients:
  - import redis              → sync client (for scripts, CLI tools)
  - import redis.asyncio      → async client (for FastAPI, asyncio apps)
"""

# ── Verify installation ──────────────────────────────────────────────────────
def check_redis_installation():
    """Run this to verify Redis is installed and reachable."""
    try:
        import redis
        print(f"✅ redis package version: {redis.__version__}")

        # Try to connect to local Redis
        r = redis.Redis(host="localhost", port=6379, db=0)
        pong = r.ping()
        print(f"✅ Redis server reachable: ping → {pong}")

        # Quick read/write test
        r.set("test:hello", "world", ex=10)  # expires in 10 seconds
        val = r.get("test:hello")
        print(f"✅ Read/write test: set 'test:hello'='world', get → {val}")

        r.close()
        return True
    except ImportError:
        print("❌ redis package not installed. Run: pip install redis[hiredis]")
        return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Is Redis running? Try: brew services start redis")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  4. REDIS DATA TYPES
# ══════════════════════════════════════════════════════════════════════════════
"""
Redis supports 5 core data types. We use STRING for caching.

┌─────────────┬─────────────────────────┬──────────────────────────┐
│ Type        │ Python Equivalent       │ Use Case                 │
├─────────────┼─────────────────────────┼──────────────────────────┤
│ STRING      │ str / bytes             │ Cache JSON predictions   │ ← We use this
│ LIST        │ list                    │ Message queues           │
│ SET         │ set                     │ Unique visitors tracking │
│ HASH        │ dict                    │ User sessions/profiles   │
│ SORTED SET  │ dict with scores       │ Leaderboards, rankings   │
└─────────────┴─────────────────────────┴──────────────────────────┘

For caching predictions, STRING is the right choice because:
  - We store a JSON blob (serialised dict)
  - We need SET with TTL (auto-expiry)
  - We only need GET and SET operations (simple!)
  - No need to query inside the cached data

If we needed to query individual fields:
  HASH would be better — you can GET/SET individual fields:
    HSET prediction:AAPL ticker AAPL prediction UP probability 0.73
    HGET prediction:AAPL prediction  → "UP"

But for our use case, we always read/write the ENTIRE prediction,
so STRING (with JSON serialisation) is simpler and faster.
"""

def demonstrate_redis_data_types():
    """Interactive demo of Redis data types."""
    import json
    import redis

    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

    # ── STRING (what we use for caching) ──────────────────────────────────
    # SET key value [EX seconds]
    # GET key
    prediction = {"ticker": "AAPL", "prediction": "UP", "probability": 0.73}
    r.set("demo:string", json.dumps(prediction), ex=60)
    result = json.loads(r.get("demo:string"))
    print(f"STRING: {result}")
    # Output: {'ticker': 'AAPL', 'prediction': 'UP', 'probability': 0.73}

    # ── SETEX shorthand (SET + EX in one command) ────────────────────────
    # SETEX key seconds value
    r.setex("demo:setex", 60, json.dumps(prediction))
    # Identical to: r.set("demo:setex", json.dumps(prediction), ex=60)

    # ── LIST (not used in our project, but good to know) ──────────────────
    # RPUSH key value [value ...]    ← push to right end
    # LPOP key                       ← pop from left end
    r.delete("demo:list")
    r.rpush("demo:list", "AAPL", "MSFT", "GOOGL")
    item = r.lpop("demo:list")
    print(f"LIST: popped → {item}")  # "AAPL"

    # ── HASH (useful for user sessions later) ─────────────────────────────
    # HSET key field value [field value ...]
    # HGET key field
    # HGETALL key
    r.hset("demo:hash", mapping={"ticker": "AAPL", "direction": "UP"})
    direction = r.hget("demo:hash", "direction")
    print(f"HASH: direction → {direction}")  # "UP"

    # ── SET (unique items, useful for tracking) ───────────────────────────
    # SADD key member [member ...]
    # SMEMBERS key
    r.delete("demo:set")
    r.sadd("demo:set", "AAPL", "MSFT", "AAPL")  # duplicate ignored
    members = r.smembers("demo:set")
    print(f"SET: {members}")  # {'AAPL', 'MSFT'}

    # ── SORTED SET (ranked, useful for leaderboards) ─────────────────────
    # ZADD key score member [score member ...]
    # ZREVRANGE key start stop WITHSCORES
    r.delete("demo:zset")
    r.zadd("demo:zset", {"AAPL": 73.2, "MSFT": 68.5, "NVDA": 81.0})
    top = r.zrevrange("demo:zset", 0, 2, withscores=True)
    print(f"SORTED SET (top stocks by confidence): {top}")
    # [('NVDA', 81.0), ('AAPL', 73.2), ('MSFT', 68.5)]

    # Cleanup
    for key in r.keys("demo:*"):
        r.delete(key)

    r.close()


# ══════════════════════════════════════════════════════════════════════════════
#  5. PYTHON REDIS CLIENT — SYNCHRONOUS
# ══════════════════════════════════════════════════════════════════════════════
"""
The SYNC client is for scripts, CLI tools, and simple programs.
NOT for FastAPI (which is async).

We show it here because it's simpler to learn with.
"""

def sync_redis_example():
    """
    Basic sync Redis usage — run this to understand the API.
    """
    import json
    import redis

    # ── Connect to Redis ──────────────────────────────────────────────────
    # redis.Redis() creates a connection pool automatically.
    # decode_responses=True means GET returns str instead of bytes.
    r = redis.Redis(
        host="localhost",          # Redis server hostname
        port=6379,                 # Default Redis port
        db=0,                      # Redis has 16 databases (0-15)
        decode_responses=True,     # Return str, not bytes
        socket_connect_timeout=2,  # Fail fast if Redis is down
    )

    # ── SET a value with TTL ──────────────────────────────────────────────
    prediction = {
        "ticker": "AAPL",
        "prediction": "UP",
        "probability": 0.73,
        "confidence_pct": 73.0,
    }

    # json.dumps converts dict → string (Redis stores strings only)
    r.setex(
        name="prediction:AAPL:1:0.500",   # key
        time=3600,                          # TTL in seconds (1 hour)
        value=json.dumps(prediction),       # value (must be string)
    )
    print("SET: stored prediction")

    # ── GET the value back ────────────────────────────────────────────────
    cached = r.get("prediction:AAPL:1:0.500")

    if cached is None:
        print("MISS: key not found or expired")
    else:
        data = json.loads(cached)  # string → dict
        print(f"HIT: {data}")

    # ── Check TTL remaining ───────────────────────────────────────────────
    ttl = r.ttl("prediction:AAPL:1:0.500")
    print(f"TTL remaining: {ttl} seconds")
    # Returns: -1 = no expiry, -2 = key doesn't exist, N = seconds left

    # ── Delete a key manually ─────────────────────────────────────────────
    r.delete("prediction:AAPL:1:0.500")
    print("Deleted key")

    # ── Check if key exists ───────────────────────────────────────────────
    exists = r.exists("prediction:AAPL:1:0.500")
    print(f"Exists: {bool(exists)}")  # False (we just deleted it)

    # ── IMPORTANT: Close the connection ───────────────────────────────────
    r.close()


# ══════════════════════════════════════════════════════════════════════════════
#  6. PYTHON REDIS ASYNC CLIENT — FOR FASTAPI
# ══════════════════════════════════════════════════════════════════════════════
"""
FastAPI is async. Using the sync client would block the event loop:

  WRONG (blocks event loop):
    import redis
    r = redis.Redis()
    r.get("key")          ← This blocks ALL other requests

  RIGHT (non-blocking):
    import redis.asyncio as aioredis
    r = aioredis.from_url(...)
    await r.get("key")    ← Event loop is free during network wait

The async client has the EXACT same API, just with 'await' before each call.
"""

import asyncio

async def async_redis_example():
    """
    Async Redis usage — this is what FastAPI uses.
    """
    import json
    import redis.asyncio as aioredis

    # ── Connect ───────────────────────────────────────────────────────────
    # from_url is the recommended way for async connections.
    # It creates an async connection pool under the hood.
    r = aioredis.from_url(
        "redis://localhost:6379/0",   # URL format: redis://host:port/db
        decode_responses=True,
        socket_connect_timeout=2,
    )

    # ── Ping (verify connection) ──────────────────────────────────────────
    pong = await r.ping()
    print(f"PING: {pong}")  # True

    # ── SET with TTL ──────────────────────────────────────────────────────
    prediction = {"ticker": "MSFT", "prediction": "DOWN", "probability": 0.38}

    await r.setex(
        name="prediction:MSFT:1:0.500",
        time=3600,
        value=json.dumps(prediction),
    )

    # ── GET ────────────────────────────────────────────────────────────────
    cached = await r.get("prediction:MSFT:1:0.500")
    if cached:
        data = json.loads(cached)
        print(f"Async GET: {data}")

    # ── TTL ────────────────────────────────────────────────────────────────
    ttl = await r.ttl("prediction:MSFT:1:0.500")
    print(f"TTL: {ttl}s")

    # ── Cleanup ───────────────────────────────────────────────────────────
    await r.delete("prediction:MSFT:1:0.500")
    await r.close()

# To run this: asyncio.run(async_redis_example())


# ══════════════════════════════════════════════════════════════════════════════
#  7. TTL — TIME-TO-LIVE (EXPIRY)
# ══════════════════════════════════════════════════════════════════════════════
"""
TTL = Time-To-Live. After this many seconds, Redis AUTOMATICALLY deletes the key.

Why automatic expiry matters:
─────────────────────────────────────────────────────────────
  Without TTL:
    - Cached predictions never expire
    - User sees yesterday's AAPL prediction with yesterday's price
    - Cache grows forever, eating RAM

  With TTL (our approach):
    - Each prediction expires after 1 hour
    - Next request recomputes with fresh data
    - Cache stays small (only recent predictions)

TTL in our project:
─────────────────────────────────────────────────────────────
  Prediction cache: 3600 seconds (1 hour)
    - During market hours: features change with each new daily bar
    - After hours: features don't change, but 1 hour is still reasonable
    - You can reduce to 900 (15 min) for more freshness during trading

TTL commands:
  SETEX key seconds value        ← SET + auto-expire
  SET key value EX seconds       ← equivalent
  EXPIRE key seconds             ← add TTL to existing key
  TTL key                        ← check remaining time
  PERSIST key                    ← remove TTL (key lives forever)

TTL return values:
  -1  = key exists but has no TTL (no expiry)
  -2  = key does not exist
   N  = key expires in N seconds
"""

def demonstrate_ttl():
    """Show how TTL works step by step."""
    import time
    import redis

    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

    # Set a key with 5-second TTL
    r.setex("demo:ttl", 5, "I will disappear in 5 seconds")

    print(f"t=0: value = {r.get('demo:ttl')}")
    print(f"t=0: TTL   = {r.ttl('demo:ttl')} seconds")

    time.sleep(3)
    print(f"\nt=3: value = {r.get('demo:ttl')}")
    print(f"t=3: TTL   = {r.ttl('demo:ttl')} seconds")

    time.sleep(3)
    print(f"\nt=6: value = {r.get('demo:ttl')}")     # None (expired!)
    print(f"t=6: TTL   = {r.ttl('demo:ttl')}")       # -2 (key gone)

    r.close()


# ══════════════════════════════════════════════════════════════════════════════
#  8. CACHE KEY DESIGN
# ══════════════════════════════════════════════════════════════════════════════
"""
How you name your Redis keys is CRITICAL. Bad keys → cache bugs.

Our key format: "prediction:{ticker}:{horizon}:{threshold:.3f}"

Examples:
  prediction:AAPL:1:0.500
  prediction:MSFT:1:0.500
  prediction:GOOGL:1:0.650

Why include horizon and threshold?
─────────────────────────────────────────────────────────────
  Same ticker + different threshold = different result.

  AAPL at threshold=0.5 → probability 0.62 → prediction: UP
  AAPL at threshold=0.7 → probability 0.62 → prediction: DOWN

  If the key was just "prediction:AAPL", the second request
  would return the cached UP prediction even though the user
  asked for a 70% confidence threshold. WRONG!

Key naming conventions:
─────────────────────────────────────────────────────────────
  Good: prediction:AAPL:1:0.500    ← namespaced, deterministic
  Good: screener:2024-01-15:0.500  ← includes date for screener
  Bad:  AAPL                       ← too generic, will clash
  Bad:  pred_aapl_1_0.5            ← inconsistent separator

The colon (:) is a Redis convention for namespacing:
  prediction:*    = all prediction keys
  screener:*      = all screener keys
  session:*       = all session keys

You can list all prediction keys with:
  redis-cli KEYS "prediction:*"
"""

def cache_key(ticker: str, horizon: int, threshold: float) -> str:
    """
    Generate a deterministic Redis cache key.

    f-string format: :.3f means always 3 decimal places.
    This ensures threshold=0.5 becomes "0.500" every time,
    not sometimes "0.5" and sometimes "0.50".
    Inconsistent formatting would create separate cache entries
    for the same logical request.
    """
    return f"prediction:{ticker}:{horizon}:{threshold:.3f}"


# Test key generation
assert cache_key("AAPL", 1, 0.5) == "prediction:AAPL:1:0.500"
assert cache_key("AAPL", 1, 0.5) == cache_key("AAPL", 1, 0.50)  # same!
assert cache_key("AAPL", 1, 0.5) != cache_key("AAPL", 1, 0.7)   # different!


# ══════════════════════════════════════════════════════════════════════════════
#  9. THE CACHE-ASIDE PATTERN (our approach)
# ══════════════════════════════════════════════════════════════════════════════
"""
There are several caching patterns. We use "Cache-Aside" (also called "Lazy Loading").

Cache-Aside Pattern:
─────────────────────────────────────────────────────────────
  1. Application receives request for AAPL prediction
  2. Check Redis: "Is prediction:AAPL:1:0.500 in cache?"
  3a. CACHE HIT → Return cached JSON immediately (5ms)
  3b. CACHE MISS → Run full ML pipeline (10 seconds)
                  → Store result in Redis with TTL
                  → Return result to client

  ┌──────────┐      GET key      ┌──────────┐
  │  FastAPI  │ ──────────────── │  Redis   │
  │  Server   │ ◄──── HIT ────── │  Cache   │
  └──────────┘   (return JSON)   └──────────┘
       │
       │ MISS (key not found)
       ▼
  ┌──────────────────────────────────────────┐
  │  Run full ML pipeline:                   │
  │   → yfinance fetch                       │
  │   → feature engineering                  │
  │   → XGBoost predict                      │
  │   → SHAP explain                         │
  │                                          │
  │  Store result in Redis:                  │
  │   → SETEX key 3600 json_result           │
  │                                          │
  │  Return result to client                 │
  └──────────────────────────────────────────┘

Why Cache-Aside and not other patterns?
─────────────────────────────────────────────────────────────
  Write-Through: Writes to cache on every data change.
    ❌ Our data changes externally (yfinance) — we can't intercept writes.

  Write-Behind:  Writes to cache first, persists later.
    ❌ We don't have a persistence layer — predictions are ephemeral.

  Refresh-Ahead: Precomputes before expiry.
    ❌ We'd need to predict all stocks on a schedule — wasteful.

  Cache-Aside: Compute on first request, cache for subsequent ones.
    ✅ Perfect for our use case — predictions are requested on demand.
"""

import json

async def cache_aside_example(ticker: str, redis_client, pipeline, feature_cols):
    """
    Pseudocode showing the cache-aside pattern.
    This is exactly what generate_prediction() in api/prediction.py does.
    """

    # Step 1: Build the cache key
    key = f"prediction:{ticker}:1:0.500"

    # Step 2: Try to get from cache
    cached = await redis_client.get(key)

    if cached is not None:
        # CACHE HIT — parse JSON and return immediately
        print(f"Cache HIT for {ticker} — returning cached result")
        return json.loads(cached)

    # Step 3: CACHE MISS — run the expensive computation
    print(f"Cache MISS for {ticker} — running full pipeline")
    result = _expensive_ml_pipeline(ticker, pipeline, feature_cols)

    # Step 4: Store in cache for next time (TTL = 1 hour)
    await redis_client.setex(
        name=key,
        time=3600,
        value=json.dumps(result, default=str),
    )

    # Step 5: Return the fresh result
    return result


def _expensive_ml_pipeline(ticker, pipeline, feature_cols):
    """Placeholder for the actual ML pipeline."""
    return {"ticker": ticker, "prediction": "UP", "probability": 0.73}


# ══════════════════════════════════════════════════════════════════════════════
#  10. INTEGRATION WITH FASTAPI LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════
"""
FastAPI has a "lifespan" pattern for startup/shutdown.
We connect to Redis on startup and close on shutdown.

Our implementation is in api/main.py:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── STARTUP ────────────────────────────────────────────────────
        try:
            import redis.asyncio as aioredis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            redis_client = aioredis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await redis_client.ping()
            app_state["redis_client"] = redis_client
            print("✅ Redis connected")
        except Exception as e:
            print(f"⚠️  Redis unavailable: {e}")
            app_state["redis_client"] = None

        yield   # ← app runs here

        # ── SHUTDOWN ───────────────────────────────────────────────────
        if app_state.get("redis_client"):
            await app_state["redis_client"].close()
            print("Redis connection closed")

Key design decisions:
─────────────────────────────────────────────────────────────

  1. Non-fatal connection:
     If Redis is down, app_state["redis_client"] = None.
     Prediction still works, just without caching.

  2. Environment variable (REDIS_URL):
     Default: redis://localhost:6379/0 (development)
     Production: Set REDIS_URL=redis://your-redis-host:6379/0

  3. socket_connect_timeout=2:
     If Redis doesn't respond in 2 seconds, give up.
     Don't make the server startup wait 30 seconds.

  4. decode_responses=True:
     Redis stores bytes by default. This returns strings.
     Without it: r.get("key") → b'{"ticker": "AAPL"}'  (bytes)
     With it:    r.get("key") → '{"ticker": "AAPL"}'    (string)

  5. Graceful shutdown:
     Always close the Redis connection to free resources.
     The try/except handles the case where Redis died mid-session.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  11. OUR FULL IMPLEMENTATION WALKTHROUGH
# ══════════════════════════════════════════════════════════════════════════════
"""
Here's exactly how Redis flows through our codebase:

File: api/main.py (startup)
─────────────────────────────────────────────────────────────
  1. Create app_state dict with "redis_client": None
  2. In lifespan(), try to connect to Redis
  3. If successful → app_state["redis_client"] = redis_client
  4. If failed → app_state["redis_client"] stays None

File: api/deps.py (dependency injection)
─────────────────────────────────────────────────────────────
  5. get_redis_client() returns app_state.get("redis_client", None)
  6. Endpoints receive redis_client via Depends(get_redis_client)

File: api/routes/predict.py (thin route)
─────────────────────────────────────────────────────────────
  7. predict_stock() receives redis_client from Depends
  8. Passes it to generate_prediction()

File: api/prediction.py (pipeline orchestrator)
─────────────────────────────────────────────────────────────
  9.  generate_prediction() calls _get_cached_prediction()
  10. If cache HIT → return cached dict immediately
  11. If cache MISS → run full pipeline
  12. After pipeline → call _cache_prediction() to store result
  13. Return result (with _cache_hit and _latency_ms metadata)

The critical insight:
─────────────────────────────────────────────────────────────
  Every function that touches Redis checks `if redis_client is None`.
  This means:
    - Redis running    → predictions cached, fast repeat requests
    - Redis not running → predictions computed from scratch, still works
    - No code changes needed to switch between modes

This is the "graceful degradation" pattern.
"""


# ── The actual caching functions (from api/prediction.py) ─────────────────────

async def _get_cached_prediction_annotated(redis_client, cache_key: str):
    """
    Try to load a cached prediction from Redis.

    SYNTAX BREAKDOWN:
    ─────────────────────────────────────────────────────────────
    redis_client.get(key)
      - Returns: str (the cached JSON) or None (cache miss)
      - await: because it's an async I/O operation
      - json.loads(): deserialise string → dict

    WHAT CAN GO WRONG:
    ─────────────────────────────────────────────────────────────
    - Redis connection dropped → ConnectionError
    - Key doesn't exist → returns None (not an error)
    - JSON corrupted → json.JSONDecodeError
    - All caught by try/except → returns None (cache miss)
    """
    if redis_client is None:
        return None        # No Redis = always cache miss

    try:
        cached = await redis_client.get(cache_key)

        if cached is not None:
            return json.loads(cached)    # string → dict

    except Exception:
        pass               # Redis down? Silently degrade.

    return None            # Cache miss


async def _cache_prediction_annotated(
    redis_client, cache_key: str, result: dict, ttl_seconds: int = 3600,
):
    """
    Store prediction in Redis with TTL.

    SYNTAX BREAKDOWN:
    ─────────────────────────────────────────────────────────────
    redis_client.setex(name, time, value)
      - name:  cache key string
      - time:  TTL in seconds (int)
      - value: must be a string → json.dumps(result)

    json.dumps(result, default=str):
      - default=str: handles datetime objects, numpy types, etc.
        Without it: TypeError on datetime or np.float64 values
        With it: those get converted to strings silently

    WHY setex AND NOT set?
    ─────────────────────────────────────────────────────────────
    setex = SET + EX (expire) in ONE atomic command
    set then expire = TWO commands (not atomic, could fail between them)

    Atomic means: either both SET and EXPIRE happen, or neither does.
    If the server crashes between two separate commands, you could
    have a key with no TTL that lives forever (memory leak).
    """
    if redis_client is None:
        return              # No Redis = silently skip caching

    try:
        await redis_client.setex(
            name=cache_key,
            time=ttl_seconds,
            value=json.dumps(result, default=str),
        )
    except Exception:
        pass                # Redis down? Don't crash the prediction.


# ══════════════════════════════════════════════════════════════════════════════
#  12. REDIS CLI — DEBUGGING AND MONITORING
# ══════════════════════════════════════════════════════════════════════════════
"""
Redis CLI is your debugging tool. Always keep a terminal open with it.

Open Redis CLI:
    redis-cli

Common commands:
─────────────────────────────────────────────────────────────

  Check all prediction keys:
    KEYS prediction:*
    # Returns:
    # 1) "prediction:AAPL:1:0.500"
    # 2) "prediction:MSFT:1:0.500"

  Get a cached prediction:
    GET prediction:AAPL:1:0.500
    # Returns: JSON string

  Check TTL remaining:
    TTL prediction:AAPL:1:0.500
    # Returns: 2847 (seconds remaining)

  Delete a specific key (force re-prediction):
    DEL prediction:AAPL:1:0.500
    # Returns: 1 (number of keys deleted)

  Delete ALL prediction keys (clear cache):
    # Use Lua pattern matching:
    redis-cli --scan --pattern "prediction:*" | xargs redis-cli DEL

  Count total keys:
    DBSIZE
    # Returns: 15

  Monitor live commands (see what your app is doing):
    MONITOR
    # Shows real-time feed of all Redis commands

  Memory usage:
    INFO memory
    # Shows used_memory, peak memory, etc.

  Flush entire database (DANGEROUS — deletes everything):
    FLUSHDB          # current database only
    FLUSHALL         # ALL databases (never use in production)

Debugging workflow:
─────────────────────────────────────────────────────────────
  1. Start monitoring:
     Terminal 1: redis-cli MONITOR

  2. Make a prediction:
     Terminal 2: curl "http://localhost:8000/predict/?ticker=AAPL"

  3. Watch Terminal 1 — you'll see:
     "GET" "prediction:AAPL:1:0.500"          ← cache check
     "SETEX" "prediction:AAPL:1:0.500" "3600" ← cache store

  4. Make the same prediction again:
     Terminal 2: curl "http://localhost:8000/predict/?ticker=AAPL"

  5. Watch Terminal 1 — you'll see ONLY:
     "GET" "prediction:AAPL:1:0.500"          ← cache HIT!
     (no SETEX because we got the cached result)
"""


# ══════════════════════════════════════════════════════════════════════════════
#  13. PRODUCTION CONSIDERATIONS
# ══════════════════════════════════════════════════════════════════════════════
"""
For deployment, here are the key Redis production practices:

1. Use a managed Redis service:
─────────────────────────────────────────────────────────────
   AWS ElastiCache, Redis Cloud, or Heroku Redis.
   These handle backups, failover, and scaling.
   Set REDIS_URL environment variable to the cloud URL.

2. Connection pooling:
─────────────────────────────────────────────────────────────
   redis.asyncio.from_url() creates a connection pool automatically.
   Default pool size: 10 connections.
   For high traffic, increase:
     from_url(redis_url, max_connections=50)

3. Memory limits:
─────────────────────────────────────────────────────────────
   Each cached prediction is ~2-3 KB of JSON.
   500 stocks × 3 KB = 1.5 MB. Redis uses very little memory.
   Set a maxmemory limit in redis.conf:
     maxmemory 100mb
     maxmemory-policy allkeys-lru    ← delete least recently used

4. Security:
─────────────────────────────────────────────────────────────
   Set a password: requirepass your_strong_password
   Use TLS in production: rediss://host:6380/0  (note: rediss with double s)
   Bind to localhost only in development: bind 127.0.0.1

5. Monitoring:
─────────────────────────────────────────────────────────────
   Add cache hit/miss metrics:
     cache_hits += 1    if cached
     cache_misses += 1  if not cached
   Log slow queries: slowlog-log-slower-than 10000 (10ms)

6. Docker Compose (for local development):
─────────────────────────────────────────────────────────────
   Create a docker-compose.yml:

   version: '3.8'
   services:
     redis:
       image: redis:7-alpine
       ports:
         - "6379:6379"
       volumes:
         - redis_data:/data
       command: redis-server --maxmemory 100mb --maxmemory-policy allkeys-lru

     backend:
       build: ./backend
       ports:
         - "8000:8000"
       environment:
         - REDIS_URL=redis://redis:6379/0
       depends_on:
         - redis

   volumes:
     redis_data:
"""


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK START — Run These to Learn
# ══════════════════════════════════════════════════════════════════════════════
"""
Step 1: Install and start Redis
    brew install redis
    brew services start redis
    redis-cli ping                    # Should print: PONG

Step 2: Install Python client
    pip install redis[hiredis]

Step 3: Run the examples in this file
    cd backend
    python -c "from docs.redis_guide import check_redis_installation; check_redis_installation()"
    python -c "from docs.redis_guide import sync_redis_example; sync_redis_example()"
    python -c "from docs.redis_guide import demonstrate_ttl; demonstrate_ttl()"
    python -c "from docs.redis_guide import demonstrate_redis_data_types; demonstrate_redis_data_types()"

Step 4: Start the API and test caching
    cd backend && .venv/bin/uvicorn api.main:app --reload --port 8000

    # First request (cache miss — takes 5-10 seconds):
    curl "http://localhost:8000/predict/?ticker=AAPL"

    # Second request (cache hit — takes <10ms):
    curl "http://localhost:8000/predict/?ticker=AAPL"

    # Check the response — look for _cache_hit and _latency_ms fields:
    # "_cache_hit": false, "_latency_ms": 8523   ← first request
    # "_cache_hit": true,  "_latency_ms": 3      ← cached!

Step 5: Monitor Redis activity
    redis-cli MONITOR
    # Then make requests and watch the GET/SETEX commands flow

That's it! Redis is now accelerating your StockSense predictions. 🚀
"""

if __name__ == "__main__":
    print("=" * 70)
    print("  Redis Learning Guide — StockSense AI")
    print("=" * 70)
    print()
    print("Running installation check...")
    check_redis_installation()
    print()
    print("Run individual examples:")
    print("  python -c 'from docs.redis_guide import sync_redis_example; sync_redis_example()'")
    print("  python -c 'from docs.redis_guide import demonstrate_ttl; demonstrate_ttl()'")
    print("  python -c 'from docs.redis_guide import demonstrate_redis_data_types; demonstrate_redis_data_types()'")
