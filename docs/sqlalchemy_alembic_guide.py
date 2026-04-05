"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SQLAlchemy · Alembic · Context Managers · joinedload               ║
║                  COMPLETE LEARNING GUIDE                                    ║
║                  For StockSense AI — Database Layer                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

This guide covers every concept used in api/database.py, explained the same
way as the Redis guide — with syntax breakdowns, diagrams, WHY decisions,
and runnable code examples.

Table of Contents:
─────────────────────────────────────────────────────────────
  PART 1 — SQLAlchemy
    1.  What is SQLAlchemy? (ORM vs Core vs Raw SQL)
    2.  Engine and Connection Pooling
    3.  Session — the Unit of Work
    4.  ORM Models — Tables as Python Classes
    5.  Column Types Syntax Reference
    6.  Relationships (one-to-many, back_populates)
    7.  CRUD Operations (Create Read Update Delete)
    8.  Async SQLAlchemy with FastAPI

  PART 2 — Context Managers
    9.  What is a Context Manager?
    10. with statement — the syntax
    11. yield-based Context Managers (async generators)
    12. Why get_db() uses yield

  PART 3 — joinedload
    13. The N+1 Query Problem (why it exists)
    14. joinedload — solve N+1 in one query
    15. selectinload — alternative for large collections
    16. Syntax comparison and when to use each

  PART 4 — Alembic
    17. What is Alembic?
    18. How Alembic Tracks Schema Versions
    19. Complete Migration Workflow (create → apply → rollback)
    20. autogenerate — Alembic writes migrations for you
    21. Migration Script Anatomy
    22. Common Migration Operations

  PART 5 — PostgreSQL Connection
    23. Installing PostgreSQL (macOS, Ubuntu, Docker)
    24. Creating the stocksense database
    25. .env file setup
    26. Testing the connection
    27. Production connection (cloud databases)
"""

# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — SQLALCHEMY
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  1. WHAT IS SQLALCHEMY?
# ─────────────────────────────────────────────────────────────────────────────
"""
SQLAlchemy is a Python library with TWO layers:

┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: ORM (Object Relational Mapper)                        │
│  Tables ↔ Python classes                                        │
│  Rows   ↔ Python objects                                        │
│  Columns ↔ Class attributes                                     │
│                                                                 │
│  session.add(DBUser(email="a@b.com"))  →  INSERT INTO users ... │
│  result.scalars().all()                →  SELECT * FROM users   │
└───────────────────────────────┬─────────────────────────────────┘
                                │ ORM uses Core underneath
┌───────────────────────────────▼─────────────────────────────────┐
│  LAYER 1: Core (SQL Expression Language)                        │
│  Builds SQL programmatically using Python                        │
│  No classes needed — works directly with tables                 │
│                                                        xx         │
│  insert(users_table).values(email="a@b.com")                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Core uses the engine
┌───────────────────────────────▼─────────────────────────────────┐
│  DRIVER (asyncpg / psycopg2)                                    │
│  Speaks PostgreSQL wire protocol over TCP                       │
│  asyncpg = async, non-blocking (we use this)                    │
│  psycopg2 = sync, blocks event loop (don't use in FastAPI)      │
└─────────────────────────────────────────────────────────────────┘

In StockSense we use the ORM layer because:
  - Models are self-documenting (read the class = understand the table)
  - Relationships are automatic (user.watchlist loads related rows)
  - Migrations are automatic (Alembic compares class vs live schema)
  - Type-safe queries (no string SQL typos)

When would you use Core instead?
  - Bulk INSERT of 100,000 rows (ORM overhead per-object is slow)
  - Complex analytical queries with window functions
  - When you know SQL better than ORM query syntax
"""


# ─────────────────────────────────────────────────────────────────────────────
#  2. ENGINE AND CONNECTION POOLING
# ─────────────────────────────────────────────────────────────────────────────
"""
The ENGINE is the entry point to the database.
It manages a POOL of reusable connections.

Without pooling (naive):
  Request 1: open TCP connection → query → close TCP connection  (100ms overhead)
  Request 2: open TCP connection → query → close TCP connection  (100ms overhead)
  Request 3: open TCP connection → query → close TCP connection  (100ms overhead)

With pooling (what SQLAlchemy does):
  Startup: open 10 TCP connections and keep them open
  Request 1: grab connection from pool → query → return to pool  (<1ms overhead)
  Request 2: grab connection from pool → query → return to pool  (<1ms overhead)
  Request 3: grab connection from pool → query → return to pool  (<1ms overhead)
"""

from sqlalchemy.ext.asyncio import create_async_engine

# SYNTAX BREAKDOWN: create_async_engine(url, **options)
engine = create_async_engine(
    # URL format: "dialect+driver://user:password@host:port/dbname"
    # postgresql+asyncpg  → use PostgreSQL with asyncpg driver
    # postgresql+psycopg2 → use PostgreSQL with psycopg2 driver (sync)
    # sqlite+aiosqlite    → SQLite with async driver (for testing)
    "postgresql+asyncpg://postgres:password@localhost:5432/stocksense",

    pool_pre_ping=True,
    # Before handing a connection to a request, send "SELECT 1"
    # If the DB was restarted, the stale connection fails and is replaced.
    # Cost: one extra round-trip per connection-reuse (~0.1ms)
    # Benefit: no cryptic "server closed connection unexpectedly" errors

    pool_size=10,
    # Permanent connections kept in the pool.
    # Rule of thumb: number of CPU cores × 2

    max_overflow=20,
    # Extra connections allowed under sudden load spikes.
    # If 30 requests arrive simultaneously: 10 from pool + 20 overflow = 30 total.
    # Overflow connections are closed after use (not returned to pool).

    echo=False,
    # echo=True: logs every SQL statement — invaluable for debugging.
    # echo=False: silent (production default)
    # You can also set echo="debug" for even more detail.
)

# The engine is created ONCE at module import time.
# It lives for the entire server lifetime (created at startup, closed at shutdown).
# Never create a new engine per request — that defeats connection pooling.


# ─────────────────────────────────────────────────────────────────────────────
#  3. SESSION — THE UNIT OF WORK
# ─────────────────────────────────────────────────────────────────────────────
"""
If the engine is a pool of TCP connections,
a SESSION is a single conversation with the database.

One session = one transaction = one HTTP request.

The session tracks:
  - What objects you've loaded from the DB ("identity map")
  - What changes you've made (pending INSERTs, UPDATEs, DELETEs)
  - When you commit(), it sends all pending changes in ONE transaction

Session lifecycle:
  ┌──────────────────────────────────────────────────────────┐
  │  HTTP request arrives                                    │
  │         ↓                                               │
  │  get_db() creates a new AsyncSession                    │
  │         ↓                                               │
  │  Route handler uses session (queries, creates objects)  │
  │         ↓                                               │
  │  commit() — writes all changes to PostgreSQL            │
  │         ↓                                               │
  │  close() — returns connection to pool                   │
  │  HTTP response sent                                     │
  └──────────────────────────────────────────────────────────┘

Why one session per request and not one global session?
─────────────────────────────────────────────────────────────
  A global session would mean all requests share the same transaction.
  Request A's uncommitted changes would be visible to Request B.
  This causes data corruption and race conditions.
  One session per request = true isolation.
"""

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    # expire_on_commit=False:
    #   After commit(), SQLAlchemy normally marks all loaded objects as "expired".
    #   The next attribute access on an expired object fires a lazy SELECT.
    #   With async SQLAlchemy, lazy SELECT inside attribute access is BROKEN
    #   (you can't await inside attribute access in Python).
    #   expire_on_commit=False disables this — objects keep their data after commit.
    expire_on_commit=False,
)


# ─────────────────────────────────────────────────────────────────────────────
#  4. ORM MODELS — TABLES AS PYTHON CLASSES
# ─────────────────────────────────────────────────────────────────────────────
"""
Each ORM model class:
  - Inherits from Base
  - Has __tablename__ = the actual PostgreSQL table name
  - Each class attribute = one Column
  - The class itself represents the TABLE
  - An instance of the class represents one ROW

SQLAlchemy ORM Model        ↔   PostgreSQL
─────────────────────────────────────────────────────────────
class DBUser(Base)           ↔   TABLE users
  id = Column(UUID)          ↔   id UUID PRIMARY KEY
  email = Column(String)     ↔   email VARCHAR(255) UNIQUE NOT NULL
  watchlist = relationship   ↔   (no column — just Python link)

Creating a new row:
  user = DBUser(email="alice@example.com", full_name="Alice")
  session.add(user)
  await session.commit()
  # SQL: INSERT INTO users (id, email, full_name, ...) VALUES (...)

Reading a row:
  from sqlalchemy import select
  result = await session.execute(select(DBUser).where(DBUser.email == "alice@example.com"))
  user = result.scalar_one_or_none()
  # SQL: SELECT * FROM users WHERE email = 'alice@example.com'

Updating:
  user.full_name = "Alice Smith"
  await session.commit()
  # SQL: UPDATE users SET full_name = 'Alice Smith' WHERE id = '...'

Deleting:
  await session.delete(user)
  await session.commit()
  # SQL: DELETE FROM users WHERE id = '...'
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Boolean, DateTime, Float, Integer, Text, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
#  5. COLUMN TYPES SYNTAX REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
"""
SQLAlchemy Type         PostgreSQL Type     Python Type
─────────────────────────────────────────────────────────
Integer                 INTEGER             int
Float                   FLOAT / REAL        float
String(n)               VARCHAR(n)          str
Text                    TEXT                str (unlimited)
Boolean                 BOOLEAN             bool
DateTime                TIMESTAMP           datetime
Date                    DATE                date
UUID(as_uuid=True)      UUID                uuid.UUID
JSON                    JSONB               dict / list
ARRAY(String)           TEXT[]              list[str]

Column() options:
─────────────────────────────────────────────────────────
  primary_key=True        → PRIMARY KEY (automatically indexed)
  nullable=False          → NOT NULL constraint
  nullable=True           → allows NULL (default)
  unique=True             → UNIQUE constraint
  index=True              → CREATE INDEX on this column
  default=value           → Python-side default (not SQL DEFAULT)
  server_default="now()"  → SQL-side DEFAULT (set by PostgreSQL)
  onupdate=datetime.utcnow → re-run this every time row is updated

Examples:
"""

class _ColumnExamples(Base):
    """Demonstration class — not a real table."""
    __tablename__ = "_demo_columns"

    # UUID primary key — random, unguessable
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    #            ↑ PostgreSQL-specific UUID type
    #                             ↑ Python uuid.UUID objects (not strings)
    #                                               ↑ call uuid.uuid4() for each new row

    # String with length limit — VARCHAR(255) in PostgreSQL
    email = Column(String(255), unique=True, nullable=False, index=True)
    #                           ↑ UNIQUE    ↑ NOT NULL     ↑ creates an index

    # Text — no length limit (for SHAP explanations, notes, etc.)
    notes = Column(Text, nullable=True)
    #                    ↑ NULL allowed (optional field)

    # Float — REAL (4-byte) or DOUBLE PRECISION (8-byte) in PostgreSQL
    probability = Column(Float, nullable=False)

    # Boolean
    is_active = Column(Boolean, default=True)
    #                           ↑ Python default — sets is_active=True
    #                             when you create a new DBUser()
    #                             without specifying is_active

    # DateTime with automatic timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    #                             ↑ datetime.utcnow is called when row is created

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    #                                                       ↑ called on every UPDATE


# ─────────────────────────────────────────────────────────────────────────────
#  6. RELATIONSHIPS (one-to-many, back_populates)
# ─────────────────────────────────────────────────────────────────────────────
"""
Relationships are Python-level links between ORM models.
No extra columns are created — they work through ForeignKey columns.

One-to-many: One user has many watchlist items.

  user.watchlist    → list of DBWatchlist objects (the "many" side)
  watchlist.user    → the DBUser object (the "one" side)

  ┌──────────┐         ┌───────────────────┐
  │  DBUser  │ 1 ───── ─ ► many DBWatchlist │
  │  id: A   │         │  user_id: A        │
  │          │         │  ticker: AAPL      │
  │          │         ├───────────────────┤
  │          │         │  user_id: A        │
  │          │         │  ticker: MSFT      │
  └──────────┘         └───────────────────┘

back_populates:
  Without back_populates: only one direction works.
  With back_populates:    both directions work and stay synchronised.

  # In DBUser:
  watchlist = relationship("DBWatchlist", back_populates="user", ...)
  #                                        ↑ "user" must match the name in DBWatchlist

  # In DBWatchlist:
  user = relationship("DBUser", back_populates="watchlist")
  #                              ↑ "watchlist" must match the name in DBUser

cascade="all, delete-orphan":
  When you delete a user, automatically delete all their watchlist items.
  Without cascade: you'd get a foreign key constraint error when deleting a user
  because watchlist rows still point to the deleted user's id.

  Options:
    "save-update, merge"       → default (no automatic deletes)
    "all"                      → propagate all operations
    "all, delete-orphan"       → also delete rows that become orphaned
    "delete"                   → only cascade deletes
"""

# Example: loading relationships
async def _relationship_example(session: AsyncSession):
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    # Load a user WITH their watchlist in one query (selectinload)
    result = await session.execute(
        select(DBUser)
        .where(DBUser.email == "alice@example.com")
        .options(selectinload(DBUser.watchlist))
        #         ↑ see Section 14-15 for full explanation of selectinload
    )
    user = result.scalar_one_or_none()

    if user:
        # No extra SQL query needed — watchlist is already loaded
        for item in user.watchlist:
            print(f"  {item.ticker} added {item.added_at.date()}")


# ─────────────────────────────────────────────────────────────────────────────
#  7. CRUD OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────
"""
CRUD = Create, Read, Update, Delete — the four basic database operations.
"""

from sqlalchemy import select


async def _crud_examples(session: AsyncSession, user_id: uuid.UUID):

    # ── CREATE ────────────────────────────────────────────────────────────────
    # Create an instance → add to session → commit
    new_user = DBUser(
        email="bob@example.com",
        full_name="Bob",
        is_active=True,
    )
    session.add(new_user)
    await session.commit()
    # After commit: new_user.id is populated (UUID was generated)
    # SQL: INSERT INTO users (id, email, full_name, ...) VALUES (...)

    # ── READ — get by primary key ─────────────────────────────────────────────
    user = await session.get(DBUser, user_id)
    # session.get is the fastest way to get by PK — uses identity map cache
    # Returns None if not found.

    # ── READ — query with WHERE ───────────────────────────────────────────────
    result = await session.execute(
        select(DBUser)
        .where(DBUser.email == "bob@example.com")
        .where(DBUser.is_active == True)
    )
    user = result.scalar_one_or_none()
    # scalar_one_or_none():  returns None if 0 rows, raises if >1 row
    # scalar_one():          raises if 0 rows or >1 row
    # scalars().all():       returns list of all matching rows

    # ── READ — get multiple rows ──────────────────────────────────────────────
    result = await session.execute(
        select(DBUser)
        .where(DBUser.is_active == True)
        .order_by(DBUser.created_at.desc())
        .limit(10)
        .offset(0)
    )
    users = result.scalars().all()   # list of DBUser objects

    # ── UPDATE ────────────────────────────────────────────────────────────────
    if user:
        user.full_name = "Robert"    # modify the attribute
        await session.commit()       # commit writes it to PostgreSQL
        # SQL: UPDATE users SET full_name = 'Robert', updated_at = now()
        #      WHERE id = '...'

    # ── DELETE ────────────────────────────────────────────────────────────────
    if user:
        await session.delete(user)
        await session.commit()
        # SQL: DELETE FROM users WHERE id = '...'

    # ── BULK INSERT (faster for many rows) ────────────────────────────────────
    from sqlalchemy import insert

    await session.execute(
        insert(DBUser),
        [
            {"email": "c@example.com", "full_name": "Charlie"},
            {"email": "d@example.com", "full_name": "Diana"},
        ],
    )
    await session.commit()
    # SQL: INSERT INTO users (email, full_name) VALUES
    #      ('c@example.com', 'Charlie'), ('d@example.com', 'Diana')


# ─────────────────────────────────────────────────────────────────────────────
#  8. ASYNC SQLALCHEMY WITH FASTAPI
# ─────────────────────────────────────────────────────────────────────────────
"""
The async_engine_from_config and AsyncSession work identically to the sync
versions except every I/O operation is awaited.

Sync vs Async comparison:
─────────────────────────────────────────────────────────────

SYNC (psycopg2 — don't use in FastAPI):
  engine = create_engine("postgresql://...")
  Session = sessionmaker(engine, ...)
  with Session() as db:
      user = db.execute(select(DBUser)).scalar()

ASYNC (asyncpg — what we use):
  engine = create_async_engine("postgresql+asyncpg://...")
  AsyncSession = sessionmaker(engine, class_=AsyncSession, ...)
  async with AsyncSessionLocal() as db:
      result = await db.execute(select(DBUser))
      user = result.scalar()

The key differences:
  - All I/O operations have await in front
  - Use create_async_engine instead of create_engine
  - Use AsyncSession instead of Session
  - The engine URL must include +asyncpg

Why does asyncpg require await everywhere?
  - DB queries involve waiting for network I/O (TCP to PostgreSQL)
  - await tells Python "pause here, let other coroutines run"
  - Without await, you'd block the event loop during the wait
  - This is the same principle as the ThreadPoolExecutor in prediction.py,
    except database I/O is natively async — no thread needed
"""


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — CONTEXT MANAGERS
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  9. WHAT IS A CONTEXT MANAGER?
# ─────────────────────────────────────────────────────────────────────────────
"""
A context manager is any object that:
  1. Does SETUP when you enter a `with` block
  2. Does CLEANUP when you leave the `with` block
     (whether you left normally OR via an exception)

The problem they solve:
  Without context managers, you must manually handle cleanup:

    file = open("data.txt")
    try:
        data = file.read()
    finally:
        file.close()  # you must remember this EVERY time

  With context manager (what open() returns):

    with open("data.txt") as file:
        data = file.read()
    # file is automatically closed here — even if an exception occurred

Classic examples in Python:
  with open(path) as f:           → closes file on exit
  with lock:                      → releases thread lock on exit
  async with AsyncSessionLocal()  → closes DB session on exit
  async with engine.begin()       → commits or rollbacks transaction on exit
"""


# ─────────────────────────────────────────────────────────────────────────────
#  10. WITH STATEMENT — THE SYNTAX
# ─────────────────────────────────────────────────────────────────────────────
"""
The with statement calls two special methods:
  __enter__() → runs on entry, its return value is bound to the `as` variable
  __exit__()  → runs on exit, receives exception info if any

SYNCHRONOUS context manager:

    class ManagedConnection:
        def __enter__(self):
            self.conn = open_database_connection()
            return self.conn        ← this becomes `conn` in `as conn`

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.conn.close()       ← always runs
            return False            ← False = don't suppress exceptions

    with ManagedConnection() as conn:
        conn.query("SELECT 1")
    # conn.close() called here automatically

ASYNC context manager (for async with):

    class AsyncManagedConnection:
        async def __aenter__(self):         # note the 'a' prefix
            self.conn = await connect()
            return self.conn

        async def __aexit__(self, *args):
            await self.conn.close()         # awaitable cleanup

    async with AsyncManagedConnection() as conn:
        await conn.query("SELECT 1")

You'll see both styles throughout our codebase:
  async with AsyncSessionLocal() as session   → database session
  async with engine.begin() as conn           → transaction
"""


# ─────────────────────────────────────────────────────────────────────────────
#  11. YIELD-BASED CONTEXT MANAGERS (async generators)
# ─────────────────────────────────────────────────────────────────────────────
"""
Python provides a shortcut for writing context managers using yield.
This is the pattern used in get_db().

SYNCHRONOUS version (simpler to understand first):

    from contextlib import contextmanager

    @contextmanager
    def managed_connection():
        conn = open_connection()   # SETUP
        try:
            yield conn             # ← PAUSE here, give conn to the with block
        finally:
            conn.close()           # CLEANUP — always runs

    with managed_connection() as conn:
        conn.query("SELECT 1")
    # conn.close() runs here

How yield splits the function:
  Everything BEFORE yield  = __enter__() / setup
  yield <value>            = the value bound to `as`
  Everything AFTER yield   = __exit__() / cleanup

ASYNC version (what get_db() uses):

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def async_managed_session():
        async with AsyncSessionLocal() as session:
            try:
                yield session          # ← PAUSE, give session to the endpoint
                await session.commit() # success: save changes
            except Exception:
                await session.rollback()  # error: undo changes
                raise                     # re-raise the exception
            finally:
                await session.close()     # ALWAYS close

But wait — get_db() doesn't use @asynccontextmanager!
It uses an async generator directly, which FastAPI Depends() treats the same way.
FastAPI knows that async generator functions (with yield) are context managers.
"""

from contextlib import asynccontextmanager


# ─────────────────────────────────────────────────────────────────────────────
#  12. WHY get_db() USES yield (FASTAPI DEPENDENCY LIFECYCLE)
# ─────────────────────────────────────────────────────────────────────────────
"""
This is the EXACT get_db() from api/database.py, annotated:
"""

from typing import AsyncGenerator

async def get_db_annotated() -> AsyncGenerator[AsyncSession, None]:
    """
    AsyncGenerator[AsyncSession, None] type hint means:
      - This is an async generator (uses yield)
      - yield produces AsyncSession objects
      - send() accepts None (we never send values into it)

    FastAPI's Depends() system handles this automatically:
      1. FastAPI sees that get_db is an async generator
      2. FastAPI creates an async iterator from it
      3. FastAPI calls __anext__() to get to the yield (setup)
      4. FastAPI injects the yielded session into the route handler
      5. Route handler runs with the session
      6. FastAPI calls __anext__() again (past the yield) for cleanup

    EXECUTION ORDER:
    ─────────────────────────────────────────────────────────────
      A. FastAPI runs get_db() until yield:
            async with AsyncSessionLocal() as session:   ← create session
                try:
                    [yield happens here — FastAPI injects session]

      B. Route handler executes:
            async def list_predictions(db: AsyncSession = Depends(get_db)):
                result = await db.execute(...)
                return result.scalars().all()

      C. FastAPI continues get_db() after yield:
            [yield happened above]
                    await session.commit()      ← success path
                except Exception:
                    await session.rollback()    ← error path
                    raise
                finally:
                    await session.close()       ← ALWAYS

    The try/except/finally structure:
    ─────────────────────────────────────────────────────────────
      try:
          yield session
          await session.commit()      ← only reached if no exception
      except Exception:
          await session.rollback()    ← undo dirty writes on error
          raise                       ← propagate so FastAPI returns 500
      finally:
          await session.close()       ← close regardless of success/error
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — joinedload AND selectinload
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  13. THE N+1 QUERY PROBLEM
# ─────────────────────────────────────────────────────────────────────────────
"""
The N+1 problem is one of the most common performance bugs in ORM code.

Scenario: Load 10 users and their watchlists.

WRONG approach (N+1 queries):
  users = session.execute(select(DBUser)).scalars().all()
  # Query 1: SELECT * FROM users  → 10 users

  for user in users:
      print(user.watchlist)   # ← TRIGGERS A QUERY FOR EACH USER
      # Query 2:  SELECT * FROM watchlists WHERE user_id = 'user_1_id'
      # Query 3:  SELECT * FROM watchlists WHERE user_id = 'user_2_id'
      # ...
      # Query 11: SELECT * FROM watchlists WHERE user_id = 'user_10_id'

  Total: 1 + 10 = 11 queries. For 1000 users: 1001 queries. 💀

Why does this happen?
  SQLAlchemy's async mode cannot auto-load relationships lazily.
  When you access user.watchlist without pre-loading it,
  SQLAlchemy would need to issue a SELECT — but in async mode,
  you can't await inside attribute access.
  Result: MissingGreenlet error or empty list.

The fix: always explicitly load relationships with joinedload or selectinload.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  14. joinedload — SOLVE N+1 IN ONE QUERY
# ─────────────────────────────────────────────────────────────────────────────
"""
joinedload tells SQLAlchemy: "fetch the related rows in the SAME query
using a SQL JOIN."

SYNTAX:
  from sqlalchemy.orm import joinedload

  result = await session.execute(
      select(DBUser)
      .options(joinedload(DBUser.watchlist))
  )
  users = result.unique().scalars().all()
  #              ↑ .unique() is REQUIRED with joinedload
  #                (JOIN produces duplicate parent rows)

SQL GENERATED:
  SELECT users.*, watchlists.*
  FROM users
  LEFT OUTER JOIN watchlists ON watchlists.user_id = users.id
  WHERE ...

One query. Zero N+1. ✅

.unique() explained:
─────────────────────────────────────────────────────────────
  The JOIN multiplies rows. One user with 3 watchlist items
  appears 3 times in the result set (one row per watchlist item).
  SQLAlchemy's unique() de-duplicates the parent objects (users)
  and collects all their children (watchlist items) into the
  watchlist list correctly.

  Without .unique():
    users[0], users[0], users[0], users[1], users[1]  ← duplicates!
  With .unique():
    users[0] (with 3 watchlist items), users[1] (with 2 items)  ✅

When to use joinedload:
  ✅ Loading ONE parent with ONE or FEW children (e.g. one user's watchlist)
  ✅ The child list is small (< 100 items)
  ✅ The parent query has a WHERE clause limiting results
  ❌ Loading MANY parents (the JOIN explodes row count)
  ❌ The child list is large (hundreds of items per parent)
"""

from sqlalchemy.orm import joinedload

async def _joinedload_example(session: AsyncSession):
    # Load one user WITH their watchlist (2 tables, 1 query)
    result = await session.execute(
        select(DBUser)
        .where(DBUser.email == "alice@example.com")
        .options(joinedload(DBUser.watchlist))
    )
    user = result.unique().scalar_one_or_none()

    if user:
        # watchlist is already loaded — no extra query
        for item in user.watchlist:
            print(f"{item.ticker} added {item.added_at.date()}")


# ─────────────────────────────────────────────────────────────────────────────
#  15. selectinload — ALTERNATIVE FOR LARGE COLLECTIONS
# ─────────────────────────────────────────────────────────────────────────────
"""
selectinload uses a SECOND query with IN (...) instead of JOIN.

SQL GENERATED:
  Query 1: SELECT * FROM users WHERE ...
  Query 2: SELECT * FROM watchlists WHERE user_id IN ('id1', 'id2', 'id3', ...)

Two queries total — but each is a simple flat query without JOINs.
No duplicate rows, no need for .unique().

When to use selectinload:
  ✅ Loading MANY parents at once (list endpoints, pagination)
  ✅ The child list is large (hundreds of items per parent)
  ✅ Multiple relationships to load (avoids huge cartesian JOIN)
  ❌ Loading a single parent (joinedload would be 1 query vs 2)

Summary table:
─────────────────────────────────────────────────────────────
                    joinedload          selectinload
  N queries:        1                   2
  Duplicates:       yes (need .unique)  no
  Best for:         single parent       many parents
  SQL:              LEFT JOIN           WHERE id IN (...)
"""

from sqlalchemy.orm import selectinload

async def _selectinload_example(session: AsyncSession):
    # Load ALL active users WITH their watchlists — selectinload is better here
    result = await session.execute(
        select(DBUser)
        .where(DBUser.is_active == True)
        .options(selectinload(DBUser.watchlist))  # no .unique() needed
    )
    users = result.scalars().all()

    for user in users:
        print(f"{user.email}: {len(user.watchlist)} tickers")


# ── Chaining multiple relationships ──────────────────────────────────────────
async def _multiple_relationships(session: AsyncSession):
    # Load user + watchlist + portfolios in 3 queries total
    result = await session.execute(
        select(DBUser)
        .where(DBUser.email == "alice@example.com")
        .options(
            selectinload(DBUser.watchlist),
            selectinload(DBUser.portfolios),
        )
    )
    user = result.scalar_one_or_none()


# ══════════════════════════════════════════════════════════════════════════════
#  PART 4 — ALEMBIC
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  17. WHAT IS ALEMBIC?
# ─────────────────────────────────────────────────────────────────────────────
"""
Alembic is a database migration tool for SQLAlchemy.

The problem it solves:
  You deploy StockSense v1 with a users table.
  Two weeks later, you need to add a phone_number column to users.

  Without Alembic:
    How do you add the column to the LIVE database without losing data?
    How do you roll back if the deploy fails?
    How does your teammate's database get the new column?
    How do you track what the schema looked like last month?

  With Alembic:
    alembic revision --autogenerate -m "add phone_number to users"
    ← Alembic reads your ORM models, compares to live DB, writes a migration script

    alembic upgrade head
    ← Applies the migration: ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)

    alembic downgrade -1
    ← Rolls back: ALTER TABLE users DROP COLUMN phone_number

Alembic file structure (after alembic init alembic):
─────────────────────────────────────────────────────────────
  backend/
  ├── alembic/
  │   ├── env.py           ← configuration (already set up)
  │   ├── script.py.mako   ← template for migration files
  │   └── versions/        ← one .py file per migration
  │       ├── abc123_initial_schema.py
  │       ├── def456_add_phone_number.py
  │       └── ghi789_add_backtest_verdict.py
  └── alembic.ini          ← config file (DB URL)
"""


# ─────────────────────────────────────────────────────────────────────────────
#  18. HOW ALEMBIC TRACKS SCHEMA VERSIONS
# ─────────────────────────────────────────────────────────────────────────────
"""
Alembic creates a special table in your PostgreSQL database:

  TABLE alembic_version:
    version_num VARCHAR(32)

  This table stores the ID of the LAST APPLIED migration.
  When you run alembic upgrade head, Alembic:
    1. Reads alembic_version to find the current version
    2. Finds all migration files with a higher version
    3. Runs them in order
    4. Updates alembic_version to the latest

  When you run alembic downgrade -1:
    1. Reads alembic_version for current version
    2. Runs the downgrade() function of that version
    3. Updates alembic_version to the previous version

Migration chain:
  None → abc123 → def456 → ghi789 (head)
  Each migration has:
    revision = "def456"      ← this migration's ID
    down_revision = "abc123" ← the previous migration
  This forms a linked list — Alembic knows the order.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  19. COMPLETE MIGRATION WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────
"""
Run ALL commands from the backend/ directory.

── INITIAL SETUP (done once) ────────────────────────────────────────────────
  cd backend

  # Create the database (see Part 5 for PostgreSQL setup)
  createdb stocksense

  # Create the first migration from your current models
  .venv/bin/alembic revision --autogenerate -m "initial schema"
  # Creates: alembic/versions/xxxx_initial_schema.py

  # Apply it
  .venv/bin/alembic upgrade head
  # Creates all tables in PostgreSQL

── DAILY WORKFLOW (adding/changing models) ───────────────────────────────────

  STEP 1: Edit your ORM model (api/database.py)
    # e.g. add a column:
    class DBUser(Base):
        ...
        phone_number = Column(String(20), nullable=True)  # ← NEW

  STEP 2: Generate a migration
    .venv/bin/alembic revision --autogenerate -m "add phone_number to users"
    # Creates: alembic/versions/xxxx_add_phone_number_to_users.py
    # ALWAYS review this file before applying!

  STEP 3: Review the generated migration
    cat alembic/versions/xxxx_add_phone_number_to_users.py
    # Should show:
    #   def upgrade():
    #       op.add_column('users', sa.Column('phone_number', sa.String(20)))
    #   def downgrade():
    #       op.drop_column('users', 'phone_number')

  STEP 4: Apply the migration
    .venv/bin/alembic upgrade head
    # ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)

── ROLLBACK (if something goes wrong) ────────────────────────────────────────

  Roll back one migration:
    .venv/bin/alembic downgrade -1

  Roll back to a specific version:
    .venv/bin/alembic downgrade abc123

  Roll back everything (back to empty database):
    .venv/bin/alembic downgrade base

── CHECK STATUS ──────────────────────────────────────────────────────────────

  Show current version:
    .venv/bin/alembic current

  Show all migrations:
    .venv/bin/alembic history --verbose

  Show pending migrations:
    .venv/bin/alembic heads
"""


# ─────────────────────────────────────────────────────────────────────────────
#  21. MIGRATION SCRIPT ANATOMY
# ─────────────────────────────────────────────────────────────────────────────
"""
Here's a real Alembic migration file with every part explained:
"""

MIGRATION_FILE_EXAMPLE = '''
"""add phone_number to users

Revision ID: a1b2c3d4e5f6
Revises: f6e5d4c3b2a1       ← the previous migration (down_revision)
Create Date: 2026-04-05 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# REQUIRED: Alembic uses these to build the chain
revision = "a1b2c3d4e5f6"
down_revision = "f6e5d4c3b2a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    What to DO when migrating forward (alembic upgrade head).
    Always make this reversible — write downgrade() at the same time.
    """
    # ADD a column:
    op.add_column(
        "users",
        sa.Column("phone_number", sa.String(20), nullable=True)
    )

    # ADD an index:
    op.create_index(
        "ix_users_phone_number",   # index name
        "users",                   # table name
        ["phone_number"],          # columns to index
    )

    # Other common operations:
    # op.create_table(...)         ← create new table
    # op.drop_table("old_table")   ← drop a table
    # op.rename_table("a", "b")    ← rename table
    # op.alter_column(...)         ← change column type/nullable
    # op.add_column(...)           ← add new column
    # op.drop_column(...)          ← remove column
    # op.create_unique_constraint(...)  ← add UNIQUE constraint
    # op.drop_constraint(...)      ← remove constraint
    # op.execute("SQL string")     ← run raw SQL (for data migrations)


def downgrade() -> None:
    """
    What to UNDO when rolling back (alembic downgrade -1).
    Must be the exact inverse of upgrade().
    """
    op.drop_index("ix_users_phone_number", table_name="users")
    op.drop_column("users", "phone_number")
'''


# ─────────────────────────────────────────────────────────────────────────────
#  22. COMMON MIGRATION OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────
"""
Quick reference for the most common Alembic migration operations:

ADDING A COLUMN (nullable — safe, no data loss):
  upgrade:   op.add_column('table', sa.Column('col', sa.String(50), nullable=True))
  downgrade: op.drop_column('table', 'col')

ADDING A NON-NULLABLE COLUMN (requires a default or data migration):
  upgrade:
    op.add_column('table', sa.Column('col', sa.Integer, nullable=True))
    op.execute("UPDATE table SET col = 0")      # fill existing rows
    op.alter_column('table', 'col', nullable=False)
  downgrade: op.drop_column('table', 'col')

RENAMING A COLUMN:
  upgrade:   op.alter_column('table', 'old_name', new_column_name='new_name')
  downgrade: op.alter_column('table', 'new_name', new_column_name='old_name')

CREATING AN INDEX:
  upgrade:   op.create_index('ix_table_col', 'table', ['col'])
  downgrade: op.drop_index('ix_table_col', table_name='table')

ADDING A FOREIGN KEY:
  upgrade:
    op.add_column('orders', sa.Column('user_id', UUID, nullable=True))
    op.create_foreign_key('fk_orders_users', 'orders', 'users', ['user_id'], ['id'])
  downgrade:
    op.drop_constraint('fk_orders_users', 'orders', type_='foreignkey')
    op.drop_column('orders', 'user_id')
"""


# ══════════════════════════════════════════════════════════════════════════════
#  PART 5 — POSTGRESQL CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  23. INSTALLING POSTGRESQL
# ─────────────────────────────────────────────────────────────────────────────
"""
── macOS (Homebrew — recommended) ────────────────────────────────────────────

  brew install postgresql@16
  brew services start postgresql@16

  # Add to PATH (add this to your ~/.zshrc):
  export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

  # Verify:
  psql --version              # should print: psql (PostgreSQL) 16.x
  pg_isready                  # should print: /tmp:5432 - accepting connections


── Ubuntu / Debian ───────────────────────────────────────────────────────────

  sudo apt update
  sudo apt install postgresql postgresql-contrib
  sudo systemctl start postgresql
  sudo systemctl enable postgresql

  # Switch to postgres user:
  sudo -u postgres psql


── Docker (recommended for consistent dev environments) ──────────────────────

  # Run PostgreSQL in a container:
  docker run -d \\
    --name stocksense-db \\
    -e POSTGRES_USER=postgres \\
    -e POSTGRES_PASSWORD=password \\
    -e POSTGRES_DB=stocksense \\
    -p 5432:5432 \\
    -v pgdata:/var/lib/postgresql/data \\
    postgres:16-alpine

  # Stop/Start:
  docker stop stocksense-db
  docker start stocksense-db

  # Connect to verify:
  docker exec -it stocksense-db psql -U postgres -d stocksense
"""


# ─────────────────────────────────────────────────────────────────────────────
#  24. CREATING THE stocksense DATABASE
# ─────────────────────────────────────────────────────────────────────────────
"""
Option A — via psql CLI:
  # Connect as superuser (default postgres user on macOS via brew):
  psql postgres

  # Inside psql:
  CREATE USER stocksense_user WITH PASSWORD 'your_password';
  CREATE DATABASE stocksense OWNER stocksense_user;
  GRANT ALL PRIVILEGES ON DATABASE stocksense TO stocksense_user;
  \\q

Option B — single command (if postgres superuser):
  createdb stocksense

Option C — Docker (database already created on container creation):
  (The -e POSTGRES_DB=stocksense flag creates it automatically)

Verify:
  psql postgresql://postgres:password@localhost:5432/stocksense
  # Should connect and show a prompt: stocksense=#
"""


# ─────────────────────────────────────────────────────────────────────────────
#  25. .ENV FILE SETUP
# ─────────────────────────────────────────────────────────────────────────────
"""
Create backend/.env (already in .gitignore — never commit this file):

    # PostgreSQL
    DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/stocksense

    # Redis
    REDIS_URL=redis://localhost:6379/0

    # NewsAPI (for FinBERT sentiment)
    NEWS_API_KEY=your_newsapi_key_here

    # JWT Secret (for user authentication later)
    SECRET_KEY=your_random_secret_key_here

The DATABASE_URL format:
  postgresql+asyncpg://  ← driver (asyncpg for async FastAPI)
  postgres:password      ← username:password
  @localhost:5432        ← host:port
  /stocksense            ← database name

Loading .env in Python (python-dotenv):
  from dotenv import load_dotenv
  load_dotenv()              ← loads .env file into os.environ
  os.getenv("DATABASE_URL")  ← reads it

This is already done in api/main.py's import section.

NEVER hardcode passwords in source code.
NEVER commit .env to git.
Add to .gitignore:
  .env
  .env.*
  !.env.example
"""

DOT_ENV_EXAMPLE = """
# backend/.env.example  ← COMMIT THIS (no real passwords)
# Copy to .env and fill in real values

DATABASE_URL=postgresql+asyncpg://postgres:CHANGE_ME@localhost:5432/stocksense
REDIS_URL=redis://localhost:6379/0
NEWS_API_KEY=CHANGE_ME
SECRET_KEY=CHANGE_ME_use_openssl_rand_hex_32
"""


# ─────────────────────────────────────────────────────────────────────────────
#  26. TESTING THE CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

async def test_db_connection():
    """
    Run this to verify your PostgreSQL connection before starting the server.
    cd backend && .venv/bin/python -c "
    import asyncio
    from docs.sqlalchemy_alembic_guide import test_db_connection
    asyncio.run(test_db_connection())
    "
    """
    import os
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text

    url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:password@localhost:5432/stocksense"
    )

    print(f"Connecting to: {url.split('@')[1]}")  # hide password in log

    try:
        engine = create_async_engine(url, echo=False)
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Connected to PostgreSQL: {version}")

        await engine.dispose()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Is PostgreSQL running?  pg_isready  or  brew services list")
        print("  2. Does the database exist?  psql postgres -c '\\l'")
        print("  3. Is the password correct?  Try: psql postgresql://postgres:password@localhost/stocksense")
        print("  4. Is the port right? Default is 5432")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  27. PRODUCTION CONNECTION (cloud databases)
# ─────────────────────────────────────────────────────────────────────────────
"""
For production, use a managed PostgreSQL service:

── Railway (easiest, free tier available) ────────────────────────────────────
  1. Create account at railway.app
  2. New Project → Add PostgreSQL
  3. Copy "Connection URL" from the Connect tab
  4. Set as environment variable: DATABASE_URL=<copied URL>
  Note: Railway gives you a postgresql:// URL — add +asyncpg:
    postgresql://... → postgresql+asyncpg://...

── Supabase (free tier, also has auth and storage) ──────────────────────────
  1. Create project at supabase.com
  2. Settings → Database → Connection String → URI
  3. Replace [YOUR-PASSWORD] with your project password
  4. Same +asyncpg swap needed

── AWS RDS (for serious production) ─────────────────────────────────────────
  1. Create PostgreSQL RDS instance in your region
  2. Set VPC security group to allow port 5432 from your app server
  3. Use the endpoint from the RDS console
  URL: postgresql+asyncpg://user:pass@your-rds-endpoint.rds.amazonaws.com:5432/stocksense

── Render (good free tier, auto-deploys from GitHub) ─────────────────────────
  1. New → PostgreSQL on render.com
  2. Copy "Internal Database URL" (faster if your app is also on Render)
  3. Same +asyncpg swap

Connection string security for production:
  NEVER hardcode credentials in your code or docker-compose.
  Use:
    - Environment variables set in your hosting platform's dashboard
    - AWS Secrets Manager / GCP Secret Manager for enterprise setups
    - Docker secrets for docker swarm
"""


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK START COMMANDS
# ══════════════════════════════════════════════════════════════════════════════
"""
Copy-paste this to get fully set up:

── 1. Install PostgreSQL ─────────────────────────────────────────────────────
  brew install postgresql@16
  brew services start postgresql@16

── 2. Create database ────────────────────────────────────────────────────────
  createdb stocksense

── 3. Create .env file ───────────────────────────────────────────────────────
  cat > backend/.env << 'EOF'
  DATABASE_URL=postgresql+asyncpg://postgres@localhost:5432/stocksense
  REDIS_URL=redis://localhost:6379/0
  EOF

── 4. Install Python packages (already done via requirements.txt) ───────────
  cd backend && .venv/bin/pip install sqlalchemy asyncpg alembic

── 5. Test connection ────────────────────────────────────────────────────────
  cd backend && .venv/bin/python -c "
  import asyncio, os, sys
  sys.path.insert(0, '.')
  from dotenv import load_dotenv; load_dotenv()
  from docs.sqlalchemy_alembic_guide import test_db_connection
  asyncio.run(test_db_connection())
  "

── 6. Create and apply initial migration ────────────────────────────────────
  cd backend
  .venv/bin/alembic revision --autogenerate -m "initial schema"
  .venv/bin/alembic upgrade head

── 7. Start the server ───────────────────────────────────────────────────────
  .venv/bin/uvicorn api.main:app --reload --port 8000
  # Watch the console — should print "✅ Database tables ready"

── 8. Verify tables were created ────────────────────────────────────────────
  psql stocksense -c "\\dt"
  # Should show: users, predictions, watchlists, portfolios, backtest_results, alembic_version
"""

if __name__ == "__main__":
    import asyncio
    print("=" * 70)
    print("  SQLAlchemy / Alembic / Context Managers Guide — StockSense AI")
    print("=" * 70)
    print()
    print("Testing database connection...")
    asyncio.run(test_db_connection())
