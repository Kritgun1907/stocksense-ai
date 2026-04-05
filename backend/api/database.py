"""
StockSense AI — api/database.py
=================================
Database connection, session management, and ORM models.

This file owns:
  - SQLAlchemy engine and session factory
  - All ORM model definitions (tables as Python classes)
  - Database dependency injection for FastAPI routes
  - Alembic base for migrations

It does NOT own:
  - Business logic           → api/crud.py  (reads/writes to these models)
  - Route handlers           → api/routes/  (calls crud functions)
  - Authentication logic     → api/auth.py  (uses DBUser model from here)

Why asyncpg over psycopg2?
─────────────────────────────────────────────────────────────
  psycopg2 is synchronous — calling it in an async FastAPI handler
  blocks the event loop during database I/O. asyncpg is a native
  async PostgreSQL driver — database queries yield control to the
  event loop while waiting, allowing other requests to be processed
  simultaneously. For a web API, this is critical for throughput.

Why UUID primary keys over integer auto-increment?
─────────────────────────────────────────────────────────────
  Integer IDs are sequential and predictable — users can enumerate
  records by incrementing the ID. UUID primary keys are random —
  you cannot guess another user's ID. This matters for security
  when IDs appear in URLs (e.g. /portfolio/[id]).
  UUIDs are also safe to generate client-side without a database round-trip.

Why store predictions permanently?
─────────────────────────────────────────────────────────────
  Storing every prediction enables:
    1. Model accuracy tracking over time (was_correct field)
    2. Historical performance charts on the stock page
    3. Regulatory audit trail (financial apps need this)
    4. Training data for future model versions
  Redis cache only stores the latest prediction — permanent storage
  provides the historical dimension.
"""

import os
import uuid
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker


# ══════════════════════════════════════════════════════════════════════════════
#  CONNECTION SETUP
# ══════════════════════════════════════════════════════════════════════════════

# DATABASE_URL format:
#   postgresql+asyncpg://<user>:<password>@<host>:<port>/<dbname>
#
#   +asyncpg tells SQLAlchemy to use the asyncpg driver (not psycopg2).
#   Without +asyncpg: "postgresql://..."  → uses psycopg2 (sync, blocks event loop)
#   With +asyncpg:    "postgresql+asyncpg://..." → async, non-blocking ✅
#
# In development: DATABASE_URL falls back to the default localhost string.
# In production:  set the DATABASE_URL environment variable in .env or
#                 your hosting provider's config panel.

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/stocksense"
)

# ── Create the async engine ───────────────────────────────────────────────────
# The engine is the LOW-LEVEL connection to PostgreSQL.
# It manages a pool of connections so every request doesn't open/close
# a brand new TCP connection to the database (expensive).
#
# pool_pre_ping=True:
#   Before using a pooled connection, send a tiny "SELECT 1" ping.
#   If the DB was restarted while the connection sat idle in the pool,
#   the ping detects the dead connection and creates a fresh one.
#   Without this: cryptic "server closed the connection unexpectedly" errors.
#
# pool_size=10:
#   Keep 10 connections open permanently. Requests grab one, use it, return it.
#
# max_overflow=20:
#   Under heavy load (>10 simultaneous queries), allow up to 20 extra
#   temporary connections. They are closed when no longer needed.
#
# echo=False:
#   Set to True during debugging to print every SQL statement.
#   e.g. "SELECT predictions.id FROM predictions WHERE predictions.ticker = 'AAPL'"

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

# ── Session factory ───────────────────────────────────────────────────────────
# A "session" is a unit of work — a conversation with the database.
# sessionmaker() creates a FACTORY that produces AsyncSession objects.
# We never call AsyncSession() directly; always use this factory.
#
# expire_on_commit=False:
#   After commit(), SQLAlchemy normally "expires" all objects,
#   meaning accessing any attribute causes a lazy SQL SELECT.
#   With async, lazy loads fail (can't await inside attribute access).
#   expire_on_commit=False keeps objects usable after commit.

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ── ORM base class ────────────────────────────────────────────────────────────
# All ORM models (tables) inherit from Base.
# Base.metadata holds the schema — used by create_all() and Alembic.

Base = declarative_base()


# ══════════════════════════════════════════════════════════════════════════════
#  ORM MODELS
#  Each class = one table in PostgreSQL.
#  Each class attribute = one column.
#  SQLAlchemy maps Python operations to SQL automatically.
# ══════════════════════════════════════════════════════════════════════════════

class DBUser(Base):
    """
    User account table.

    Column decisions explained:
    - hashed_password is nullable because OAuth users (Google, GitHub login)
      authenticate via their provider — they never set a password here.
    - is_verified: set to True after email confirmation link is clicked.
    - cascade="all, delete-orphan": when a user is deleted, SQLAlchemy
      automatically deletes all their watchlist entries and portfolios.
      Without cascade, deleting a user would leave orphaned rows.
    """
    __tablename__ = "users"

    id              = Column(UUID(as_uuid=True), primary_key=True,
                             default=uuid.uuid4)
    email           = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=True)
    full_name       = Column(String(255), nullable=True)
    is_active       = Column(Boolean, default=True)
    is_verified     = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow,
                             onupdate=datetime.utcnow)

    # Relationships — these are Python-level only, not extra columns.
    # SQLAlchemy uses the ForeignKey in the child table to find related rows.
    # back_populates="user" means DBWatchlist.user points back to DBUser.watchlist.
    watchlist  = relationship("DBWatchlist", back_populates="user",
                              cascade="all, delete-orphan")
    portfolios = relationship("DBPortfolio", back_populates="user",
                              cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"


class DBPrediction(Base):
    """
    One row per ML prediction generated.
    was_correct and actual_direction are NULL at creation time.
    A nightly background job resolves them after the horizon passes.

    UniqueConstraint:
      One prediction per ticker per horizon per day.
      The nightly job can re-run safely — ON CONFLICT DO NOTHING.
    """
    __tablename__ = "predictions"

    id                   = Column(UUID(as_uuid=True), primary_key=True,
                                  default=uuid.uuid4)
    ticker               = Column(String(20),  nullable=False, index=True)
    prediction           = Column(String(10),  nullable=False)   # "UP" or "DOWN"
    probability          = Column(Float,        nullable=False)
    confidence           = Column(Float,        nullable=False)   # 0–100
    horizon_days         = Column(Integer,      default=1)
    threshold_used       = Column(Float,        default=0.5)
    n_features_used      = Column(Integer,      nullable=True)
    model_version        = Column(String(50),   nullable=True)
    generated_at         = Column(DateTime,     default=datetime.utcnow, index=True)
    price_at_prediction  = Column(Float,        nullable=True)

    # Resolved N days later by background job
    actual_direction     = Column(String(10),  nullable=True)
    was_correct          = Column(Boolean,     nullable=True)
    price_at_resolution  = Column(Float,       nullable=True)
    resolved_at          = Column(DateTime,    nullable=True)

    explanation_json     = Column(Text,        nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "ticker", "horizon_days",
            name="uq_prediction_ticker_horizon",
        ),
    )

    def __repr__(self):
        return (
            f"<Prediction {self.ticker} {self.prediction} "
            f"{self.confidence:.0f}% @ {self.generated_at.date()}>"
        )


class DBWatchlist(Base):
    """
    Stocks a user is watching. One row per user-ticker pair.

    UniqueConstraint on (user_id, ticker) prevents duplicates —
    a user can't add AAPL to their watchlist twice.
    """
    __tablename__ = "watchlists"

    id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id  = Column(UUID(as_uuid=True), ForeignKey("users.id"),
                      nullable=False, index=True)
    ticker   = Column(String(20),  nullable=False)
    added_at = Column(DateTime,    default=datetime.utcnow)
    notes    = Column(Text,        nullable=True)

    user = relationship("DBUser", back_populates="watchlist")

    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_watchlist_user_ticker"),
    )

    def __repr__(self):
        return f"<Watchlist user={self.user_id} ticker={self.ticker}>"


class DBPortfolio(Base):
    """
    Paper trading position. One row per open/closed position.
    sell_price and sold_at are NULL while the position is open.
    """
    __tablename__ = "portfolios"

    id            = Column(UUID(as_uuid=True), primary_key=True,
                           default=uuid.uuid4)
    user_id       = Column(UUID(as_uuid=True), ForeignKey("users.id"),
                           nullable=False, index=True)
    ticker        = Column(String(20), nullable=False)
    shares        = Column(Float,      nullable=False)
    avg_buy_price = Column(Float,      nullable=False)
    bought_at     = Column(DateTime,   default=datetime.utcnow)
    is_open       = Column(Boolean,    default=True, index=True)
    sell_price    = Column(Float,      nullable=True)
    sold_at       = Column(DateTime,   nullable=True)
    notes         = Column(Text,       nullable=True)

    user = relationship("DBUser", back_populates="portfolios")

    @property
    def total_cost(self) -> float:
        """Total amount spent buying this position."""
        return self.shares * self.avg_buy_price

    @property
    def pnl(self) -> float | None:
        """Realised P&L — only available after position is closed."""
        if self.sell_price is None:
            return None
        return (self.sell_price - self.avg_buy_price) * self.shares

    def __repr__(self):
        status = "open" if self.is_open else "closed"
        return f"<Portfolio {self.ticker} {self.shares}sh @ {self.avg_buy_price} [{status}]>"


class DBBacktestResult(Base):
    """
    Cached backtest metrics per ticker/config combination.
    Recomputed nightly — fresh results replace stale ones.

    UniqueConstraint on (ticker, config_name) ensures:
      - Only one row per ticker per config (e.g. "default", "aggressive")
      - ON CONFLICT → UPDATE replaces the old result cleanly
    """
    __tablename__ = "backtest_results"

    id                = Column(UUID(as_uuid=True), primary_key=True,
                               default=uuid.uuid4)
    ticker            = Column(String(20),  nullable=False, index=True)
    period_start      = Column(DateTime,    nullable=False)
    period_end        = Column(DateTime,    nullable=False)
    sharpe_ratio      = Column(Float,       nullable=True)
    sortino_ratio     = Column(Float,       nullable=True)
    calmar_ratio      = Column(Float,       nullable=True)
    max_drawdown      = Column(Float,       nullable=True)
    win_rate          = Column(Float,       nullable=True)
    total_return      = Column(Float,       nullable=True)
    annualised_return = Column(Float,       nullable=True)
    n_trades          = Column(Integer,     nullable=True)
    beats_benchmark   = Column(Boolean,     nullable=True)
    config_name       = Column(String(50),  default="default")
    computed_at       = Column(DateTime,    default=datetime.utcnow)
    verdict           = Column(Text,        nullable=True)

    __table_args__ = (
        UniqueConstraint("ticker", "config_name",
                         name="uq_backtest_ticker_config"),
    )

    def __repr__(self):
        return (
            f"<Backtest {self.ticker} config={self.config_name} "
            f"sharpe={self.sharpe_ratio:.2f}>"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

async def init_db() -> None:
    """
    Create all tables if they don't exist.
    Called once at FastAPI startup (inside lifespan).

    Why create_all instead of Alembic here?
    ─────────────────────────────────────────────────────────────
    create_all() works perfectly while you're still designing the schema.
    It creates missing tables but never modifies existing ones.
    Once you have production data, switch to:
      alembic upgrade head
    because Alembic handles adding/removing columns safely.
    See docs/sqlalchemy_alembic_guide.py for full migration workflow.
    """
    async with engine.begin() as conn:
        # engine.begin() returns an async context manager
        # conn.run_sync() runs the synchronous create_all in a thread
        # (Base.metadata.create_all is synchronous SQLAlchemy Core)
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """
    Drop all tables. ONLY for tests — never in production.
    Tests call this in teardown to reset the database state.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI DEPENDENCY — get_db()
# ══════════════════════════════════════════════════════════════════════════════

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides one database session per HTTP request.

    How to use in a route:
    ─────────────────────────────────────────────────────────────
      from sqlalchemy.ext.asyncio import AsyncSession
      from api.database import get_db

      @router.get("/predictions")
      async def list_predictions(db: AsyncSession = Depends(get_db)):
          result = await db.execute(select(DBPrediction))
          return result.scalars().all()

    Why yield and not return?
    ─────────────────────────────────────────────────────────────
    yield turns get_db into a GENERATOR (context manager Depends).
    FastAPI:
      1. Calls get_db() up to the yield → creates session
      2. Injects the session into the route handler
      3. Runs the route handler
      4. Returns to get_db() after the yield → runs cleanup
         (commit or rollback, then close)

    This guarantees the session is ALWAYS closed, even if the
    route raises an exception. Without this, connections would leak.

    The try/except/finally pattern:
    ─────────────────────────────────────────────────────────────
      try:
          yield session          ← route runs here
          await session.commit() ← success: save changes
      except Exception:
          await session.rollback() ← error: undo changes
          raise                    ← re-raise so FastAPI returns 500
      finally:
          await session.close()  ← ALWAYS runs — no connection leak
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
