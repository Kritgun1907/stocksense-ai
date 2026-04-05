"""
StockSense AI — api/crud.py
=============================
Database CRUD operations.

CRUD = Create, Read, Update, Delete.
This file owns all database queries — no SQL anywhere else.

All functions are async and accept an AsyncSession parameter.
Called from routes.py via FastAPI dependency injection.

Why separate crud.py from routes.py?
─────────────────────────────────────────────────────────────
  Routes handle HTTP (request parsing, response formatting).
  CRUD handles database (queries, data shaping).
  Mixing them makes both harder to test and maintain.
  With separate files: routes are thin, CRUD is testable in isolation.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import select, update, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import (
    DBUser, DBPrediction, DBWatchlist,
    DBPortfolio, DBBacktestResult,
)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

async def save_prediction(
    db:         AsyncSession,
    ticker:     str,
    prediction: str,
    probability: float,
    confidence: float,
    horizon_days: int   = 1,
    threshold:  float   = 0.5,
    price:      Optional[float] = None,
    n_features: Optional[int]   = None,
    model_version: str  = "v1",
    explanation_dict: Optional[dict] = None,
) -> DBPrediction:
    """
    Save a new prediction to the database.
    Uses INSERT ... ON CONFLICT DO UPDATE to handle the unique constraint —
    if a prediction for this ticker+day+horizon already exists, update it.

    Why upsert rather than pure insert?
    ─────────────────────────────────────────────────────────────
    The nightly job might run multiple times (reruns, retries).
    A pure insert would fail with unique constraint violation.
    Upsert updates existing predictions with fresh data instead.
    """
    from sqlalchemy.dialects.postgresql import insert

    values = {
        "id":               uuid.uuid4(),
        "ticker":           ticker.upper(),
        "prediction":       prediction,
        "probability":      round(probability, 4),
        "confidence":       round(confidence, 1),
        "horizon_days":     horizon_days,
        "threshold_used":   threshold,
        "price_at_prediction": price,
        "n_features_used":  n_features,
        "model_version":    model_version,
        "generated_at":     datetime.utcnow(),
        "explanation_json": json.dumps(explanation_dict) if explanation_dict else None,
    }

    stmt = insert(DBPrediction).values(**values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_prediction_ticker_day_horizon",
        set_={
            "prediction":    values["prediction"],
            "probability":   values["probability"],
            "confidence":    values["confidence"],
            "generated_at":  values["generated_at"],
        }
    )

    await db.execute(stmt)
    await db.commit()

    # Fetch and return the saved record
    result = await db.execute(
        select(DBPrediction)
        .where(DBPrediction.ticker == ticker.upper())
        .order_by(desc(DBPrediction.generated_at))
        .limit(1)
    )
    return result.scalar_one()


async def get_prediction_history(
    db:          AsyncSession,
    ticker:      str,
    days_back:   int = 30,
    horizon:     int = 1,
) -> List[DBPrediction]:
    """
    Get prediction history for a ticker over the last N days.
    Used for the 'Prediction History' chart on the stock page.
    """
    since = datetime.utcnow() - timedelta(days=days_back)
    result = await db.execute(
        select(DBPrediction)
        .where(
            and_(
                DBPrediction.ticker       == ticker.upper(),
                DBPrediction.horizon_days == horizon,
                DBPrediction.generated_at >= since,
            )
        )
        .order_by(desc(DBPrediction.generated_at))
    )
    return result.scalars().all()


async def resolve_predictions(
    db:        AsyncSession,
    days_ago:  int = 1,
) -> int:
    """
    Check predictions made N days ago and mark whether they were correct.
    Called nightly by Celery background job.

    Returns number of predictions resolved.
    """
    import yfinance as yf

    target_date = datetime.utcnow() - timedelta(days=days_ago)
    since       = target_date - timedelta(hours=12)
    until       = target_date + timedelta(hours=12)

    result = await db.execute(
        select(DBPrediction)
        .where(
            and_(
                DBPrediction.generated_at >= since,
                DBPrediction.generated_at <= until,
                DBPrediction.was_correct  == None,
            )
        )
    )
    unresolved = result.scalars().all()

    resolved_count = 0

    for pred in unresolved:
        try:
            # Fetch actual price N days after prediction
            ticker_data = yf.download(
                pred.ticker,
                start=target_date.strftime("%Y-%m-%d"),
                end=(target_date + timedelta(days=pred.horizon_days + 2))
                    .strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )

            if ticker_data.empty or len(ticker_data) < 2:
                continue

            entry_price = float(ticker_data["Close"].iloc[0])
            exit_price  = float(ticker_data["Close"].iloc[
                min(pred.horizon_days, len(ticker_data) - 1)
            ])

            actual = "UP" if exit_price > entry_price else "DOWN"

            await db.execute(
                update(DBPrediction)
                .where(DBPrediction.id == pred.id)
                .values(
                    actual_direction    = actual,
                    was_correct         = (pred.prediction == actual),
                    price_at_resolution = exit_price,
                    resolved_at         = datetime.utcnow(),
                )
            )
            resolved_count += 1

        except Exception as e:
            continue

    await db.commit()
    return resolved_count


async def get_model_accuracy_stats(
    db:        AsyncSession,
    days_back: int = 30,
) -> dict:
    """
    Compute model accuracy statistics over the last N days.
    Used for the 'Model Performance' dashboard widget.
    """
    since = datetime.utcnow() - timedelta(days=days_back)
    result = await db.execute(
        select(DBPrediction)
        .where(
            and_(
                DBPrediction.generated_at >= since,
                DBPrediction.was_correct  != None,
            )
        )
    )
    resolved = result.scalars().all()

    if not resolved:
        return {"total": 0, "correct": 0, "accuracy": None}

    total   = len(resolved)
    correct = sum(1 for p in resolved if p.was_correct)

    return {
        "total":    total,
        "correct":  correct,
        "accuracy": round(correct / total * 100, 1),
        "up_correct":   sum(1 for p in resolved
                            if p.prediction == "UP" and p.was_correct),
        "down_correct": sum(1 for p in resolved
                            if p.prediction == "DOWN" and p.was_correct),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  USERS
# ══════════════════════════════════════════════════════════════════════════════

async def get_user_by_email(
    db: AsyncSession, email: str
) -> Optional[DBUser]:
    """Fetch user by email. Returns None if not found."""
    result = await db.execute(
        select(DBUser).where(DBUser.email == email.lower())
    )
    return result.scalar_one_or_none()


async def create_user(
    db:            AsyncSession,
    email:         str,
    hashed_password: str,
    full_name:     Optional[str] = None,
) -> DBUser:
    """Create a new user account."""
    user = DBUser(
        email           = email.lower(),
        hashed_password = hashed_password,
        full_name       = full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# ══════════════════════════════════════════════════════════════════════════════
#  WATCHLISTS
# ══════════════════════════════════════════════════════════════════════════════

async def get_watchlist(
    db: AsyncSession, user_id: uuid.UUID
) -> List[DBWatchlist]:
    """Get all stocks in a user's watchlist."""
    result = await db.execute(
        select(DBWatchlist)
        .where(DBWatchlist.user_id == user_id)
        .order_by(DBWatchlist.added_at.desc())
    )
    return result.scalars().all()


async def add_to_watchlist(
    db:      AsyncSession,
    user_id: uuid.UUID,
    ticker:  str,
    notes:   Optional[str] = None,
) -> DBWatchlist:
    """Add a stock to user's watchlist. Ignores if already present."""
    from sqlalchemy.dialects.postgresql import insert

    stmt = insert(DBWatchlist).values(
        user_id  = user_id,
        ticker   = ticker.upper(),
        notes    = notes,
        added_at = datetime.utcnow(),
    ).on_conflict_do_nothing(constraint="uq_watchlist_user_ticker")

    await db.execute(stmt)
    await db.commit()

    result = await db.execute(
        select(DBWatchlist)
        .where(
            and_(
                DBWatchlist.user_id == user_id,
                DBWatchlist.ticker  == ticker.upper(),
            )
        )
    )
    return result.scalar_one()


async def remove_from_watchlist(
    db:      AsyncSession,
    user_id: uuid.UUID,
    ticker:  str,
) -> bool:
    """Remove a stock from watchlist. Returns True if removed."""
    result = await db.execute(
        select(DBWatchlist)
        .where(
            and_(
                DBWatchlist.user_id == user_id,
                DBWatchlist.ticker  == ticker.upper(),
            )
        )
    )
    item = result.scalar_one_or_none()
    if item:
        await db.delete(item)
        await db.commit()
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIOS
# ══════════════════════════════════════════════════════════════════════════════

async def get_open_positions(
    db: AsyncSession, user_id: uuid.UUID
) -> List[DBPortfolio]:
    """Get all open portfolio positions for a user."""
    result = await db.execute(
        select(DBPortfolio)
        .where(
            and_(
                DBPortfolio.user_id == user_id,
                DBPortfolio.is_open == True,
            )
        )
        .order_by(DBPortfolio.bought_at.desc())
    )
    return result.scalars().all()


async def open_position(
    db:            AsyncSession,
    user_id:       uuid.UUID,
    ticker:        str,
    shares:        float,
    buy_price:     float,
    notes:         Optional[str] = None,
) -> DBPortfolio:
    """Open a new paper trading position."""
    position = DBPortfolio(
        user_id       = user_id,
        ticker        = ticker.upper(),
        shares        = shares,
        avg_buy_price = buy_price,
        notes         = notes,
    )
    db.add(position)
    await db.commit()
    await db.refresh(position)
    return position


async def close_position(
    db:         AsyncSession,
    position_id: uuid.UUID,
    sell_price: float,
) -> Optional[DBPortfolio]:
    """Close an open paper trading position."""
    result = await db.execute(
        select(DBPortfolio).where(DBPortfolio.id == position_id)
    )
    position = result.scalar_one_or_none()

    if not position or not position.is_open:
        return None

    position.is_open    = False
    position.sell_price = sell_price
    position.sold_at    = datetime.utcnow()

    await db.commit()
    await db.refresh(position)
    return position


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST RESULTS
# ══════════════════════════════════════════════════════════════════════════════

async def save_backtest_result(
    db:      AsyncSession,
    ticker:  str,
    report_dict: dict,
    config_name: str = "default",
) -> DBBacktestResult:
    """
    Save backtest performance metrics.
    Called by the nightly Celery job after running backtester.py.
    """
    from sqlalchemy.dialects.postgresql import insert

    values = {
        "id":               uuid.uuid4(),
        "ticker":           ticker.upper(),
        "period_start":     datetime.fromisoformat(report_dict["period_start"])
                            if isinstance(report_dict["period_start"], str)
                            else report_dict["period_start"],
        "period_end":       datetime.fromisoformat(report_dict["period_end"])
                            if isinstance(report_dict["period_end"], str)
                            else report_dict["period_end"],
        "sharpe_ratio":     report_dict.get("sharpe_ratio"),
        "sortino_ratio":    report_dict.get("sortino_ratio"),
        "calmar_ratio":     report_dict.get("calmar_ratio"),
        "max_drawdown":     report_dict.get("max_drawdown"),
        "win_rate":         report_dict.get("win_rate"),
        "total_return":     report_dict.get("total_return"),
        "annualised_return": report_dict.get("annualised_return"),
        "n_trades":         report_dict.get("n_trades"),
        "beats_benchmark":  report_dict.get("beats_benchmark"),
        "verdict":          report_dict.get("verdict"),
        "config_name":      config_name,
        "computed_at":      datetime.utcnow(),
    }

    stmt = insert(DBBacktestResult).values(**values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_backtest_ticker_config",
        set_={k: v for k, v in values.items()
              if k not in ("id", "ticker", "config_name")}
    )

    await db.execute(stmt)
    await db.commit()

    result = await db.execute(
        select(DBBacktestResult)
        .where(
            and_(
                DBBacktestResult.ticker      == ticker.upper(),
                DBBacktestResult.config_name == config_name,
            )
        )
    )
    return result.scalar_one()


async def get_backtest_result(
    db:          AsyncSession,
    ticker:      str,
    config_name: str = "default",
) -> Optional[DBBacktestResult]:
    """Get cached backtest result for a ticker."""
    result = await db.execute(
        select(DBBacktestResult)
        .where(
            and_(
                DBBacktestResult.ticker      == ticker.upper(),
                DBBacktestResult.config_name == config_name,
            )
        )
    )
    return result.scalar_one_or_none()