"""
StockSense AI — api/schemas.py
================================
Shared Pydantic models used across multiple endpoints.
"""

from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    status:       str
    version:      str
    model_loaded: bool
    n_features:   int


class ErrorResponse(BaseModel):
    error:   str
    detail:  str
    ticker:  Optional[str] = None


class TickerInfo(BaseModel):
    """
    Validated ticker that gets reused across endpoints.
    Using a shared model ensures /predict and /backtest
    accept the same ticker format.
    """
    ticker: str
    period: str = "2y"