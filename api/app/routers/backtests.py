"""Backtests router — backtest run results with optional filtering."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from sqlalchemy import select

from api.app.deps import SessionDep
from core.models import BacktestRun
from core.schemas import BacktestOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.get("", response_model=list[BacktestOut])
def list_backtests(
    session: SessionDep,
    signal: str | None = Query(default=None, description="Filter by signal name"),
    horizon_days: int | None = Query(default=None, description="Filter by horizon in days"),
) -> list[BacktestRun]:
    """Return backtest run results, newest first.

    Both ``signal`` and ``horizon_days`` filters are applied together (AND
    semantics) when provided.

    Args:
        session: Injected database session.
        signal: Optional signal name to filter on (e.g. ``lm_negative``).
        horizon_days: Optional forward-return horizon to filter on.

    Returns:
        A list of ``BacktestOut`` objects sorted by creation date descending.
    """
    logger.debug("Fetching backtests signal=%s horizon_days=%s", signal, horizon_days)

    stmt = select(BacktestRun)

    if signal is not None:
        stmt = stmt.where(BacktestRun.signal == signal)
    if horizon_days is not None:
        stmt = stmt.where(BacktestRun.horizon_days == horizon_days)

    stmt = stmt.order_by(BacktestRun.created_at.desc())

    rows = session.execute(stmt).scalars().all()
    return list(rows)
