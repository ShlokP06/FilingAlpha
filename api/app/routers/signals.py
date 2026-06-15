"""Signals router — per-ticker time-series of NLP signals."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from api.app.deps import SessionDep
from core.models import Company, Filing, Signal
from core.schemas import SignalPoint, SignalSeries

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/{ticker}", response_model=SignalSeries)
def get_signals(ticker: str, session: SessionDep) -> SignalSeries:
    """Return the full NLP signal time-series for a given ticker.

    Joins Filing and Signal tables, ordered by filing date ascending.

    Args:
        ticker: The company ticker symbol (e.g. ``AAPL``).
        session: Injected database session.

    Returns:
        A ``SignalSeries`` containing the ticker and a list of ``SignalPoint``
        objects, one per filing that has computed signals.

    Raises:
        HTTPException: 404 if the ticker is not found in the database.
    """
    ticker_upper = ticker.upper()
    logger.debug("Fetching signals for ticker=%s", ticker_upper)

    company = session.execute(
        select(Company).where(Company.ticker == ticker_upper)
    ).scalar_one_or_none()

    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{ticker_upper}' not found.",
        )

    rows = session.execute(
        select(Filing, Signal)
        .join(Signal, Signal.filing_id == Filing.id)
        .where(Filing.company_id == company.id)
        .order_by(Filing.filing_date.asc())
    ).all()

    points: list[SignalPoint] = [
        SignalPoint(
            filing_date=signal.filing_date,
            fiscal_period=filing.fiscal_period,
            lm_negative=signal.lm_negative,
            lm_uncertainty=signal.lm_uncertainty,
            lm_litigious=signal.lm_litigious,
            yoy_similarity=signal.yoy_similarity,
            risk_factor_delta=signal.risk_factor_delta,
            fog_readability=signal.fog_readability,
        )
        for filing, signal in rows
    ]

    return SignalSeries(ticker=ticker_upper, points=points)
