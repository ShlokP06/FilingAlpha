"""Forward-return computation with strict filing-lag (no lookahead).

For each filing we compute the forward stock return over a horizon of ``h``
*trading* days. The window starts on the first trading day **strictly after**
the filing date (``t+1``), never on the filing date itself. This filing-lag
removes same-day lookahead: at the close of the filing day the market may not
yet have fully incorporated the disclosure, and using the filing-day price as
the entry would let information leak backward into the entry price.

    fwd_return = adj_close[t+1+h] / adj_close[t+1] - 1

where ``t+1`` is the first trading day after ``filing_date`` and ``t+1+h`` is
``h`` trading days later in the firm's own price series. Filings without enough
forward price history are skipped.

**Market adjustment.** The persisted ``fwd_return`` is an *excess* (abnormal)
return: the firm's forward return minus the market's return (proxied by ``SPY``)
over the identical entry-to-exit window. Raw forward returns are dominated by
broad market beta, which is noise for a stock-selection signal; subtracting the
market isolates the firm-specific move the filing text might predict. When the
benchmark series is unavailable for a window, the raw return is stored as a
fallback.
"""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models import Company, Filing, ForwardReturn, Price

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS: tuple[int, ...] = (21, 63)

# Market benchmark ticker used to compute excess (market-adjusted) returns.
MARKET_TICKER = "SPY"


def _price_series(session: Session, company_id: int) -> list[tuple[date, float]]:
    """Return the firm's ``(date, adj_close)`` series sorted ascending by date.

    Args:
        session: Active database session.
        company_id: Company whose prices to fetch.

    Returns:
        Date-ascending list of ``(date, adj_close)`` tuples.
    """
    rows = session.execute(
        select(Price.date, Price.adj_close)
        .where(Price.company_id == company_id)
        .order_by(Price.date.asc())
    ).all()
    return [(row[0], float(row[1])) for row in rows]


def _first_index_after(dates: list[date], filing_date: date) -> int | None:
    """Index of the first date strictly greater than ``filing_date``.

    Args:
        dates: Date-ascending list of trading dates.
        filing_date: The filing date to lag from.

    Returns:
        The index of the first ``date > filing_date``, or ``None`` if none
        exists.
    """
    # Linear scan keeps the dependency surface minimal; price series per firm
    # are small. bisect would also work since ``dates`` is sorted.
    for i, d in enumerate(dates):
        if d > filing_date:
            return i
    return None


def forward_return_for_series(
    dates: list[date],
    closes: list[float],
    filing_date: date,
    horizon: int,
) -> float | None:
    """Compute a single forward return from a price series with filing-lag.

    Pure helper (no DB) so the no-lookahead logic is unit-testable in isolation.

    Args:
        dates: Date-ascending trading dates.
        closes: Adjusted closes aligned with ``dates``.
        filing_date: Filing date; entry is the first trading day strictly after.
        horizon: Forward horizon in trading days.

    Returns:
        ``adj_close[t+1+h] / adj_close[t+1] - 1`` or ``None`` if there is no
        trading day after the filing date or insufficient forward history.
    """
    entry_idx = _first_index_after(dates, filing_date)
    if entry_idx is None:
        return None
    exit_idx = entry_idx + horizon
    if exit_idx >= len(closes):
        return None
    entry_price = closes[entry_idx]
    if entry_price == 0:
        return None
    return closes[exit_idx] / entry_price - 1.0


def _market_series(session: Session) -> dict[date, float]:
    """Return the market benchmark's ``{date: adj_close}`` map, or empty if absent.

    Args:
        session: Active database session.

    Returns:
        Mapping of date to adjusted close for :data:`MARKET_TICKER`. Empty when
        the benchmark has not been ingested (returns then stay raw).
    """
    company_id = session.execute(
        select(Company.id).where(Company.ticker == MARKET_TICKER)
    ).scalar_one_or_none()
    if company_id is None:
        logger.warning(
            "Market benchmark %s not in DB; forward returns will be raw (not "
            "market-adjusted). Run ingest_market_benchmark() to enable.",
            MARKET_TICKER,
        )
        return {}
    rows = session.execute(
        select(Price.date, Price.adj_close).where(Price.company_id == company_id)
    ).all()
    return {row[0]: float(row[1]) for row in rows}


def _market_return(
    market_close: dict[date, float], entry_date: date, exit_date: date
) -> float | None:
    """Market return between two calendar dates, or None if either is missing.

    Args:
        market_close: ``{date: adj_close}`` benchmark map from :func:`_market_series`.
        entry_date: Window start (the firm's filing-lagged entry date).
        exit_date: Window end.

    Returns:
        ``close[exit] / close[entry] - 1`` or None when a date is absent from the
        benchmark series (so the caller can fall back to the raw return).
    """
    entry = market_close.get(entry_date)
    exit_ = market_close.get(exit_date)
    if entry is None or exit_ is None or entry == 0:
        return None
    return exit_ / entry - 1.0


def compute_forward_returns(
    session: Session,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> int:
    """Compute and persist forward returns for every filing, idempotently.

    For each filing and each horizon, computes the filing-lagged forward return
    (see module docstring) and upserts a :class:`ForwardReturn` row keyed by
    ``(filing_id, horizon_days)``. Filings without enough forward price history
    are skipped. Existing rows are updated in place so re-runs are idempotent.

    Args:
        session: Active database session.
        horizons: Forward horizons in trading days.

    Returns:
        Number of ``ForwardReturn`` rows written or updated.
    """
    filings = session.execute(
        select(Filing.id, Filing.company_id, Filing.filing_date).order_by(Filing.id.asc())
    ).all()

    # Cache price series per company to avoid repeated queries.
    series_cache: dict[int, list[tuple[date, float]]] = {}
    market_close = _market_series(session)
    written = 0

    for filing_id, company_id, filing_date in filings:
        if company_id not in series_cache:
            series_cache[company_id] = _price_series(session, company_id)
        series = series_cache[company_id]
        if not series:
            continue
        dates = [d for d, _ in series]
        closes = [c for _, c in series]

        entry_idx = _first_index_after(dates, filing_date)
        if entry_idx is None:
            continue
        entry_price = closes[entry_idx]
        if entry_price == 0:
            continue
        entry_date = dates[entry_idx]

        for horizon in horizons:
            exit_idx = entry_idx + horizon
            if exit_idx >= len(closes):
                continue
            # Excess return: firm move minus the market move over the same window.
            stock_ret = closes[exit_idx] / entry_price - 1.0
            mkt_ret = _market_return(market_close, entry_date, dates[exit_idx])
            fwd = stock_ret - mkt_ret if mkt_ret is not None else stock_ret

            existing = session.execute(
                select(ForwardReturn).where(
                    ForwardReturn.filing_id == filing_id,
                    ForwardReturn.horizon_days == horizon,
                )
            ).scalar_one_or_none()
            if existing is None:
                session.add(
                    ForwardReturn(
                        filing_id=filing_id,
                        horizon_days=horizon,
                        fwd_return=fwd,
                    )
                )
            else:
                existing.fwd_return = fwd
            written += 1

    session.commit()
    logger.info("Computed forward returns for %d (filing, horizon) pairs.", written)
    return written


__all__ = [
    "compute_forward_returns",
    "forward_return_for_series",
    "DEFAULT_HORIZONS",
    "MARKET_TICKER",
]
