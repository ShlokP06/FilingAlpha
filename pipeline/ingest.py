"""SEC filing and price ingestion for FilingAlpha.

Pulls 10-K filings from EDGAR (via edgartools) and daily adjusted-close
prices from Yahoo Finance (via yfinance), then persists them through the
shared SQLAlchemy ORM.

All functions are idempotent: re-running over already-ingested data is safe.
Section extraction (Item 1A / Item 7) degrades gracefully to None rather
than aborting the run when a filing does not expose clean HTML.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import edgar
import pandas as pd
import yfinance as yf
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.config import settings
from core.db import SessionLocal
from core.models import Company, Filing, Price

# Register the SEC identity once at import time so every edgar call is
# properly attributed.  EDGAR requires a name + email in the User-Agent.
edgar.set_identity(settings.sec_identity)

logger = logging.getLogger(__name__)

__all__ = [
    "ingest_company",
    "ingest_filings",
    "ingest_prices",
    "ingest_universe",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_date(value: Optional[str]) -> Optional[date]:
    """Parse a YYYY-MM-DD string into a :class:`datetime.date`, or return None.

    Args:
        value: ISO-format date string, or None/empty.

    Returns:
        Parsed date, or None if the input is falsy or cannot be parsed.
    """
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        logger.debug("Could not parse date string %r", value)
        return None


def _extract_section(report: object, attr: str) -> Optional[str]:
    """Extract a named section from an edgartools report data object.

    edgartools exposes 10-K sections as properties on the object returned by
    ``filing.obj()`` (e.g. ``TenK.risk_factors`` for Item 1A,
    ``TenK.management_discussion`` for Item 7), not via a ``get_sec_section``
    method.  The property may return ``None`` or a rich section object; we
    coerce to stripped text.

    Args:
        report: Data object returned by ``filing.obj()`` (e.g. ``TenK``).
        attr: Property name, e.g. ``"risk_factors"`` or ``"management_discussion"``.

    Returns:
        Section text, or None if the section is absent or extraction raises.
    """
    try:
        value = getattr(report, attr, None)
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    except Exception:
        logger.debug("Section %r not available on report object", attr)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_company(session: Session, ticker: str) -> Company:
    """Upsert a company row resolved from its ticker via edgartools.

    Looks up the CIK, name, and SIC-based industry description from the SEC
    EDGAR submissions API.  If the ticker cannot be resolved, a
    :class:`ValueError` is raised.

    Args:
        session: Active SQLAlchemy session.
        ticker: Upper-case ticker symbol, e.g. ``"AAPL"``.

    Returns:
        The upserted :class:`~core.models.Company` ORM object (already added
        to *session* but not yet committed).

    Raises:
        ValueError: When edgartools cannot resolve *ticker* to an SEC filer.
    """
    ticker = ticker.upper().strip()
    logger.info("Resolving company for ticker %s", ticker)

    edgar_co = edgar.Company(ticker)
    if edgar_co.not_found:
        raise ValueError(f"Ticker {ticker!r} not found on EDGAR")

    cik_str = str(edgar_co.cik)
    name: Optional[str] = edgar_co.name or None
    sector: Optional[str] = edgar_co.industry or None

    # Upsert: update in place if the row already exists.
    existing: Optional[Company] = session.query(Company).filter_by(ticker=ticker).first()
    if existing is not None:
        existing.cik = cik_str
        existing.name = name
        existing.sector = sector
        logger.debug("Updated existing company row for %s", ticker)
        return existing

    company = Company(ticker=ticker, cik=cik_str, name=name, sector=sector)
    session.add(company)
    session.flush()  # populate company.id without full commit
    logger.info("Inserted new company %s (CIK %s)", ticker, cik_str)
    return company


def ingest_filings(
    session: Session,
    company: Company,
    form: str = "10-K",
    limit: int = 6,
) -> int:
    """Pull and persist recent SEC filings for *company*.

    For each filing the function attempts to:

    * Cache the full filing text to ``settings.filings_text_dir()`` and store
      the path in ``Filing.text_path``.
    * Extract Item 1A (Risk Factors) text into ``Filing.item1a_text``.
    * Extract Item 7 (MD&A) text into ``Filing.mdna_text``.

    Section extraction failures are swallowed and fall back to ``None`` so
    that a single malformed filing cannot abort the entire batch.

    Idempotency is guaranteed via the ``(company_id, form_type, filing_date)``
    unique constraint: duplicate filing rows trigger a per-filing savepoint
    rollback and are skipped gracefully.

    Args:
        session: Active SQLAlchemy session.
        company: Already-persisted :class:`~core.models.Company` row.
        form: SEC form type to fetch.  Defaults to ``"10-K"``.
        limit: Maximum number of recent filings to ingest.

    Returns:
        Number of filings actually inserted (duplicates excluded).
    """
    logger.info(
        "Ingesting up to %d %s filings for %s (company_id=%d)",
        limit,
        form,
        company.ticker,
        company.id,
    )

    try:
        edgar_co = edgar.Company(company.ticker)
        if edgar_co.not_found:
            logger.warning("Ticker %s not found on EDGAR; skipping filings", company.ticker)
            return 0
        entity_filings = edgar_co.get_filings(form=form, trigger_full_load=True)
    except Exception:
        logger.exception("Failed to retrieve filing list for %s", company.ticker)
        return 0

    text_dir: Path = settings.filings_text_dir()
    ingested = 0

    for ef in entity_filings.head(limit):
        # ------------------------------------------------------------------
        # 1. Parse dates from the index row (no extra network calls)
        # ------------------------------------------------------------------
        filing_date_raw = getattr(ef, "filing_date", None)
        report_date_raw = getattr(ef, "report_date", None)

        filing_date = _parse_date(
            filing_date_raw.isoformat() if isinstance(filing_date_raw, date) else filing_date_raw
        )
        period_end = _parse_date(
            report_date_raw.isoformat() if isinstance(report_date_raw, date) else report_date_raw
        )

        if filing_date is None:
            logger.warning("No filing_date for accession %s; skipping", ef.accession_no)
            continue

        fiscal_period: Optional[str] = f"FY{period_end.year}" if period_end is not None else None

        # ------------------------------------------------------------------
        # 2. Extract text sections (all failures → None)
        # ------------------------------------------------------------------
        item1a: Optional[str] = None
        mdna: Optional[str] = None
        text_path: Optional[str] = None

        try:
            report = ef.obj()
        except Exception:
            logger.warning(
                "obj() parse failed for %s accession %s",
                company.ticker,
                ef.accession_no,
            )
            report = None

        if report is not None:
            item1a = _extract_section(report, "risk_factors")
            mdna = _extract_section(report, "management_discussion")

        # Cache full filing text to disk
        try:
            full_text: Optional[str] = ef.text()
            if full_text:
                safe_acc = ef.accession_no.replace("-", "")
                fname = f"{company.ticker}_{safe_acc}.txt"
                text_file = text_dir / fname
                text_file.write_text(full_text, encoding="utf-8", errors="replace")
                text_path = str(text_file)
                logger.debug("Cached filing text to %s", text_path)
        except Exception:
            logger.warning(
                "Could not download full text for %s accession %s",
                company.ticker,
                ef.accession_no,
            )

        # ------------------------------------------------------------------
        # 3. Upsert via savepoint — catch UniqueConstraint violations
        # ------------------------------------------------------------------
        try:
            sp = session.begin_nested()
            filing_row = Filing(
                company_id=company.id,
                form_type=form,
                filing_date=filing_date,
                fiscal_period=fiscal_period,
                period_end=period_end,
                text_path=text_path,
                item1a_text=item1a,
                mdna_text=mdna,
            )
            session.add(filing_row)
            session.flush()
            sp.commit()
            ingested += 1
            logger.info(
                "Inserted filing %s / %s (period_end=%s)",
                company.ticker,
                filing_date,
                period_end,
            )
        except IntegrityError:
            sp.rollback()
            logger.debug(
                "Duplicate filing skipped: %s %s %s",
                company.id,
                form,
                filing_date,
            )

    logger.info("Ingested %d/%d %s filings for %s", ingested, limit, form, company.ticker)
    return ingested


def ingest_prices(
    session: Session,
    company: Company,
    start: date,
    end: date,
) -> int:
    """Pull daily adjusted-close prices from Yahoo Finance and upsert them.

    Uses ``auto_adjust=True`` which renames the adjusted close column to
    ``"Close"`` in the returned DataFrame.

    Args:
        session: Active SQLAlchemy session.
        company: Already-persisted :class:`~core.models.Company` row.
        start: First date (inclusive) of the price range.
        end: Last date (inclusive) of the price range.  yfinance's *end*
             parameter is exclusive, so one day is added internally.

    Returns:
        Number of price rows actually inserted (duplicates excluded).
    """
    logger.info("Ingesting prices for %s from %s to %s", company.ticker, start, end)

    yf_end = end + timedelta(days=1)  # yfinance end is exclusive
    try:
        hist = yf.Ticker(company.ticker).history(
            start=start.isoformat(),
            end=yf_end.isoformat(),
            auto_adjust=True,
        )
    except Exception:
        logger.exception("yfinance fetch failed for %s", company.ticker)
        return 0

    if hist is None or hist.empty:
        logger.warning("No price data returned for %s", company.ticker)
        return 0

    # With auto_adjust=True, the adjusted close is in "Close".
    # Guard against column name differences across yfinance versions.
    if "Close" in hist.columns:
        close_series = hist["Close"]
    elif "Adj Close" in hist.columns:
        close_series = hist["Adj Close"]
    else:
        logger.error(
            "Neither 'Close' nor 'Adj Close' found in yfinance result for %s; " "columns: %s",
            company.ticker,
            list(hist.columns),
        )
        return 0

    ingested = 0
    for ts, adj_close in zip(hist.index, close_series):
        try:
            price_date: date = ts.date() if hasattr(ts, "date") else ts
        except Exception:
            logger.debug("Could not convert timestamp %r to date", ts)
            continue

        if pd.isna(adj_close):
            # Skip NaN / pd.NA / None rows (handles float nan, np.nan, pd.NA)
            continue

        try:
            sp = session.begin_nested()
            session.add(
                Price(
                    company_id=company.id,
                    date=price_date,
                    adj_close=float(adj_close),
                )
            )
            session.flush()
            sp.commit()
            ingested += 1
        except IntegrityError:
            sp.rollback()
            logger.debug("Duplicate price skipped: company_id=%d date=%s", company.id, price_date)

    logger.info("Ingested %d price rows for %s", ingested, company.ticker)
    return ingested


def ingest_universe(tickers: list[str], years: int = 6) -> None:
    """Orchestrate full ingestion for a list of tickers.

    For each ticker the function:

    1. Upserts the company row.
    2. Pulls the most recent ``years`` worth of 10-K filings (approximate: up
       to ``years`` filings, one per fiscal year).
    3. Pulls daily adjusted-close prices for the preceding ``years`` years.

    Failures for individual tickers are logged and do not abort the batch.

    Args:
        tickers: List of ticker symbols (case-insensitive).
        years: Look-back window in years for both filings and prices.
    """
    end_date = date.today()
    start_date = end_date.replace(year=end_date.year - years)

    logger.info(
        "Starting universe ingestion: %d tickers, %d-year window",
        len(tickers),
        years,
    )

    session: Session = SessionLocal()
    try:
        for ticker in tickers:
            try:
                company = ingest_company(session, ticker)
                session.commit()
            except Exception:
                session.rollback()
                logger.exception("Failed to ingest company %s; skipping", ticker)
                continue

            try:
                ingest_filings(session, company, form="10-K", limit=years)
                session.commit()
            except Exception:
                session.rollback()
                logger.exception("Failed to ingest filings for %s; continuing with prices", ticker)

            try:
                ingest_prices(session, company, start=start_date, end=end_date)
                session.commit()
            except Exception:
                session.rollback()
                logger.exception("Failed to ingest prices for %s", ticker)

        logger.info("Universe ingestion complete")
    finally:
        session.close()
