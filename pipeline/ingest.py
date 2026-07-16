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
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import edgar
import pandas as pd
import yfinance as yf
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from tqdm import tqdm

from core.config import settings
from core.db import SessionLocal
from core.models import Company, Filing, Price
from pipeline.net import (
    STATE_COMPLETE,
    STATE_FAILED,
    STATE_NO_DATA,
    STATE_PENDING,
    TRANSIENT_ERRORS,
    chunked,
    make_yf_session,
    retry_edgar,
    retry_yf,
)

# Register the SEC identity once at import time so every edgar call is
# properly attributed.  EDGAR requires a name + email in the User-Agent.
edgar.set_identity(settings.sec_identity)

logger = logging.getLogger(__name__)

__all__ = [
    "ingest_company",
    "ingest_filings",
    "ingest_prices",
    "ingest_prices_bulk",
    "ingest_universe",
    "ingest_market_benchmark",
    "MARKET_TICKER",
]

# Market benchmark used to compute excess (market-adjusted) forward returns.
MARKET_TICKER = "SPY"

# Lazily-built shared yfinance session (cached / rate-limit friendly). Kept at
# module level so every price call in a run reuses one HTTP cache + connection.
_YF_SESSION: object | None = None
_YF_SESSION_READY = False


def _yf_session() -> object | None:
    """Return the process-wide yfinance session, building it once on first use."""
    global _YF_SESSION, _YF_SESSION_READY
    if not _YF_SESSION_READY:
        _YF_SESSION = make_yf_session()
        _YF_SESSION_READY = True
    return _YF_SESSION


# Retry-wrapped network primitives (transient errors only; see pipeline.net)


@retry_edgar
def _edgar_company(ticker: str) -> object:
    """Resolve an EDGAR company handle by ticker, retrying transient network errors."""
    return edgar.Company(ticker)


@retry_edgar
def _edgar_company_by_cik(cik: int) -> object:
    """Resolve an EDGAR company handle by CIK, retrying transient network errors.

    CIK resolution is stable across ticker changes and survives delistings, so it
    is the reliable identifier for a 2018-anchored universe where many small/mid
    caps have since been renamed or acquired. A ticker lookup would raise
    ``not_found`` for those; the CIK never changes.
    """
    return edgar.Company(cik)


@retry_edgar
def _get_entity_filings(edgar_co: object, form: str) -> object:
    """Fetch a company's filing index for *form*, retrying transient errors."""
    return edgar_co.get_filings(form=form, trigger_full_load=True)


@retry_yf
def _fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch one ticker's daily history, retrying transient / rate-limit errors.

    Raises the underlying transient error (e.g. ``YFRateLimitError``) once
    retries are exhausted, so callers can distinguish a throttled fetch from a
    genuinely empty (delisted) result. A clean empty DataFrame is *not* an error.
    """
    yf_end = end + timedelta(days=1)  # yfinance end is exclusive
    return yf.Ticker(ticker, session=_yf_session()).history(
        start=start.isoformat(),
        end=yf_end.isoformat(),
        auto_adjust=True,
    )


@retry_yf
def _download_chunk(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Bulk-download several tickers in one HTTP call, retrying transient errors."""
    yf_end = end + timedelta(days=1)
    return yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=yf_end.isoformat(),
        auto_adjust=True,
        group_by="ticker",
        threads=False,
        progress=False,
        session=_yf_session(),
    )


def _extract_close(data: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    """Pull one ticker's adjusted-close series out of a (possibly bulk) frame.

    Handles both the single-ticker layout (flat columns) and the multi-ticker
    ``group_by="ticker"`` layout (``MultiIndex`` columns). Returns ``None`` when
    the ticker is absent or exposes no close column.
    """
    if data is None or data.empty:
        return None
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        if ticker not in cols.get_level_values(0):
            return None
        sub = data[ticker]
    else:
        sub = data
    for name in ("Close", "Adj Close"):
        if name in sub.columns:
            return sub[name]
    return None


# Internal helpers


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


# Public API


def ingest_company(
    session: Session, ticker: str, cik: str | int | None = None
) -> Company:
    """Upsert a company row resolved via edgartools, preferring CIK when given.

    Looks up the name and SIC-based industry description from the SEC EDGAR
    submissions API. When *cik* is supplied it is used as the lookup key (stable
    across ticker renames and delistings); otherwise the *ticker* is resolved.
    The *ticker* is always stored as the row's identity, so a historical (2018)
    ticker is preserved even when EDGAR now indexes the filer under a new one.

    Args:
        session: Active SQLAlchemy session.
        ticker: Upper-case ticker symbol, e.g. ``"AAPL"`` — stored as identity.
        cik: Optional SEC CIK. When provided, resolution goes through the CIK
            (robust for renamed/delisted firms) instead of the current ticker.

    Returns:
        The upserted :class:`~core.models.Company` ORM object (already added
        to *session* but not yet committed).

    Raises:
        ValueError: When edgartools cannot resolve the firm to an SEC filer.
    """
    ticker = ticker.upper().strip()

    if cik is not None:
        logger.info("Resolving company for CIK %s (ticker %s)", cik, ticker)
        edgar_co = _edgar_company_by_cik(int(cik))
    else:
        logger.info("Resolving company for ticker %s", ticker)
        edgar_co = _edgar_company(ticker)
    if edgar_co.not_found:
        raise ValueError(f"Firm {ticker!r} (cik={cik}) not found on EDGAR")

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
        edgar_co = (
            _edgar_company_by_cik(int(company.cik))
            if company.cik and str(company.cik).isdigit()
            else _edgar_company(company.ticker)
        )
        if edgar_co.not_found:
            logger.warning("Firm %s not found on EDGAR; skipping filings", company.ticker)
            return 0
        entity_filings = _get_entity_filings(edgar_co, form)
    except TRANSIENT_ERRORS:
        # A throttle / network blip survived retries. Surface it so the caller
        # marks this firm ``failed`` and retries it next run — recording a false
        # "0 filings" here would let the firm look ``complete`` with no data.
        logger.warning(
            "Transient error retrieving %s filings for %s; will retry next run",
            form,
            company.ticker,
        )
        raise
    except Exception:
        logger.exception("Failed to retrieve filing list for %s", company.ticker)
        return 0

    text_dir: Path = settings.filings_text_dir()
    ingested = 0

    for ef in entity_filings.head(limit):
        # 1. Parse dates from the index row (no extra network calls)
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

        # Cheap idempotency short-circuit: the (company, form, filing_date) key is
        # known from the *index* row alone, so if this filing is already stored we
        # skip the expensive obj()/text() download entirely. This makes resumes and
        # re-runs near-free instead of re-parsing every already-ingested filing.
        already = (
            session.query(Filing.id)
            .filter_by(company_id=company.id, form_type=form, filing_date=filing_date)
            .first()
        )
        if already is not None:
            logger.debug(
                "Filing already stored; skipping download: %s %s %s",
                company.ticker,
                form,
                filing_date,
            )
            continue

        fiscal_period: Optional[str] = f"FY{period_end.year}" if period_end is not None else None

        # 2. Extract text sections (all failures → None)
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

        # Cache full filing text to disk (skippable for a faster, lighter run)
        try:
            full_text: Optional[str] = ef.text() if settings.ingest_full_text else None
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

        # 3. Upsert via savepoint — catch UniqueConstraint violations
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


def _upsert_prices(session: Session, company: Company, close_series: pd.Series) -> int:
    """Upsert an adjusted-close series for *company*, skipping NaN/duplicate rows.

    Args:
        session: Active SQLAlchemy session.
        company: Already-persisted company row.
        close_series: Adjusted-close values indexed by timestamp.

    Returns:
        Number of price rows actually inserted (duplicates and NaNs excluded).
    """
    ingested = 0
    for ts, adj_close in zip(close_series.index, close_series):
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
            session.add(Price(company_id=company.id, date=price_date, adj_close=float(adj_close)))
            session.flush()
            sp.commit()
            ingested += 1
        except IntegrityError:
            sp.rollback()
            logger.debug("Duplicate price skipped: company_id=%d date=%s", company.id, price_date)
    return ingested


def ingest_prices(
    session: Session,
    company: Company,
    start: date,
    end: date,
) -> int:
    """Pull daily adjusted-close prices from Yahoo Finance and upsert them.

    Uses ``auto_adjust=True`` which puts the adjusted close in the ``"Close"``
    column. This is the *safe* single-ticker path: it never raises, returning 0
    on any failure (transient errors are already retried inside
    :func:`_fetch_history`). The state-tracking batch path uses
    :func:`ingest_prices_bulk`, which distinguishes rate-limited from empty.

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

    try:
        hist = _fetch_history(company.ticker, start, end)
    except Exception:
        logger.exception("yfinance fetch failed for %s (after retries)", company.ticker)
        return 0

    close_series = _extract_close(hist, company.ticker)
    if close_series is None or close_series.dropna().empty:
        logger.warning("No price data returned for %s", company.ticker)
        return 0

    ingested = _upsert_prices(session, company, close_series)
    logger.info("Ingested %d price rows for %s", ingested, company.ticker)
    return ingested


def ingest_prices_bulk(
    session: Session,
    companies: list[Company],
    start: date,
    end: date,
) -> dict[int, str]:
    """Fetch prices for many companies in bulk, returning a per-firm outcome.

    Prices are downloaded in chunks of ``settings.yf_chunk_size`` via one
    ``yf.download`` call each — far fewer HTTP requests than one-per-ticker,
    which is the biggest lever against Yahoo rate limits. A short pause between
    chunks throttles proactively.

    A chunk-level throttle (the whole ``yf.download`` raising a transient error
    after retries) marks *every* firm in that chunk ``failed`` for the next run —
    it deliberately does **not** re-probe each ticker individually, which would
    amplify load on an already-throttled Yahoo. Per-ticker re-probing is reserved
    for the case where the bulk call *succeeded* but one ticker came back empty:
    :func:`_probe_prices_one` then tells a genuinely delisted firm (``no_data``)
    apart from a single-ticker gap. After :data:`settings.yf_max_failed_chunks`
    consecutive chunk failures a circuit breaker trips: the remaining firms are
    marked ``failed`` without further calls, so one throttled run never hammers
    Yahoo for the whole batch.

    Args:
        session: Active SQLAlchemy session (committed per firm).
        companies: Already-persisted company rows to fetch prices for.
        start: First date (inclusive).
        end: Last date (inclusive).

    Returns:
        Mapping of ``company.id`` to a price outcome: :data:`STATE_COMPLETE`,
        :data:`STATE_NO_DATA`, or :data:`STATE_FAILED`.
    """
    outcomes: dict[int, str] = {}
    consecutive_failures = 0
    breaker_tripped = False

    size = max(1, settings.yf_chunk_size)
    n_chunks = (len(companies) + size - 1) // size
    for chunk in tqdm(
        chunked(companies, settings.yf_chunk_size),
        total=n_chunks,
        desc="Prices",
        unit="chunk",
        dynamic_ncols=True,
    ):
        if breaker_tripped:
            # Circuit open: don't touch the network again this run.
            for company in chunk:
                outcomes[company.id] = STATE_FAILED
            continue

        symbols = [c.ticker for c in chunk]
        logger.info("Bulk price fetch for %d tickers: %s", len(symbols), symbols)
        try:
            data = _download_chunk(symbols, start, end)
            consecutive_failures = 0
        except TRANSIENT_ERRORS:
            # Whole chunk throttled — mark all failed (retried next run) instead
            # of re-probing each ticker, which would only deepen the throttle.
            consecutive_failures += 1
            logger.warning(
                "Bulk chunk failed after retries (%d consecutive); marking %d firms failed",
                consecutive_failures,
                len(chunk),
            )
            for company in chunk:
                outcomes[company.id] = STATE_FAILED
            if consecutive_failures >= settings.yf_max_failed_chunks:
                logger.error(
                    "Circuit breaker tripped after %d consecutive throttled chunks; "
                    "stopping price fetches this run (remaining firms marked failed)",
                    consecutive_failures,
                )
                breaker_tripped = True
            time.sleep(settings.yf_chunk_pause)
            continue

        for company in chunk:
            close = _extract_close(data, company.ticker) if data is not None else None
            if close is not None and not close.dropna().empty:
                rows = _upsert_prices(session, company, close)
                session.commit()
                outcomes[company.id] = STATE_COMPLETE if rows > 0 else STATE_NO_DATA
            else:
                # Bulk call succeeded but this ticker was empty — probe it alone
                # to distinguish delisted (no_data) from a transient gap (failed).
                outcomes[company.id] = _probe_prices_one(session, company, start, end)

        time.sleep(settings.yf_chunk_pause)

    return outcomes


def _probe_prices_one(session: Session, company: Company, start: date, end: date) -> str:
    """Fetch one firm's prices strictly, classifying the outcome.

    Returns:
        :data:`STATE_COMPLETE` when rows land, :data:`STATE_NO_DATA` on a clean
        empty result (delisted/illiquid), or :data:`STATE_FAILED` when a
        transient/rate-limit error survives all retries.
    """
    try:
        hist = _fetch_history(company.ticker, start, end)
    except TRANSIENT_ERRORS:
        logger.warning("Price probe rate-limited/failed for %s; marking failed", company.ticker)
        return STATE_FAILED
    except Exception:
        logger.exception("Unexpected price error for %s; marking failed", company.ticker)
        return STATE_FAILED

    close = _extract_close(hist, company.ticker)
    if close is None or close.dropna().empty:
        logger.info("No price data for %s; marking no_data", company.ticker)
        return STATE_NO_DATA

    rows = _upsert_prices(session, company, close)
    session.commit()
    logger.info("Ingested %d price rows for %s", rows, company.ticker)
    return STATE_COMPLETE if rows > 0 else STATE_NO_DATA


def _resolve_window(
    start: date | None, end: date | None, years: int
) -> tuple[date, date]:
    """Return the ``(start, end)`` ingest window, defaulting a ``years`` look-back.

    An explicit *start*/*end* wins; otherwise the window is *years* back from
    *end* (or today). The subtraction is Feb-29-safe: a leap-day *end* whose
    look-back year is not a leap year falls back to Feb 28.
    """
    end = end or date.today()
    if start is None:
        try:
            start = end.replace(year=end.year - years)
        except ValueError:  # end is Feb 29 in a year whose look-back isn't a leap year
            start = end.replace(year=end.year - years, day=28)
    return start, end


def ingest_universe(
    universe: list[dict | str],
    *,
    start: date | None = None,
    end: date | None = None,
    years: int = 8,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
) -> dict[str, str]:
    """Orchestrate full ingestion for a universe of firms.

    Each entry of *universe* is either a plain ticker string or a record dict
    ``{"ticker": ..., "cik": ...}`` (as produced by the universe builder). When a
    CIK is present it drives EDGAR resolution — stable across ticker renames and
    delistings — while the given ticker is stored as the row's identity. This is
    what lets a 2018-anchored small/mid universe resolve firms that have since
    been renamed or acquired instead of dropping them.

    For each firm the function:

    1. Upserts the company row (by CIK when available).
    2. Pulls filings for each form in *forms* over the window, sized to its span
       (10-Ks file annually, 10-Qs roughly quarterly, so the quarterly cap is
       scaled up ~4x).
    3. Pulls daily adjusted-close prices over the window (bulk; see
       :func:`ingest_prices_bulk`).

    Each firm carries an ``ingest_state`` set to ``complete`` only once *both* its
    filings and prices have landed, so a run interrupted by rate limits never
    leaves a firm looking done with missing data. A firm whose prices come back
    empty (delisted) is marked ``no_data`` (terminal) but keeps its filings; one
    hit by a transient/rate-limit error is marked ``failed`` and retried next run.

    Args:
        universe: Ticker strings and/or ``{"ticker","cik"}`` record dicts.
        start: Explicit window start (inclusive). Defaults to *years* before *end*.
        end: Explicit window end (inclusive). Defaults to today.
        years: Look-back used only when *start* is not given.
        forms: SEC form types to ingest per company.

    Returns:
        Mapping of each processed ticker to its final ``ingest_state``.
    """
    start_date, end_date = _resolve_window(start, end, years)
    span_years = max(1, end_date.year - start_date.year + 1)

    logger.info(
        "Starting universe ingestion: %d firms, window %s..%s, forms=%s",
        len(universe),
        start_date,
        end_date,
        ",".join(forms),
    )

    session: Session = SessionLocal()
    final_states: dict[str, str] = {}
    try:
        # -- Phase 1 & 2: resolve companies and ingest their filings --------
        # ``filing_failed`` tracks a *transient* filing error (retries exhausted)
        # so we don't mark such a firm complete; a firm with simply zero filings
        # of a form is not an error.
        resolved: list[Company] = []
        filing_failed: dict[int, bool] = {}
        progress = tqdm(universe, desc="Filings", unit="firm", dynamic_ncols=True)
        for rec in progress:
            ticker = (rec if isinstance(rec, str) else str(rec["ticker"])).upper().strip()
            cik = None if isinstance(rec, str) else rec.get("cik")
            progress.set_postfix_str(ticker, refresh=False)
            try:
                company = ingest_company(session, ticker, cik=cik)
                company.ingest_state = STATE_PENDING
                session.commit()
            except ValueError:
                # Not an SEC filer — nothing to persist; skip quietly.
                session.rollback()
                logger.warning("Firm %s (cik=%s) not found on EDGAR; skipping", ticker, cik)
                final_states[ticker] = STATE_NO_DATA
                continue
            except Exception:
                session.rollback()
                logger.exception("Failed to resolve company %s; will retry next run", ticker)
                final_states[ticker] = STATE_FAILED
                continue

            resolved.append(company)
            filing_failed[company.id] = False
            for form in forms:
                # 10-Qs file ~quarterly, 10-Ks annually — size the per-form cap to the span.
                limit = span_years * 4 if form == "10-Q" else span_years + 1
                try:
                    ingest_filings(session, company, form=form, limit=limit)
                    session.commit()
                except Exception:
                    session.rollback()
                    filing_failed[company.id] = True
                    logger.exception("Failed to ingest %s filings for %s; continuing", form, ticker)

        # -- Phase 3: bulk price fetch for every resolved firm --------------
        price_states = ingest_prices_bulk(session, resolved, start=start_date, end=end_date)

        # -- Phase 4: record each firm's terminal state ---------------------
        for company in resolved:
            price_state = price_states.get(company.id, STATE_FAILED)
            if filing_failed[company.id] or price_state == STATE_FAILED:
                state, err = STATE_FAILED, "transient error during filings/prices"
            elif price_state == STATE_NO_DATA:
                state, err = STATE_NO_DATA, "no price data (delisted/illiquid)"
            else:
                state, err = STATE_COMPLETE, None
            company.ingest_state = state
            company.ingest_error = err
            final_states[company.ticker] = state
            session.commit()

        summary = {
            s: sum(1 for v in final_states.values() if v == s) for s in set(final_states.values())
        }
        logger.info("Universe ingestion complete: %s", summary)
    finally:
        session.close()

    return final_states


def ingest_market_benchmark(
    *, start: date | None = None, end: date | None = None, years: int = 8
) -> None:
    """Ingest the market benchmark (``SPY``) price series, prices only.

    Stores ``SPY`` as a price-only :class:`~core.models.Company` row (no filings)
    so :mod:`pipeline.returns` can compute forward returns *in excess of the
    market*. Idempotent: re-runs upsert prices and never duplicate the company.

    Args:
        start: Explicit window start (inclusive). Defaults to *years* before *end*.
        end: Explicit window end (inclusive). Defaults to today.
        years: Look-back used only when *start* is not given.
    """
    start_date, end_date = _resolve_window(start, end, years)

    session: Session = SessionLocal()
    try:
        benchmark: Optional[Company] = (
            session.query(Company).filter_by(ticker=MARKET_TICKER).first()
        )
        if benchmark is None:
            benchmark = Company(
                ticker=MARKET_TICKER,
                cik="0000884394",  # SPDR S&P 500 ETF Trust
                name="SPDR S&P 500 ETF Trust",
                sector="ETF",
            )
            session.add(benchmark)
            session.flush()
        ingest_prices(session, benchmark, start=start_date, end=end_date)
        session.commit()
        logger.info("Market benchmark %s prices ingested.", MARKET_TICKER)
    except Exception:
        session.rollback()
        logger.exception("Failed to ingest market benchmark %s", MARKET_TICKER)
    finally:
        session.close()
