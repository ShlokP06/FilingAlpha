"""Offline unit tests for pipeline/ingest.py.

All tests use an in-memory SQLite engine — no Postgres, no network.
edgartools and yfinance are monkeypatched so these tests remain green in CI
without any external dependencies.

Tests that genuinely require a live Postgres + network are marked with
``@pytest.mark.integration`` and are skipped by the default ``uv run pytest``
invocation.
"""

from __future__ import annotations

import math
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.models import Base, Company, Filing, Price

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_engine():
    """In-memory SQLite engine with the full FilingAlpha schema created."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def session(sqlite_engine) -> Session:
    """Transactional SQLAlchemy session bound to the SQLite engine."""
    SessionLocal = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    s = SessionLocal()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Helpers for building fake edgar / yfinance objects
# ---------------------------------------------------------------------------


def _make_edgar_company(
    ticker: str = "AAPL",
    cik: int = 320193,
    name: str = "Apple Inc.",
    industry: str = "Electronic Computers",
    not_found: bool = False,
) -> MagicMock:
    """Return a MagicMock that mimics edgar.Company."""
    mock = MagicMock()
    mock.not_found = not_found
    mock.cik = cik
    mock.name = name
    mock.industry = industry
    return mock


def _make_entity_filing(
    accession_no: str = "0000320193-24-000001",
    filing_date: str = "2024-11-01",
    report_date: str = "2024-09-28",
    form: str = "10-K",
) -> MagicMock:
    """Return a MagicMock that mimics an edgar EntityFiling row."""
    ef = MagicMock()
    ef.accession_no = accession_no
    ef.filing_date = filing_date
    ef.report_date = report_date
    ef.form = form
    # obj() returns a mock report exposing the 10-K section properties
    report = MagicMock()
    report.risk_factors = "<Item 1A text>"
    report.management_discussion = "<Item 7 text>"
    ef.obj.return_value = report
    # text() returns a short dummy string
    ef.text.return_value = "Full filing text content."
    return ef


def _make_entity_filings(*entity_filings: MagicMock) -> MagicMock:
    """Return a MagicMock whose .head(n) returns an iterable of EntityFiling mocks."""
    filings_container = MagicMock()
    filings_container.head.return_value = list(entity_filings)
    return filings_container


def _make_price_df(
    dates: list[str],
    closes: list[float],
) -> pd.DataFrame:
    """Build a minimal DataFrame matching the yfinance auto_adjust=True layout."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="Date")
    df = pd.DataFrame({"Close": closes}, index=idx)
    return df


# ---------------------------------------------------------------------------
# Tests: ingest_company
# ---------------------------------------------------------------------------


class TestIngestCompany:
    def test_inserts_new_company(self, session: Session, monkeypatch):
        """ingest_company should insert a new Company row when ticker is absent."""
        from pipeline import ingest

        mock_co = _make_edgar_company()
        monkeypatch.setattr(ingest, "edgar", MagicMock(Company=lambda t: mock_co))

        company = ingest.ingest_company(session, "AAPL")
        session.commit()

        assert company.ticker == "AAPL"
        assert company.cik == "320193"
        assert company.name == "Apple Inc."
        assert company.sector == "Electronic Computers"
        assert session.query(Company).count() == 1

    def test_upserts_existing_company(self, session: Session, monkeypatch):
        """ingest_company called twice for the same ticker should not duplicate rows."""
        from pipeline import ingest

        # First call — original values
        mock_co1 = _make_edgar_company(name="Apple Inc.", industry="Electronic Computers")
        monkeypatch.setattr(ingest, "edgar", MagicMock(Company=lambda t: mock_co1))
        ingest.ingest_company(session, "AAPL")
        session.commit()

        # Second call — updated name/industry
        mock_co2 = _make_edgar_company(name="Apple Inc. (Updated)", industry="Tech Hardware")
        monkeypatch.setattr(ingest, "edgar", MagicMock(Company=lambda t: mock_co2))
        ingest.ingest_company(session, "AAPL")
        session.commit()

        rows = session.query(Company).all()
        assert len(rows) == 1, "Expected exactly one row after two upserts"
        assert rows[0].name == "Apple Inc. (Updated)"
        assert rows[0].sector == "Tech Hardware"

    def test_raises_on_ticker_not_found(self, session: Session, monkeypatch):
        """ingest_company should raise ValueError for unknown tickers."""
        from pipeline import ingest

        mock_co = _make_edgar_company(not_found=True)
        monkeypatch.setattr(ingest, "edgar", MagicMock(Company=lambda t: mock_co))

        with pytest.raises(ValueError, match="not found on EDGAR"):
            ingest.ingest_company(session, "ZZZZ")

    def test_sector_none_when_industry_missing(self, session: Session, monkeypatch):
        """ingest_company should accept None sector without crashing."""
        from pipeline import ingest

        mock_co = _make_edgar_company(industry=None)
        monkeypatch.setattr(ingest, "edgar", MagicMock(Company=lambda t: mock_co))

        company = ingest.ingest_company(session, "AAPL")
        session.commit()
        assert company.sector is None


# ---------------------------------------------------------------------------
# Tests: ingest_filings
# ---------------------------------------------------------------------------


def _patch_settings(monkeypatch, tmp_path) -> None:
    """Patch pipeline.ingest.settings so filings_text_dir() returns tmp_path.

    ``settings`` is a frozen Pydantic model; we cannot setattr a method on
    it.  Instead we replace the module-level ``settings`` reference with a
    MagicMock that delegates every attribute access to the real settings
    object except ``filings_text_dir``.
    """
    from pipeline import ingest

    mock_settings = MagicMock(wraps=ingest.settings)
    mock_settings.filings_text_dir = lambda: tmp_path
    monkeypatch.setattr(ingest, "settings", mock_settings)


class TestIngestFilings:
    def _make_company(self, session: Session, ticker: str = "AAPL") -> Company:
        """Insert and return a minimal Company fixture."""
        co = Company(ticker=ticker, cik="320193", name="Apple Inc.", sector="Tech")
        session.add(co)
        session.flush()
        return co

    def test_inserts_filings(self, session: Session, monkeypatch, tmp_path):
        """ingest_filings should insert one row per unique filing."""
        from pipeline import ingest

        _patch_settings(monkeypatch, tmp_path)

        ef1 = _make_entity_filing(
            accession_no="0000320193-24-000001",
            filing_date="2024-11-01",
            report_date="2024-09-28",
        )
        ef2 = _make_entity_filing(
            accession_no="0000320193-23-000001",
            filing_date="2023-11-03",
            report_date="2023-09-30",
        )

        edgar_mock = MagicMock()
        edgar_mock.Company.return_value = _make_edgar_company()
        edgar_mock.Company.return_value.get_filings.return_value = _make_entity_filings(ef1, ef2)
        monkeypatch.setattr(ingest, "edgar", edgar_mock)

        company = self._make_company(session)
        count = ingest.ingest_filings(session, company, limit=6)
        session.commit()

        assert count == 2
        rows = session.query(Filing).all()
        assert len(rows) == 2

    def test_idempotent_no_duplicate_on_second_run(self, session: Session, monkeypatch, tmp_path):
        """Running ingest_filings twice for the same filing should not duplicate rows."""
        from pipeline import ingest

        _patch_settings(monkeypatch, tmp_path)

        ef = _make_entity_filing(
            accession_no="0000320193-24-000001",
            filing_date="2024-11-01",
            report_date="2024-09-28",
        )

        def make_edgar_mock():
            edgar_mock = MagicMock()
            edgar_mock.Company.return_value = _make_edgar_company()
            edgar_mock.Company.return_value.get_filings.return_value = _make_entity_filings(ef)
            return edgar_mock

        company = self._make_company(session)

        monkeypatch.setattr(ingest, "edgar", make_edgar_mock())
        ingest.ingest_filings(session, company, limit=6)
        session.commit()

        monkeypatch.setattr(ingest, "edgar", make_edgar_mock())
        count2 = ingest.ingest_filings(session, company, limit=6)
        session.commit()

        assert count2 == 0, "Second run should skip the already-inserted filing"
        assert session.query(Filing).count() == 1

    def test_section_extraction_failure_does_not_crash(
        self, session: Session, monkeypatch, tmp_path
    ):
        """If obj() raises, the filing is still inserted with None sections."""
        from pipeline import ingest

        _patch_settings(monkeypatch, tmp_path)

        ef = _make_entity_filing()
        ef.obj.side_effect = RuntimeError("HTML parse error")
        ef.text.return_value = None  # also fails to fetch text

        edgar_mock = MagicMock()
        edgar_mock.Company.return_value = _make_edgar_company()
        edgar_mock.Company.return_value.get_filings.return_value = _make_entity_filings(ef)
        monkeypatch.setattr(ingest, "edgar", edgar_mock)

        company = self._make_company(session)
        count = ingest.ingest_filings(session, company, limit=6)
        session.commit()

        assert count == 1
        row = session.query(Filing).one()
        assert row.item1a_text is None
        assert row.mdna_text is None

    def test_fiscal_period_derived_from_report_date(self, session: Session, monkeypatch, tmp_path):
        """fiscal_period should be 'FY<year>' derived from report_date."""
        from pipeline import ingest

        _patch_settings(monkeypatch, tmp_path)

        ef = _make_entity_filing(report_date="2023-09-30")

        edgar_mock = MagicMock()
        edgar_mock.Company.return_value = _make_edgar_company()
        edgar_mock.Company.return_value.get_filings.return_value = _make_entity_filings(ef)
        monkeypatch.setattr(ingest, "edgar", edgar_mock)

        company = self._make_company(session)
        ingest.ingest_filings(session, company, limit=6)
        session.commit()

        row = session.query(Filing).one()
        assert row.fiscal_period == "FY2023"
        assert row.period_end == date(2023, 9, 30)

    def test_text_cached_to_disk(self, session: Session, monkeypatch, tmp_path):
        """Filing text should be cached to disk and the path stored in text_path."""
        from pipeline import ingest

        _patch_settings(monkeypatch, tmp_path)

        ef = _make_entity_filing(accession_no="0000320193-24-000001")
        ef.text.return_value = "Sample full text content."

        edgar_mock = MagicMock()
        edgar_mock.Company.return_value = _make_edgar_company()
        edgar_mock.Company.return_value.get_filings.return_value = _make_entity_filings(ef)
        monkeypatch.setattr(ingest, "edgar", edgar_mock)

        company = self._make_company(session)
        ingest.ingest_filings(session, company, limit=6)
        session.commit()

        row = session.query(Filing).one()
        assert row.text_path is not None
        assert "AAPL" in row.text_path

    def test_returns_zero_when_edgar_not_found(self, session: Session, monkeypatch):
        """ingest_filings should return 0 cleanly when the company is not on EDGAR."""
        from pipeline import ingest

        edgar_mock = MagicMock()
        edgar_mock.Company.return_value = _make_edgar_company(not_found=True)
        monkeypatch.setattr(ingest, "edgar", edgar_mock)

        company = self._make_company(session)
        count = ingest.ingest_filings(session, company, limit=6)
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: ingest_prices
# ---------------------------------------------------------------------------


class TestIngestPrices:
    def _make_company(self, session: Session, ticker: str = "AAPL") -> Company:
        co = Company(ticker=ticker, cik="320193", name="Apple Inc.", sector="Tech")
        session.add(co)
        session.flush()
        return co

    def _patch_yf(self, monkeypatch, df: pd.DataFrame) -> None:
        """Patch yfinance Ticker.history to return *df*."""
        from pipeline import ingest

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        monkeypatch.setattr(ingest.yf, "Ticker", lambda t: mock_ticker)

    def test_inserts_prices(self, session: Session, monkeypatch):
        """ingest_prices should insert one Price row per trading day."""
        from pipeline import ingest

        df = _make_price_df(["2024-01-02", "2024-01-03", "2024-01-04"], [185.0, 186.5, 184.9])
        self._patch_yf(monkeypatch, df)

        company = self._make_company(session)
        count = ingest.ingest_prices(session, company, start=date(2024, 1, 1), end=date(2024, 1, 5))
        session.commit()

        assert count == 3
        assert session.query(Price).count() == 3

    def test_idempotent_no_duplicate_prices(self, session: Session, monkeypatch):
        """Calling ingest_prices twice for the same date range must not duplicate rows."""
        from pipeline import ingest

        df = _make_price_df(["2024-01-02", "2024-01-03"], [185.0, 186.5])
        self._patch_yf(monkeypatch, df)

        company = self._make_company(session)
        ingest.ingest_prices(session, company, start=date(2024, 1, 1), end=date(2024, 1, 5))
        session.commit()

        # Second call with same data
        self._patch_yf(monkeypatch, df)
        count2 = ingest.ingest_prices(
            session, company, start=date(2024, 1, 1), end=date(2024, 1, 5)
        )
        session.commit()

        assert count2 == 0, "Second run should skip already-inserted dates"
        assert session.query(Price).count() == 2

    def test_empty_dataframe_returns_zero(self, session: Session, monkeypatch):
        """ingest_prices should return 0 and not crash on an empty DataFrame."""
        from pipeline import ingest

        df = pd.DataFrame(columns=["Close"])
        self._patch_yf(monkeypatch, df)

        company = self._make_company(session)
        count = ingest.ingest_prices(session, company, start=date(2024, 1, 1), end=date(2024, 1, 5))
        assert count == 0

    def test_adj_close_column_fallback(self, session: Session, monkeypatch):
        """ingest_prices should handle 'Adj Close' column (older yfinance behaviour)."""
        from pipeline import ingest

        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="Date")
        df = pd.DataFrame({"Adj Close": [185.0]}, index=idx)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        monkeypatch.setattr(ingest.yf, "Ticker", lambda t: mock_ticker)

        company = self._make_company(session)
        count = ingest.ingest_prices(session, company, start=date(2024, 1, 1), end=date(2024, 1, 5))
        session.commit()

        assert count == 1
        assert session.query(Price).one().adj_close == pytest.approx(185.0)

    def test_nan_close_is_skipped(self, session: Session, monkeypatch):
        """NaN adj_close values should be silently skipped."""
        from pipeline import ingest

        df = _make_price_df(["2024-01-02", "2024-01-03"], [math.nan, 186.5])
        self._patch_yf(monkeypatch, df)

        company = self._make_company(session)
        count = ingest.ingest_prices(session, company, start=date(2024, 1, 1), end=date(2024, 1, 5))
        session.commit()

        assert count == 1
        assert session.query(Price).one().adj_close == pytest.approx(186.5)

    def test_price_values_stored_correctly(self, session: Session, monkeypatch):
        """The adj_close stored in DB should match the yfinance data exactly."""
        from pipeline import ingest

        df = _make_price_df(["2024-03-15"], [172.62])
        self._patch_yf(monkeypatch, df)

        company = self._make_company(session)
        ingest.ingest_prices(session, company, start=date(2024, 3, 14), end=date(2024, 3, 16))
        session.commit()

        row = session.query(Price).one()
        assert row.date == date(2024, 3, 15)
        assert row.adj_close == pytest.approx(172.62, rel=1e-5)


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_valid_date(self):
        from pipeline.ingest import _parse_date

        assert _parse_date("2024-11-01") == date(2024, 11, 1)

    def test_none_input(self):
        from pipeline.ingest import _parse_date

        assert _parse_date(None) is None

    def test_empty_string(self):
        from pipeline.ingest import _parse_date

        assert _parse_date("") is None

    def test_invalid_format_returns_none(self):
        from pipeline.ingest import _parse_date

        assert _parse_date("not-a-date") is None

    def test_datetime_prefix_truncated(self):
        """Should handle ISO datetimes by slicing to 10 chars."""
        from pipeline.ingest import _parse_date

        assert _parse_date("2024-11-01T00:00:00") == date(2024, 11, 1)


# ---------------------------------------------------------------------------
# Integration smoke tests (skipped by default — require postgres + network)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_live_ingest_company():
    """Verify ingest_company against the live EDGAR API and Postgres.

    Requires:
        - docker compose stack running (postgres)
        - network access to EDGAR
    """
    from core.db import SessionLocal

    session = SessionLocal()
    try:
        from pipeline.ingest import ingest_company

        company = ingest_company(session, "AAPL")
        session.commit()
        assert company.cik is not None
        assert company.name is not None
    finally:
        session.close()


@pytest.mark.integration
def test_live_ingest_prices():
    """Verify ingest_prices against the live Yahoo Finance API and Postgres."""
    from core.db import SessionLocal

    session = SessionLocal()
    try:
        from pipeline.ingest import ingest_company, ingest_prices

        company = ingest_company(session, "AAPL")
        session.commit()
        count = ingest_prices(
            session,
            company,
            start=date(2024, 1, 1),
            end=date(2024, 1, 10),
        )
        session.commit()
        assert count > 0
    finally:
        session.close()
