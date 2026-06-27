"""Offline tests for the FilingAlpha FastAPI read API.

Uses an in-memory SQLite database so no Postgres or Docker stack is required.
The ``get_session`` dependency is overridden with a SQLite-backed session so
every endpoint can be exercised against seeded fixture data.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# ---------------------------------------------------------------------------
# SQLite engine + session factory
#
# SQLite "":memory:"" databases are per-connection by default; using
# ``?check_same_thread=false`` alone is insufficient when SQLAlchemy opens
# multiple connections from the pool.  We force a single shared connection
# by using a static connection URL with the ``StaticPool`` so every session
# in the test process uses the same in-memory database.
# ---------------------------------------------------------------------------
from sqlalchemy.pool import StaticPool  # noqa: E402

from api.app.main import app
from core.db import get_session
from core.models import BacktestRun, Base, Company, Filing, ModelRun, Signal

SQLITE_URL = "sqlite://"
_engine = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)


def override_get_session():
    """Dependency override: yield a SQLite session instead of Postgres."""
    session: Session = _TestSession()
    try:
        yield session
    finally:
        session.close()


app.dependency_overrides[get_session] = override_get_session


# ---------------------------------------------------------------------------
# Schema + seed fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def create_schema():
    """Create all tables once for the test session."""
    Base.metadata.create_all(_engine)
    yield
    Base.metadata.drop_all(_engine)


@pytest.fixture(scope="session")
def seeded_db(create_schema):  # noqa: ANN001
    """Seed the SQLite database with representative fixture rows.

    Returns:
        None; the seeded data persists for the entire test session.
    """
    session: Session = _TestSession()

    # Two companies so we can test 404 logic separately
    apple = Company(ticker="AAPL", cik="0000320193", name="Apple Inc.", sector="Technology")
    oracle = Company(ticker="ORCL", cik="0001341439", name="Oracle Corp.", sector="Technology")
    session.add_all([apple, oracle])
    session.flush()

    # Filing + Signal for AAPL
    filing = Filing(
        company_id=apple.id,
        form_type="10-K",
        filing_date=date(2023, 10, 27),
        fiscal_period="FY2023",
    )
    session.add(filing)
    session.flush()

    signal = Signal(
        filing_id=filing.id,
        company_id=apple.id,
        filing_date=date(2023, 10, 27),
        lm_negative=0.04,
        lm_uncertainty=0.02,
        lm_litigious=0.01,
        yoy_similarity=0.85,
        risk_factor_delta=0.15,
        fog_readability=18.3,
    )
    session.add(signal)

    # Two backtest rows with different signals so filter tests are meaningful
    bt1 = BacktestRun(
        signal="lm_negative",
        horizon_days=30,
        ic=0.05,
        ic_tstat=2.1,
        ls_spread=0.012,
        spread_tstat=1.4,
        created_at=datetime(2024, 1, 15, 12, 0, 0),
    )
    bt2 = BacktestRun(
        signal="yoy_similarity",
        horizon_days=60,
        ic=0.08,
        ic_tstat=3.2,
        ls_spread=0.025,
        spread_tstat=2.3,
        created_at=datetime(2024, 2, 1, 9, 0, 0),
    )
    session.add_all([bt1, bt2])

    # One model run
    mr = ModelRun(
        model_type="GradientBoosting",
        features_json='["lm_negative","yoy_similarity"]',
        metrics_json='{"roc_auc": 0.62}',
        created_at=datetime(2024, 3, 1, 8, 0, 0),
    )
    session.add(mr)
    session.commit()
    session.close()


# ---------------------------------------------------------------------------
# TestClient
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def client(seeded_db):  # noqa: ANN001
    """Return a TestClient backed by the seeded SQLite database."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self, client: TestClient) -> None:
        """GET /health should return 200 with status=ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestCompanies:
    def test_returns_seeded_companies(self, client: TestClient) -> None:
        """GET /companies should include both seeded companies."""
        resp = client.get("/companies")
        assert resp.status_code == 200
        tickers = {c["ticker"] for c in resp.json()}
        assert "AAPL" in tickers
        assert "ORCL" in tickers

    def test_response_schema(self, client: TestClient) -> None:
        """Each company object must have the expected keys."""
        resp = client.get("/companies")
        company = next(c for c in resp.json() if c["ticker"] == "AAPL")
        assert company["cik"] == "0000320193"
        assert company["name"] == "Apple Inc."
        assert company["sector"] == "Technology"


class TestSignals:
    def test_returns_signal_series(self, client: TestClient) -> None:
        """GET /signals/AAPL should return a SignalSeries with one point."""
        resp = client.get("/signals/AAPL")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ticker"] == "AAPL"
        assert len(body["points"]) == 1
        point = body["points"][0]
        assert point["filing_date"] == "2023-10-27"
        assert point["fiscal_period"] == "FY2023"
        assert point["lm_negative"] == pytest.approx(0.04)
        assert point["fog_readability"] == pytest.approx(18.3)

    def test_case_insensitive_ticker(self, client: TestClient) -> None:
        """Lowercase ticker should be normalised to uppercase and found."""
        resp = client.get("/signals/aapl")
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "AAPL"

    def test_unknown_ticker_returns_404(self, client: TestClient) -> None:
        """GET /signals/UNKNOWN should return HTTP 404."""
        resp = client.get("/signals/UNKNOWN")
        assert resp.status_code == 404

    def test_known_company_no_signals_returns_empty_series(self, client: TestClient) -> None:
        """ORCL exists but has no signals — should return empty points list, not 404."""
        resp = client.get("/signals/ORCL")
        assert resp.status_code == 200
        assert resp.json()["points"] == []


class TestBacktests:
    def test_returns_all_backtests(self, client: TestClient) -> None:
        """GET /backtests without filters should return both seeded rows."""
        resp = client.get("/backtests")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_filter_by_signal(self, client: TestClient) -> None:
        """GET /backtests?signal=lm_negative should return one row."""
        resp = client.get("/backtests?signal=lm_negative")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["signal"] == "lm_negative"

    def test_filter_by_horizon_days(self, client: TestClient) -> None:
        """GET /backtests?horizon_days=60 should return one row."""
        resp = client.get("/backtests?horizon_days=60")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["horizon_days"] == 60

    def test_filter_both_params(self, client: TestClient) -> None:
        """Both signal and horizon_days filters applied together (AND semantics)."""
        resp = client.get("/backtests?signal=lm_negative&horizon_days=30")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        # Cross-filter: lm_negative at horizon 60 doesn't exist
        resp2 = client.get("/backtests?signal=lm_negative&horizon_days=60")
        assert resp2.status_code == 200
        assert len(resp2.json()) == 0

    def test_newest_first(self, client: TestClient) -> None:
        """Backtests should be returned newest-first by created_at."""
        resp = client.get("/backtests")
        dates = [r["created_at"] for r in resp.json()]
        assert dates == sorted(dates, reverse=True)


class TestPredictions:
    def test_returns_model_runs(self, client: TestClient) -> None:
        """GET /predictions should return the seeded model run."""
        resp = client.get("/predictions")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["model_type"] == "GradientBoosting"
        assert results[0]["features_json"] == '["lm_negative","yoy_similarity"]'


class TestMetrics:
    def test_metrics_endpoint_returns_exposition_text(self, client: TestClient) -> None:
        """GET /metrics should return Prometheus text exposition."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        # Prometheus exposition always starts with # HELP or a metric name
        assert b"http_request" in resp.content or b"# HELP" in resp.content

    def test_metrics_contains_request_counter(self, client: TestClient) -> None:
        """After hitting /health, the request counter should appear in /metrics."""
        client.get("/health")
        resp = client.get("/metrics")
        assert b"http_requests_total" in resp.content
