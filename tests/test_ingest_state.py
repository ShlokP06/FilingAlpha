"""Offline tests for the ingestion state machine (bulk prices + terminal states).

Verifies the "fool-proof to 300" guarantees without any network or Postgres:

* ``_extract_close`` pulls the right series out of bulk / flat yfinance frames.
* A rate-limited (transient) price fetch yields ``failed`` (retried next run),
  a clean-empty fetch yields ``no_data`` (terminal), and real data yields
  ``complete``.
* ``ingest_prices_bulk`` re-probes tickers that come back empty from the bulk
  call, so a throttled ticker is never mistaken for a delisted one.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.models import Base, Company, Price
from pipeline import ingest
from pipeline.net import STATE_COMPLETE, STATE_FAILED, STATE_NO_DATA


@pytest.fixture()
def session() -> Session:
    """In-memory SQLite session with the full schema."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    s = SessionLocal()
    yield s
    s.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def _no_real_session(monkeypatch):
    """Never build a real cached yfinance session, and never sleep between chunks."""
    monkeypatch.setattr(ingest, "_yf_session", lambda: None)
    monkeypatch.setattr(ingest.time, "sleep", lambda *_: None)


def _company(session: Session, ticker: str) -> Company:
    co = Company(ticker=ticker, cik="1", name=ticker, sector="Tech")
    session.add(co)
    session.flush()
    return co


def _flat_df(dates: list[str], closes: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="Date")
    return pd.DataFrame({"Close": closes}, index=idx)


def _bulk_df(tickers: list[str], dates: list[str], closes: dict[str, list[float]]) -> pd.DataFrame:
    """Build a group_by='ticker' MultiIndex frame; NaN closes mark a failed ticker."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="Date")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([tickers, fields])
    frame = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for t in tickers:
        frame[(t, "Close")] = closes.get(t, [float("nan")] * len(dates))
    return frame


# _extract_close


class TestExtractClose:
    def test_flat_frame(self):
        df = _flat_df(["2024-01-02", "2024-01-03"], [10.0, 11.0])
        series = ingest._extract_close(df, "AAA")
        assert list(series) == [10.0, 11.0]

    def test_multiindex_frame(self):
        df = _bulk_df(["AAA", "BBB"], ["2024-01-02"], {"AAA": [10.0], "BBB": [20.0]})
        assert ingest._extract_close(df, "BBB").iloc[0] == 20.0

    def test_missing_ticker_returns_none(self):
        df = _bulk_df(["AAA"], ["2024-01-02"], {"AAA": [10.0]})
        assert ingest._extract_close(df, "ZZZ") is None

    def test_empty_frame_returns_none(self):
        assert ingest._extract_close(pd.DataFrame(), "AAA") is None


# _probe_prices_one — the transient-vs-empty classifier


class TestProbePricesOne:
    def test_transient_error_marks_failed(self, session, monkeypatch):
        co = _company(session, "AAA")

        def boom(*_a, **_k):
            raise ConnectionError("rate limited")

        monkeypatch.setattr(ingest, "_fetch_history", boom)
        assert (
            ingest._probe_prices_one(session, co, date(2024, 1, 1), date(2024, 2, 1))
            == STATE_FAILED
        )
        assert session.query(Price).count() == 0

    def test_empty_marks_no_data(self, session, monkeypatch):
        co = _company(session, "AAA")
        monkeypatch.setattr(ingest, "_fetch_history", lambda *_a, **_k: pd.DataFrame())
        state = ingest._probe_prices_one(session, co, date(2024, 1, 1), date(2024, 2, 1))
        assert state == STATE_NO_DATA
        assert session.query(Price).count() == 0

    def test_data_marks_complete_and_inserts(self, session, monkeypatch):
        co = _company(session, "AAA")
        df = _flat_df(["2024-01-02", "2024-01-03"], [10.0, 11.0])
        monkeypatch.setattr(ingest, "_fetch_history", lambda *_a, **_k: df)
        state = ingest._probe_prices_one(session, co, date(2024, 1, 1), date(2024, 2, 1))
        assert state == STATE_COMPLETE
        assert session.query(Price).count() == 2


# ingest_prices_bulk — bulk fast path + per-ticker re-probe


class TestIngestPricesBulk:
    def test_bulk_data_marks_complete_without_probe(self, session, monkeypatch):
        aaa, bbb = _company(session, "AAA"), _company(session, "BBB")
        df = _bulk_df(["AAA", "BBB"], ["2024-01-02"], {"AAA": [10.0], "BBB": [20.0]})
        monkeypatch.setattr(ingest, "_download_chunk", lambda *_a, **_k: df)
        # If the probe were called, this would raise — asserting we didn't need it.
        monkeypatch.setattr(
            ingest, "_probe_prices_one", MagicMock(side_effect=AssertionError("should not probe"))
        )

        out = ingest.ingest_prices_bulk(session, [aaa, bbb], date(2024, 1, 1), date(2024, 2, 1))
        assert out == {aaa.id: STATE_COMPLETE, bbb.id: STATE_COMPLETE}
        assert session.query(Price).count() == 2

    def test_empty_ticker_is_reprobed(self, session, monkeypatch):
        """A ticker that is all-NaN in the bulk frame must be re-probed individually."""
        aaa, bbb = _company(session, "AAA"), _company(session, "BBB")
        # AAA has data; BBB is all-NaN in the bulk result.
        df = _bulk_df(["AAA", "BBB"], ["2024-01-02"], {"AAA": [10.0]})
        monkeypatch.setattr(ingest, "_download_chunk", lambda *_a, **_k: df)
        probe = MagicMock(return_value=STATE_NO_DATA)
        monkeypatch.setattr(ingest, "_probe_prices_one", probe)

        out = ingest.ingest_prices_bulk(session, [aaa, bbb], date(2024, 1, 1), date(2024, 2, 1))
        assert out[aaa.id] == STATE_COMPLETE
        assert out[bbb.id] == STATE_NO_DATA
        probe.assert_called_once()  # only the empty ticker was probed

    def test_bulk_chunk_failure_marks_failed_without_probe(self, session, monkeypatch):
        """A transient chunk failure marks every firm ``failed`` WITHOUT per-ticker probing.

        Re-probing each ticker when the whole chunk is already throttled would only
        amplify load on Yahoo, so the firms are simply queued for the next run.
        """
        aaa, bbb = _company(session, "AAA"), _company(session, "BBB")

        def boom(*_a, **_k):
            raise ConnectionError("chunk throttled")

        monkeypatch.setattr(ingest, "_download_chunk", boom)
        probe = MagicMock(side_effect=AssertionError("must not probe on a chunk failure"))
        monkeypatch.setattr(ingest, "_probe_prices_one", probe)

        out = ingest.ingest_prices_bulk(session, [aaa, bbb], date(2024, 1, 1), date(2024, 2, 1))
        assert out == {aaa.id: STATE_FAILED, bbb.id: STATE_FAILED}
        probe.assert_not_called()

    def test_circuit_breaker_stops_after_consecutive_failures(self, session, monkeypatch):
        """After yf_max_failed_chunks consecutive throttles, no further calls are made."""
        firms = [_company(session, f"T{i}") for i in range(5)]
        monkeypatch.setattr(ingest.settings, "yf_chunk_size", 1)
        monkeypatch.setattr(ingest.settings, "yf_max_failed_chunks", 2)

        calls = {"n": 0}

        def boom(*_a, **_k):
            calls["n"] += 1
            raise ConnectionError("chunk throttled")

        monkeypatch.setattr(ingest, "_download_chunk", boom)
        monkeypatch.setattr(
            ingest, "_probe_prices_one", MagicMock(side_effect=AssertionError("must not probe"))
        )

        out = ingest.ingest_prices_bulk(session, firms, date(2024, 1, 1), date(2024, 2, 1))

        # All five firms failed, but the network was touched only until the breaker tripped.
        assert out == {f.id: STATE_FAILED for f in firms}
        assert calls["n"] == 2
