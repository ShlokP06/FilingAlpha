"""Offline tests for forward returns, backtest, and walk-forward.

A synthetic in-memory SQLite database is built per test from the canonical ORM
models, so these tests need no Postgres and no network. The headline assertions
are the no-lookahead invariants:

* forward returns never enter on a price dated on/before the filing date;
* in walk-forward, every fold has ``max(train date) < min(test date)``.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.models import (
    BacktestRun,
    Company,
    Filing,
    ForwardReturn,
    ModelRun,
    Price,
    Signal,
)
from pipeline.backtest import (
    compute_event_study_spread,
    compute_ic,
    load_observations,
    run_backtest,
)
from pipeline.model import run_walkforward
from pipeline.returns import (
    compute_forward_returns,
    forward_return_for_series,
)


@pytest.fixture()
def session() -> Session:
    """Yield a session bound to a fresh in-memory SQLite DB with the schema."""
    engine = create_engine("sqlite:///:memory:", future=True)
    # Import Base from the same module hierarchy as the models.
    from core.models import Base

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    sess = factory()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


# --------------------------------------------------------------------------- #
# Pure forward-return helper: no-lookahead at the unit level
# --------------------------------------------------------------------------- #
def test_forward_return_enters_strictly_after_filing() -> None:
    """Entry is the first trading day strictly after the filing date."""
    dates = [date(2022, 1, d) for d in (3, 4, 5, 6, 7, 10, 11)]
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    # Filing on the 4th: entry must be the 5th (index 2), not the 4th.
    ret = forward_return_for_series(dates, closes, filing_date=date(2022, 1, 4), horizon=2)
    # entry close = 102 (the 5th), exit = 104 (the 7th, 2 trading days later).
    assert ret == pytest.approx(104.0 / 102.0 - 1.0)


def test_forward_return_filing_on_trading_day_excludes_that_day() -> None:
    """A filing dated on a trading day must not use that day's price as entry."""
    dates = [date(2022, 3, d) for d in (1, 2, 3, 4)]
    closes = [10.0, 20.0, 40.0, 80.0]
    # Filing on the 2nd (a trading day). Entry must be the 3rd (40.0), not 20.0.
    ret = forward_return_for_series(dates, closes, filing_date=date(2022, 3, 2), horizon=1)
    assert ret == pytest.approx(80.0 / 40.0 - 1.0)


def test_forward_return_none_when_insufficient_history() -> None:
    """Returns None when there is no entry day or not enough forward history."""
    dates = [date(2022, 1, 3), date(2022, 1, 4)]
    closes = [100.0, 101.0]
    # No trading day after the filing date.
    assert forward_return_for_series(dates, closes, date(2022, 1, 4), 1) is None
    # Entry exists but horizon overshoots the series.
    assert forward_return_for_series(dates, closes, date(2022, 1, 3), 5) is None


# --------------------------------------------------------------------------- #
# DB-level forward returns: no price on/before filing_date is ever the entry
# --------------------------------------------------------------------------- #
def _seed_prices(session: Session, company_id: int, start: date, n: int) -> list[date]:
    """Insert ``n`` consecutive daily prices and return their dates."""
    dates: list[date] = []
    for i in range(n):
        d = start + timedelta(days=i)
        session.add(Price(company_id=company_id, date=d, adj_close=100.0 + i))
        dates.append(d)
    session.commit()
    return dates


def test_compute_forward_returns_no_lookahead_invariant(session: Session) -> None:
    """The persisted forward return's implied entry price is strictly post-filing."""
    company = Company(ticker="AAA", cik="0000001", name="Alpha")
    session.add(company)
    session.commit()

    price_dates = _seed_prices(session, company.id, date(2022, 1, 1), 60)
    filing_date = date(2022, 1, 10)
    filing = Filing(
        company_id=company.id,
        form_type="10-K",
        filing_date=filing_date,
    )
    session.add(filing)
    session.commit()

    written = compute_forward_returns(session, horizons=(5,))
    assert written == 1

    fwd = session.query(ForwardReturn).filter_by(filing_id=filing.id, horizon_days=5).one()

    # Reconstruct the entry: first price strictly after the filing date.
    closes_by_date = {d: 100.0 + i for i, d in enumerate(price_dates)}
    entry_date = min(d for d in price_dates if d > filing_date)
    assert entry_date > filing_date  # the invariant, explicitly.
    entry_price = closes_by_date[entry_date]
    exit_date = entry_date + timedelta(days=5)
    expected = closes_by_date[exit_date] / entry_price - 1.0
    assert fwd.fwd_return == pytest.approx(expected)


def test_compute_forward_returns_idempotent(session: Session) -> None:
    """Re-running does not duplicate rows and overwrites in place."""
    company = Company(ticker="BBB", cik="0000002")
    session.add(company)
    session.commit()
    _seed_prices(session, company.id, date(2022, 1, 1), 60)
    session.add(Filing(company_id=company.id, form_type="10-K", filing_date=date(2022, 1, 10)))
    session.commit()

    compute_forward_returns(session, horizons=(5, 21))
    first = session.query(ForwardReturn).count()
    compute_forward_returns(session, horizons=(5, 21))
    second = session.query(ForwardReturn).count()
    assert first == second == 2


# --------------------------------------------------------------------------- #
# IC sanity: perfectly-correlated synthetic signal -> IC ~ 1
# --------------------------------------------------------------------------- #
def test_ic_of_perfectly_correlated_signal_is_one(session: Session) -> None:
    """A signal monotonically equal to forward return yields rank-IC ~ 1."""
    company = Company(ticker="CCC", cik="0000003")
    session.add(company)
    session.commit()

    n = 40
    base = date(2022, 1, 3)
    for i in range(n):
        fd = base + timedelta(days=i)
        filing = Filing(company_id=company.id, form_type="10-K", filing_date=fd)
        session.add(filing)
        session.flush()
        value = float(i)  # strictly increasing signal
        session.add(
            Signal(
                filing_id=filing.id,
                company_id=company.id,
                filing_date=fd,
                lm_negative=value,
            )
        )
        # Forward return strictly increasing in the same order => Spearman = 1.
        session.add(ForwardReturn(filing_id=filing.id, horizon_days=21, fwd_return=value / 100.0))
    session.commit()

    frame = load_observations(session, "lm_negative", 21)
    assert len(frame) == n
    ic, tstat, method = compute_ic(frame)
    assert ic == pytest.approx(1.0, abs=1e-9)
    # Pooled fallback expected here (one obs per filing date).
    assert method == "pooled"

    result = run_backtest(session, "lm_negative", 21, cost_bps=10.0)
    assert result.ic == pytest.approx(1.0, abs=1e-9)
    assert session.query(BacktestRun).count() == 1


def test_ic_cross_sectional_mode_when_breadth_exists(session: Session) -> None:
    """With many firms per date, IC uses the cross-sectional method."""
    companies = []
    for c in range(10):
        comp = Company(ticker=f"X{c:02d}", cik=f"{c:07d}")
        session.add(comp)
        companies.append(comp)
    session.commit()

    base = date(2022, 1, 3)
    rng = np.random.default_rng(0)
    for p in range(8):  # 8 filing dates
        fd = base + timedelta(days=p * 30)
        for comp in companies:
            filing = Filing(company_id=comp.id, form_type="10-K", filing_date=fd)
            session.add(filing)
            session.flush()
            val = rng.normal()
            session.add(
                Signal(
                    filing_id=filing.id,
                    company_id=comp.id,
                    filing_date=fd,
                    lm_negative=val,
                )
            )
            session.add(
                ForwardReturn(
                    filing_id=filing.id, horizon_days=21, fwd_return=val + rng.normal() * 0.01
                )
            )
    session.commit()

    frame = load_observations(session, "lm_negative", 21)
    _, _, method = compute_ic(frame)
    assert method == "cross_sectional"


# --------------------------------------------------------------------------- #
# Event-study tercile spread
# --------------------------------------------------------------------------- #
def test_event_study_spread_positive_when_signal_predicts_returns() -> None:
    """A signal monotonically aligned with forward returns yields a positive spread."""
    n = 30
    signal = np.arange(n, dtype=float)
    frame = pd.DataFrame({"signal": signal, "fwd_return": signal / 100.0})
    spread, tstat, n_long, n_short = compute_event_study_spread(frame, cost_bps=0.0)
    assert spread > 0.0
    assert tstat > 0.0
    assert n_long == n_short == n // 3


def test_event_study_spread_charges_round_trip_cost() -> None:
    """The spread is reduced by twice the per-side cost."""
    n = 30
    signal = np.arange(n, dtype=float)
    frame = pd.DataFrame({"signal": signal, "fwd_return": signal / 100.0})
    gross, _, _, _ = compute_event_study_spread(frame, cost_bps=0.0)
    net, _, _, _ = compute_event_study_spread(frame, cost_bps=10.0)
    assert net == pytest.approx(gross - 2.0 * 10.0 / 1e4)


def test_event_study_spread_too_few_events_returns_zeros() -> None:
    """Fewer than two observations per tercile yields a neutral, zeroed result."""
    frame = pd.DataFrame({"signal": [1.0, 2.0, 3.0], "fwd_return": [0.0, 0.1, 0.2]})
    assert compute_event_study_spread(frame, cost_bps=10.0) == (0.0, 0.0, 0, 0)


# --------------------------------------------------------------------------- #
# Walk-forward: temporal no-lookahead invariant
# --------------------------------------------------------------------------- #
def test_walkforward_train_dates_precede_test_dates(session: Session) -> None:
    """Every fold satisfies max(train filing_date) < min(test filing_date)."""
    company = Company(ticker="DDD", cik="0000004")
    session.add(company)
    session.commit()

    n = 120
    base = date(2018, 1, 1)
    rng = np.random.default_rng(7)
    for i in range(n):
        fd = base + timedelta(days=i * 7)
        filing = Filing(company_id=company.id, form_type="10-K", filing_date=fd)
        session.add(filing)
        session.flush()
        feats = rng.normal(size=6)
        session.add(
            Signal(
                filing_id=filing.id,
                company_id=company.id,
                filing_date=fd,
                lm_negative=feats[0],
                lm_uncertainty=feats[1],
                lm_litigious=feats[2],
                yoy_similarity=feats[3],
                risk_factor_delta=feats[4],
                fog_readability=feats[5],
            )
        )
        # Label depends on a feature so the model can learn something.
        ret = 0.01 if feats[0] + rng.normal() * 0.1 > 0 else -0.01
        session.add(ForwardReturn(filing_id=filing.id, horizon_days=21, fwd_return=ret))
    session.commit()

    result = run_walkforward(session, horizon=21, n_folds=5)

    assert result.n_folds >= 1
    assert result.n_oos > 0
    # The headline invariant, checked on the reported fold boundaries.
    for fold in result.fold_boundaries:
        assert fold["train_max_date"] < fold["test_min_date"]

    assert 0.0 <= result.oos_accuracy <= 1.0
    assert session.query(ModelRun).count() == 1


def test_walkforward_no_overlap_between_folds(session: Session) -> None:
    """Consecutive test folds are contiguous and non-overlapping in time."""
    company = Company(ticker="EEE", cik="0000005")
    session.add(company)
    session.commit()

    n = 90
    base = date(2019, 1, 1)
    rng = np.random.default_rng(3)
    for i in range(n):
        fd = base + timedelta(days=i * 5)
        filing = Filing(company_id=company.id, form_type="10-K", filing_date=fd)
        session.add(filing)
        session.flush()
        feats = rng.normal(size=6)
        session.add(
            Signal(
                filing_id=filing.id,
                company_id=company.id,
                filing_date=fd,
                lm_negative=feats[0],
                lm_uncertainty=feats[1],
                lm_litigious=feats[2],
                yoy_similarity=feats[3],
                risk_factor_delta=feats[4],
                fog_readability=feats[5],
            )
        )
        ret = 0.01 if feats[1] > 0 else -0.01
        session.add(ForwardReturn(filing_id=filing.id, horizon_days=21, fwd_return=ret))
    session.commit()

    result = run_walkforward(session, horizon=21, n_folds=4, persist=False)
    boundaries = result.fold_boundaries
    for prev, nxt in zip(boundaries, boundaries[1:]):
        # Next fold's test window starts after the previous fold's test window.
        assert prev["test_max_date"] <= nxt["test_min_date"]
