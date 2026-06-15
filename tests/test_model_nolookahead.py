"""Regression test: walk-forward must not split a shared filing_date across folds.

Many firms file on the same calendar date. A row-count fold boundary that lands
inside a same-date group would put one ``filing_date`` in both train and test,
violating the strict no-lookahead invariant. This test reproduces that condition
(every firm shares each year's filing date) and asserts the walk-forward both
runs and keeps every fold's training dates strictly before its test dates.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.models import Base, Company, Filing, ForwardReturn, Signal
from pipeline.model import run_walkforward


@pytest.fixture()
def session_with_shared_dates():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    sess = sessionmaker(bind=engine)()
    rng = np.random.default_rng(0)
    fid = 0
    for ci in range(6):
        company = Company(ticker=f"T{ci}", cik=str(ci))
        sess.add(company)
        sess.flush()
        for year in range(2019, 2024):
            fid += 1
            filing_date = date(year, 2, 1)  # identical across all 6 firms
            sess.add(
                Filing(id=fid, company_id=company.id, form_type="10-K", filing_date=filing_date)
            )
            sess.add(
                Signal(
                    filing_id=fid,
                    company_id=company.id,
                    filing_date=filing_date,
                    lm_negative=rng.random(),
                    lm_uncertainty=rng.random(),
                    lm_litigious=rng.random(),
                    yoy_similarity=rng.random(),
                    risk_factor_delta=rng.random(),
                    fog_readability=rng.random(),
                )
            )
            sess.add(ForwardReturn(filing_id=fid, horizon_days=21, fwd_return=float(rng.normal())))
    sess.commit()
    yield sess
    sess.close()


def test_walkforward_runs_with_shared_dates(session_with_shared_dates):
    result = run_walkforward(session_with_shared_dates, horizon=21, persist=False)
    assert result.n_folds >= 1
    assert result.n_oos > 0


def test_every_fold_train_precedes_test(session_with_shared_dates):
    result = run_walkforward(session_with_shared_dates, horizon=21, persist=False)
    for fold in result.fold_boundaries:
        assert fold["train_max_date"] < fold["test_min_date"]
