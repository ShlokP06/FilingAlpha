"""Offline tests for the per-filing signal orchestration (SQLite, no network)."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.models import Base, Company, Filing, Signal
from pipeline.compute_signals import compute_signals

_CURR = "the company faces moderate risks and stable conditions across its markets " * 25
_PREV = "the company faces significant risks and uncertainty in litigation worldwide " * 25


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine)
    sess = maker()
    company = Company(ticker="TST", cik="1", name="Test Co")
    sess.add(company)
    sess.commit()
    sess.add(
        Filing(
            company_id=company.id,
            form_type="10-K",
            filing_date=date(2022, 2, 1),
            item1a_text=_PREV,
            mdna_text=_PREV,
        )
    )
    sess.add(
        Filing(
            company_id=company.id,
            form_type="10-K",
            filing_date=date(2023, 2, 1),
            item1a_text=_CURR,
            mdna_text=_CURR,
        )
    )
    sess.commit()
    yield sess
    sess.close()


def test_first_filing_has_no_yoy_signals(session):
    compute_signals(session)
    first = session.query(Signal).join(Filing).filter(Filing.filing_date == date(2022, 2, 1)).one()
    assert first.yoy_similarity is None
    assert first.risk_factor_delta is None
    assert first.fog_readability is not None


def test_second_filing_has_yoy_signals_in_range(session):
    compute_signals(session)
    second = session.query(Signal).join(Filing).filter(Filing.filing_date == date(2023, 2, 1)).one()
    assert 0.0 <= second.yoy_similarity <= 1.0
    assert 0.0 <= second.risk_factor_delta <= 1.0


def test_idempotent(session):
    compute_signals(session)
    compute_signals(session)
    assert session.query(Signal).count() == 2
