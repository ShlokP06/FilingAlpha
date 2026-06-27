"""SQLAlchemy ORM models — the canonical FilingAlpha data schema.

Pipeline writes these tables; the API reads them. Foreign keys and frequently
queried columns are indexed; unique constraints make ingestion idempotent.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    cik: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str | None] = mapped_column(String(256))
    sector: Mapped[str | None] = mapped_column(String(128))

    filings: Mapped[list["Filing"]] = relationship(back_populates="company")
    prices: Mapped[list["Price"]] = relationship(back_populates="company")


class Filing(Base):
    __tablename__ = "filings"
    __table_args__ = (UniqueConstraint("company_id", "form_type", "filing_date", name="uq_filing"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    form_type: Mapped[str] = mapped_column(String(16))
    filing_date: Mapped[date] = mapped_column(Date, index=True)
    fiscal_period: Mapped[str | None] = mapped_column(String(16))
    period_end: Mapped[date | None] = mapped_column(Date)
    text_path: Mapped[str | None] = mapped_column(String(512))  # cached full text on disk
    item1a_text: Mapped[str | None] = mapped_column(Text)  # Risk Factors
    mdna_text: Mapped[str | None] = mapped_column(Text)  # Management Discussion & Analysis

    company: Mapped["Company"] = relationship(back_populates="filings")
    signal: Mapped["Signal | None"] = relationship(back_populates="filing", uselist=False)


class Price(Base):
    __tablename__ = "prices"
    __table_args__ = (UniqueConstraint("company_id", "date", name="uq_price"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    adj_close: Mapped[float] = mapped_column(Float)

    company: Mapped["Company"] = relationship(back_populates="prices")


class Signal(Base):
    """One row of classical-NLP features per filing."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id"), unique=True, index=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    filing_date: Mapped[date] = mapped_column(Date, index=True)

    # Loughran-McDonald tone (fraction of words in each category)
    lm_negative: Mapped[float | None] = mapped_column(Float)
    lm_uncertainty: Mapped[float | None] = mapped_column(Float)
    lm_litigious: Mapped[float | None] = mapped_column(Float)
    # Lazy Prices: cosine similarity to the prior year's filing (low => big change)
    yoy_similarity: Mapped[float | None] = mapped_column(Float)
    # Risk-factor (Item 1A) change vs prior year (1 - cosine)
    risk_factor_delta: Mapped[float | None] = mapped_column(Float)
    # Gunning-Fog readability index
    fog_readability: Mapped[float | None] = mapped_column(Float)

    filing: Mapped["Filing"] = relationship(back_populates="signal")


class ForwardReturn(Base):
    __tablename__ = "forward_returns"
    __table_args__ = (UniqueConstraint("filing_id", "horizon_days", name="uq_fwd_return"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id"), index=True)
    horizon_days: Mapped[int] = mapped_column(Integer)
    fwd_return: Mapped[float] = mapped_column(Float)


class BacktestRun(Base):
    """Result of evaluating one signal at one horizon."""

    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    signal: Mapped[str] = mapped_column(String(64))
    horizon_days: Mapped[int] = mapped_column(Integer)
    config_json: Mapped[str | None] = mapped_column(Text)
    ic: Mapped[float | None] = mapped_column(Float)  # information coefficient
    ic_tstat: Mapped[float | None] = mapped_column(Float)
    # Event-study tercile spread: top-minus-bottom mean forward return (net of cost)
    ls_spread: Mapped[float | None] = mapped_column(Float)
    spread_tstat: Mapped[float | None] = mapped_column(Float)  # Welch t-stat of the spread


class ModelRun(Base):
    """Walk-forward out-of-sample metrics for the ML model."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    model_type: Mapped[str] = mapped_column(String(64))
    features_json: Mapped[str | None] = mapped_column(Text)
    metrics_json: Mapped[str | None] = mapped_column(Text)
