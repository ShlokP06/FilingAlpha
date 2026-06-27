"""Pydantic schemas — the API request/response contract.

``from_attributes`` lets these be built directly from ORM rows.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class CompanyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticker: str
    cik: str
    name: str | None = None
    sector: str | None = None


class SignalPoint(BaseModel):
    """One filing's signal values, plus the contemporaneous price for charting."""

    model_config = ConfigDict(from_attributes=True)

    filing_date: date
    fiscal_period: str | None = None
    lm_negative: float | None = None
    lm_uncertainty: float | None = None
    lm_litigious: float | None = None
    yoy_similarity: float | None = None
    risk_factor_delta: float | None = None
    fog_readability: float | None = None


class SignalSeries(BaseModel):
    ticker: str
    points: list[SignalPoint]


class BacktestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    signal: str
    horizon_days: int
    ic: float | None = None
    ic_tstat: float | None = None
    ls_spread: float | None = None
    spread_tstat: float | None = None
    created_at: datetime | None = None


class ModelRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_type: str
    features_json: str | None = None
    metrics_json: str | None = None
    created_at: datetime | None = None
