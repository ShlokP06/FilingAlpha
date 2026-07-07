"""Assemble the latest per-form results from the database into report dataclasses.

This is the only module in :mod:`reporting` that touches the database. Everything
downstream (plots, narrative, LaTeX) consumes these plain dataclasses, which keeps
those stages testable with synthetic data and no DB.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.models import BacktestRun, Company, Filing, ForwardReturn, ModelRun, Signal

logger = logging.getLogger(__name__)

# A two-sided t-stat threshold for flagging significance in the report. Nothing
# in this small-universe project is expected to clear it; the flag exists to make
# the honest "not significant" verdict explicit rather than buried.
SIGNIFICANT_T = 2.0
SUGGESTIVE_T = 1.5

# Human-readable labels for the signal columns, used in prose and tables so the
# report never exposes raw code identifiers (which also avoids LaTeX underscores).
SIGNAL_LABELS: dict[str, str] = {
    "lm_negative": "negative tone (Loughran-McDonald)",
    "lm_uncertainty": "uncertainty tone",
    "lm_litigious": "litigious tone",
    "yoy_similarity": "year-over-year filing similarity (Lazy Prices)",
    "risk_factor_delta": "risk-factor section change",
    "fog_readability": "Gunning-Fog readability",
}


def signal_label(signal: str) -> str:
    """Return the human-readable label for a signal column."""
    return SIGNAL_LABELS.get(signal, signal.replace("_", " "))


@dataclass(frozen=True)
class SignalResult:
    """One signal's backtest metrics at one horizon and form."""

    form: str
    signal: str
    horizon_days: int
    n_obs: int
    ic: float
    ic_tstat: float
    ls_spread: float
    spread_tstat: float

    @property
    def stars(self) -> str:
        """Significance marker on the long-short spread t-stat."""
        t = abs(self.spread_tstat)
        if t >= SIGNIFICANT_T:
            return "**"
        if t >= SUGGESTIVE_T:
            return "*"
        return ""


@dataclass(frozen=True)
class ModelResult:
    """Walk-forward out-of-sample metrics for one form and horizon."""

    form: str
    horizon_days: int
    n_oos: int
    n_folds: int
    oos_accuracy: float
    oos_auc: float | None
    feature_importances: dict[str, float]


@dataclass(frozen=True)
class EquityCurve:
    """Event-ordered cumulative long-short return for the headline signal."""

    form: str
    signal: str
    horizon_days: int
    dates: list[date]
    cumulative: list[float]


@dataclass
class ReportData:
    """Everything the report needs, already resolved from the database."""

    generated_at: date
    n_companies: int
    n_filings_by_form: dict[str, int]
    date_range: tuple[date | None, date | None]
    signal_results: list[SignalResult]
    model_results: list[ModelResult]
    equity_curve: EquityCurve | None = field(default=None)

    def headline(self) -> SignalResult | None:
        """Return the most economically meaningful 10-K result.

        Picks the 10-K signal/horizon with the largest absolute long-short
        spread t-stat, since that is the result the note leads with.
        """
        tens = [r for r in self.signal_results if r.form == "10-K" and r.n_obs > 0]
        if not tens:
            return None
        return max(tens, key=lambda r: abs(r.spread_tstat))


def _latest_backtests(session: Session) -> list[SignalResult]:
    """Return the most recent backtest row per ``(form, signal, horizon)``."""
    rows = session.execute(
        select(BacktestRun).order_by(BacktestRun.created_at.asc())
    ).scalars()

    # Later rows overwrite earlier ones for the same key, so iterating ascending
    # by ``created_at`` leaves the latest run per key in the dict.
    latest: dict[tuple[str, str, int], SignalResult] = {}
    for run in rows:
        config = json.loads(run.config_json) if run.config_json else {}
        form = config.get("form") or "all"
        n_obs = int(config.get("n_obs") or 0)
        key = (form, run.signal, run.horizon_days)
        latest[key] = SignalResult(
            form=form,
            signal=run.signal,
            horizon_days=run.horizon_days,
            n_obs=n_obs,
            ic=float(run.ic or 0.0),
            ic_tstat=float(run.ic_tstat or 0.0),
            ls_spread=float(run.ls_spread or 0.0),
            spread_tstat=float(run.spread_tstat or 0.0),
        )
    return list(latest.values())


def _latest_models(session: Session) -> list[ModelResult]:
    """Return the most recent model row per ``(form, horizon)``."""
    rows = session.execute(select(ModelRun).order_by(ModelRun.created_at.asc())).scalars()

    latest: dict[tuple[str, int], ModelResult] = {}
    for run in rows:
        metrics = json.loads(run.metrics_json) if run.metrics_json else {}
        form = metrics.get("form") or "all"
        horizon = int(metrics.get("horizon_days") or 0)
        latest[(form, horizon)] = ModelResult(
            form=form,
            horizon_days=horizon,
            n_oos=int(metrics.get("n_oos") or 0),
            n_folds=int(metrics.get("n_folds") or 0),
            oos_accuracy=float(metrics.get("oos_accuracy") or 0.0),
            oos_auc=(None if metrics.get("oos_auc") is None else float(metrics["oos_auc"])),
            feature_importances={
                k: float(v) for k, v in (metrics.get("feature_importances") or {}).items()
            },
        )
    return list(latest.values())


def _equity_curve(session: Session, result: SignalResult) -> EquityCurve | None:
    """Build an event-ordered cumulative long-short curve for a signal.

    Ranks every filing event of the result's form by the signal, marks the top
    tercile long (+1) and the bottom tercile short (-1), then orders events by
    filing date and accumulates the signed forward return. This mirrors the
    backtest's tercile construction and is labelled as such — it is a
    visualisation of the same spread, not a tradeable equity curve.

    Args:
        session: Active database session.
        result: The headline signal result to chart.

    Returns:
        An :class:`EquityCurve`, or ``None`` if there are too few events.
    """
    column = getattr(Signal, result.signal)
    rows = session.execute(
        select(Signal.filing_date, column, ForwardReturn.fwd_return)
        .join(ForwardReturn, ForwardReturn.filing_id == Signal.filing_id)
        .join(Filing, Filing.id == Signal.filing_id)
        .where(ForwardReturn.horizon_days == result.horizon_days)
        .where(Filing.form_type == result.form)
    ).all()

    events = [
        (d, float(s), float(r))
        for d, s, r in rows
        if s is not None and r is not None
    ]
    if len(events) < 6:
        return None

    ranked = sorted(events, key=lambda e: e[1])
    k = len(ranked) // 3
    if k < 2:
        return None
    bottom = {id(e) for e in ranked[:k]}
    top = {id(e) for e in ranked[-k:]}

    contributions: list[tuple[date, float]] = []
    for event in events:
        if id(event) in top:
            contributions.append((event[0], event[2]))
        elif id(event) in bottom:
            contributions.append((event[0], -event[2]))
    contributions.sort(key=lambda c: c[0])

    dates: list[date] = []
    cumulative: list[float] = []
    running = 0.0
    for d, contribution in contributions:
        running += contribution
        dates.append(d)
        cumulative.append(running)

    return EquityCurve(
        form=result.form,
        signal=result.signal,
        horizon_days=result.horizon_days,
        dates=dates,
        cumulative=cumulative,
    )


def build_report_data(session: Session) -> ReportData:
    """Resolve all report inputs from the database.

    Args:
        session: Active database session.

    Returns:
        A fully-populated :class:`ReportData`.
    """
    n_companies = int(
        session.execute(
            select(func.count()).select_from(Company).where(Company.sector != "ETF")
        ).scalar_one()
    )

    form_counts = session.execute(
        select(Filing.form_type, func.count()).group_by(Filing.form_type)
    ).all()
    n_filings_by_form = {form: int(count) for form, count in form_counts}

    date_range = session.execute(
        select(func.min(Filing.filing_date), func.max(Filing.filing_date))
    ).one()

    signal_results = _latest_backtests(session)
    model_results = _latest_models(session)

    # The current methodology evaluates each form separately. Drop any legacy
    # pooled ("all") rows left in the tables by pre-form-aware runs so the note
    # never mixes a stale pooled result in with the per-form ones.
    if any(r.form != "all" for r in signal_results):
        signal_results = [r for r in signal_results if r.form != "all"]
    if any(m.form != "all" for m in model_results):
        model_results = [m for m in model_results if m.form != "all"]

    data = ReportData(
        generated_at=date.today(),
        n_companies=n_companies,
        n_filings_by_form=n_filings_by_form,
        date_range=(date_range[0], date_range[1]),
        signal_results=signal_results,
        model_results=model_results,
    )

    headline = data.headline()
    if headline is not None:
        data.equity_curve = _equity_curve(session, headline)

    logger.info(
        "Report data: %d companies, %s filings, %d backtest rows, %d model rows.",
        n_companies,
        n_filings_by_form,
        len(signal_results),
        len(model_results),
    )
    return data


__all__ = [
    "ReportData",
    "SignalResult",
    "ModelResult",
    "EquityCurve",
    "build_report_data",
    "signal_label",
    "SIGNAL_LABELS",
    "SIGNIFICANT_T",
    "SUGGESTIVE_T",
]
