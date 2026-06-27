"""Signal backtesting: information coefficient and an event-study spread.

Evaluates one signal column at one horizon against filing-lagged forward
returns (computed in :mod:`pipeline.returns`). Two measures are produced and
persisted as a :class:`BacktestRun`:

* **Information Coefficient (IC):** Spearman rank correlation between the signal
  and the forward return, with a t-statistic
  ``t = ic * sqrt((n - 2) / (1 - ic ** 2))``.

* **Event-study tercile spread:** annual 10-K filings are sparse and scattered
  across the calendar, so almost no single filing date has enough firms to form
  a cross-sectional long-short portfolio — a per-date Sharpe is undefined. The
  appropriate unit of analysis is the *filing event*. We therefore sort **all**
  filing events by the signal, take the top and bottom terciles, and report the
  difference in their mean forward return (net of round-trip cost) together with
  a Welch two-sample t-statistic. This is the standard event-study construction
  and is honest about what the data can support.

**IC methodology — cross-sectional, averaged across filing dates, with a pooled
fallback.** The academically correct construction is a *cross-sectional* IC:
within each period rank signals across firms, correlate with that period's
forward returns, then average the per-period ICs (and t-test them across
periods). We do exactly this whenever periods have enough cross-sectional
breadth. Because 10-K filings are sparse and scattered across dates, many
filing dates contain only one or two observations where a rank correlation is
undefined; if too few periods qualify we fall back to a single **pooled**
rank-IC across all observations. The chosen mode is recorded in
``config_json`` so results are never silently conflated.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models import BacktestRun, ForwardReturn, Signal

logger = logging.getLogger(__name__)

SIGNAL_COLUMNS: frozenset[str] = frozenset(
    {
        "lm_negative",
        "lm_uncertainty",
        "lm_litigious",
        "yoy_similarity",
        "risk_factor_delta",
        "fog_readability",
    }
)

# Minimum observations a period needs to contribute a cross-sectional IC.
_MIN_PERIOD_OBS = 5
# Minimum qualifying periods before we trust cross-sectional averaging.
_MIN_PERIODS = 5
# Minimum events required to form top/bottom terciles of at least 2 each.
_MIN_SPREAD_OBS = 6


@dataclass(frozen=True)
class BacktestResult:
    """Backtest metrics for one signal / horizon."""

    signal: str
    horizon_days: int
    n_obs: int
    ic: float
    ic_tstat: float
    ls_spread: float  # top-minus-bottom tercile mean forward return (net of cost)
    spread_tstat: float  # Welch two-sample t-stat of the tercile spread
    ic_method: str


def load_observations(session: Session, signal_col: str, horizon: int) -> pd.DataFrame:
    """Load aligned (filing_date, signal, forward_return) observations.

    Args:
        session: Active database session.
        signal_col: Name of the signal column on :class:`Signal`.
        horizon: Forward horizon in trading days.

    Returns:
        DataFrame with columns ``filing_date``, ``signal``, ``fwd_return``,
        dropping rows with a null signal or forward return.
    """
    if signal_col not in SIGNAL_COLUMNS:
        raise ValueError(f"Unknown signal column: {signal_col!r}")

    column = getattr(Signal, signal_col)
    rows = session.execute(
        select(Signal.filing_date, column, ForwardReturn.fwd_return)
        .join(ForwardReturn, ForwardReturn.filing_id == Signal.filing_id)
        .where(ForwardReturn.horizon_days == horizon)
    ).all()

    frame = pd.DataFrame(rows, columns=["filing_date", "signal", "fwd_return"])
    frame = frame.dropna(subset=["signal", "fwd_return"]).reset_index(drop=True)
    return frame


def _ic_tstat(ic: float, n: int) -> float:
    """T-statistic of a correlation coefficient with ``n`` observations.

    Args:
        ic: Correlation coefficient.
        n: Number of observations.

    Returns:
        ``ic * sqrt((n - 2) / (1 - ic ** 2))``; ``0.0`` if undefined.
    """
    if n < 3 or not np.isfinite(ic) or abs(ic) >= 1.0:
        return 0.0
    return float(ic * math.sqrt((n - 2) / (1.0 - ic**2)))


def compute_ic(frame: pd.DataFrame) -> tuple[float, float, str]:
    """Compute the information coefficient and its t-stat.

    Prefers a cross-sectional IC averaged across filing dates with a t-stat on
    the per-period IC series; falls back to a pooled rank-IC when too few
    periods have enough cross-sectional breadth.

    Args:
        frame: Output of :func:`load_observations`.

    Returns:
        ``(ic, ic_tstat, method)`` where ``method`` is ``"cross_sectional"`` or
        ``"pooled"``.
    """
    if len(frame) < 3:
        return 0.0, 0.0, "pooled"

    period_ics: list[float] = []
    for _, group in frame.groupby("filing_date"):
        if len(group) < _MIN_PERIOD_OBS:
            continue
        if group["signal"].nunique() < 2 or group["fwd_return"].nunique() < 2:
            continue
        rho, _ = spearmanr(group["signal"], group["fwd_return"])
        if np.isfinite(rho):
            period_ics.append(float(rho))

    if len(period_ics) >= _MIN_PERIODS:
        ics = np.asarray(period_ics, dtype=float)
        mean_ic = float(ics.mean())
        n_periods = len(ics)
        std = float(ics.std(ddof=1)) if n_periods > 1 else 0.0
        if std > 0:
            tstat = float(mean_ic / (std / math.sqrt(n_periods)))
        else:
            tstat = 0.0
        return mean_ic, tstat, "cross_sectional"

    # Pooled fallback.
    if frame["signal"].nunique() < 2 or frame["fwd_return"].nunique() < 2:
        return 0.0, 0.0, "pooled"
    rho, _ = spearmanr(frame["signal"], frame["fwd_return"])
    rho = float(rho) if np.isfinite(rho) else 0.0
    return rho, _ic_tstat(rho, len(frame)), "pooled"


def compute_event_study_spread(
    frame: pd.DataFrame, cost_bps: float
) -> tuple[float, float, int, int]:
    """Event-study tercile spread across all filing events.

    Treats each filing as an independent event (the correct unit for sparse
    annual filings, where a per-rebalance-date portfolio is undefined). Sorts
    every event by the signal, takes the top and bottom terciles, and reports
    the difference in mean forward return — long the top tercile, short the
    bottom — net of a round-trip transaction cost, together with a Welch
    two-sample t-statistic between the two terciles' returns.

    Args:
        frame: Output of :func:`load_observations`.
        cost_bps: Per-side transaction cost in basis points.

    Returns:
        ``(ls_spread, spread_tstat, n_long, n_short)``. The spread and t-stat
        are ``0.0`` and the counts ``0`` when there are too few events to form
        terciles of at least two observations each.
    """
    cost = cost_bps / 1e4
    n = len(frame)
    if n < _MIN_SPREAD_OBS:
        return 0.0, 0.0, 0, 0

    ranked = frame.sort_values("signal", kind="stable")
    k = n // 3
    if k < 2:
        return 0.0, 0.0, 0, 0

    bottom = ranked.head(k)["fwd_return"].to_numpy(dtype=float)
    top = ranked.tail(k)["fwd_return"].to_numpy(dtype=float)

    # Long the top tercile, short the bottom; charge cost on both sides.
    spread = float(top.mean() - bottom.mean()) - 2.0 * cost

    if top.std(ddof=1) == 0.0 and bottom.std(ddof=1) == 0.0:
        tstat = 0.0
    else:
        t, _ = ttest_ind(top, bottom, equal_var=False)
        tstat = float(t) if np.isfinite(t) else 0.0

    return spread, tstat, len(top), len(bottom)


def run_backtest(
    session: Session,
    signal_col: str,
    horizon: int,
    cost_bps: float = 10.0,
    persist: bool = True,
) -> BacktestResult:
    """Backtest one signal at one horizon and persist the result.

    Args:
        session: Active database session.
        signal_col: Signal column on :class:`Signal` to evaluate.
        horizon: Forward horizon in trading days.
        cost_bps: Per-side transaction cost in basis points.
        persist: If ``True``, write a :class:`BacktestRun` row.

    Returns:
        The computed :class:`BacktestResult`. Metrics are reported honestly: a
        weak or insignificant signal yields a near-zero IC and Sharpe rather
        than a fabricated edge.
    """
    frame = load_observations(session, signal_col, horizon)
    n_obs = len(frame)

    if n_obs < 3:
        logger.warning(
            "Only %d observations for %s@%d; metrics are unreliable.",
            n_obs,
            signal_col,
            horizon,
        )

    ic, ic_tstat, ic_method = compute_ic(frame)
    ls_spread, spread_tstat, n_long, n_short = compute_event_study_spread(frame, cost_bps)

    result = BacktestResult(
        signal=signal_col,
        horizon_days=horizon,
        n_obs=n_obs,
        ic=ic,
        ic_tstat=ic_tstat,
        ls_spread=ls_spread,
        spread_tstat=spread_tstat,
        ic_method=ic_method,
    )

    if persist:
        config = {
            "cost_bps": cost_bps,
            "ic_method": ic_method,
            "n_obs": n_obs,
            "n_long": n_long,
            "n_short": n_short,
        }
        session.add(
            BacktestRun(
                signal=signal_col,
                horizon_days=horizon,
                config_json=json.dumps(config),
                ic=ic,
                ic_tstat=ic_tstat,
                ls_spread=ls_spread,
                spread_tstat=spread_tstat,
            )
        )
        session.commit()

    logger.info("Backtest %s@%d: %s", signal_col, horizon, asdict(result))
    return result


__all__ = [
    "run_backtest",
    "load_observations",
    "compute_ic",
    "compute_event_study_spread",
    "BacktestResult",
    "SIGNAL_COLUMNS",
]
