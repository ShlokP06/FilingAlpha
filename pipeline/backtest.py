"""Signal backtesting: information coefficient and long-short portfolio.

Evaluates one signal column at one horizon against filing-lagged forward
returns (computed in :mod:`pipeline.returns`). Two standard measures are
produced and persisted as a :class:`BacktestRun`:

* **Information Coefficient (IC):** Spearman rank correlation between the signal
  and the forward return, with a t-statistic
  ``t = ic * sqrt((n - 2) / (1 - ic ** 2))``.

* **Long-short portfolio:** each period, rank observations by the signal, go
  long the top tercile and short the bottom tercile, charge ``cost_bps`` per
  side, and build the per-period return series, from which we report annualised
  Sharpe, hit rate, and cumulative return.

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
from scipy.stats import spearmanr
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

# Trading days per year, used to annualise the long-short Sharpe ratio. Each
# long-short observation corresponds to one filing event held over the horizon,
# so we annualise by horizon length rather than by 252 directly.
_TRADING_DAYS_PER_YEAR = 252

# Minimum observations a period needs to contribute a cross-sectional IC.
_MIN_PERIOD_OBS = 5
# Minimum qualifying periods before we trust cross-sectional averaging.
_MIN_PERIODS = 5


@dataclass(frozen=True)
class BacktestResult:
    """Backtest metrics for one signal / horizon."""

    signal: str
    horizon_days: int
    n_obs: int
    ic: float
    ic_tstat: float
    ls_sharpe: float
    hit_rate: float
    cum_return: float
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


def _tercile_long_short(group: pd.DataFrame, cost: float) -> float | None:
    """Net long-short return for one period's cross-section.

    Long the top signal tercile, short the bottom, equal-weighted, charging
    ``cost`` (fraction) per side.

    Args:
        group: Observations for one period with ``signal`` and ``fwd_return``.
        cost: Per-side transaction cost as a fraction (e.g. 0.001 for 10 bps).

    Returns:
        Net long-short return, or ``None`` if the cross-section is too thin to
        form distinct terciles.
    """
    n = len(group)
    if n < 3:
        return None
    ranked = group.sort_values("signal")
    k = n // 3
    if k < 1:
        return None
    short_leg = ranked.head(k)["fwd_return"].mean()
    long_leg = ranked.tail(k)["fwd_return"].mean()
    gross = float(long_leg - short_leg)
    # Two sides traded (long and short), cost charged on each.
    return gross - 2.0 * cost


def compute_long_short(
    frame: pd.DataFrame, horizon: int, cost_bps: float
) -> tuple[float, float, float, pd.Series]:
    """Build the long-short return series and its summary statistics.

    When multiple periods have enough breadth, the portfolio is rebalanced per
    period (per filing date). When the data are too sparse for cross-sectional
    terciles, a single pooled tercile sort across all observations is used so a
    statistic can still be reported.

    Args:
        frame: Output of :func:`load_observations`.
        horizon: Forward horizon in trading days (used for annualisation).
        cost_bps: Per-side transaction cost in basis points.

    Returns:
        ``(ls_sharpe, hit_rate, cum_return, returns)`` where ``returns`` is the
        per-period net long-short return series.
    """
    cost = cost_bps / 1e4

    period_returns: list[float] = []
    for _, group in frame.groupby("filing_date"):
        ret = _tercile_long_short(group, cost)
        if ret is not None:
            period_returns.append(ret)

    if len(period_returns) < 2:
        # Pooled single-sort fallback: treat the whole sample as one rebalance.
        pooled = _tercile_long_short(frame, cost)
        returns = pd.Series([pooled] if pooled is not None else [], dtype=float)
    else:
        returns = pd.Series(period_returns, dtype=float)

    if returns.empty:
        return 0.0, 0.0, 0.0, returns

    hit_rate = float((returns > 0).mean())
    cum_return = float((1.0 + returns).prod() - 1.0)

    if len(returns) > 1 and returns.std(ddof=1) > 0:
        # Each observation spans ``horizon`` trading days; scale to annual.
        periods_per_year = _TRADING_DAYS_PER_YEAR / horizon
        mean_r = returns.mean()
        std_r = returns.std(ddof=1)
        sharpe = float(mean_r / std_r * math.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    return sharpe, hit_rate, cum_return, returns


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
    ls_sharpe, hit_rate, cum_return, _ = compute_long_short(frame, horizon, cost_bps)

    result = BacktestResult(
        signal=signal_col,
        horizon_days=horizon,
        n_obs=n_obs,
        ic=ic,
        ic_tstat=ic_tstat,
        ls_sharpe=ls_sharpe,
        hit_rate=hit_rate,
        cum_return=cum_return,
        ic_method=ic_method,
    )

    if persist:
        config = {
            "cost_bps": cost_bps,
            "ic_method": ic_method,
            "n_obs": n_obs,
        }
        session.add(
            BacktestRun(
                signal=signal_col,
                horizon_days=horizon,
                config_json=json.dumps(config),
                ic=ic,
                ic_tstat=ic_tstat,
                ls_sharpe=ls_sharpe,
                hit_rate=hit_rate,
                cum_return=cum_return,
            )
        )
        session.commit()

    logger.info("Backtest %s@%d: %s", signal_col, horizon, asdict(result))
    return result


__all__ = [
    "run_backtest",
    "load_observations",
    "compute_ic",
    "compute_long_short",
    "BacktestResult",
    "SIGNAL_COLUMNS",
]
