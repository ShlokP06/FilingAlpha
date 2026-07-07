"""Render report figures from :class:`~reporting.data.ReportData`.

Uses the non-interactive Agg backend so figures render headless (in CI, in a
container, over SSH). Every figure is saved as a PNG and its path returned for
embedding in the LaTeX document. Bars are coloured by statistical significance so
the honest "weak / not significant" reading is visible at a glance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display required

import matplotlib.pyplot as plt  # noqa: E402

from reporting.data import (  # noqa: E402
    SIGNIFICANT_T,
    SUGGESTIVE_T,
    ReportData,
    SignalResult,
)

logger = logging.getLogger(__name__)

# Significance colour ramp (colour-blind-safe-ish): grey = noise, amber =
# suggestive, green = significant.
_COLOR_NOISE = "#9aa0a6"
_COLOR_SUGGESTIVE = "#e8a33d"
_COLOR_SIGNIFICANT = "#2e9e5b"

# Short, axis-friendly signal labels (the table uses the long names).
_SHORT_LABELS = {
    "lm_negative": "Neg. tone",
    "lm_uncertainty": "Uncertainty",
    "lm_litigious": "Litigious",
    "yoy_similarity": "YoY similarity",
    "risk_factor_delta": "Risk-factor Δ",
    "fog_readability": "Fog readability",
}


def _short(signal: str) -> str:
    """Short axis label for a signal column."""
    return _SHORT_LABELS.get(signal, signal.replace("_", " "))


def _t_color(tstat: float) -> str:
    """Map an absolute t-stat to a significance colour."""
    t = abs(tstat)
    if t >= SIGNIFICANT_T:
        return _COLOR_SIGNIFICANT
    if t >= SUGGESTIVE_T:
        return _COLOR_SUGGESTIVE
    return _COLOR_NOISE


def _headline_form_results(data: ReportData, horizon: int) -> list[SignalResult]:
    """10-K results at one horizon, sorted by signal name for stable plots."""
    rows = [
        r
        for r in data.signal_results
        if r.form == "10-K" and r.horizon_days == horizon and r.n_obs > 0
    ]
    return sorted(rows, key=lambda r: r.signal)


def _ic_bar(data: ReportData, out_dir: Path) -> Path | None:
    """Bar chart of information coefficient by signal (10-K), per horizon."""
    horizons = sorted({r.horizon_days for r in data.signal_results if r.form == "10-K"})
    if not horizons:
        return None

    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 4.0), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        rows = _headline_form_results(data, horizon)
        names = [_short(r.signal) for r in rows]
        ics = [r.ic for r in rows]
        colors = [_t_color(r.ic_tstat) for r in rows]
        ax.bar(names, ics, color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{horizon}-day horizon")
        ax.set_ylabel("Information coefficient (Spearman)")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig.suptitle("Signal information coefficient — 10-K filings", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "ic_by_signal.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _spread_bar(data: ReportData, out_dir: Path) -> Path | None:
    """Bar chart of long-short tercile spread by signal (10-K), per horizon."""
    horizons = sorted({r.horizon_days for r in data.signal_results if r.form == "10-K"})
    if not horizons:
        return None

    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 4.0), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        rows = _headline_form_results(data, horizon)
        names = [_short(r.signal) for r in rows]
        spreads = [r.ls_spread * 100.0 for r in rows]  # percent
        colors = [_t_color(r.spread_tstat) for r in rows]
        ax.bar(names, spreads, color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{horizon}-day horizon")
        ax.set_ylabel("Long-short spread, net of cost (%)")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig.suptitle("Event-study long-short tercile spread — 10-K filings", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "spread_by_signal.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _equity_curve(data: ReportData, out_dir: Path) -> Path | None:
    """Event-ordered cumulative long-short curve for the headline signal."""
    curve = data.equity_curve
    if curve is None or not curve.dates:
        return None

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    # matplotlib accepts date sequences on the x-axis at runtime; its stubs do not.
    ax.plot(
        curve.dates,  # type: ignore[arg-type]
        [c * 100.0 for c in curve.cumulative],
        color="#1a73e8",
        linewidth=1.6,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative long-short return (%)")
    ax.set_xlabel("Filing date (event-ordered)")
    ax.set_title(
        f"Cumulative long-short return — {_short(curve.signal)} "
        f"@ {curve.horizon_days}d ({curve.form})",
        fontweight="bold",
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out_dir / "equity_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _feature_importance(data: ReportData, out_dir: Path) -> Path | None:
    """Walk-forward model feature importances for the 10-K model."""
    models = [m for m in data.model_results if m.form == "10-K" and m.feature_importances]
    if not models:
        return None
    model = max(models, key=lambda m: m.n_oos)
    items = sorted(model.feature_importances.items(), key=lambda kv: kv[1])
    names = [_short(k) for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.barh(names, vals, color="#6741d9")
    ax.set_xlabel("Mean Gini importance")
    ax.set_title(
        f"Walk-forward feature importance — 10-K, {model.horizon_days}d horizon",
        fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_figures(data: ReportData, out_dir: Path) -> dict[str, Path]:
    """Render all report figures to ``out_dir``.

    Args:
        data: Resolved report data.
        out_dir: Directory to write PNGs into (created if needed).

    Returns:
        Mapping of figure key to file path, including only figures that had
        enough data to render.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "ic": _ic_bar,
        "spread": _spread_bar,
        "equity_curve": _equity_curve,
        "feature_importance": _feature_importance,
    }
    figures: dict[str, Path] = {}
    for key, builder in builders.items():
        path = builder(data, out_dir)
        if path is not None:
            figures[key] = path
    logger.info("Rendered %d report figures to %s", len(figures), out_dir)
    return figures


__all__ = ["render_figures"]
