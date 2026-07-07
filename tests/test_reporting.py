"""Offline unit tests for the reporting package (no database, no network)."""

from __future__ import annotations

from datetime import date

import pytest

from reporting.data import (
    EquityCurve,
    ModelResult,
    ReportData,
    SignalResult,
    signal_label,
)
from reporting.latex import build_tex
from reporting.narrative import generate_narrative
from reporting.plots import render_figures


def _signal(
    form: str,
    signal: str,
    horizon: int,
    ic: float,
    ic_t: float,
    spread: float,
    spread_t: float,
) -> SignalResult:
    return SignalResult(
        form=form,
        signal=signal,
        horizon_days=horizon,
        n_obs=60,
        ic=ic,
        ic_tstat=ic_t,
        ls_spread=spread,
        spread_tstat=spread_t,
    )


@pytest.fixture
def report_data() -> ReportData:
    """A small synthetic ReportData covering both forms and horizons."""
    results = [
        _signal("10-K", "yoy_similarity", 63, 0.159, 1.22, 0.0415, 1.49),
        _signal("10-K", "fog_readability", 21, -0.157, -1.33, -0.0344, -1.59),
        _signal("10-K", "lm_negative", 21, 0.128, 1.08, 0.0209, 1.00),
        _signal("10-K", "risk_factor_delta", 63, 0.20, 2.4, 0.0556, 2.30),
        _signal("10-Q", "lm_litigious", 21, -0.065, -1.09, -0.0162, -1.59),
    ]
    models = [
        ModelResult(
            form="10-K",
            horizon_days=21,
            n_oos=49,
            n_folds=5,
            oos_accuracy=0.51,
            oos_auc=0.47,
            feature_importances={
                "lm_negative": 0.2,
                "yoy_similarity": 0.5,
                "fog_readability": 0.3,
            },
        )
    ]
    curve = EquityCurve(
        form="10-K",
        signal="fog_readability",
        horizon_days=21,
        dates=[date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)],
        cumulative=[0.0, -0.2, -0.35],
    )
    return ReportData(
        generated_at=date(2026, 6, 28),
        n_companies=12,
        n_filings_by_form={"10-K": 72, "10-Q": 286},
        date_range=(date(2018, 7, 25), date(2026, 5, 29)),
        signal_results=results,
        model_results=models,
        equity_curve=curve,
    )


def test_significance_stars() -> None:
    assert _signal("10-K", "x", 21, 0, 0, 0, 2.30).stars == "**"
    assert _signal("10-K", "x", 21, 0, 0, 0, 1.60).stars == "*"
    assert _signal("10-K", "x", 21, 0, 0, 0, 0.90).stars == ""


def test_signal_label_human_readable() -> None:
    assert "Lazy Prices" in signal_label("yoy_similarity")
    # Unknown columns degrade gracefully without leaving an underscore.
    assert "_" not in signal_label("some_new_signal")


def test_headline_picks_largest_abs_spread_t(report_data: ReportData) -> None:
    head = report_data.headline()
    assert head is not None
    # risk_factor_delta has |t| = 2.30, the largest among 10-K rows.
    assert head.signal == "risk_factor_delta"


def test_template_narrative_is_honest_without_key(report_data: ReportData) -> None:
    # With no API key configured, generate_narrative uses the deterministic
    # template, which must state significance honestly and quote real numbers.
    text = generate_narrative(report_data)
    assert "significant" in text.lower()
    assert "coin flip" in text.lower()
    # It must not invent a tradeable edge.
    assert "guaranteed" not in text.lower()


def test_build_tex_escapes_and_structures(report_data: ReportData) -> None:
    tex = build_tex(report_data, figures={}, narrative="Line one.\n\nLine two.")
    assert "\\begin{document}" in tex and "\\end{document}" in tex
    # Percent signs in the table header must be escaped.
    assert "L-S \\%" in tex
    # The significant row carries the ** marker.
    assert "2.30**" in tex
    # Human-readable labels, not raw identifiers.
    assert "year-over-year filing similarity" in tex
    assert "yoy_similarity" not in tex


def test_render_figures_writes_pngs(report_data: ReportData, tmp_path) -> None:
    figures = render_figures(report_data, tmp_path)
    assert {"ic", "spread", "equity_curve", "feature_importance"} <= set(figures)
    for path in figures.values():
        assert path.exists() and path.stat().st_size > 0
