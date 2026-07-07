"""End-to-end pipeline runner: ingest -> signals -> returns -> backtest -> model.

Usage (from the project root, with Postgres up and migrated):

    uv run python scripts/run_pipeline.py
    uv run python scripts/run_pipeline.py --no-ingest          # recompute only
    uv run python scripts/run_pipeline.py --tickers AAPL,MSFT  # custom universe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the project root importable when run directly: ``python scripts/run_pipeline.py``
# puts ``scripts/`` on sys.path (not the repo root), so ``import scripts.*`` would fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import SessionLocal  # noqa: E402
from pipeline.backtest import run_backtest  # noqa: E402
from pipeline.compute_signals import compute_signals  # noqa: E402
from pipeline.ingest import ingest_market_benchmark, ingest_universe  # noqa: E402
from pipeline.model import run_walkforward  # noqa: E402
from pipeline.returns import compute_forward_returns  # noqa: E402
from reporting.report import generate_report  # noqa: E402
from scripts.seed_universe import DEFAULT_TICKERS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")

SIGNAL_COLUMNS = (
    "lm_negative",
    "lm_uncertainty",
    "lm_litigious",
    "yoy_similarity",
    "risk_factor_delta",
    "fog_readability",
)
HORIZONS = (21, 63)
# Backtest each form separately: the text-change signals are an annual-report
# phenomenon, so 10-Ks carry them and 10-Qs dilute them — pooling hides that.
FORMS = ("10-K", "10-Q")


def main() -> None:
    """Parse args and run the full pipeline."""
    parser = argparse.ArgumentParser(description="Run the FilingAlpha pipeline.")
    parser.add_argument("--tickers", help="Comma-separated ticker override.")
    parser.add_argument(
        "--universe-file",
        help="Path to a universe JSON (list of {'ticker': ...}) from build_universe.py.",
    )
    parser.add_argument("--years", type=int, default=6, help="Years of history to ingest.")
    parser.add_argument("--no-ingest", action="store_true", help="Skip ingestion; recompute only.")
    parser.add_argument(
        "--report", action="store_true", help="Generate the LaTeX/PDF research note at the end."
    )
    args = parser.parse_args()

    if args.universe_file:
        records = json.loads(Path(args.universe_file).read_text(encoding="utf-8"))
        tickers = [str(r["ticker"]).strip().upper() for r in records]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = DEFAULT_TICKERS

    if not args.no_ingest:
        logger.info("Ingesting %d tickers (10-K + 10-Q)...", len(tickers))
        ingest_universe(tickers, years=args.years)
        logger.info("Ingesting market benchmark for excess returns...")
        ingest_market_benchmark(years=args.years)

    with SessionLocal() as session:
        logger.info("Computing signals...")
        compute_signals(session)

        logger.info("Computing forward returns...")
        compute_forward_returns(session, horizons=HORIZONS)

        logger.info(
            "Backtesting %d signals x %d horizons x %d forms...",
            len(SIGNAL_COLUMNS),
            len(HORIZONS),
            len(FORMS),
        )
        results = []
        for form in FORMS:
            for horizon in HORIZONS:
                for col in SIGNAL_COLUMNS:
                    results.append(run_backtest(session, col, horizon, form=form))

        logger.info("Walk-forward model (per form)...")
        for form in FORMS:
            for horizon in HORIZONS:
                run_walkforward(session, horizon=horizon, form=form)

        if args.report:
            logger.info("Generating research note (LaTeX/PDF)...")
            outputs = generate_report(session)
            logger.info("Report artifacts: %s", {k: str(v) for k, v in outputs.items()})

    _print_summary(results)


def _print_summary(results: list) -> None:
    """Print a compact backtest results table to stdout."""
    print(
        f"\n{'form':<7}{'signal':<20}{'horizon':>8}{'IC':>9}{'IC t':>7}"
        f"{'L-S spread':>12}{'spr t':>7}"
    )
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x.form or "", -abs(x.spread_tstat or 0))):
        print(
            f"{(r.form or 'all'):<7}{r.signal:<20}{r.horizon_days:>8}"
            f"{(r.ic or 0):>9.3f}{(r.ic_tstat or 0):>7.2f}"
            f"{(r.ls_spread or 0):>12.4f}{(r.spread_tstat or 0):>7.2f}"
        )
    print("\nResults are reported as-is; weak signals are expected, not a failure.\n")


if __name__ == "__main__":
    main()
