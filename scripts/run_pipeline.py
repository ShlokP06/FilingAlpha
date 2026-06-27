"""End-to-end pipeline runner: ingest -> signals -> returns -> backtest -> model.

Usage (from the project root, with Postgres up and migrated):

    uv run python scripts/run_pipeline.py
    uv run python scripts/run_pipeline.py --no-ingest          # recompute only
    uv run python scripts/run_pipeline.py --tickers AAPL,MSFT  # custom universe
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable when run directly: ``python scripts/run_pipeline.py``
# puts ``scripts/`` on sys.path (not the repo root), so ``import scripts.*`` would fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import SessionLocal  # noqa: E402
from pipeline.backtest import run_backtest  # noqa: E402
from pipeline.compute_signals import compute_signals  # noqa: E402
from pipeline.ingest import ingest_universe  # noqa: E402
from pipeline.model import run_walkforward  # noqa: E402
from pipeline.returns import compute_forward_returns  # noqa: E402
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


def main() -> None:
    """Parse args and run the full pipeline."""
    parser = argparse.ArgumentParser(description="Run the FilingAlpha pipeline.")
    parser.add_argument("--tickers", help="Comma-separated ticker override.")
    parser.add_argument("--years", type=int, default=6, help="Years of history to ingest.")
    parser.add_argument("--no-ingest", action="store_true", help="Skip ingestion; recompute only.")
    args = parser.parse_args()

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else DEFAULT_TICKERS
    )

    if not args.no_ingest:
        logger.info("Ingesting %d tickers...", len(tickers))
        ingest_universe(tickers, years=args.years)

    with SessionLocal() as session:
        logger.info("Computing signals...")
        compute_signals(session)

        logger.info("Computing forward returns...")
        compute_forward_returns(session, horizons=HORIZONS)

        logger.info("Backtesting %d signals x %d horizons...", len(SIGNAL_COLUMNS), len(HORIZONS))
        results = []
        for horizon in HORIZONS:
            for col in SIGNAL_COLUMNS:
                results.append(run_backtest(session, col, horizon))

        logger.info("Walk-forward model...")
        for horizon in HORIZONS:
            run_walkforward(session, horizon=horizon)

    _print_summary(results)


def _print_summary(results: list) -> None:
    """Print a compact backtest results table to stdout."""
    print(f"\n{'signal':<20}{'horizon':>8}{'IC':>9}{'IC t':>7}{'L-S spread':>12}{'spr t':>7}")
    print("-" * 63)
    for r in sorted(results, key=lambda x: abs(x.spread_tstat or 0), reverse=True):
        print(
            f"{r.signal:<20}{r.horizon_days:>8}{(r.ic or 0):>9.3f}{(r.ic_tstat or 0):>7.2f}"
            f"{(r.ls_spread or 0):>12.4f}{(r.spread_tstat or 0):>7.2f}"
        )
    print("\nResults are reported as-is; weak signals are expected, not a failure.\n")


if __name__ == "__main__":
    main()
