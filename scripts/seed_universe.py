"""Seed the FilingAlpha database with a default large-cap universe.

Run with:

    uv run python scripts/seed_universe.py

Override the ticker list:

    uv run python scripts/seed_universe.py --tickers AAPL,MSFT,GOOGL

The default universe covers ~12 large-cap tickers across several sectors with
long enough 10-K histories to populate at least 6 annual filings each.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable when the script is run directly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.ingest import ingest_universe  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default universe: 12 large-cap tickers across 5 sectors.
# All have continuous 10-K filing histories going back at least 10 years.
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: list[str] = [
    # Technology
    "AAPL",
    "MSFT",
    "NVDA",
    # Financials
    "JPM",
    # Energy
    "XOM",
    # Healthcare
    "JNJ",
    # Consumer Staples
    "PG",
    "WMT",
    "KO",
    # Industrials
    "CAT",
    "BA",
    # Consumer Discretionary
    "DIS",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when None).

    Returns:
        Parsed namespace with attribute ``tickers`` (list of str) and
        ``years`` (int).
    """
    parser = argparse.ArgumentParser(
        description="Seed FilingAlpha with SEC filings and price data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers to ingest (overrides the default universe).",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=6,
        help="Look-back window in years for filings and prices.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the seed script.

    Args:
        argv: Optional argument list for programmatic invocation / testing.
    """
    args = _parse_args(argv)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = DEFAULT_TICKERS

    logger.info(
        "Seeding universe: %d tickers, %d-year window — %s",
        len(tickers),
        args.years,
        ", ".join(tickers),
    )

    ingest_universe(tickers=tickers, years=args.years)
    logger.info("Seed complete.")


if __name__ == "__main__":
    main()
