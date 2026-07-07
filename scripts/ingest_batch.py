"""Resumable batch ingester: ingest only the next N *un-ingested* firms.

The full universe is large and EDGAR section-parsing is slow, so a single
end-to-end ingest is impractical here. This script makes each run productive:
it reads the universe, subtracts firms already present in the DB, and ingests
only the next ``--batch`` new tickers. Re-run it repeatedly to chip toward the
full universe — it never re-processes a firm that is already ingested.

Usage::

    python scripts/ingest_batch.py --batch 15
    python scripts/ingest_batch.py --batch 15 --forms 10-K   # 10-K only (faster)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import SessionLocal  # noqa: E402
from core.models import Company  # noqa: E402
from pipeline.ingest import ingest_market_benchmark, ingest_universe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# The edgar HTML/TOC extractors log an INFO line per section — far too noisy
# for a 300-firm chip. Keep our own ingest logs, silence theirs.
logging.getLogger("edgar").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logger = logging.getLogger("ingest_batch")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the next N un-ingested firms.")
    ap.add_argument("--universe-file", default="data/universe/smallmid_2018.json")
    ap.add_argument("--batch", type=int, default=15, help="How many new firms to ingest this run.")
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument(
        "--forms",
        default="10-K,10-Q",
        help="Comma-separated SEC forms to ingest (e.g. '10-K' for a faster run).",
    )
    args = ap.parse_args()

    records = json.loads(Path(args.universe_file).read_text(encoding="utf-8"))
    universe = [str(r["ticker"]).strip().upper() for r in records]

    with SessionLocal() as s:
        done = {t for (t,) in s.query(Company.ticker).all()}

    remaining = [t for t in universe if t not in done]
    logger.info(
        "universe=%d  already_ingested=%d  remaining=%d",
        len(universe),
        len(done),
        len(remaining),
    )
    if not remaining:
        print("NOTHING_TO_DO — universe fully ingested.")
        return

    batch = remaining[: args.batch]
    forms = tuple(f.strip() for f in args.forms.split(","))
    logger.info("Ingesting batch of %d (forms=%s): %s", len(batch), forms, batch)

    ingest_universe(batch, years=args.years, forms=forms)
    ingest_market_benchmark(years=args.years)  # idempotent; ensures SPY present

    with SessionLocal() as s:
        total = s.query(Company).count()
    still_left = len(remaining) - len(batch)
    print(f"BATCH_DONE  companies_now={total}  remaining_after={still_left}")


if __name__ == "__main__":
    main()
