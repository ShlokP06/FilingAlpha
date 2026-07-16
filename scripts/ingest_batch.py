"""Resumable batch ingester: ingest only the next N firms that still need work.

The full universe is large and EDGAR section-parsing is slow, so a single
end-to-end ingest is impractical here. This script makes each run productive:
it reads the universe and selects the next ``--batch`` firms whose
``ingest_state`` is unset, ``pending``, or ``failed`` — i.e. never ingested or
previously interrupted by a rate limit. Firms marked ``complete`` (fully
ingested) or ``no_data`` (delisted/illiquid — nothing to fetch) are skipped, so
re-running repeatedly chips toward the full universe and automatically retries
only the firms that actually failed. Nothing is ever silently frozen mid-way.

Usage::

    python scripts/ingest_batch.py --batch 15
    python scripts/ingest_batch.py --batch 15 --forms 10-K   # 10-K only (faster)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import SessionLocal  # noqa: E402
from core.models import Company  # noqa: E402
from pipeline.ingest import ingest_market_benchmark, ingest_universe  # noqa: E402
from pipeline.net import STATE_FAILED, STATE_PENDING  # noqa: E402

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
    ap.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(2018, 1, 1),
        help="Window start (YYYY-MM-DD). Defaults to the 2018 universe anchor.",
    )
    ap.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="Window end (YYYY-MM-DD). Defaults to today.",
    )
    ap.add_argument(
        "--forms",
        default="10-K,10-Q",
        help="Comma-separated SEC forms to ingest (e.g. '10-K' for a faster run).",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel shards (processes) cooperating on the universe.",
    )
    ap.add_argument(
        "--shard",
        type=int,
        default=0,
        help="This process's shard index in [0, num-shards). Each shard owns a "
        "disjoint slice of the universe (by file position), so N processes can run "
        "concurrently without overlapping — the fastest safe speedup, since ingest "
        "is latency/CPU-bound and far under SEC's 10 req/s limit.",
    )
    args = ap.parse_args()
    end = args.end or date.today()
    if not 0 <= args.shard < args.num_shards:
        ap.error("--shard must be in [0, --num-shards)")

    # Keep full records (ticker + cik): the CIK drives EDGAR resolution so
    # renamed/delisted firms resolve instead of being dropped.
    all_records = json.loads(Path(args.universe_file).read_text(encoding="utf-8"))
    universe_tickers = {str(r["ticker"]).strip().upper() for r in all_records}
    # Disjoint sharding by file position: deterministic and independent of live DB
    # state, so concurrent shards never touch the same firm. Shard 0 also refreshes
    # the SPY benchmark (idempotent) so it isn't fetched N times in parallel.
    records = [r for i, r in enumerate(all_records) if i % args.num_shards == args.shard]

    with SessionLocal() as s:
        states = {t: st for t, st in s.query(Company.ticker, Company.ingest_state).all()}

    # A firm needs work if it's absent (None), pending, or previously failed.
    # 'complete' and 'no_data' are terminal and skipped.
    needs_work = (None, STATE_PENDING, STATE_FAILED)
    remaining = [r for r in records if states.get(str(r["ticker"]).strip().upper()) in needs_work]
    logger.info(
        "shard=%d/%d  shard_firms=%d  needs_work=%d  (complete/no_data skipped=%d)",
        args.shard,
        args.num_shards,
        len(records),
        len(remaining),
        len(records) - len(remaining),
    )
    if not remaining:
        print("NOTHING_TO_DO — universe fully ingested.")
        return

    batch = remaining[: args.batch]
    forms = tuple(f.strip() for f in args.forms.split(","))
    logger.info(
        "Ingesting batch of %d (window %s..%s, forms=%s): %s",
        len(batch),
        args.start,
        end,
        forms,
        [r["ticker"] for r in batch],
    )

    batch_states = ingest_universe(batch, start=args.start, end=end, forms=forms)
    if args.shard == 0:
        ingest_market_benchmark(start=args.start, end=end)  # idempotent; ensures SPY present

    # Report this batch's outcome and the universe-wide progress.
    batch_summary = {
        s: sum(1 for v in batch_states.values() if v == s)
        for s in sorted(set(batch_states.values()))
    }
    with SessionLocal() as s:
        rows = s.query(Company.ticker, Company.ingest_state).all()
    overall = {}
    for ticker, st in rows:
        if ticker in universe_tickers:
            overall[st] = overall.get(st, 0) + 1
    still_left = len(remaining) - len(batch)
    print(
        f"BATCH_DONE  batch={batch_summary}  universe_states={overall}  "
        f"remaining_after={still_left}"
    )


if __name__ == "__main__":
    main()
