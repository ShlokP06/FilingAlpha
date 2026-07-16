"""Run one shard of signal computation over a disjoint slice of companies.

Signal computation reads each filing's full text from disk and computes TF-IDF
cosine similarity against the prior-year filing — CPU/disk-bound and, over the
full ~12K-filing universe, slow single-threaded. This runner computes signals
for only the companies where ``company_id % num_shards == shard``, so N of these
processes can run concurrently over disjoint companies (no row contention, no
cross-shard prior-year lookups) for a near-linear speedup.

Usage (launch one process per shard, e.g. 5-way)::

    python scripts/compute_signals_shard.py --num-shards 5 --shard 0
    python scripts/compute_signals_shard.py --num-shards 5 --shard 1
    ...

Once every shard prints ``SHARD_DONE``, run the rest of the pipeline once with
``scripts/run_pipeline.py --no-ingest --no-signals`` (returns -> backtest -> model).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import SessionLocal  # noqa: E402
from pipeline.compute_signals import compute_signals  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("compute_signals_shard")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute signals for one shard of companies.")
    ap.add_argument("--num-shards", type=int, default=1, help="Total cooperating shards.")
    ap.add_argument("--shard", type=int, default=0, help="This shard's index in [0, num-shards).")
    args = ap.parse_args()
    if not 0 <= args.shard < args.num_shards:
        ap.error("--shard must be in [0, --num-shards)")

    logger.info("Computing signals for shard %d/%d ...", args.shard, args.num_shards)
    with SessionLocal() as session:
        written = compute_signals(session, shard=args.shard, num_shards=args.num_shards)
    print(f"SHARD_DONE  shard={args.shard}/{args.num_shards}  signals_written={written}")


if __name__ == "__main__":
    main()
