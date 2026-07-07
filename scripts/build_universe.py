"""Build a 2018-anchored small/mid-cap universe for a powered backtest.

Selection methodology (designed to limit survivorship/selection bias):

1. **Point-in-time population.** Download SEC EDGAR's 2018 quarterly
   ``master.idx`` files and keep every filer that filed a ``10-K`` in 2018. This
   anchors selection at the study's start, not today, so we are not implicitly
   choosing firms *because* they survived.
2. **Tradeable + data-available.** Intersect those CIKs with SEC's current
   ``company_tickers.json``. Firms delisted since 2018 have no current ticker and
   drop out here — the residual survivorship bias, which we disclose rather than
   hide (delisted firms simply lack forward returns; they are not retroactively
   removed from results).
3. **Cap band.** Seeded-random shuffle, then accept only firms with a market cap
   in ``[$300M, $10B]`` — dropping mega-caps (over-arbitraged, no anomaly) and
   micro-caps (illiquid, poor data).
4. **Sector spread.** Soft-cap any one SIC division at a fixed fraction of the
   universe so the result is not dominated by a single sector's beta.

The output is a reproducible JSON list written to ``data/universe/``. Run::

    uv run python scripts/build_universe.py --target 300
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yfinance as yf  # noqa: E402

from core.config import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

ANCHOR_YEAR = 2018
CAP_MIN = 300_000_000.0  # $300M floor: exclude illiquid micro-caps.
CAP_MAX = 10_000_000_000.0  # $10B ceiling: exclude over-arbitraged mega-caps.
DEFAULT_TARGET = 300
SECTOR_CAP_FRACTION = 0.20  # no single SIC division may exceed 20% of the universe.
SEED = 42
# Politeness delay between SEC requests (their guidance is <= 10 req/s).
_SEC_DELAY = 0.12
# Bound total probes so a sparse pool cannot run unboundedly.
_MAX_PROBE_FACTOR = 8

_MASTER_IDX = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def _headers() -> dict[str, str]:
    """SEC-required identifying User-Agent header."""
    return {"User-Agent": settings.sec_identity, "Accept-Encoding": "gzip, deflate"}


def _sic_division(sic: int | None) -> str:
    """Map a numeric SIC code to a coarse industry division.

    Args:
        sic: SIC code, or ``None`` when unknown.

    Returns:
        A coarse division label used only for sector-spread balancing.
    """
    if not sic:
        return "Unknown"
    ranges = [
        (100, 999, "Agriculture"),
        (1000, 1499, "Mining"),
        (1500, 1799, "Construction"),
        (2000, 3999, "Manufacturing"),
        (4000, 4999, "Transport/Utilities"),
        (5000, 5199, "Wholesale"),
        (5200, 5999, "Retail"),
        (6000, 6799, "Finance"),
        (7000, 8999, "Services"),
    ]
    for low, high, label in ranges:
        if low <= sic <= high:
            return label
    return "Other"


def fetch_2018_tenk_ciks() -> dict[int, str]:
    """Return ``{cik: company_name}`` for every 10-K filed in the anchor year.

    Returns:
        Mapping of CIK to the filer name, deduplicated across all four quarters.
    """
    filers: dict[int, str] = {}
    for quarter in range(1, 5):
        url = _MASTER_IDX.format(year=ANCHOR_YEAR, q=quarter)
        logger.info("Fetching EDGAR index %s", url)
        resp = requests.get(url, headers=_headers(), timeout=60)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            parts = line.split("|")
            if len(parts) != 5:
                continue
            cik_raw, name, form_type, _date, _file = parts
            if form_type.strip() != "10-K":
                continue
            try:
                cik = int(cik_raw)
            except ValueError:
                continue
            filers.setdefault(cik, name.strip())
        time.sleep(_SEC_DELAY)
    logger.info("Found %d distinct 10-K filers in %d.", len(filers), ANCHOR_YEAR)
    return filers


def load_ticker_map() -> dict[int, str]:
    """Return ``{cik: ticker}`` from SEC's current ticker directory."""
    resp = requests.get(_TICKERS_URL, headers=_headers(), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    mapping: dict[int, str] = {}
    for row in data.values():
        mapping[int(row["cik_str"])] = str(row["ticker"]).upper()
    logger.info("Loaded %d CIK->ticker mappings.", len(mapping))
    return mapping


def _market_cap(ticker: str) -> float | None:
    """Best-effort current market cap for a ticker, or ``None`` on failure."""
    try:
        fast = yf.Ticker(ticker).fast_info
        cap = getattr(fast, "market_cap", None)
        if cap is None and isinstance(fast, dict):  # pragma: no cover - version drift
            cap = fast.get("market_cap") or fast.get("marketCap")
        return float(cap) if cap else None
    except Exception:  # pragma: no cover - network/SDK variability
        return None


def _sic_for_cik(cik: int) -> int | None:
    """Fetch the firm's SIC code from EDGAR submissions, or ``None``."""
    try:
        resp = requests.get(_SUBMISSIONS_URL.format(cik=cik), headers=_headers(), timeout=30)
        time.sleep(_SEC_DELAY)
        if resp.status_code != 200:
            return None
        sic = resp.json().get("sic")
        return int(sic) if sic else None
    except (requests.RequestException, ValueError):  # pragma: no cover
        return None


def build_universe(target: int = DEFAULT_TARGET, seed: int = SEED) -> list[dict]:
    """Select the 2018-anchored small/mid-cap universe.

    Args:
        target: Desired number of firms.
        seed: RNG seed for a reproducible sample.

    Returns:
        List of ``{ticker, cik, market_cap, sector}`` dicts, length ``<= target``.
    """
    filers = fetch_2018_tenk_ciks()
    tickers = load_ticker_map()

    pool = [(cik, tickers[cik]) for cik in filers if cik in tickers]
    logger.info("Candidate pool (2018 10-K filers with a current ticker): %d", len(pool))

    rng = random.Random(seed)
    rng.shuffle(pool)

    sector_cap = max(1, math.ceil(target * SECTOR_CAP_FRACTION))
    selected: list[dict] = []
    sector_counts: Counter[str] = Counter()
    probes = 0
    max_probes = target * _MAX_PROBE_FACTOR

    for cik, ticker in pool:
        if len(selected) >= target or probes >= max_probes:
            break
        probes += 1

        cap = _market_cap(ticker)
        if cap is None or not (CAP_MIN <= cap <= CAP_MAX):
            continue

        sector = _sic_division(_sic_for_cik(cik))
        if sector_counts[sector] >= sector_cap:
            continue

        selected.append(
            {"ticker": ticker, "cik": cik, "market_cap": round(cap, 2), "sector": sector}
        )
        sector_counts[sector] += 1
        if len(selected) % 25 == 0:
            logger.info("Selected %d/%d (probed %d)...", len(selected), target, probes)

    logger.info(
        "Done: %d firms selected from %d probes. Sector mix: %s",
        len(selected),
        probes,
        dict(sector_counts),
    )
    return selected


def main(argv: list[str] | None = None) -> None:
    """CLI entry: build the universe and write it to JSON."""
    parser = argparse.ArgumentParser(description="Build the 2018-anchored small/mid-cap universe.")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Number of firms.")
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=str,
        default="data/universe/smallmid_2018.json",
        help="Output JSON path.",
    )
    args = parser.parse_args(argv)

    universe = build_universe(target=args.target, seed=args.seed)
    out_path = _PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(universe, indent=2), encoding="utf-8")
    logger.info("Wrote %d firms to %s", len(universe), out_path)


if __name__ == "__main__":
    main()


__all__ = ["build_universe", "fetch_2018_tenk_ciks", "load_ticker_map"]
