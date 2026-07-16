"""Compute per-filing classical-NLP signals and persist :class:`Signal` rows.

This is the orchestration glue between the pure signal functions
(``pipeline/signals/*``) and the database: for every filing it locates the
prior-year filing of the same company, computes the six signal values, and
upserts a ``Signal`` row idempotently.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from core.models import Filing, Signal
from pipeline.signals.loughran_mcdonald import lm_tone
from pipeline.signals.readability import fog_readability
from pipeline.signals.risk_factors import risk_factor_delta
from pipeline.signals.similarity import yoy_similarity

logger = logging.getLogger(__name__)

# Year-over-year comparison tolerance: a prior filing qualifies as the
# "same period last year" if its reference date is 365 +/- 90 days earlier.
_YOY_TOLERANCE_DAYS = 90


def _read_full_text(filing: Filing) -> str:
    """Return the filing's cached full text, falling back to stored sections."""
    if filing.text_path:
        path = Path(filing.text_path)
        if path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("Could not read %s: %s", path, exc)
    return " ".join(filter(None, [filing.item1a_text, filing.mdna_text]))


def _ref_date(filing: Filing) -> date:
    """Reference date for year-over-year matching (period end, else filing date)."""
    return filing.period_end or filing.filing_date


def _prior_year_filing(filing: Filing, earlier: list[Filing]) -> Filing | None:
    """Pick the same-period prior-year filing from chronologically earlier ones.

    Among earlier same-form filings, returns the one whose reference date is
    closest to one year before ``filing`` (within ``_YOY_TOLERANCE_DAYS``). For
    annual 10-Ks this is the immediately preceding filing; for quarterly 10-Qs
    it is the same quarter a year earlier, which avoids comparing, say, a Q3
    against a Q2 and picking up seasonal rather than informative text change.

    Args:
        filing: The filing whose prior-year comparison base is wanted.
        earlier: Chronologically earlier filings of the same company and form.

    Returns:
        The best matching prior-year filing, or None if none falls in range.
    """
    target = _ref_date(filing)
    best: Filing | None = None
    best_gap: int | None = None
    for candidate in earlier:
        days = (target - _ref_date(candidate)).days
        if days <= 0:
            continue
        gap = abs(days - 365)
        if gap <= _YOY_TOLERANCE_DAYS and (best_gap is None or gap < best_gap):
            best, best_gap = candidate, gap
    return best


def compute_signals(session: Session, *, shard: int = 0, num_shards: int = 1) -> int:
    """Compute and upsert signal rows for every filing.

    For each filing, the prior filing of the same company and form type (by
    filing date) supplies the year-over-year comparison. Signals that need a
    prior filing (``yoy_similarity``, ``risk_factor_delta``) are left ``None``
    for a company's first filing.

    The work is CPU/disk-bound (full-text reads + TF-IDF cosine per filing), so
    it can be sharded across processes: pass ``num_shards``/``shard`` to have
    this process own only companies where ``company_id % num_shards == shard``.
    Sharding by *company* keeps every company's whole filing history — all forms
    and prior-year comparisons — inside one shard, so shards never contend and
    each writes a disjoint set of ``Signal`` rows.

    Args:
        session: Active database session.
        shard: This process's shard index in ``[0, num_shards)``.
        num_shards: Total number of cooperating shards (1 = no sharding).

    Returns:
        Number of ``Signal`` rows written or updated.
    """
    if not 0 <= shard < num_shards:
        raise ValueError("shard must be in [0, num_shards)")

    query = select(Filing).order_by(Filing.company_id.asc(), Filing.filing_date.asc())
    if num_shards > 1:
        query = query.where(Filing.company_id % num_shards == shard)
    filings = list(session.execute(query).scalars())

    # Group filings by (company, form_type) so we can find each one's predecessor.
    by_group: dict[tuple[int, str], list[Filing]] = {}
    for filing in filings:
        by_group.setdefault((filing.company_id, filing.form_type), []).append(filing)

    desc = f"Signals[{shard}/{num_shards}]" if num_shards > 1 else "Signals"
    progress = tqdm(total=len(filings), desc=desc, unit="filing", dynamic_ncols=True)

    written = 0
    for group in by_group.values():
        for idx, filing in enumerate(group):  # group is chronologically ordered
            progress.update(1)
            full_text = _read_full_text(filing)
            tone = lm_tone(full_text) if full_text else {}

            yoy = None
            rf_delta = None
            base = _prior_year_filing(filing, group[:idx])
            if base is not None:
                base_text = _read_full_text(base)
                if full_text and base_text:
                    yoy = yoy_similarity(full_text, base_text)
                if filing.item1a_text and base.item1a_text:
                    rf_delta = risk_factor_delta(filing.item1a_text, base.item1a_text)

            fog = fog_readability(filing.mdna_text or full_text)

            values = {
                "company_id": filing.company_id,
                "filing_date": filing.filing_date,
                "lm_negative": tone.get("lm_negative"),
                "lm_uncertainty": tone.get("lm_uncertainty"),
                "lm_litigious": tone.get("lm_litigious"),
                "yoy_similarity": yoy,
                "risk_factor_delta": rf_delta,
                "fog_readability": fog,
            }

            existing = session.execute(
                select(Signal).where(Signal.filing_id == filing.id)
            ).scalar_one_or_none()
            if existing is None:
                session.add(Signal(filing_id=filing.id, **values))
            else:
                for key, val in values.items():
                    setattr(existing, key, val)
            written += 1

    progress.close()
    session.commit()
    logger.info("Computed signals for %d filings.", written)
    return written
