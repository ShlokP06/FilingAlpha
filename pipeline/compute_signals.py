"""Compute per-filing classical-NLP signals and persist :class:`Signal` rows.

This is the orchestration glue between the pure signal functions
(``pipeline/signals/*``) and the database: for every filing it locates the
prior-year filing of the same company, computes the six signal values, and
upserts a ``Signal`` row idempotently.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models import Filing, Signal
from pipeline.signals.loughran_mcdonald import lm_tone
from pipeline.signals.readability import fog_readability
from pipeline.signals.risk_factors import risk_factor_delta
from pipeline.signals.similarity import yoy_similarity

logger = logging.getLogger(__name__)


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


def compute_signals(session: Session) -> int:
    """Compute and upsert signal rows for every filing.

    For each filing, the prior filing of the same company and form type (by
    filing date) supplies the year-over-year comparison. Signals that need a
    prior filing (``yoy_similarity``, ``risk_factor_delta``) are left ``None``
    for a company's first filing.

    Args:
        session: Active database session.

    Returns:
        Number of ``Signal`` rows written or updated.
    """
    filings = list(
        session.execute(
            select(Filing).order_by(Filing.company_id.asc(), Filing.filing_date.asc())
        ).scalars()
    )

    # Group filings by (company, form_type) so we can find each one's predecessor.
    by_group: dict[tuple[int, str], list[Filing]] = {}
    for filing in filings:
        by_group.setdefault((filing.company_id, filing.form_type), []).append(filing)

    written = 0
    for group in by_group.values():
        prev: Filing | None = None
        for filing in group:  # already chronologically ordered
            full_text = _read_full_text(filing)
            tone = lm_tone(full_text) if full_text else {}

            yoy = None
            rf_delta = None
            if prev is not None:
                prev_text = _read_full_text(prev)
                if full_text and prev_text:
                    yoy = yoy_similarity(full_text, prev_text)
                if filing.item1a_text and prev.item1a_text:
                    rf_delta = risk_factor_delta(filing.item1a_text, prev.item1a_text)

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
            prev = filing

    session.commit()
    logger.info("Computed signals for %d filings.", written)
    return written
