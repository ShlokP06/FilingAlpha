"""Readability signal via the Gunning-Fog index.

Filing complexity is a classical text feature linked to information processing
costs (e.g. Li 2008; Loughran & McDonald 2014 on the "fog" index in 10-Ks).
Higher Fog scores indicate harder-to-read disclosures.
"""

from __future__ import annotations

import logging

import textstat

logger = logging.getLogger(__name__)


def fog_readability(text: str) -> float:
    """Compute the Gunning-Fog readability index for ``text``.

    Args:
        text: Raw document text.

    Returns:
        The Gunning-Fog index (roughly, US school grade level required to read
        the text on a first pass). Returns ``0.0`` for empty/whitespace input or
        if the underlying computation fails.
    """
    if not text or not text.strip():
        return 0.0
    try:
        return float(textstat.gunning_fog(text))
    except Exception:  # noqa: BLE001 - readability libs can raise on odd input
        logger.warning("gunning_fog failed on input of length %d; returning 0.0", len(text))
        return 0.0


__all__ = ["fog_readability"]
