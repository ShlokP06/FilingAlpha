"""Risk-factor (Item 1A) change signal.

Measures how much a firm's "Item 1A. Risk Factors" section changed versus the
prior year. Defined as ``1 - cosine_similarity`` of the Item 1A text across
years, so larger values indicate larger disclosed-risk revisions. This mirrors
the change-in-disclosure intuition of Cohen, Malloy & Nguyen (2020) applied to
the risk-factors section specifically.
"""

from __future__ import annotations

from pipeline.signals.similarity import tfidf_cosine


def risk_factor_delta(curr_item1a: str, prev_item1a: str) -> float:
    """Compute the year-over-year change in Item 1A risk-factor text.

    Args:
        curr_item1a: Current year's Item 1A (Risk Factors) text.
        prev_item1a: Prior year's Item 1A text.

    Returns:
        ``1 - cosine_similarity`` in ``[0, 1]``. ``0.0`` for identical text
        (no change) and larger values for greater divergence.
    """
    similarity = tfidf_cosine(curr_item1a, prev_item1a)
    delta = 1.0 - similarity
    # Guard against tiny floating-point negatives.
    return max(0.0, delta)


__all__ = ["risk_factor_delta"]
