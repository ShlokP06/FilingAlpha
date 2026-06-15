"""Offline unit tests for the classical-NLP signal functions.

These tests use only hand-built fixtures and the committed minimal LM lexicon
so they run with no database and no network access.
"""

from __future__ import annotations

import math

import pytest

from pipeline.signals.loughran_mcdonald import lm_tone, load_lm_dictionary
from pipeline.signals.readability import fog_readability
from pipeline.signals.risk_factors import risk_factor_delta
from pipeline.signals.similarity import yoy_similarity


# --------------------------------------------------------------------------- #
# Loughran-McDonald tone
# --------------------------------------------------------------------------- #
def test_lm_dictionary_fallback_has_all_categories() -> None:
    """The offline fallback lexicon exposes all three sentiment categories."""
    lexicon = load_lm_dictionary(path="data/raw/__does_not_exist__.csv")
    assert set(lexicon) == {"negative", "uncertainty", "litigious"}
    assert len(lexicon["negative"]) >= 30
    assert len(lexicon["uncertainty"]) >= 30
    assert len(lexicon["litigious"]) >= 30


def test_lm_tone_known_counts() -> None:
    """Tone fractions match hand-counted category words over total tokens.

    Text has 11 tokens: 'loss' and 'adverse' (negative), 'risk' and 'uncertain'
    (uncertainty), 'lawsuit' (litigious). So negative=2/11, uncertainty=2/11,
    litigious=1/11.
    """
    text = "The loss and adverse risk created uncertain lawsuit outcomes for shareholders"
    tone = lm_tone(text)
    total = 11
    assert tone["lm_negative"] == pytest.approx(2 / total)
    assert tone["lm_uncertainty"] == pytest.approx(2 / total)
    assert tone["lm_litigious"] == pytest.approx(1 / total)


def test_lm_tone_is_case_and_punctuation_insensitive() -> None:
    """Tokenisation upper-cases and strips punctuation/digits."""
    tone = lm_tone("LOSS, loss; Loss! 123")
    # 4 tokens (LOSS, loss, Loss, plus '123' is dropped -> 3 word tokens).
    assert tone["lm_negative"] == pytest.approx(3 / 3)


def test_lm_tone_empty_text_is_zero() -> None:
    """Empty / whitespace text yields zero tone, never a division error."""
    tone = lm_tone("   ")
    assert tone == {"lm_negative": 0.0, "lm_uncertainty": 0.0, "lm_litigious": 0.0}


# --------------------------------------------------------------------------- #
# YoY similarity (Lazy Prices)
# --------------------------------------------------------------------------- #
def test_yoy_similarity_identical_is_one() -> None:
    """Identical documents have cosine similarity exactly 1.0."""
    doc = "the company expanded operations and increased revenue across all segments"
    assert yoy_similarity(doc, doc) == pytest.approx(1.0)


def test_yoy_similarity_disjoint_is_zero() -> None:
    """Documents with no shared vocabulary have similarity 0.0."""
    a = "alpha bravo charlie delta echo"
    b = "foxtrot golf hotel india juliet"
    assert yoy_similarity(a, b) == pytest.approx(0.0, abs=1e-9)


def test_yoy_similarity_partial_overlap_in_unit_interval() -> None:
    """Partially overlapping documents score strictly between 0 and 1."""
    a = "revenue increased due to strong product demand this fiscal year"
    b = "revenue decreased due to weak product demand this fiscal year"
    sim = yoy_similarity(a, b)
    assert 0.0 < sim < 1.0


def test_yoy_similarity_empty_input_is_zero() -> None:
    """Empty input is treated as maximal divergence (0.0)."""
    assert yoy_similarity("", "anything") == 0.0
    assert yoy_similarity("anything", "") == 0.0


# --------------------------------------------------------------------------- #
# Risk-factor delta
# --------------------------------------------------------------------------- #
def test_risk_factor_delta_identical_is_zero() -> None:
    """Identical Item 1A text gives delta ~ 0 (no change)."""
    item = "our business faces competition regulatory risk and supply chain disruption"
    assert risk_factor_delta(item, item) == pytest.approx(0.0, abs=1e-9)


def test_risk_factor_delta_complete_change_is_one() -> None:
    """Completely different risk text gives delta ~ 1 (maximal change)."""
    a = "competition regulatory supply chain"
    b = "cybersecurity pandemic geopolitical inflation"
    assert risk_factor_delta(a, b) == pytest.approx(1.0, abs=1e-9)


def test_risk_factor_delta_is_one_minus_similarity() -> None:
    """Delta equals 1 - yoy_similarity for the same inputs."""
    a = "we depend on key customers and face interest rate risk"
    b = "we depend on key suppliers and face currency exchange risk"
    assert risk_factor_delta(a, b) == pytest.approx(1.0 - yoy_similarity(a, b))


# --------------------------------------------------------------------------- #
# Readability
# --------------------------------------------------------------------------- #
def test_fog_readability_positive_for_real_text() -> None:
    """Gunning-Fog returns a positive finite grade for ordinary prose."""
    text = (
        "The corporation experienced significant deterioration in operating "
        "performance during the period, attributable to unprecedented "
        "macroeconomic headwinds and intensifying competitive dynamics."
    )
    fog = fog_readability(text)
    assert fog > 0.0
    assert math.isfinite(fog)


def test_fog_readability_empty_is_zero() -> None:
    """Empty input returns 0.0 without raising."""
    assert fog_readability("") == 0.0
    assert fog_readability("   ") == 0.0
