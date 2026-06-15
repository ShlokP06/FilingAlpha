"""Loughran-McDonald tone signal.

Implements the sentiment-tone measures of Loughran & McDonald (2011),
"When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and
10-Ks", *Journal of Finance* 66(1). For each LM sentiment category the signal
is the count of category words divided by the total word count of the document
(a normalised frequency).

The official LM Master Dictionary is loaded from
``data/raw/lm_master_dictionary.csv`` when present; otherwise a committed
minimal lexicon (:mod:`pipeline.signals.lm_lexicon_min`) is used so the signal
works fully offline.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from pipeline.signals import lm_lexicon_min

logger = logging.getLogger(__name__)

DEFAULT_DICT_PATH = Path("data/raw/lm_master_dictionary.csv")

# LM category name -> the column flag in the master dictionary CSV.
_CATEGORY_COLUMNS: dict[str, str] = {
    "negative": "Negative",
    "uncertainty": "Uncertainty",
    "litigious": "Litigious",
}

# Word tokeniser: contiguous A-Z letters (LM dictionaries are letter-only).
_WORD_RE = re.compile(r"[A-Za-z]+")


def _tokenize(text: str) -> list[str]:
    """Tokenise ``text`` into upper-cased alphabetic words.

    Args:
        text: Raw document text.

    Returns:
        List of upper-cased word tokens (punctuation and digits stripped).
    """
    return [m.group(0).upper() for m in _WORD_RE.finditer(text)]


def _fallback_dictionary() -> dict[str, frozenset[str]]:
    """Return the committed minimal lexicon keyed by category name."""
    return {
        "negative": lm_lexicon_min.NEGATIVE,
        "uncertainty": lm_lexicon_min.UNCERTAINTY,
        "litigious": lm_lexicon_min.LITIGIOUS,
    }


@lru_cache(maxsize=4)
def load_lm_dictionary(path: str | None = None) -> dict[str, frozenset[str]]:
    """Load the Loughran-McDonald sentiment lexicon.

    Reads the official LM Master Dictionary CSV if it exists, mapping each
    category-flag column (``Negative``/``Uncertainty``/``Litigious``) to the set
    of words flagged for that category. A word is flagged when its column value
    is non-zero (the official file encodes membership as the year of inclusion).
    If the CSV is missing or unreadable, falls back to the committed minimal
    lexicon so the signal runs offline.

    Args:
        path: Optional path to the master dictionary CSV. Defaults to
            ``data/raw/lm_master_dictionary.csv``.

    Returns:
        Mapping of category name (``"negative"``, ``"uncertainty"``,
        ``"litigious"``) to a frozenset of upper-cased member words.
    """
    csv_path = Path(path) if path is not None else DEFAULT_DICT_PATH
    if not csv_path.exists():
        logger.info(
            "LM master dictionary not found at %s; using minimal fallback lexicon.",
            csv_path,
        )
        return _fallback_dictionary()

    try:
        frame = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001 - any read error must fall back gracefully
        logger.warning("Failed to read LM dictionary at %s; using fallback.", csv_path)
        return _fallback_dictionary()

    if "Word" not in frame.columns:
        logger.warning("LM dictionary at %s lacks 'Word' column; using fallback.", csv_path)
        return _fallback_dictionary()

    words = frame["Word"].astype(str).str.upper()
    lexicon: dict[str, frozenset[str]] = {}
    for category, column in _CATEGORY_COLUMNS.items():
        if column not in frame.columns:
            logger.warning("LM dictionary missing column %s; category empty.", column)
            lexicon[category] = frozenset()
            continue
        flags = pd.to_numeric(frame[column], errors="coerce").fillna(0) != 0
        lexicon[category] = frozenset(words[flags].tolist())

    logger.info(
        "Loaded LM dictionary from %s (neg=%d, unc=%d, lit=%d).",
        csv_path,
        len(lexicon["negative"]),
        len(lexicon["uncertainty"]),
        len(lexicon["litigious"]),
    )
    return lexicon


def lm_tone(text: str, path: str | None = None) -> dict[str, float]:
    """Compute Loughran-McDonald tone fractions for ``text``.

    Each value is the number of words belonging to the category divided by the
    total token count, matching the normalised word-frequency measure of
    Loughran & McDonald (2011).

    Args:
        text: Raw document text.
        path: Optional override path for the LM dictionary CSV.

    Returns:
        Mapping with keys ``lm_negative``, ``lm_uncertainty``, ``lm_litigious``,
        each a float in ``[0, 1]``. Returns zeros for empty/whitespace input.
    """
    tokens = _tokenize(text)
    total = len(tokens)
    if total == 0:
        return {"lm_negative": 0.0, "lm_uncertainty": 0.0, "lm_litigious": 0.0}

    lexicon = load_lm_dictionary(path)
    counts = {category: 0 for category in lexicon}
    for token in tokens:
        for category, members in lexicon.items():
            if token in members:
                counts[category] += 1

    return {
        "lm_negative": counts["negative"] / total,
        "lm_uncertainty": counts["uncertainty"] / total,
        "lm_litigious": counts["litigious"] / total,
    }


__all__ = ["load_lm_dictionary", "lm_tone"]
