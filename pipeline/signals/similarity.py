"""Year-over-year textual similarity signal ("Lazy Prices").

Implements the document-similarity measure of Cohen, Malloy & Nguyen (2020),
"Lazy Prices", *Journal of Finance* 75(3). The signal is the TF-IDF cosine
similarity between consecutive years' filings for the same firm. A *low*
similarity (a large textual change relative to the prior year) is the
predictive event the paper documents.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _has_tokens(text: str) -> bool:
    """Return whether ``text`` contains at least one non-whitespace character."""
    return bool(text and text.strip())


def tfidf_cosine(curr_text: str, prev_text: str) -> float:
    """Compute TF-IDF cosine similarity between two documents.

    The vectoriser is fit on exactly the two documents being compared (as in the
    pairwise "Lazy Prices" construction), then their TF-IDF vectors are compared
    with cosine similarity.

    Args:
        curr_text: Current-period document text.
        prev_text: Prior-period document text.

    Returns:
        Cosine similarity in ``[0, 1]``. Returns ``0.0`` when either document is
        empty (maximal change) and ``1.0`` for two empty documents is *not*
        defined, so empty input yields ``0.0``.
    """
    if not _has_tokens(curr_text) or not _has_tokens(prev_text):
        return 0.0

    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform([curr_text, prev_text])
    except ValueError:
        # Raised when the documents share no usable vocabulary (e.g. all
        # stop-words / punctuation stripped). Treat as maximal divergence.
        return 0.0

    similarity = cosine_similarity(matrix[0], matrix[1])[0, 0]
    return float(similarity)


def yoy_similarity(curr_text: str, prev_text: str) -> float:
    """Year-over-year TF-IDF cosine similarity between consecutive filings.

    Args:
        curr_text: Current year's filing text.
        prev_text: Prior year's filing text.

    Returns:
        Cosine similarity in ``[0, 1]``; ``1.0`` for identical text, lower for
        larger textual changes.
    """
    return tfidf_cosine(curr_text, prev_text)


__all__ = ["yoy_similarity", "tfidf_cosine"]
