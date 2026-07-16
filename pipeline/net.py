"""Network-robustness helpers for ingestion: retries, throttling, HTTP caching.

Ingesting 300 firms means hundreds of calls to Yahoo Finance (which has no
official API and rate-limits aggressively) and SEC EDGAR. This module centralises
the three defences that keep a large run from silently corrupting data:

1. **A shared, cached, rate-limit-friendly yfinance session** (:func:`make_yf_session`).
2. **Exponential backoff + jitter retries** on *transient* errors only
   (:data:`retry_yf`, :data:`retry_edgar`) — rate limits and network blips are
   retried; a clean "no data" (delisted) result is not an error and is left to
   the caller to record as terminal.
3. **Chunking** (:func:`chunked`) so prices are fetched in bulk batches rather
   than one HTTP request per ticker.

The durable checkpoint is the database itself: firms carry an ``ingest_state``
(:data:`STATE_*`) so anything that still fails after retries is marked
``failed`` and retried by the next batch run, never frozen mid-way.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from core.config import settings

logger = logging.getLogger(__name__)

__all__ = [
    "STATE_PENDING",
    "STATE_COMPLETE",
    "STATE_NO_DATA",
    "STATE_FAILED",
    "TRANSIENT_ERRORS",
    "make_yf_session",
    "retry_yf",
    "retry_edgar",
    "chunked",
]

# Ingestion lifecycle states (mirrors core.models.Company.ingest_state)
STATE_PENDING = "pending"
STATE_COMPLETE = "complete"
STATE_NO_DATA = "no_data"
STATE_FAILED = "failed"


# Transient error taxonomy — what is worth retrying
def _transient_error_types() -> tuple[type[BaseException], ...]:
    """Collect the exception types that represent *retryable* failures.

    Rate limits, timeouts, and connection resets are transient; a malformed
    response or a genuinely empty result is not (the caller records those as
    terminal ``no_data``). Imports are guarded so a missing optional transport
    (e.g. ``curl_cffi``) never breaks the import.

    Returns:
        Tuple of exception classes for :func:`tenacity.retry_if_exception_type`.
    """
    types: list[type[BaseException]] = [ConnectionError, TimeoutError, OSError]

    try:
        from yfinance.exceptions import YFRateLimitError

        types.append(YFRateLimitError)
    except Exception:  # pragma: no cover - yfinance internals vary by version
        logger.debug("YFRateLimitError not importable; relying on generic network errors")

    try:
        import requests

        types.append(requests.exceptions.RequestException)
    except Exception:  # pragma: no cover
        pass

    try:  # curl_cffi is yfinance's default transport in recent versions
        from curl_cffi.requests.exceptions import RequestsError

        types.append(RequestsError)
    except Exception:  # pragma: no cover
        pass

    return tuple(types)


TRANSIENT_ERRORS: tuple[type[BaseException], ...] = _transient_error_types()


# Retry decorators (tenacity)
def _make_retry(attempts: int):
    """Build a tenacity retry decorator with exponential backoff + jitter.

    Args:
        attempts: Maximum number of attempts (>= 1).

    Returns:
        A decorator that retries the wrapped callable on any transient error,
        backing off exponentially (2s, 4s, 8s, ... capped at
        ``settings.yf_backoff_max``) with jitter, then re-raises on exhaustion.
    """
    return retry(
        retry=retry_if_exception_type(TRANSIENT_ERRORS),
        wait=wait_exponential_jitter(initial=2.0, max=settings.yf_backoff_max, jitter=2.0),
        stop=stop_after_attempt(max(1, attempts)),
        reraise=True,
        before_sleep=lambda rs: logger.warning(
            "Transient error (attempt %d/%d): %r — backing off %.1fs",
            rs.attempt_number,
            max(1, attempts),
            rs.outcome.exception() if rs.outcome else None,
            rs.next_action.sleep if rs.next_action else 0.0,
        ),
    )


#: Retry decorator for yfinance calls (rate-limit / network aware).
retry_yf = _make_retry(settings.yf_max_retries)

#: Retry decorator for EDGAR calls (fewer attempts; SEC throttles server-side).
retry_edgar = _make_retry(settings.sec_max_retries)


# yfinance session
def make_yf_session():
    """Return the session yfinance should use, or ``None`` to let it self-manage.

    yfinance >= 1.x manages its own ``curl_cffi`` session with browser
    impersonation — itself a strong rate-limit deterrent — and explicitly
    *rejects* external caching sessions (``requests_cache``) with a
    ``YFDataException``. So the robust choice is to let yfinance handle its own
    transport and rely on our other defences (throttling between chunks,
    exponential-backoff retries, and the per-firm ``ingest_state`` machine,
    with the database itself as the durable cache).

    Returns:
        ``None`` — yfinance builds and impersonates its own session.
    """
    return None


# Chunking
T = TypeVar("T")


def chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield successive ``size``-length chunks of ``items``.

    Args:
        items: The sequence to split.
        size: Chunk length (coerced to at least 1).

    Yields:
        Lists of up to ``size`` items, preserving order.
    """
    step = max(1, size)
    for i in range(0, len(items), step):
        yield list(items[i : i + step])
