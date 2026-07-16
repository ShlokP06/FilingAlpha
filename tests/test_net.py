"""Unit tests for pipeline/net.py — retry, chunking, and session helpers.

Fully offline. Retry timing is neutralised by replacing the tenacity retrying
object's ``sleep`` with a no-op, so these tests never actually back off.
"""

from __future__ import annotations

import pytest

from core.config import settings
from pipeline import net


class TestChunked:
    def test_splits_into_expected_sizes(self):
        assert list(net.chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    def test_exact_multiple(self):
        assert list(net.chunked([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_empty_input(self):
        assert list(net.chunked([], 3)) == []

    def test_size_coerced_to_at_least_one(self):
        assert list(net.chunked([1, 2], 0)) == [[1], [2]]


class TestTransientErrors:
    def test_includes_common_network_errors(self):
        assert ConnectionError in net.TRANSIENT_ERRORS
        assert TimeoutError in net.TRANSIENT_ERRORS

    def test_includes_yfinance_rate_limit(self):
        from yfinance.exceptions import YFRateLimitError

        assert YFRateLimitError in net.TRANSIENT_ERRORS


class TestRetryYf:
    def test_retries_transient_then_succeeds(self):
        """A transient error should be retried until the call succeeds."""
        calls = {"n": 0}

        @net.retry_yf
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        flaky.retry.sleep = lambda *_: None  # no real backoff
        assert flaky() == "ok"
        assert calls["n"] == 3

    def test_reraises_after_exhausting_attempts(self):
        """A persistent transient error should reraise after max attempts."""
        calls = {"n": 0}

        @net.retry_yf
        def always_fail():
            calls["n"] += 1
            raise ConnectionError("still down")

        always_fail.retry.sleep = lambda *_: None
        with pytest.raises(ConnectionError):
            always_fail()
        assert calls["n"] == settings.yf_max_retries

    def test_does_not_retry_non_transient(self):
        """A non-transient error (e.g. ValueError) must fail on the first try."""
        calls = {"n": 0}

        @net.retry_yf
        def bad():
            calls["n"] += 1
            raise ValueError("logic error")

        bad.retry.sleep = lambda *_: None
        with pytest.raises(ValueError):
            bad()
        assert calls["n"] == 1


class TestRetryEdgar:
    def test_uses_sec_attempt_budget(self):
        calls = {"n": 0}

        @net.retry_edgar
        def always_fail():
            calls["n"] += 1
            raise ConnectionError("sec down")

        always_fail.retry.sleep = lambda *_: None
        with pytest.raises(ConnectionError):
            always_fail()
        assert calls["n"] == settings.sec_max_retries


class TestMakeYfSession:
    def test_returns_none_so_yfinance_self_manages(self):
        # yfinance >= 1.x rejects external caching sessions and manages its own
        # curl_cffi impersonation transport, so we hand it None.
        assert net.make_yf_session() is None
