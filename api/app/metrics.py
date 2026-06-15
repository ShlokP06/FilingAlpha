"""Prometheus metric objects shared across the API.

These are module-level singletons so they are registered once with the
default registry, not recreated per-request.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_LATENCY: Histogram = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

REQUEST_COUNT: Counter = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    labelnames=["path", "method", "status_code"],
)

__all__ = ["REQUEST_LATENCY", "REQUEST_COUNT"]
