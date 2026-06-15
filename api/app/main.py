"""FilingAlpha FastAPI application entry point.

Wires together:
- CORS middleware (permissive for demo + Vite dev origin)
- Prometheus latency/count middleware
- Health and metrics endpoints
- Domain routers: companies, signals, backtests, predictions
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.app.metrics import REQUEST_COUNT, REQUEST_LATENCY
from api.app.routers import backtests, companies, predictions, signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FilingAlpha API",
    description=(
        "Read-only REST API over the FilingAlpha signal and backtest tables. "
        "Exposes SEC-filing-derived NLP signals and backtest results for the React dashboard."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Prometheus middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next: object) -> Response:
    """Record request latency and total count per path template.

    The matched route template (e.g. ``/signals/{ticker}``) is extracted
    *after* dispatch so that FastAPI has already resolved the route.

    Args:
        request: The incoming HTTP request.
        call_next: The next middleware or route handler.

    Returns:
        The HTTP response from the next handler.
    """
    start = time.perf_counter()
    response: Response = await call_next(request)  # type: ignore[operator]
    duration = time.perf_counter() - start

    route = request.scope.get("route")
    path_template: str = route.path if route else request.url.path

    REQUEST_LATENCY.labels(path=path_template).observe(duration)
    REQUEST_COUNT.labels(
        path=path_template,
        method=request.method,
        status_code=str(response.status_code),
    ).inc()

    return response


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Liveness probe — no database required.

    Returns:
        A dict ``{"status": "ok"}`` when the process is alive.
    """
    return {"status": "ok"}


@app.get("/metrics", tags=["system"])
def metrics_endpoint() -> Response:
    """Expose Prometheus metrics in text exposition format.

    Returns:
        Plain-text Prometheus exposition (content-type ``text/plain; version=0.0.4``).
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Domain routers
# ---------------------------------------------------------------------------
app.include_router(companies.router)
app.include_router(signals.router)
app.include_router(backtests.router)
app.include_router(predictions.router)
