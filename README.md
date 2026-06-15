# FilingAlpha

**Backtested text signals from SEC filings — classical NLP, no LLM.**

FilingAlpha turns the *text* of SEC 10-K filings into quantitative trading signals
using established finance-NLP methods, then tests — with rigorous, lookahead-free
backtesting — whether those signals predict forward stock returns. It is a
production-shaped backend (FastAPI + PostgreSQL + Docker + Prometheus/Grafana),
not a notebook.

The point is not to claim a money-printing edge. It is to **replicate documented
academic signals and measure them honestly** with the same discipline a
quant-research team would demand: point-in-time data, no lookahead, transaction
costs, information coefficients, and long-short Sharpe ratios.

---

## Signals (each anchored to published research)

| Signal | Definition | Reference |
|--------|-----------|-----------|
| **Loughran-McDonald tone** | Fraction of words in the LM finance dictionary's *Negative / Uncertainty / Litigious* categories | Loughran & McDonald (2011), *J. Finance* |
| **YoY textual change ("Lazy Prices")** | TF-IDF cosine similarity to the prior year's 10-K — low similarity (big change) is the predictive event | Cohen, Malloy & Nguyen (2020), *J. Finance* |
| **Risk-factor delta** | `1 − cosine` of Item 1A (Risk Factors) year-over-year | — |
| **Readability (Fog)** | Gunning-Fog index of the MD&A | Li (2008), *J. Acct. Econ.* |

> The Loughran-McDonald **Master Dictionary** is not redistributed here. Download it
> from [sraf.nd.edu](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) to
> `data/raw/lm_master_dictionary.csv`; without it, a small bundled fallback lexicon
> is used so the code runs offline.

## Backtest rigor

- **Point-in-time / filing-lag** — a filing's signal is only actionable *after* it is
  filed; forward returns start from the first trading day **strictly after** the
  filing date. No same-day lookahead.
- **Walk-forward, expanding window** — the ML model only ever trains on data that
  precedes the test fold. A unit test asserts `max(train date) < min(test date)` for
  every fold.
- **Honest metrics** — Spearman information coefficient (with t-stat), long-short
  tercile Sharpe (net of transaction costs), hit rate. Weak signals are reported as
  weak.

## Architecture

```
React/Vite dashboard ──REST──► FastAPI (:8000) ──SQLAlchemy──► PostgreSQL (:5432)
                                     │                              ▲
                                  /metrics                          │ pipeline writes
                                     ▼                              │
                          Prometheus (:9090) ──► Grafana (:3000)    │
  pipeline:  ingest (edgartools + yfinance) ─► signals ─► forward returns ─► backtest ─► walk-forward model
```

- `core/` — shared SQLAlchemy models, Pydantic schemas, DB session, settings.
- `pipeline/` — ingest, the classical-NLP signals, returns, backtest, model.
- `api/` — FastAPI read API over the computed tables.
- `frontend/` — Vite + React + Tremor dashboard (Signal Explorer, Backtest, Model).
- `migrations/` — Alembic.
- `monitoring/` — Prometheus + Grafana provisioning.

## Quickstart

```bash
cp .env.example .env
uv venv --python 3.11 && uv pip install -e ".[dev]"

docker compose up -d postgres                 # or: docker compose up -d  (full stack)
uv run alembic upgrade head                   # create schema
uv run python scripts/run_pipeline.py         # ingest -> signals -> returns -> backtest -> model
uv run pytest                                  # offline suite (integration tests auto-skip)

# explore
curl localhost:8000/signals/AAPL
curl localhost:8000/backtests
# UI: http://localhost:5173   Grafana: http://localhost:3000   Prometheus: http://localhost:9090
```

Run the full stack with `docker compose up`. Integration tests require Postgres and
network; enable them with `RUN_INTEGRATION=1 uv run pytest -m integration`.

## Backtest results

Run `scripts/run_pipeline.py` to populate the universe and print the IC / Sharpe
table; paste the table here once you've run it on your chosen universe. Report it
straight — a correctly-measured null result is a feature of this project, not a bug.

## Tech

FastAPI · PostgreSQL · SQLAlchemy 2.0 · Alembic · scikit-learn · pandas · edgartools ·
yfinance · textstat · Vite/React/TypeScript/Tremor · Docker Compose · Prometheus · Grafana

## Resume angles

- **Backend/SDE** — typed FastAPI (routers by domain, DI'd sessions, Pydantic contracts),
  Postgres + Alembic migrations, idempotent ingestion, Docker, CI, Prometheus metrics.
- **Hedge-Fund DS** — published text signals, point-in-time/no-lookahead backtest,
  transaction costs, IC + t-stat + long-short Sharpe.
- **Applied ML** — walk-forward classifier on the engineered features, OOS evaluation.
