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
- **Honest metrics** — Spearman information coefficient (with t-stat) and an
  event-study tercile spread: top-minus-bottom forward return (net of transaction
  costs) with a Welch t-stat. Annual filings are too sparse for a per-rebalance-date
  long-short portfolio, so each filing is treated as an independent event. Weak,
  insignificant signals are reported as weak.

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

Universe: **12 large-cap US companies** (AAPL, BA, CAT, DIS, JNJ, JPM, KO, MSFT, NVDA, PG, WMT, XOM) —
**72 10-K filings**, 6-year daily price history (18,135 closes), two forward horizons (21 and 63 trading
days). Reported straight: a correctly-measured null result is a feature of this project, not a bug.

**Signal evaluation** (rank-IC with t-stat; event-study top-minus-bottom tercile spread, net of 10 bps/side,
with a Welch t-stat):

| Signal | Horizon | IC | IC t | L-S spread | Spread t |
|--------|--------:|------:|------:|-----------:|---------:|
| fog_readability    | 21 | -0.151 | -1.28 | -0.0334 | -1.33 |
| lm_litigious       | 21 |  0.081 |  0.68 |  0.0198 |  1.03 |
| lm_litigious       | 63 |  0.035 |  0.29 |  0.0257 |  0.77 |
| yoy_similarity     | 63 |  0.096 |  0.74 |  0.0183 |  0.73 |
| risk_factor_delta  | 63 | -0.042 | -0.31 |  0.0333 |  0.66 |
| fog_readability    | 63 |  0.006 |  0.05 | -0.0227 | -0.57 |
| lm_uncertainty     | 63 |  0.097 |  0.82 |  0.0143 |  0.39 |
| lm_negative        | 63 | -0.037 | -0.31 | -0.0146 | -0.36 |
| risk_factor_delta  | 21 |  0.068 |  0.50 |  0.0055 |  0.27 |
| yoy_similarity     | 21 | -0.012 | -0.09 | -0.0075 | -0.25 |
| lm_negative        | 21 | -0.032 | -0.27 |  0.0005 |  0.11 |
| lm_uncertainty     | 21 | -0.002 | -0.02 | -0.0010 |  0.04 |

**Walk-forward classifier** (GradientBoostingClassifier, 5 expanding-window folds, 47 out-of-sample obs):

| Horizon | OOS accuracy | OOS ROC-AUC |
|--------:|-------------:|------------:|
| 21d | 0.489 | 0.475 |
| 63d | 0.574 | 0.500 |

No signal is statistically significant (every \|t\| < 1.4) and out-of-sample accuracy is ~coin-flip — the
expected, honest outcome on a small single-name universe. The deliverable is the **leakage-free measurement
harness** (point-in-time filing lag, cross-sectional IC, event-study spread, an expanding-window walk-forward
that asserts `max(train date) < min(test date)`), not an alpha claim.

## Tech

FastAPI · PostgreSQL · SQLAlchemy 2.0 · Alembic · scikit-learn · pandas · edgartools ·
yfinance · textstat · Vite/React/TypeScript/Tremor · Docker Compose · Prometheus · Grafana

## Resume angles

- **Backend/SDE** — typed FastAPI (routers by domain, DI'd sessions, Pydantic contracts),
  Postgres + Alembic migrations, idempotent ingestion, Docker, CI, Prometheus metrics.
- **Hedge-Fund DS** — published text signals, point-in-time/no-lookahead backtest,
  transaction costs, rank-IC + t-stat + event-study tercile spread.
- **Applied ML** — walk-forward classifier on the engineered features, OOS evaluation.
