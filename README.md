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

Universe: a **2018-anchored small/mid-cap universe** (\$300M–\$10B cap band, sector-balanced, survivorship-aware —
built point-in-time by `scripts/build_universe.py`). This run ingested **60 firms** (55 with complete price
history), **1,588 filings** (406 10-K + 1,182 10-Q), **98,347 daily closes**, and two forward horizons (21 and 63
trading days). Results are reported straight.

**Headline:** on 10-Ks, the **Loughran-McDonald negative-tone** signal reproduces the documented anomaly with
statistical significance — more negative tone predicts *lower* filing-lagged forward returns (rank-IC with t-stat;
event-study top-minus-bottom tercile spread, net of 10 bps/side, with a Welch t-stat):

| Signal (10-K)      | Horizon | IC | IC t | L-S spread | Spread t |
|--------------------|--------:|------:|------:|-----------:|---------:|
| lm_negative        | 21 | -0.447 | **-3.38** | -0.0608 | **-2.66** |
| fog_readability    | 21 | -0.187 | -0.93 | -0.0406 | **-2.24** |
| yoy_similarity     | 21 | -0.113 | **-2.02** | -0.0139 | -0.48 |
| lm_negative        | 63 | -0.090 | -0.37 | -0.0615 | -1.55 |

On 10-Qs a few spreads clear \|t\|>2 (fog_readability 63d t=3.08; lm_negative 63d t=2.83) but with inconsistent
sign — likely noise at this sample size. The remaining signal×horizon cells are insignificant. (The full 24-row
table is printed by `scripts/run_pipeline.py`.)

**Walk-forward classifier** (GradientBoostingClassifier, 5 expanding-window folds; **10-K only**, since
`yoy_similarity` and `risk_factor_delta` are year-over-year signals and undefined for 10-Qs):

| Horizon | Folds | OOS obs | OOS accuracy | OOS ROC-AUC |
|--------:|------:|--------:|-------------:|------------:|
| 21d | 5 | 255 | 0.541 | 0.514 |
| 63d | 5 | 255 | 0.467 | 0.433 |

The nuance is the point: a **statistically significant cross-sectional event-study signal** (LM negative tone,
spread t=-2.66) that **does not compound into an out-of-sample tradeable classifier** (OOS accuracy ~coin-flip).
That gap — real in-sample signal, no OOS edge once the temporal split is enforced — is exactly what a rigorous,
leakage-free harness is built to reveal. The deliverable is the **measurement discipline** (point-in-time filing
lag, cross-sectional IC, event-study tercile spread, an expanding-window walk-forward that asserts
`max(train date) < min(test date)`), not an alpha claim.

## Tech

FastAPI · PostgreSQL · SQLAlchemy 2.0 · Alembic · scikit-learn · pandas · edgartools ·
yfinance · textstat · Vite/React/TypeScript/Tremor · Docker Compose · Prometheus · Grafana

## Resume angles

- **Backend/SDE** — typed FastAPI (routers by domain, DI'd sessions, Pydantic contracts),
  Postgres + Alembic migrations, idempotent ingestion, Docker, CI, Prometheus metrics.
- **Hedge-Fund DS** — published text signals, point-in-time/no-lookahead backtest,
  transaction costs, rank-IC + t-stat + event-study tercile spread.
- **Applied ML** — walk-forward classifier on the engineered features, OOS evaluation.
