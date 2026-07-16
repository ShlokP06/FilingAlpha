"""Application settings loaded from the environment / ``.env`` file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration shared by the API and the pipeline.

    Attributes:
        database_url: SQLAlchemy URL (psycopg3 driver).
        sec_identity: Identity string SEC EDGAR requires on every request
            (a name and contact email).
        data_dir: Directory where raw filing text is cached.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://filing:filing@localhost:5432/filing_alpha"
    sec_identity: str = "FilingAlpha research che240008027@iiti.ac.in"
    data_dir: Path = Path("data/raw")
    reports_dir: Path = Path("reports")

    # Ingestion robustness (see pipeline/net.py). These govern how the
    # price/filing ingester survives Yahoo/SEC rate limits at 300-firm scale:
    # bulk-fetch in chunks, throttle between chunks, and back off + retry on
    # transient errors. Firms that still fail are marked ``failed`` and retried
    # on the next batch run rather than silently frozen. yfinance manages its
    # own curl_cffi impersonation session, so there is no HTTP-cache knob here.
    yf_chunk_size: int = 30  # tickers per bulk yf.download() call
    yf_chunk_pause: float = 1.0  # seconds to sleep between bulk chunks
    yf_max_retries: int = 5  # attempts per transient-failing yfinance call
    yf_backoff_max: float = 60.0  # cap (seconds) on exponential backoff wait
    sec_max_retries: int = 3  # attempts per transient-failing EDGAR call
    # Circuit breaker: after this many *consecutive* throttled bulk chunks, stop
    # issuing new price calls this run and mark the rest ``failed`` (retried next
    # run) rather than hammering an already-throttled Yahoo for the whole batch.
    yf_max_failed_chunks: int = 3
    # When False, skip the per-filing full-text download (``ef.text()``) and rely
    # on the extracted Item 1A / Item 7 sections instead — a faster, lighter run.
    # Whole-document signals (yoy_similarity, fog_readability) then fall back to
    # those sections, so keep this True for the full-fidelity backtest.
    ingest_full_text: bool = True

    # Report narrative LLM. Optional: when no key is set, the report falls back
    # to a deterministic template, so the pipeline never depends on a network
    # call. ``llm_provider`` selects which key/SDK to use.
    llm_provider: str = "cerebras"  # "cerebras" | "groq"
    llm_model: str = "llama-3.3-70b"
    cerebras_api_key: str | None = None
    groq_api_key: str | None = None

    def filings_text_dir(self) -> Path:
        """Return (creating if needed) the directory for cached filing text."""
        path = self.data_dir / "filings"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def reports_figures_dir(self) -> Path:
        """Return (creating if needed) the directory for generated report figures."""
        path = self.reports_dir / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
