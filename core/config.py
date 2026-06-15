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

    def filings_text_dir(self) -> Path:
        """Return (creating if needed) the directory for cached filing text."""
        path = self.data_dir / "filings"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
