"""Orchestrate report generation: data -> figures -> narrative -> LaTeX/PDF."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from core.config import settings
from reporting.data import build_report_data
from reporting.latex import write_report
from reporting.narrative import generate_narrative
from reporting.plots import render_figures

logger = logging.getLogger(__name__)


def generate_report(session: Session, out_dir: Path | None = None) -> dict[str, Path]:
    """Generate the full research note from current database results.

    Pulls the latest per-form backtest and model results, renders figures, writes
    the interpretation prose (LLM or deterministic fallback), and assembles a
    LaTeX document compiled to PDF when an engine is available.

    Args:
        session: Active database session.
        out_dir: Output directory; defaults to ``settings.reports_dir``.

    Returns:
        Mapping of output artifacts (``tex`` always; ``pdf`` when compiled), plus
        the figure paths under their keys.
    """
    out_dir = out_dir or settings.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_report_data(session)
    figures = render_figures(data, out_dir / "figures")
    narrative = generate_narrative(data)
    outputs = write_report(data, figures, narrative, out_dir)

    logger.info(
        "Report generated: %s",
        ", ".join(f"{k}={v.name}" for k, v in outputs.items()),
    )
    return outputs


def main() -> None:
    """CLI entry point: generate the report from the configured database."""
    from core.db import SessionLocal

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    with SessionLocal() as session:
        outputs = generate_report(session)
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
