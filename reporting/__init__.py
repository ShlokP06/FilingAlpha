"""Report generation: turn backtest/model results into a LaTeX/PDF research note.

The package is deliberately split so the heavy parts are unit-testable without a
database or a network call:

* :mod:`reporting.data` — pull the latest per-form results into plain dataclasses.
* :mod:`reporting.plots` — render matplotlib figures from those dataclasses.
* :mod:`reporting.narrative` — write the prose interpretation (LLM or a
  deterministic fallback). The LLM only ever *narrates* numbers it is handed; it
  never originates a quantitative result.
* :mod:`reporting.latex` — assemble the ``.tex`` document and compile it.
* :mod:`reporting.report` — orchestrate the above into a finished report.
"""

from __future__ import annotations

from reporting.report import generate_report

__all__ = ["generate_report"]
