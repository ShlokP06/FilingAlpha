"""Write the report's prose interpretation, optionally via an LLM.

**Safety boundary.** Every quantitative figure in the report (tables, charts) is
filled deterministically from the database. This module produces *only prose*,
and when it uses an LLM the model is handed a frozen set of already-computed facts
and instructed to interpret *only* those numbers — never to originate a metric,
inflate significance, or claim an edge the statistics do not support. If no API
key is configured, or the call fails, a deterministic template is used instead, so
the pipeline never depends on a network call and can never silently fabricate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from core.config import settings
from reporting.data import SIGNIFICANT_T, SUGGESTIVE_T, ReportData, signal_label

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a quantitative research assistant writing the interpretation section "
    "of an internal research note. You will be given a JSON object of ALREADY-"
    "COMPUTED results from a backtest. Write 2-3 short paragraphs of plain, sober "
    "prose interpreting ONLY those numbers. Hard rules: (1) never invent or alter a "
    "number; quote only values present in the JSON. (2) Be scrupulously honest about "
    "statistical significance — if |t| < 2 the result is NOT significant, say so "
    "plainly; do not imply a tradeable edge. (3) No marketing language, no hype, no "
    "emojis. (4) Plain text only: no markdown, no LaTeX commands, no underscores. "
    "(5) It is correct and expected for the conclusion to be a cautious null; frame "
    "the value as the rigour of the measurement, not the size of the result."
)


@dataclass(frozen=True)
class _Facts:
    """The frozen fact set handed to the narrator (LLM or template)."""

    payload: dict

    def as_json(self) -> str:
        return json.dumps(self.payload, indent=2, default=str)


def _build_facts(data: ReportData) -> _Facts:
    """Distil :class:`ReportData` into a compact, quotable fact set."""
    headline = data.headline()
    tenk_model = max(
        (m for m in data.model_results if m.form == "10-K"),
        key=lambda m: m.n_oos,
        default=None,
    )
    payload: dict = {
        "universe_companies": data.n_companies,
        "filings_by_form": data.n_filings_by_form,
        "return_label": "market-adjusted (excess vs SPY) forward returns",
        "cost_model": "10 bps per side, charged round-trip",
        "headline_10k_signal": None,
        "walk_forward_10k": None,
        "form_comparison": "Signals are evaluated separately on 10-K and 10-Q "
        "filings; the text-change signals are an annual-report phenomenon.",
    }
    if headline is not None:
        payload["headline_10k_signal"] = {
            "signal": signal_label(headline.signal),
            "horizon_days": headline.horizon_days,
            "information_coefficient": round(headline.ic, 3),
            "ic_tstat": round(headline.ic_tstat, 2),
            "long_short_spread_pct": round(headline.ls_spread * 100, 2),
            "spread_tstat": round(headline.spread_tstat, 2),
            "n_events": headline.n_obs,
            "significant_at_95pct": abs(headline.spread_tstat) >= SIGNIFICANT_T,
        }
    if tenk_model is not None:
        payload["walk_forward_10k"] = {
            "horizon_days": tenk_model.horizon_days,
            "oos_accuracy": round(tenk_model.oos_accuracy, 3),
            "oos_auc": (None if tenk_model.oos_auc is None else round(tenk_model.oos_auc, 3)),
            "n_out_of_sample": tenk_model.n_oos,
            "interpretation_hint": "~0.5 accuracy / ~0.5 AUC is a coin-flip null",
        }
    return _Facts(payload)


def _template_narrative(facts: _Facts) -> str:
    """Deterministic, honest fallback narrative (no network)."""
    p = facts.payload
    lines: list[str] = []
    n_companies = p["universe_companies"]
    forms = p.get("filings_by_form", {})
    filings_desc = ", ".join(f"{count} {form}" for form, count in sorted(forms.items()))
    lines.append(
        f"Across a universe of {n_companies} companies ({filings_desc}), each signal "
        "was evaluated against market-adjusted forward returns, with 10-K and 10-Q "
        "filings measured separately. Returns are excess of SPY over an identical "
        "holding window, and the long-short construction is charged 10 bps per side."
    )

    head = p.get("headline_10k_signal")
    if head:
        sig = (
            "statistically significant"
            if head["significant_at_95pct"]
            else "not statistically significant"
        )
        lines.append(
            f"On 10-K filings, the strongest result is {head['signal']} at a "
            f"{head['horizon_days']}-day horizon: an information coefficient of "
            f"{head['information_coefficient']} (t = {head['ic_tstat']}) and a "
            f"long-short tercile spread of {head['long_short_spread_pct']}% net of cost "
            f"(t = {head['spread_tstat']}) over {head['n_events']} filing events. The "
            f"sign and magnitude are economically sensible, but the result is {sig} at "
            "the 95% level — consistent with a real but small anomaly that this "
            "universe is too narrow to resolve."
        )

    model = p.get("walk_forward_10k")
    if model:
        auc = "n/a" if model["oos_auc"] is None else model["oos_auc"]
        lines.append(
            f"The expanding-window walk-forward classifier confirms this: out-of-sample "
            f"accuracy of {model['oos_accuracy']} and AUC of {auc} over "
            f"{model['n_out_of_sample']} held-out filings is indistinguishable from a "
            "coin flip. The contribution here is the leakage-free measurement harness — "
            "an honest null is the correct, credible outcome at this scale, not a "
            "failure."
        )
    return "\n\n".join(lines)


def _llm_narrative(facts: _Facts) -> str | None:
    """Generate the narrative via the configured LLM, or ``None`` on any failure."""
    provider = settings.llm_provider.lower()
    api_key = settings.cerebras_api_key if provider == "cerebras" else settings.groq_api_key
    if not api_key:
        logger.info("No %s API key configured; using template narrative.", provider)
        return None

    try:
        if provider == "cerebras":
            from cerebras.cloud.sdk import Cerebras

            client = Cerebras(api_key=api_key)
        else:
            from groq import Groq

            client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Interpret these results:\n" + facts.as_json(),
                },
            ],
            temperature=0.2,
            max_tokens=600,
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            logger.info("Generated narrative via %s.", provider)
            return text
        return None
    except Exception:  # pragma: no cover - network/SDK variability
        logger.exception("LLM narrative failed; falling back to template.")
        return None


def generate_narrative(data: ReportData) -> str:
    """Return the report's prose interpretation.

    Tries the configured LLM (which only narrates the provided numbers); falls
    back to a deterministic template if no key is set or the call fails.

    Args:
        data: Resolved report data.

    Returns:
        Plain-text prose (no markdown / LaTeX), safe to embed in the document.
    """
    facts = _build_facts(data)
    return _llm_narrative(facts) or _template_narrative(facts)


__all__ = ["generate_narrative", "SIGNIFICANT_T", "SUGGESTIVE_T"]
