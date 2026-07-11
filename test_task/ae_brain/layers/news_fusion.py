"""Bounded, optional news-context fusion for AE Brain.

News only **slightly** adjusts confidence/EV. It never creates a trade, never
changes a SKIP, and the adjustment is hard-capped by ``max_conf_delta`` /
``max_ev_multiplier_delta``. With no active news the signal is returned
unchanged — so AE Brain behaves exactly as before.

Score → bias mapping (per spec)::

    news_bias = (score - 5.5) / 4.5
    1 → -1.00 ... 5 → -0.11, 6 → +0.11 ... 10 → +1.00

    news_strength = abs(news_bias) * relevance * confidence   (weighted-avg over active signals)

For a given math-model decision:
* LONG  → aligned if news_bias > 0, opposed if news_bias < 0
* SHORT → aligned if news_bias < 0, opposed if news_bias > 0
* SKIP  → unchanged (news never flips SKIP into LONG/SHORT)

Confidence adjustment::

    delta = max_conf_delta * news_strength        (default 0.05)
    aligned  -> min(1.0, base + delta)
    opposed  -> max(0.0, base - delta)
    neutral  -> base

EV adjustment::

    ev_delta = max_ev_multiplier_delta * news_strength   (default 0.10)
    aligned  -> base_ev * (1 + ev_delta)
    opposed  -> base_ev * (1 - ev_delta)
    neutral  -> base_ev

Safety (provable from this module):
* The decision enum is never changed; a SKIP in stays a SKIP out.
* Confidence deltas are clamped to [0, 1] and bounded by ``max_conf_delta``.
* EV is only scaled by a factor in ``[1 - max_ev_mult, 1 + max_ev_mult]``.
* ``apply_news_to_signal`` with an empty aggregate is a strict no-op.

Diagnostic logs: ``news_context_applied`` / ``news_context_absent`` /
``news_adjusted_confidence``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict

from ae_brain.contracts import Decision, FinalSignal
from ae_brain.messaging.news_context_store import NewsAggregate, score_to_bias

log = logging.getLogger("ae_brain.news_fusion")


def classify_alignment(decision: Decision, news_bias: float) -> str:
    """Return ``"aligned"`` / ``"opposed"`` / ``"neutral"`` for a decision + bias.

    SKIP is always neutral (and never reaches the adjuster, but be defensive).
    """
    if decision == Decision.SKIP:
        return "neutral"
    if decision == Decision.LONG:
        if news_bias > 1e-9:
            return "aligned"
        if news_bias < -1e-9:
            return "opposed"
        return "neutral"
    if decision == Decision.SHORT:
        if news_bias < -1e-9:
            return "aligned"
        if news_bias > 1e-9:
            return "opposed"
        return "neutral"
    return "neutral"


def apply_news_to_signal(
    signal: FinalSignal,
    agg: NewsAggregate,
    *,
    max_conf_delta: float = 0.05,
    max_ev_multiplier_delta: float = 0.10,
) -> FinalSignal:
    """Return a news-adjusted copy of ``signal`` (or the same signal if no news).

    Never mutates the input. A SKIP decision or an empty aggregate ⇒ the input
    is returned unchanged. Emits ``news_context_applied`` /
    ``news_context_absent`` / ``news_adjusted_confidence``.
    """
    # SKIP is sacred: news must never turn it into a trade.
    if signal.decision == Decision.SKIP:
        log.info("news_context_absent", extra={"event": "news_context_absent",
                 "symbol": signal.symbol, "reason": "skip_decision"})
        return signal

    # No active news ⇒ identical behavior.
    if not agg.has_news:
        log.info("news_context_absent", extra={"event": "news_context_absent",
                 "symbol": signal.symbol, "reason": "no_active_news"})
        return signal

    news_bias = score_to_bias(agg.score_avg)
    # Aggregate strength already folds in relevance × confidence × age-decay.
    news_strength = agg.news_strength
    alignment = classify_alignment(signal.decision, news_bias)

    # --- Confidence adjustment (bounded by max_conf_delta) -------------------
    conf_delta = max_conf_delta * news_strength
    base_conf = float(signal.confidence)
    if alignment == "aligned":
        adjusted_conf = min(1.0, base_conf + conf_delta)
    elif alignment == "opposed":
        adjusted_conf = max(0.0, base_conf - conf_delta)
    else:
        adjusted_conf = base_conf

    # --- EV adjustment (bounded multiplicative) -------------------------------
    ev_mult_delta = max_ev_multiplier_delta * news_strength
    base_ev = float(signal.expected_value_usd)
    if alignment == "aligned":
        adjusted_ev = base_ev * (1.0 + ev_mult_delta)
    elif alignment == "opposed":
        adjusted_ev = base_ev * (1.0 - ev_mult_delta)
    else:
        adjusted_ev = base_ev

    log.info(
        "news_context_applied",
        extra={
            "event": "news_context_applied",
            "symbol": signal.symbol,
            "decision": signal.decision.value,
            "alignment": alignment,
            "base_confidence": base_conf,
            "adjusted_confidence": adjusted_conf,
            "base_ev": base_ev,
            "adjusted_ev": adjusted_ev,
            "score_avg": round(agg.score_avg, 4),
            "news_bias": round(news_bias, 4),
            "news_strength": round(news_strength, 4),
            "signal_count": agg.signal_count,
        },
    )
    log.info(
        "news_adjusted_confidence",
        extra={"event": "news_adjusted_confidence",
               "symbol": signal.symbol, "base": base_conf,
               "adjusted": adjusted_conf, "delta": round(adjusted_conf - base_conf, 5)},
    )

    # Mirror the adjusted EV into the ev dict (best-effort; never invent keys).
    new_ev: Dict[str, Any] = dict(signal.ev or {})
    if new_ev:
        # Scale the headline expected_value consistently with the multiplier.
        if "expected_value" in new_ev:
            base_ev_inner = float(new_ev["expected_value"])
            new_ev["expected_value"] = base_ev_inner * (
                (1.0 + ev_mult_delta) if alignment == "aligned"
                else (1.0 - ev_mult_delta) if alignment == "opposed"
                else 1.0
            )

    # Record the news adjustment in components for auditability.
    new_components = dict(signal.components or {})
    news_meta = dict(new_components.get("news") or {})
    news_meta.update({
        "alignment": alignment,
        "score_avg": round(agg.score_avg, 4),
        "news_bias": round(news_bias, 4),
        "news_strength": round(news_strength, 4),
        "signal_count": agg.signal_count,
        "confidence_delta": round(adjusted_conf - base_conf, 5),
        "ev_multiplier_delta": round(
            (adjusted_ev / base_ev - 1.0) if base_ev != 0 else 0.0, 5
        ),
        "base_confidence": base_conf,
        "adjusted_confidence": adjusted_conf,
    })
    new_components["news"] = news_meta

    return replace(
        signal,
        confidence=adjusted_conf,
        expected_value_usd=adjusted_ev,
        ev=new_ev,
        components=new_components,
    )
