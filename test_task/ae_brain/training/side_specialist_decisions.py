"""Final side_specialists decision vector from calibrated specialist probabilities.

Used by fusion inference and by evaluate/backtest so summary.json / publishable_report
share one decision rule (never copy diagnostic counts from side_specialists_report).
"""

from __future__ import annotations

from typing import Any

from ae_brain.layers.side_specialists import resolve_side_specialist_decision

PUBLISH_THRESHOLD = 0.70


def decision_from_calibrated_probs(
    long_prob: float,
    short_prob: float,
    *,
    long_ev: float | None = None,
    short_ev: float | None = None,
    publish_threshold: float = PUBLISH_THRESHOLD,
) -> tuple[str, str | None]:
    """Map calibrated specialist probs (+ optional EV) to LONG/SHORT/SKIP."""
    # CostModel / tabular heuristic EV is asymmetric and must not silently wipe SHORT.
    # Only use EV for tie-break when both sides have strictly positive EV; otherwise
    # treat EV as unavailable and fall back to higher calibrated probability.
    long_ev_arg = long_ev if (long_ev is not None and long_ev > 0.0) else None
    short_ev_arg = short_ev if (short_ev is not None and short_ev > 0.0) else None
    if long_ev_arg is None or short_ev_arg is None:
        long_ev_arg, short_ev_arg = None, None
    return resolve_side_specialist_decision(
        float(long_prob),
        float(short_prob),
        publish_threshold=publish_threshold,
        long_ev=long_ev_arg,
        short_ev=short_ev_arg,
    )


def extract_calibrated_probs(components: dict[str, Any] | None) -> tuple[float, float, float | None, float | None]:
    """Pull calibrated LONG/SHORT probs (and optional EV) from a FinalSignal.components dict."""
    ss = (components or {}).get("side_specialists") or {}
    long_blk = ss.get("long") or {}
    short_blk = ss.get("short") or {}
    long_prob = float(
        long_blk.get("p_profitable_calibrated", long_blk.get("p_long_profitable_calibrated", 0.0)) or 0.0
    )
    short_prob = float(
        short_blk.get("p_profitable_calibrated", short_blk.get("p_short_profitable_calibrated", 0.0)) or 0.0
    )
    long_ev = long_blk.get("ev_usd", long_blk.get("ev_long"))
    short_ev = short_blk.get("ev_usd", short_blk.get("ev_short"))
    long_ev_f = float(long_ev) if long_ev is not None else None
    short_ev_f = float(short_ev) if short_ev is not None else None
    return long_prob, short_prob, long_ev_f, short_ev_f


def rebuild_decision_from_components(
    components: dict[str, Any] | None,
    *,
    publish_threshold: float = PUBLISH_THRESHOLD,
) -> tuple[str, str | None, float, float]:
    """Return (decision, reason, long_prob, short_prob) from specialist components."""
    long_prob, short_prob, long_ev, short_ev = extract_calibrated_probs(components)
    decision, reason = decision_from_calibrated_probs(
        long_prob,
        short_prob,
        long_ev=long_ev,
        short_ev=short_ev,
        publish_threshold=publish_threshold,
    )
    return decision, reason, long_prob, short_prob
