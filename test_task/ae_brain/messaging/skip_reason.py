"""Derive human-readable skip reasons from fusion output."""

from __future__ import annotations

from ae_brain.contracts import Decision, FinalSignal


def extract_skip_reason(signal: FinalSignal) -> str:
    if signal.decision != Decision.SKIP:
        return ""

    components = signal.components or {}
    sizing = components.get("sizing") or {}
    rejected = sizing.get("rejected_reason")
    if rejected:
        return str(rejected)

    ev = signal.ev or {}
    if ev and not ev.get("is_positive_ev", True):
        return "negative_ev"

    meta = components.get("meta") or {}
    skip_reason = meta.get("skip_reason")
    if skip_reason == "directional_ambiguity":
        return "directional_ambiguity"
    source = components.get("decision_source") or ""
    if meta.get("directional_class") is None and source.startswith("meta_"):
        return "meta_model_no_direction"

    if components.get("decision_source") == "heuristic_ev_gate":
        return "heuristic_ev_gate_skip"

    if components.get("decision_source") == "training_regime_filter":
        return str(components.get("skip_reason") or "outside_training_regime")

    regime = components.get("regime_filter") or components.get("skip_reason")
    if regime in ("below_min_vol_z", "outside_training_regime"):
        return str(regime)

    return "no_actionable_edge"
