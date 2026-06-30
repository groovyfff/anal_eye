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
    if meta.get("directional_class") is None and components.get("decision_source") == "meta_model":
        return "meta_model_no_direction"

    if components.get("decision_source") == "heuristic_ev_gate":
        return "heuristic_ev_gate_skip"

    return "no_actionable_edge"
