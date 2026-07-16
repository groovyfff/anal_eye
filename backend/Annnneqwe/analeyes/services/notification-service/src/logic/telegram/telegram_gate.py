"""Defensive Telegram gate — last line of defense before Bot API delivery."""

from __future__ import annotations

import logging
import os

from shared.symbol_universe import default_allowed_symbols, is_symbol_allowed

logger = logging.getLogger(__name__)

_ACTIONABLE_DECISIONS = frozenset({"LONG", "SHORT"})


def _min_confidence_from_env() -> float:
    raw = os.environ.get("NOTIFICATION_MIN_CONFIDENCE") or os.environ.get(
        "AEB_MIN_PUBLISH_CONFIDENCE", "0.70"
    )
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return 0.70
    if value > 1.0:
        value = value / 100.0
    return value


def normalize_confidence(confidence: object) -> float | None:
    if confidence is None:
        return None
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None
    if value > 1.0:
        value = value / 100.0
    return value


def evaluate_telegram_signal(
    payload: dict,
    *,
    allowed_symbols: frozenset[str] | None = None,
    min_confidence: float | None = None,
) -> tuple[bool, str | None]:
    """Return (should_send, reject_reason)."""
    universe = allowed_symbols if allowed_symbols is not None else default_allowed_symbols()
    threshold = min_confidence if min_confidence is not None else _min_confidence_from_env()

    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol or not is_symbol_allowed(symbol, universe):
        return False, "unsupported_symbol"

    decision = str(payload.get("decision") or payload.get("signal_type") or "").strip().upper()
    if decision not in _ACTIONABLE_DECISIONS:
        return False, "skip_decision"

    normalized = normalize_confidence(payload.get("confidence"))
    if normalized is None:
        return False, "invalid_confidence"
    if normalized < threshold:
        return False, "confidence_below_threshold"

    return True, None


def log_telegram_rejection(payload: dict, reason: str) -> None:
    logger.info(
        "telegram_signal_rejected symbol=%s decision=%s confidence=%s reason=%s",
        payload.get("symbol"),
        payload.get("decision") or payload.get("signal_type"),
        payload.get("confidence"),
        reason,
    )
