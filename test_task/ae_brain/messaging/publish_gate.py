"""Symbol normalization and publish eligibility for AE Brain.

Publishing gates (confidence >= 0.70, LONG/SHORT only) affect RabbitMQ/Telegram
output only — they never change inference labels or model math.
"""

from __future__ import annotations

from ae_brain.contracts import Decision, FinalSignal
from ae_brain.symbols import (
    DEFAULT_SYMBOL_UNIVERSE,
    UNKNOWN_SYMBOL,
    normalize_symbol,
    parse_symbol_list,
)

_BTC_RAW_ALIASES = frozenset(
    {
        "BTCUSDT",
        "BTC/USD",
        "BTC-USD",
        "BTCUSD",
        "XBTUSDT",
        "XBTUSD",
        "XBT",
        "BTC",
    }
)


def parse_allowed_symbols(raw: str | None, *, only_btc: bool) -> frozenset[str]:
    if only_btc:
        return frozenset({"BTCUSDT"})
    return frozenset(parse_symbol_list(raw))


def normalize_candidate_symbol(symbol: str | None, *, only_btc: bool) -> str:
    """Normalize inbound symbol; map Bitcoin aliases to BTCUSDT when only_btc."""
    if symbol is None:
        return ""
    cleaned = str(symbol).strip().upper()
    if not cleaned:
        return ""
    if only_btc and cleaned in _BTC_RAW_ALIASES:
        return "BTCUSDT"
    compact = cleaned.replace("/", "").replace("-", "")
    if only_btc and compact in {"BTCUSDT", "BTCUSD", "XBTUSDT", "XBTUSD", "BTC", "XBT"}:
        return "BTCUSDT"
    return normalize_symbol(cleaned)


def is_symbol_allowed(symbol: str, allowed_symbols: frozenset[str]) -> bool:
    normalized = normalize_symbol(symbol)
    if normalized == UNKNOWN_SYMBOL:
        return False
    return normalized in allowed_symbols


def normalize_confidence(confidence: float | int | str | None) -> float | None:
    """Map confidence to [0, 1]; treat values > 1 as percent (e.g. 72 -> 0.72)."""
    if confidence is None:
        return None
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None
    if value > 1.0:
        value = value / 100.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def is_sizing_valid(signal: FinalSignal) -> bool:
    """Return True when position sizing is actionable for publish."""
    components = signal.components or {}
    sizing = components.get("sizing") or {}
    if sizing.get("rejected_reason"):
        return False
    if signal.position_size_pct <= 0.0:
        return False
    if signal.leverage <= 0.0:
        return False
    if signal.take_profit <= 0.0 or signal.stop_loss <= 0.0:
        return False
    return True


def evaluate_publish(
    signal: FinalSignal,
    *,
    allowed_symbols: frozenset[str],
    min_confidence: float,
) -> tuple[bool, str | None, float | None]:
    """Return (should_publish, suppress_reason, normalized_confidence).

    Production policy: publish LONG/SHORT to signal.final only when confidence
    >= min_confidence (default 0.70), EV > 0, sizing is valid, and symbol is
    in the allowed universe. SKIP is suppressed unless debug mode enables
    publish_skipped_decisions elsewhere.
    """
    if not is_symbol_allowed(signal.symbol, allowed_symbols):
        return False, "unsupported_symbol", normalize_confidence(signal.confidence)

    if signal.decision not in (Decision.LONG, Decision.SHORT):
        if signal.decision == Decision.SKIP:
            return False, "skip_decision", normalize_confidence(signal.confidence)
        return False, "empty_decision", normalize_confidence(signal.confidence)

    normalized = normalize_confidence(signal.confidence)
    if normalized is None:
        return False, "invalid_confidence", None
    if normalized < min_confidence:
        return False, "confidence_below_threshold", normalized

    if signal.expected_value_usd <= 0.0:
        return False, "negative_ev", normalized

    ev_meta = signal.ev or {}
    if ev_meta.get("is_positive_ev") is False:
        return False, "negative_ev", normalized

    if not is_sizing_valid(signal):
        return False, "invalid_sizing", normalized

    return True, None, normalized


def default_allowed_symbol_set() -> frozenset[str]:
    return frozenset(DEFAULT_SYMBOL_UNIVERSE)
