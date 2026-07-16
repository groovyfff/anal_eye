"""Telegram defensive gate tests."""

from __future__ import annotations

import pytest

from src.logic.telegram.telegram_gate import evaluate_telegram_signal


ALLOWED = frozenset(
    {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"}
)


@pytest.mark.parametrize("symbol", ["ADAUSDT", "AVAXUSDT", "LINKUSDT"])
def test_disallowed_symbols_rejected(symbol: str) -> None:
    ok, reason = evaluate_telegram_signal(
        {"symbol": symbol, "decision": "LONG", "confidence": 0.95},
        allowed_symbols=ALLOWED,
        min_confidence=0.70,
    )
    assert ok is False
    assert reason == "unsupported_symbol"


def test_low_confidence_rejected() -> None:
    ok, reason = evaluate_telegram_signal(
        {"symbol": "BTCUSDT", "decision": "LONG", "confidence": 0.395},
        allowed_symbols=ALLOWED,
        min_confidence=0.70,
    )
    assert ok is False
    assert reason == "confidence_below_threshold"


def test_skip_rejected() -> None:
    ok, reason = evaluate_telegram_signal(
        {"symbol": "ETHUSDT", "decision": "SKIP", "confidence": 0.95},
        allowed_symbols=ALLOWED,
        min_confidence=0.70,
    )
    assert ok is False
    assert reason == "skip_decision"


def test_valid_signal_passes_gate() -> None:
    ok, reason = evaluate_telegram_signal(
        {"symbol": "BTCUSDT", "decision": "LONG", "confidence": 0.72},
        allowed_symbols=ALLOWED,
        min_confidence=0.70,
    )
    assert ok is True
    assert reason is None
