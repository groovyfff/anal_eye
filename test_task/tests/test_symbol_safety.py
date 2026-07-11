"""Symbol-safety regression tests for AE Brain publish and consume gates."""

from __future__ import annotations

import pytest

from ae_brain.config import Settings, get_settings
from ae_brain.contracts import Decision, FinalSignal
from ae_brain.messaging.publish_gate import evaluate_publish, is_symbol_allowed
from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE


def _signal(
    *,
    decision: Decision = Decision.LONG,
    confidence: float = 0.75,
    symbol: str = "BTCUSDT",
    expected_value_usd: float = 10.0,
    position_size_pct: float = 0.02,
    leverage: float = 2.0,
) -> FinalSignal:
    return FinalSignal(
        symbol=symbol,
        decision=decision,
        position_size_pct=position_size_pct,
        leverage=leverage,
        take_profit=110.0,
        stop_loss=95.0,
        entry_reference=100.0,
        expected_value_usd=expected_value_usd,
        confidence=confidence,
        components={"sizing": {"rejected_reason": None}},
    )


ALLOWED = frozenset(DEFAULT_SYMBOL_UNIVERSE)


@pytest.mark.parametrize("symbol", ["ADAUSDT", "AVAXUSDT", "LINKUSDT"])
def test_disallowed_symbols_rejected_at_publish_gate(symbol: str) -> None:
    ok, reason, _ = evaluate_publish(
        _signal(symbol=symbol, confidence=0.95),
        allowed_symbols=ALLOWED,
        min_confidence=0.70,
    )
    assert ok is False
    assert reason == "unsupported_symbol"


def test_confidence_0395_not_published() -> None:
    ok, reason, conf = evaluate_publish(
        _signal(confidence=0.395),
        allowed_symbols=ALLOWED,
        min_confidence=0.70,
    )
    assert ok is False
    assert reason == "confidence_below_threshold"
    assert conf == pytest.approx(0.395)


def test_skip_never_published() -> None:
    ok, reason, _ = evaluate_publish(
        _signal(decision=Decision.SKIP, confidence=0.95),
        allowed_symbols=ALLOWED,
        min_confidence=0.70,
    )
    assert ok is False
    assert reason == "skip_decision"


def test_negative_ev_not_published() -> None:
    ok, reason, _ = evaluate_publish(
        _signal(expected_value_usd=-1.0),
        allowed_symbols=ALLOWED,
        min_confidence=0.70,
    )
    assert ok is False
    assert reason == "negative_ev"


def test_invalid_sizing_not_published() -> None:
    signal = _signal()
    signal.components = {"sizing": {"rejected_reason": "confidence_below_threshold"}}
    ok, reason, _ = evaluate_publish(signal, allowed_symbols=ALLOWED, min_confidence=0.70)
    assert ok is False
    assert reason == "invalid_sizing"


def test_allowed_symbol_high_confidence_published() -> None:
    for symbol in ("BTCUSDT", "ETHUSDT"):
        ok, reason, conf = evaluate_publish(
            _signal(symbol=symbol, confidence=0.71),
            allowed_symbols=ALLOWED,
            min_confidence=0.70,
        )
        assert ok is True
        assert reason is None
        assert conf == pytest.approx(0.71)


def test_publish_skipped_decisions_forbidden_in_prod(monkeypatch) -> None:
    monkeypatch.setenv("AEB_ENV", "prod")
    monkeypatch.setenv("AEB_PUBLISH_SKIPPED_DECISIONS", "true")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="AEB_PUBLISH_SKIPPED_DECISIONS"):
        Settings()
    get_settings.cache_clear()


def test_direct_telegram_forbidden_in_prod(monkeypatch) -> None:
    monkeypatch.setenv("AEB_ENV", "prod")
    monkeypatch.setenv("AEB_DIRECT_TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("AEB_PUBLISH_SKIPPED_DECISIONS", "false")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="AEB_DIRECT_TELEGRAM_ENABLED"):
        Settings()
    get_settings.cache_clear()


def test_is_symbol_allowed_matches_universe() -> None:
    assert is_symbol_allowed("BTCUSDT", ALLOWED)
    assert not is_symbol_allowed("ADAUSDT", ALLOWED)
