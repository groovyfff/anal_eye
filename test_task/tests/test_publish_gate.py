"""Publish gate and multi-symbol universe tests."""

from __future__ import annotations

from ae_brain.config import get_settings
from ae_brain.contracts import Decision, FinalSignal
from ae_brain.messaging.publish_gate import (
    evaluate_publish,
    parse_allowed_symbols,
)
from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE, default_allowed_symbols_csv


def _signal(
    *,
    decision: Decision = Decision.LONG,
    confidence: float = 0.75,
    symbol: str = "BTCUSDT",
    expected_value_usd: float = 12.5,
) -> FinalSignal:
    return FinalSignal(
        symbol=symbol,
        decision=decision,
        position_size_pct=0.02,
        leverage=2.0,
        take_profit=110.0,
        stop_loss=95.0,
        entry_reference=100.0,
        expected_value_usd=expected_value_usd,
        confidence=confidence,
        components={"sizing": {"rejected_reason": None}},
    )


def test_default_universe_has_six_evidenced_symbols() -> None:
    assert len(DEFAULT_SYMBOL_UNIVERSE) == 6
    assert "BTCUSDT" in DEFAULT_SYMBOL_UNIVERSE
    assert "ETHUSDT" in DEFAULT_SYMBOL_UNIVERSE
    assert "DOGEUSDT" in DEFAULT_SYMBOL_UNIVERSE


def test_settings_only_btc_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setenv("AEB_ONLY_BTC", "false")
    monkeypatch.setenv("AEB_ALLOWED_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.only_btc is False
    assert settings.allowed_symbol_set == frozenset(DEFAULT_SYMBOL_UNIVERSE)
    get_settings.cache_clear()


def test_only_btc_mode_still_works() -> None:
    assert parse_allowed_symbols(default_allowed_symbols_csv(), only_btc=True) == frozenset({"BTCUSDT"})


def test_long_confidence_069_not_published() -> None:
    allowed = frozenset(DEFAULT_SYMBOL_UNIVERSE)
    ok, reason, conf = evaluate_publish(_signal(confidence=0.69), allowed_symbols=allowed, min_confidence=0.70)
    assert ok is False
    assert reason == "confidence_below_threshold"
    assert conf == 0.69


def test_long_confidence_070_published() -> None:
    allowed = frozenset(DEFAULT_SYMBOL_UNIVERSE)
    ok, reason, _ = evaluate_publish(_signal(confidence=0.70), allowed_symbols=allowed, min_confidence=0.70)
    assert ok is True
    assert reason is None


def test_short_confidence_070_published() -> None:
    allowed = frozenset(DEFAULT_SYMBOL_UNIVERSE)
    ok, _, _ = evaluate_publish(
        _signal(decision=Decision.SHORT, confidence=0.72),
        allowed_symbols=allowed,
        min_confidence=0.70,
    )
    assert ok is True


def test_skip_not_published_by_default() -> None:
    allowed = frozenset(DEFAULT_SYMBOL_UNIVERSE)
    ok, reason, _ = evaluate_publish(
        _signal(decision=Decision.SKIP, confidence=0.95),
        allowed_symbols=allowed,
        min_confidence=0.70,
    )
    assert ok is False
    assert reason == "skip_decision"


def test_non_btc_symbol_eth_accepted_for_publish() -> None:
    allowed = frozenset(DEFAULT_SYMBOL_UNIVERSE)
    ok, reason, _ = evaluate_publish(
        _signal(symbol="ETHUSDT", confidence=0.71),
        allowed_symbols=allowed,
        min_confidence=0.70,
    )
    assert ok is True
    assert reason is None


def test_non_btc_symbol_rejected_when_not_in_universe() -> None:
    allowed = frozenset({"BTCUSDT"})
    ok, reason, _ = evaluate_publish(
        _signal(symbol="ETHUSDT", confidence=0.95),
        allowed_symbols=allowed,
        min_confidence=0.70,
    )
    assert ok is False
    assert reason == "unsupported_symbol"


def test_negative_ev_suppressed() -> None:
    allowed = frozenset(DEFAULT_SYMBOL_UNIVERSE)
    ok, reason, _ = evaluate_publish(
        _signal(expected_value_usd=0.0),
        allowed_symbols=allowed,
        min_confidence=0.70,
    )
    assert ok is False
    assert reason == "negative_ev"
