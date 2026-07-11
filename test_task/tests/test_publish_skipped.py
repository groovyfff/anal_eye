"""Tests for publishing skipped decisions when test mode is enabled."""

from __future__ import annotations

from ae_brain.config import AmqpInputConfig, AmqpOutputConfig
from ae_brain.contracts import Decision, FinalSignal, TradeCandidate
from ae_brain.messaging.publish_gate import evaluate_publish
from ae_brain.messaging.rabbitmq import SignalBroker, build_signal_final_payload
from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE
from ae_brain.messaging.skip_reason import extract_skip_reason


def _skip_signal(symbol: str = "ETHUSDT") -> FinalSignal:
    return FinalSignal(
        symbol=symbol,
        decision=Decision.SKIP,
        position_size_pct=0.0,
        leverage=0.0,
        take_profit=0.0,
        stop_loss=0.0,
        entry_reference=100.0,
        expected_value_usd=0.0,
        confidence=0.41,
        components={
            "decision_source": "heuristic_ev_gate",
            "sizing": {"rejected_reason": "confidence_below_threshold"},
        },
        ev={"is_positive_ev": False},
    )


def test_extract_skip_reason_from_components() -> None:
    reason = extract_skip_reason(_skip_signal())
    assert reason == "confidence_below_threshold"


def test_build_signal_final_payload_includes_skip_fields() -> None:
    signal = _skip_signal("SOLUSDT")
    candidate = TradeCandidate(
        symbol="SOLUSDT",
        interval="1h",
        candles=[],
        signal_log_db_id=0,
    )
    payload = build_signal_final_payload(signal, candidate)
    assert payload["symbol"] == "SOLUSDT"
    assert payload["decision"] == "SKIP"
    assert payload["skip_reason"] == "confidence_below_threshold"
    assert payload["consensus_achieved"] is False


def test_should_publish_signal_respects_flag() -> None:
    broker = SignalBroker(
        AmqpInputConfig(),
        AmqpOutputConfig(),
        publish_skipped_decisions=False,
        allowed_symbols=frozenset({"BTCUSDT"}),
        min_publish_confidence=0.70,
    )
    signal = _skip_signal("BTCUSDT")
    candidate = TradeCandidate(symbol="BTCUSDT", interval="1h", candles=[], signal_log_db_id=0)
    should_publish, reason, _ = broker._should_publish_signal(signal, candidate)
    assert should_publish is False
    assert reason == "skip_decision"

    broker_on = SignalBroker(
        AmqpInputConfig(),
        AmqpOutputConfig(),
        publish_skipped_decisions=True,
        allowed_symbols=frozenset({"BTCUSDT"}),
        min_publish_confidence=0.0,
    )
    should_publish, _, _ = broker_on._should_publish_signal(signal, candidate)
    assert should_publish is True
