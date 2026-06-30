"""Tests for publishing skipped decisions when test mode is enabled."""

from __future__ import annotations

from ae_brain.contracts import Decision, FinalSignal
from ae_brain.messaging.rabbitmq import SignalBroker, build_signal_final_payload
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
    from ae_brain.contracts import TradeCandidate

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
    from ae_brain.config import AmqpInputConfig, AmqpOutputConfig

    broker = SignalBroker(
        AmqpInputConfig(),
        AmqpOutputConfig(),
        publish_skipped_decisions=False,
    )
    from ae_brain.contracts import TradeCandidate

    signal = _skip_signal()
    candidate = TradeCandidate(symbol="ETHUSDT", interval="1h", candles=[], signal_log_db_id=0)
    assert broker._should_publish_signal(signal, candidate) is False

    broker_on = SignalBroker(
        AmqpInputConfig(),
        AmqpOutputConfig(),
        publish_skipped_decisions=True,
    )
    assert broker_on._should_publish_signal(signal, candidate) is True
