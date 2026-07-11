"""Tests for the optional ``news.market_signal`` fusion in AE Brain.

Covers the spec's required AE Brain test list:
* starts and runs without a news queue/messages (no-news ⇒ identical output);
* a malformed news message does not break inference;
* the news cache TTL expires correctly;
* LONG confidence increases with bullish news / decreases with bearish news;
* SHORT confidence increases with bearish news / decreases with bullish news;
* SKIP remains SKIP even with extreme news;
* news cannot bypass the 0.70 threshold by more than max delta;
* no news ⇒ exact same output as before;
* the consumer binds ``q_news_market_signal`` correctly.

No real broker / real LLM / paid API is used. The fusion math is tested
directly against ``apply_news_to_signal`` with hand-built ``FinalSignal``s.
"""

from __future__ import annotations

import time

from ae_brain.contracts import Decision, FinalSignal
from ae_brain.layers.news_fusion import apply_news_to_signal, classify_alignment
from ae_brain.messaging.news_context_store import (
    NewsAggregate,
    NewsContextStore,
    score_to_bias,
)
from ae_brain.messaging.news_signal_consumer import NewsSignalConsumer


# --- helpers -----------------------------------------------------------------

def _signal(decision: Decision, confidence: float = 0.80, ev: float = 50.0,
            symbol: str = "BTCUSDT") -> FinalSignal:
    return FinalSignal(
        symbol=symbol,
        decision=decision,
        position_size_pct=0.1,
        leverage=2.0,
        take_profit=100.0,
        stop_loss=90.0,
        entry_reference=95.0,
        expected_value_usd=ev,
        confidence=confidence,
        ev={"expected_value": ev, "is_positive_ev": ev > 0},
        components={},
    )


def _agg(symbol: str, score: float, relevance: float = 0.9,
         confidence: float = 0.85, count: int = 1) -> NewsAggregate:
    """Build a synthetic aggregate with a fixed score/strength."""
    bias_abs = abs(score_to_bias(score))
    strength = bias_abs * relevance * confidence
    return NewsAggregate(
        symbol=symbol,
        score_avg=score,
        relevance_avg=relevance,
        confidence_avg=confidence,
        news_strength=strength,
        signal_count=count,
        weight=1.0,
    )


def _store_with(symbol: str, score: int, relevance: float = 0.9,
                confidence: float = 0.85, ttl: float = 10000.0) -> NewsContextStore:
    store = NewsContextStore(ttl_s=ttl)
    store.add_signal_dict(symbol, {
        "symbol": symbol, "score": score, "relevance": relevance,
        "confidence": confidence, "horizon": "short", "source_type": "macro",
        "reason": "r", "risk_flags": [],
    })
    return store


# --- score → bias mapping ----------------------------------------------------

class TestScoreMapping:
    def test_extremes(self):
        assert score_to_bias(1) == -1.0
        assert score_to_bias(10) == 1.0
        assert abs(score_to_bias(5.5)) < 1e-9

    def test_near_neutral(self):
        assert round(score_to_bias(5), 2) == -0.11
        assert round(score_to_bias(6), 2) == 0.11

    def test_monotonic(self):
        vals = [score_to_bias(s) for s in range(1, 11)]
        assert vals == sorted(vals)


# --- alignment classification ------------------------------------------------

class TestAlignment:
    def test_long_bullish_aligned(self):
        assert classify_alignment(Decision.LONG, 0.5) == "aligned"

    def test_long_bearish_opposed(self):
        assert classify_alignment(Decision.LONG, -0.5) == "opposed"

    def test_short_bearish_aligned(self):
        assert classify_alignment(Decision.SHORT, -0.5) == "aligned"

    def test_short_bullish_opposed(self):
        assert classify_alignment(Decision.SHORT, 0.5) == "opposed"

    def test_skip_neutral_regardless(self):
        assert classify_alignment(Decision.SKIP, 0.9) == "neutral"
        assert classify_alignment(Decision.SKIP, -0.9) == "neutral"

    def test_zero_bias_neutral(self):
        assert classify_alignment(Decision.LONG, 0.0) == "neutral"


# --- confidence / EV adjustment ----------------------------------------------

class TestConfidenceAdjustment:
    def test_long_confidence_increases_with_bullish(self):
        sig = _signal(Decision.LONG, confidence=0.80)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=9),
                                   max_conf_delta=0.05, max_ev_multiplier_delta=0.10)
        assert out.confidence > 0.80
        # delta = 0.05 * (abs((9-5.5)/4.5) * 0.9 * 0.85) = 0.05 * 0.595 = 0.02975
        assert abs(out.confidence - (0.80 + 0.05 * 0.595)) < 1e-6

    def test_long_confidence_decreases_with_bearish(self):
        sig = _signal(Decision.LONG, confidence=0.80)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=2),
                                   max_conf_delta=0.05, max_ev_multiplier_delta=0.10)
        assert out.confidence < 0.80

    def test_short_confidence_increases_with_bearish(self):
        sig = _signal(Decision.SHORT, confidence=0.75)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=2),
                                   max_conf_delta=0.05, max_ev_multiplier_delta=0.10)
        assert out.confidence > 0.75

    def test_short_confidence_decreases_with_bullish(self):
        sig = _signal(Decision.SHORT, confidence=0.75)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=9),
                                   max_conf_delta=0.05, max_ev_multiplier_delta=0.10)
        assert out.confidence < 0.75

    def test_confidence_capped_at_1(self):
        sig = _signal(Decision.LONG, confidence=0.99)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=10),
                                   max_conf_delta=0.05, max_ev_multiplier_delta=0.10)
        assert out.confidence <= 1.0

    def test_confidence_floored_at_0(self):
        sig = _signal(Decision.LONG, confidence=0.01)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=1),
                                   max_conf_delta=0.05, max_ev_multiplier_delta=0.10)
        assert out.confidence >= 0.0


class TestEVAdjustment:
    def test_long_aligned_ev_increases(self):
        sig = _signal(Decision.LONG, ev=100.0)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=9),
                                   max_conf_delta=0.05, max_ev_multiplier_delta=0.10)
        assert out.expected_value_usd > 100.0
        # ev_delta = 0.10 * 0.595 = 0.0595 -> 100 * 1.0595
        assert abs(out.expected_value_usd - 100.0 * 1.0595) < 1e-4

    def test_long_opposed_ev_decreases(self):
        sig = _signal(Decision.LONG, ev=100.0)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=2),
                                   max_conf_delta=0.05, max_ev_multiplier_delta=0.10)
        assert out.expected_value_usd < 100.0

    def test_ev_dict_mirrored(self):
        sig = _signal(Decision.LONG, ev=100.0)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=9),
                                   max_conf_delta=0.05, max_ev_multiplier_delta=0.10)
        assert out.ev["expected_value"] > 100.0


# --- safety: SKIP + threshold + no-op ----------------------------------------

class TestSafety:
    def test_skip_remains_skip_with_extreme_news(self):
        sig = _signal(Decision.SKIP, confidence=0.40, ev=50.0)
        extreme = _agg("BTCUSDT", score=10)  # extremely bullish
        out = apply_news_to_signal(sig, extreme)
        assert out.decision == Decision.SKIP
        assert out.confidence == 0.40  # unchanged
        assert out.expected_value_usd == 50.0  # unchanged (input EV preserved)

    def test_skip_with_extreme_bearish(self):
        sig = _signal(Decision.SKIP, confidence=0.40)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=1))
        assert out.decision == Decision.SKIP
        assert out.confidence == 0.40

    def test_news_cannot_bypass_threshold_by_more_than_max_delta(self):
        # A LONG at 0.68 (below 0.70 threshold) with maximal bullish news.
        sig = _signal(Decision.LONG, confidence=0.68)
        out = apply_news_to_signal(sig, _agg("BTCUSDT", score=10, relevance=1.0, confidence=1.0),
                                   max_conf_delta=0.05, max_ev_multiplier_delta=0.10)
        # Max possible delta is exactly max_conf_delta = 0.05.
        assert out.confidence <= 0.68 + 0.05 + 1e-9
        assert out.confidence < 0.731  # cannot jump far past 0.70

    def test_no_news_means_identical_output(self):
        sig = _signal(Decision.LONG, confidence=0.80, ev=50.0)
        empty = NewsAggregate(symbol="BTCUSDT", score_avg=5.5, relevance_avg=0.0,
                              confidence_avg=0.0, news_strength=0.0,
                              signal_count=0, weight=0.0)
        out = apply_news_to_signal(sig, empty)
        assert out.confidence == 0.80
        assert out.expected_value_usd == 50.0
        assert out.decision == Decision.LONG

    def test_input_signal_not_mutated(self):
        sig = _signal(Decision.LONG, confidence=0.80, ev=50.0)
        original_conf = sig.confidence
        apply_news_to_signal(sig, _agg("BTCUSDT", score=9))
        assert sig.confidence == original_conf  # input unchanged


# --- NewsContextStore TTL + aggregation --------------------------------------

class TestNewsContextStore:
    def test_empty_store_aggregate_is_neutral(self):
        store = NewsContextStore(ttl_s=1000)
        agg = store.aggregate("BTCUSDT")
        assert agg.signal_count == 0
        assert agg.has_news is False

    def test_add_and_aggregate(self):
        store = _store_with("BTCUSDT", score=9)
        agg = store.aggregate("BTCUSDT")
        assert agg.signal_count == 1
        assert agg.score_avg == 9.0
        assert agg.news_strength > 0

    def test_ttl_expires_signals(self):
        store = NewsContextStore(ttl_s=0.02)
        store.add_signal_dict("BTCUSDT", {
            "symbol": "BTCUSDT", "score": 9, "relevance": 0.9,
            "confidence": 0.85, "horizon": "short", "source_type": "ETF",
            "reason": "r", "risk_flags": [],
        })
        time.sleep(0.05)
        agg = store.aggregate("BTCUSDT")
        assert agg.signal_count == 0  # expired

    def test_invalid_signal_rejected(self):
        store = NewsContextStore(ttl_s=1000)
        assert store.add_signal_dict("BTCUSDT", {"score": 99}) is False
        assert store.add_signal_dict("BTCUSDT", {"score": 5, "relevance": 2.0,
                                                  "confidence": 0.5}) is False
        assert store.size() == 0

    def test_multiple_signals_weighted_average(self):
        store = NewsContextStore(ttl_s=10000)
        # Two equal-weight signals: score 9 (bull) and score 3 (bear).
        for s in (9, 3):
            store.add_signal_dict("BTCUSDT", {
                "symbol": "BTCUSDT", "score": s, "relevance": 0.9,
                "confidence": 0.9, "horizon": "short", "source_type": "macro",
                "reason": "r", "risk_flags": [],
            })
        agg = store.aggregate("BTCUSDT")
        assert agg.signal_count == 2
        # Average of 9 and 3 is 6.
        assert abs(agg.score_avg - 6.0) < 1e-6

    def test_malformed_message_dict_does_not_crash(self):
        """A malformed signal dict is rejected, not raised."""
        store = NewsContextStore(ttl_s=1000)
        # Various malformed inputs.
        for bad in [
            {"symbol": "BTCUSDT"},  # missing score
            {"symbol": "BTCUSDT", "score": "abc", "relevance": 0.5, "confidence": 0.5},
            {"symbol": "", "score": 5, "relevance": 0.5, "confidence": 0.5},
            {"symbol": "BTCUSDT", "score": 5, "relevance": -1.0, "confidence": 0.5},
        ]:
            assert store.add_signal_dict(bad.get("symbol", "BTCUSDT"), bad) is False
        assert store.size() == 0


# --- consumer wiring ---------------------------------------------------------

class TestConsumerWiring:
    def test_binds_correct_queue_routing_exchange(self):
        c = NewsSignalConsumer("amqp://analeyes:x@rabbitmq:5672/analeyes",
                               NewsContextStore())
        assert c.QUEUE == "q_news_market_signal"
        assert c.ROUTING_KEY == "news.market_signal"
        assert c.EXCHANGE == "analeyes.events"

    def test_defaults_match_spec(self):
        assert NewsSignalConsumer.QUEUE == "q_news_market_signal"
        assert NewsSignalConsumer.ROUTING_KEY == "news.market_signal"


# --- end-to-end: store → aggregate → fusion ----------------------------------

class TestEndToEnd:
    def test_store_drives_fusion_long_bullish(self):
        store = _store_with("BTCUSDT", score=9, relevance=0.9, confidence=0.85)
        sig = _signal(Decision.LONG, confidence=0.80, ev=50.0)
        out = apply_news_to_signal(
            sig, store.aggregate("BTCUSDT"),
            max_conf_delta=0.05, max_ev_multiplier_delta=0.10,
        )
        assert out.confidence > 0.80
        assert out.expected_value_usd > 50.0
        # components carry the audit trail
        assert out.components["news"]["alignment"] == "aligned"
        assert out.components["news"]["signal_count"] == 1

    def test_store_drives_fusion_short_bearish(self):
        store = _store_with("BTCUSDT", score=2, relevance=0.9, confidence=0.85)
        sig = _signal(Decision.SHORT, confidence=0.75, ev=40.0)
        out = apply_news_to_signal(
            sig, store.aggregate("BTCUSDT"),
            max_conf_delta=0.05, max_ev_multiplier_delta=0.10,
        )
        assert out.confidence > 0.75
        assert out.expected_value_usd > 40.0
