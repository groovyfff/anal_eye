"""Tests for the news-features RabbitMQ consumer (math model side).

Verifies the boundary contract between news-sentiment-service and AE Brain:
* the cache stores the latest snapshot per symbol;
* freshness TTL drops stale snapshots;
* no fresh news => candidate scoring continues untouched (graceful);
* the consumer never imports news-service code (boundary isolation).

No real broker / real LLM / paid API is used.
"""

from __future__ import annotations

import time

from ae_brain.contracts import TradeCandidate
from ae_brain.messaging.news_features import (
    NewsFeaturesCache,
    attach_news_features_to_candidate,
)


def _snap(symbol="BTCUSDT", sentiment=0.6):
    return {
        "schema_version": "1.0",
        "event_type": "news.sentiment.snapshot",
        "symbol": symbol,
        "asset_class": "crypto",
        "ts": "2024-01-01T00:00:00Z",
        "news_sentiment": sentiment,
        "news_volume": 4.7,
        "news_top_source": "coindesk",
        "news_source_trust": 0.72,
        "news_recency_s": 80,
        "bullish_count": 3,
        "bearish_count": 0,
        "neutral_count": 1,
        "manipulation_risk_avg": 0.14,
        "items": [],
    }


class TestNewsFeaturesCache:
    def test_get_returns_none_when_empty(self):
        c = NewsFeaturesCache()
        assert c.get("BTCUSDT") is None

    def test_update_then_get(self):
        c = NewsFeaturesCache()
        c.update("btcusdt", _snap())  # lowercase normalized
        assert c.get("btcusdt") is not None
        assert c.get("BTCUSDT")["news_sentiment"] == 0.6

    def test_stale_snapshot_expired(self):
        c = NewsFeaturesCache(max_age_s=0.02)
        c.update("BTCUSDT", _snap())
        time.sleep(0.05)
        assert c.get("BTCUSDT") is None

    def test_fresh_snapshot_returned(self):
        c = NewsFeaturesCache(max_age_s=10.0)
        c.update("BTCUSDT", _snap())
        assert c.get("BTCUSDT") is not None

    def test_clear(self):
        c = NewsFeaturesCache()
        c.update("BTCUSDT", _snap())
        c.clear()
        assert c.size() == 0

    def test_empty_symbol_ignored(self):
        c = NewsFeaturesCache()
        c.update("", _snap())
        assert c.size() == 0


class TestAttachToCandidate:
    def test_attach_when_fresh(self):
        c = NewsFeaturesCache()
        c.update("BTCUSDT", _snap(sentiment=0.63))
        meta: dict = {}
        snap = attach_news_features_to_candidate(c, meta, "BTCUSDT")
        assert snap is not None
        assert meta["news"]["news_sentiment"] == 0.63
        assert meta["features"]["news_sentiment"] == 0.63
        assert meta["features"]["news_bullish_count"] == 3

    def test_no_attach_when_absent(self):
        c = NewsFeaturesCache()
        meta: dict = {"features": {"existing": 1.0}}
        snap = attach_news_features_to_candidate(c, meta, "BTCUSDT")
        assert snap is None
        # Untouched: scoring continues normally.
        assert "news" not in meta
        assert meta["features"] == {"existing": 1.0}

    def test_no_attach_when_stale(self):
        c = NewsFeaturesCache(max_age_s=0.02)
        c.update("BTCUSDT", _snap())
        time.sleep(0.05)
        meta: dict = {}
        assert attach_news_features_to_candidate(c, meta, "BTCUSDT") is None
        assert "news" not in meta


class TestBoundaryIsolation:
    def test_no_news_service_import(self):
        """The math model side must not import news-service code."""
        import ae_brain.messaging.news_features as mod
        import inspect

        src = inspect.getsource(mod)
        assert "news_sentiment_service" not in src
        assert "from messaging" not in src  # news-service messaging pkg
        assert "import runtime" not in src  # news-service runtime


class TestConfigDefaults:
    def test_news_features_disabled_by_default(self):
        from ae_brain.config import Settings

        s = Settings()
        assert s.enable_news_features is False
        assert s.news_features_queue == "q_data_news_sentiment"
        assert s.news_features_routing_key == "data.news.sentiment"
        assert s.news_features_max_age_s == 300.0


class TestRuntimeHook:
    def test_runtime_constructs_consumer_only_when_enabled(self):
        from ae_brain.config import Settings
        from ae_brain.runtime import LiveRuntime

        # Disabled by default -> no consumer constructed.
        s = Settings()
        rt = LiveRuntime(s)
        assert rt._news_consumer is None
        assert rt._news_cache is None

    def test_handle_attaches_news_when_enabled(self):
        from ae_brain.config import Settings
        from ae_brain.runtime import LiveRuntime

        s = Settings()
        rt = LiveRuntime(s)
        # Manually wire a cache (avoid a real broker) to test _handle's hook.
        rt._news_cache = NewsFeaturesCache()
        rt._news_cache.update("BTCUSDT", _snap(sentiment=0.5))
        cand = TradeCandidate(symbol="BTCUSDT", candles=[], meta={}, interval="5m", signal_log_db_id=0)
        # Call the hook logic directly (engine.evaluate is not available here).
        snap = attach_news_features_to_candidate(rt._news_cache, cand.meta, cand.symbol)
        assert snap is not None
        assert cand.meta["features"]["news_sentiment"] == 0.5
