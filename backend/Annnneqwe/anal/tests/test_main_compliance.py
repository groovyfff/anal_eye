import asyncio
import datetime as dt
import json
from typing import Any
import pandas as pd
import pytest
from src.collectors.base_collector import Quote
from src.logic.pattern_engine import PatternDetection
from src.logic.trigger_engine import TriggerResult
from src.main import ExternalMarketsService

class _AlwaysOpenHours:

    def is_market_open(self, asset_class: str, now_utc: dt.datetime | None=None) -> bool:
        _ = (asset_class, now_utc)
        return True

class _FakeRabbit:

    def __init__(self) -> None:
        self.published: list[tuple[str, str, str]] = []

    async def publish_async(self, exchange_name: str, routing_key: str, body: str) -> None:
        self.published.append((exchange_name, routing_key, body))

class _DummyFeatureGenerator:

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def compute_indicators(self, frame: Any, asset_class: str, benchmark_close: Any=None, dxy_close: Any=None) -> pd.DataFrame:
        _ = (frame, asset_class, benchmark_close, dxy_close)
        return self.frame

    def build_feature_payload(self, indicator_df: pd.DataFrame, asset_class: str, bid: float | None, ask: float | None) -> dict[str, Any]:
        _ = (indicator_df, asset_class, bid, ask)
        return {'current_price': 101.0, 'rsi': 60.0, 'vol_rel': 2.1, 'macd_hist': 0.3, 'adx': 25.0, 'sp500_correlation': 0.65}

    def build_historical_snapshots(self, indicator_df: pd.DataFrame, count: int=2) -> list[dict[str, Any]]:
        _ = (indicator_df, count)
        return []

class _DummyTriggerEngine:

    def evaluate(self, indicator_df: pd.DataFrame, context_trends: dict[str, dict[str, Any]] | None=None) -> TriggerResult:
        _ = (indicator_df, context_trends)
        return TriggerResult(triggered=True, reasons=['EMA_CROSSOVER_BULLISH', 'VOLUME_SPIKE'], heuristic_signal_consensus='LONG', indicators={'consensus': 'BULLISH', 'consensus_strength': 1.0, 'signals': []})

class _CaptureComposite:

    def __init__(self) -> None:
        self.pattern_scores: list[float] = []

    def calculate(self, features: dict[str, Any], pattern_score: float=0.0) -> float:
        _ = features
        self.pattern_scores.append(pattern_score)
        return 1.0

    def should_publish(self, score: float) -> bool:
        _ = score
        return True

class _DummyPatternEngine:

    def analyze(self, frame: pd.DataFrame) -> PatternDetection:
        _ = frame
        return PatternDetection(patterns_payload={'consensus': 'BULLISH', 'consensus_strength': 0.7, 'detected_patterns_info': [{'pattern_name': 'Engulfing', 'signal': 'BULLISH', 'strength': 0.7, 'candle_offset': 0}]}, pattern_score=0.77)

def _settings(history_depth: int=200, max_retries: int=1) -> dict[str, Any]:
    return {'rabbitmq': {'url': 'amqp://guest:guest@localhost:5672/', 'exchange': 'analeyes_exchange'}, 'data_provider': {'name': 'yahoo_finance', 'request_delay_s': 0.0, 'quote_request_delay_s': 0.0, 'max_retries': max_retries}, 'watchlist': {'stocks': [{'symbol': 'AAPL', 'name': 'Apple'}], 'indices': [], 'metals': [], 'forex': []}, 'market_hours': {'timezone': 'America/New_York', 'stock_open': '09:30', 'stock_close': '16:00', 'pre_market_enabled': False, 'after_hours_enabled': False, 'metal_breaks_utc': []}, 'triggers': {}, 'patterns': {'enabled': True}, 'composite_score': {'weights': {'vol_rel': 1.0}}, 'logging': {'level': 'INFO', 'service_name': 'external-markets-service'}, 'main_timeframe': '5m', 'history_depth': history_depth}

def test_service_clamps_history_depth_and_max_retries() -> None:
    service = ExternalMarketsService(_settings(history_depth=999, max_retries=42))
    assert service.history_depth == 200
    assert service.collector.max_retries == 5

def test_service_rejects_unknown_data_provider() -> None:
    settings = _settings()
    settings['data_provider']['name'] = 'unsupported_provider'
    with pytest.raises(ValueError, match='Unsupported data_provider.name'):
        ExternalMarketsService(settings)

def test_publish_json_skips_invalid_live_payload() -> None:
    service = ExternalMarketsService(_settings())
    rabbit = _FakeRabbit()
    service.rabbit = rabbit
    invalid_payload = {'symbol': 'AAPL', 'asset_class': 'stock', 'price': 101.0, 'bid': 100.9, 'ask': 101.1, 'timestamp': '2026-01-01T00:00:00Z'}
    asyncio.run(service._publish_json(service.exchange_name, service.price_routing_key, invalid_payload))
    assert rabbit.published == []
    valid_payload = dict(invalid_payload)
    valid_payload['ts'] = int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000)
    asyncio.run(service._publish_json(service.exchange_name, service.price_routing_key, valid_payload))
    assert len(rabbit.published) == 1
    published = json.loads(rabbit.published[0][2])
    assert published['symbol'] == 'AAPL'

def test_publish_json_skips_stale_live_payload() -> None:
    service = ExternalMarketsService(_settings())
    rabbit = _FakeRabbit()
    service.rabbit = rabbit
    stale_payload = {'symbol': 'AAPL', 'asset_class': 'stock', 'price': 101.0, 'bid': 100.9, 'ask': 101.1, 'timestamp': '2026-01-01T00:00:00Z', 'ts': int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000) - 10000}
    asyncio.run(service._publish_json(service.exchange_name, service.price_routing_key, stale_payload))
    assert rabbit.published == []


def test_missing_asset_specific_features_are_not_hard_rejected() -> None:
    service = ExternalMarketsService(_settings())
    assert service._has_required_asset_specific_features(symbol='AAPL', asset_class='stock', features={'sp500_correlation': None})
    assert service._has_required_asset_specific_features(symbol='GC=F', asset_class='metal', features={'dxy_correlation': None})
    assert service._has_required_asset_specific_features(symbol='EURUSD=X', asset_class='forex', features={'bid_ask_spread_pips': None})

def test_scan_passes_pattern_score_and_payload() -> None:
    frame = pd.DataFrame({'open': [100.0, 101.0], 'high': [101.0, 102.0], 'low': [99.5, 100.5], 'close': [100.5, 101.5]}, index=pd.DatetimeIndex([pd.Timestamp('2026-01-01T10:00:00Z'), pd.Timestamp('2026-01-01T10:05:00Z')]))
    service = ExternalMarketsService(_settings(history_depth=10))
    service.market_hours = _AlwaysOpenHours()
    service.feature_generator = _DummyFeatureGenerator(frame)
    service.trigger_engine = _DummyTriggerEngine()
    capture_composite = _CaptureComposite()
    service.composite_score = capture_composite
    service.pattern_engine = _DummyPatternEngine()

    async def _fake_fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        _ = (symbol, timeframe, limit)
        return frame

    async def _fake_fetch_quote_with_cache(symbol: str, now_utc: dt.datetime, allow_stale: bool) -> Quote:
        _ = (symbol, now_utc, allow_stale)
        return Quote(symbol=symbol, price=101.0, bid=100.9, ask=101.1, timestamp_ms=int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000))

    async def _fake_build_context_trends(symbol: str, asset_class: str, now_utc: dt.datetime) -> dict[str, dict[str, Any]]:
        _ = (symbol, asset_class, now_utc)
        return {}
    published: list[dict[str, Any]] = []

    async def _fake_publish_json(exchange_name: str, routing_key: str, payload: dict[str, Any]) -> None:
        _ = (exchange_name, routing_key)
        published.append(payload)
    service._fetch_ohlcv = _fake_fetch_ohlcv
    service._fetch_quote_with_cache = _fake_fetch_quote_with_cache
    service._build_context_trends = _fake_build_context_trends
    service._publish_json = _fake_publish_json
    asyncio.run(service._scan_once())
    assert capture_composite.pattern_scores == [0.77]
    assert len(published) == 1
    assert published[0]['patterns']['consensus'] == 'BULLISH'
    assert published[0]['trigger_reason'] == 'EMA_CROSSOVER_BULLISH'
    assert published[0]['trigger_reasons'] == ['EMA_CROSSOVER_BULLISH', 'VOLUME_SPIKE']
