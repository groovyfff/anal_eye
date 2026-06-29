import asyncio
import datetime as dt
from typing import Any
import pandas as pd
from src.collectors.base_collector import Quote
from src.logic.trigger_engine import TriggerResult
from src.main import ExternalMarketsService

class _AlwaysOpenHours:

    def is_market_open(self, asset_class: str, now_utc: dt.datetime | None=None) -> bool:
        _ = (asset_class, now_utc)
        return True

class _DummyFeatureGenerator:

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def compute_indicators(self, frame: Any, asset_class: str, benchmark_close: Any=None, dxy_close: Any=None) -> pd.DataFrame:
        _ = (frame, asset_class, benchmark_close, dxy_close)
        return self.frame

    def build_feature_payload(self, indicator_df: pd.DataFrame, asset_class: str, bid: float | None, ask: float | None) -> dict[str, Any]:
        _ = (indicator_df, asset_class, bid, ask)
        return {'current_price': 101.0, 'rsi': 60.0, 'vol_rel': 2.1, 'macd_hist': 0.3, 'adx': 25.0, 'sp500_correlation': 0.7}

    def build_historical_snapshots(self, indicator_df: pd.DataFrame, count: int=2) -> list[dict[str, Any]]:
        _ = (indicator_df, count)
        return []

class _DummyTriggerEngine:

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons

    def evaluate(self, indicator_df: pd.DataFrame, context_trends: dict[str, dict[str, Any]] | None=None) -> TriggerResult:
        _ = (indicator_df, context_trends)
        return TriggerResult(triggered=True, reasons=list(self.reasons), heuristic_signal_consensus='LONG', indicators={'consensus': 'BULLISH', 'consensus_strength': 1.0, 'signals': []})

class _DummyCompositeScore:

    def calculate(self, features: dict[str, Any], pattern_score: float=0.0) -> float:
        _ = (features, pattern_score)
        return 1.0

    def should_publish(self, score: float) -> bool:
        _ = score
        return True

def _settings() -> dict[str, Any]:
    return {'rabbitmq': {'url': 'amqp://guest:guest@localhost:5672/', 'exchange': 'analeyes_exchange'}, 'data_provider': {'name': 'yahoo_finance', 'request_delay_s': 0.0, 'max_retries': 1}, 'watchlist': {'stocks': [{'symbol': 'AAPL', 'name': 'Apple'}], 'indices': [], 'metals': [], 'forex': []}, 'market_hours': {'timezone': 'America/New_York', 'stock_open': '09:30', 'stock_close': '16:00', 'pre_market_enabled': False, 'after_hours_enabled': False, 'metal_breaks_utc': []}, 'triggers': {}, 'composite_score': {'weights': {}}, 'database': {'enabled': False}, 'logging': {'level': 'INFO', 'service_name': 'external-markets-service'}, 'main_timeframe': '5m', 'history_depth': 10}

def _indicator_frame(last_open_ts: str) -> pd.DataFrame:
    last_open = pd.Timestamp(last_open_ts, tz='UTC')
    previous_open = last_open - pd.Timedelta(minutes=5)
    index = pd.DatetimeIndex([previous_open, last_open])
    return pd.DataFrame({'close': [100.0, 101.0]}, index=index)

def _build_service(frame: pd.DataFrame, reasons: list[str]) -> tuple[ExternalMarketsService, list[dict[str, Any]], _DummyFeatureGenerator, _DummyTriggerEngine]:
    service = ExternalMarketsService(_settings())
    service.market_hours = _AlwaysOpenHours()
    generator = _DummyFeatureGenerator(frame)
    trigger = _DummyTriggerEngine(reasons)
    service.feature_generator = generator
    service.trigger_engine = trigger
    service.composite_score = _DummyCompositeScore()

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
    return (service, published, generator, trigger)

def test_scan_deduplicates_same_candle_same_reasons_even_if_order_changes() -> None:
    frame = _indicator_frame('2026-01-01T10:05:00Z')
    service, published, _generator, trigger = _build_service(frame=frame, reasons=['EMA_CROSSOVER_BULLISH', 'VOLUME_SPIKE'])
    asyncio.run(service._scan_once())
    trigger.reasons = ['VOLUME_SPIKE', 'EMA_CROSSOVER_BULLISH']
    asyncio.run(service._scan_once())
    assert len(published) == 1
    assert published[0]['timestamp'] == '2026-01-01T10:10:00Z'

def test_scan_does_not_republish_when_reasons_change_on_same_candle() -> None:
    frame = _indicator_frame('2026-01-01T10:05:00Z')
    service, published, _generator, trigger = _build_service(frame=frame, reasons=['EMA_CROSSOVER_BULLISH'])
    asyncio.run(service._scan_once())
    trigger.reasons = ['RSI_OVERSOLD_EXIT']
    asyncio.run(service._scan_once())
    assert len(published) == 1

def test_scan_publishes_again_for_new_candle_with_same_reasons() -> None:
    frame = _indicator_frame('2026-01-01T10:05:00Z')
    service, published, generator, _trigger = _build_service(frame=frame, reasons=['EMA_CROSSOVER_BULLISH'])
    asyncio.run(service._scan_once())
    generator.frame = _indicator_frame('2026-01-01T10:10:00Z')
    asyncio.run(service._scan_once())
    assert len(published) == 2
    assert published[1]['timestamp'] == '2026-01-01T10:15:00Z'
