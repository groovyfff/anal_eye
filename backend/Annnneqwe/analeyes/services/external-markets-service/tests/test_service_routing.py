import datetime as dt
import json
import asyncio
from src.collectors.base_collector import Quote
from src.main import ExternalMarketsService

class _AlwaysOpenHours:

    def is_market_open(self, asset_class: str, now_utc: dt.datetime | None=None) -> bool:
        _ = (asset_class, now_utc)
        return True

class _FakeCollector:

    def __init__(self) -> None:
        self.calls = 0

    def fetch_quote(self, symbol: str) -> Quote:
        self.calls += 1
        now_ms = int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000)
        return Quote(symbol=symbol, price=100.0, bid=99.9, ask=100.1, timestamp_ms=now_ms)

class _AlwaysStaleCollector:

    def fetch_quote(self, symbol: str) -> Quote:
        stale_ms = int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000) - 10000
        return Quote(symbol=symbol, price=100.0, bid=99.9, ask=100.1, timestamp_ms=stale_ms)


class _FixedTimestampCollector:

    def __init__(self, ts_ms: int) -> None:
        self.ts_ms = ts_ms

    def fetch_quote(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=100.0, bid=99.9, ask=100.1, timestamp_ms=self.ts_ms)

class _FailingCollector:

    def fetch_quote(self, symbol: str) -> Quote:
        _ = symbol
        raise RuntimeError('provider unavailable')

class _FakeRabbit:

    def __init__(self) -> None:
        self.published: list[tuple[str, str, str]] = []

    def publish(self, exchange_name: str, routing_key: str, body: str) -> None:
        self.published.append((exchange_name, routing_key, body))

    async def publish_async(self, exchange_name: str, routing_key: str, body: str) -> None:
        self.publish(exchange_name=exchange_name, routing_key=routing_key, body=body)

def _settings() -> dict:
    return {'rabbitmq': {'url': 'amqp://guest:guest@localhost:5672/', 'exchange': 'analeyes_exchange'}, 'data_provider': {'name': 'yahoo_finance', 'request_delay_s': 0.0, 'max_retries': 1}, 'watchlist': {'stocks': [{'symbol': 'AAPL', 'name': 'Apple'}], 'indices': [{'symbol': '^GSPC', 'name': 'S&P500', 'use_for_correlation': True}]}, 'market_hours': {'timezone': 'America/New_York', 'stock_open': '09:30', 'stock_close': '16:00', 'pre_market_enabled': False, 'after_hours_enabled': False, 'metal_breaks_utc': []}, 'triggers': {}, 'composite_score': {'weights': {}}, 'logging': {'level': 'INFO', 'service_name': 'external-markets-service'}}

def test_indices_without_correlation_flag_are_scanned_as_stock_candidates() -> None:
    settings = _settings()
    settings['watchlist']['indices'].append({'symbol': '^NDX', 'name': 'NASDAQ 100', 'use_for_correlation': False})
    service = ExternalMarketsService(settings)
    scan_symbols = {item['symbol'] for item in service.scan_candidates}
    assert '^NDX' in scan_symbols
    ndx_item = next((item for item in service.scan_candidates if item['symbol'] == '^NDX'))
    assert ndx_item['asset_class'] == 'stock'

def test_correlation_only_indices_are_not_broadcasted_or_scanned() -> None:
    service = ExternalMarketsService(_settings())
    service.market_hours = _AlwaysOpenHours()
    service.collector = _FakeCollector()
    service.rabbit = _FakeRabbit()
    service._quote_cache = {item['symbol']: service.collector.fetch_quote(item['symbol']) for item in service.broadcast_items}
    assert [item['symbol'] for item in service.scan_candidates] == ['AAPL']
    assert {item['symbol'] for item in service.broadcast_items} == {'AAPL'}
    asyncio.run(asyncio.wait_for(service._broadcast_once(), timeout=1.0))
    symbols = {json.loads(body)['symbol'] for _exchange, routing_key, body in service.rabbit.published if routing_key == 'data.live_prices.external'}
    assert symbols == {'AAPL'}

def test_broadcast_refreshes_stale_cache_before_publish() -> None:
    service = ExternalMarketsService(_settings())
    service.market_hours = _AlwaysOpenHours()
    collector = _FakeCollector()
    service.collector = collector
    service.rabbit = _FakeRabbit()
    stale_ts_ms = int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000) - 10000
    service._quote_cache = {item['symbol']: Quote(symbol=item['symbol'], price=100.0, bid=99.9, ask=100.1, timestamp_ms=stale_ts_ms) for item in service.broadcast_items}
    asyncio.run(asyncio.wait_for(service._broadcast_once(), timeout=1.0))
    symbols = {json.loads(body)['symbol'] for _exchange, routing_key, body in service.rabbit.published if routing_key == 'data.live_prices.external'}
    assert symbols == {'AAPL'}
    assert collector.calls >= 1

def test_broadcast_skips_publish_when_quote_stays_stale() -> None:
    service = ExternalMarketsService(_settings())
    service.market_hours = _AlwaysOpenHours()
    service.collector = _AlwaysStaleCollector()
    service.rabbit = _FakeRabbit()
    service._quote_cache = {}
    asyncio.run(asyncio.wait_for(service._broadcast_once(), timeout=1.0))
    symbols = {json.loads(body)['symbol'] for _exchange, routing_key, body in service.rabbit.published if routing_key == 'data.live_prices.external'}
    assert symbols == set()

def test_broadcast_skips_publish_when_quote_fetch_fails() -> None:
    service = ExternalMarketsService(_settings())
    service.market_hours = _AlwaysOpenHours()
    service.collector = _FailingCollector()
    service.rabbit = _FakeRabbit()
    service._quote_cache = {}
    asyncio.run(asyncio.wait_for(service._broadcast_once(), timeout=1.0))
    symbols = {json.loads(body)['symbol'] for _exchange, routing_key, body in service.rabbit.published if routing_key == 'data.live_prices.external'}
    assert symbols == set()


def test_broadcast_uses_source_quote_timestamp_for_ts_and_timestamp() -> None:
    service = ExternalMarketsService(_settings())
    service.market_hours = _AlwaysOpenHours()
    source_ts_ms = int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000) - 1000
    service.collector = _FixedTimestampCollector(ts_ms=source_ts_ms)
    service.rabbit = _FakeRabbit()
    service.quote_freshness_ms = 60000
    service._quote_cache = {}
    asyncio.run(asyncio.wait_for(service._broadcast_once(), timeout=1.0))
    payload = json.loads(service.rabbit.published[0][2])
    assert payload['ts'] == source_ts_ms
    expected_timestamp = dt.datetime.fromtimestamp(source_ts_ms / 1000, tz=dt.timezone.utc).isoformat().replace('+00:00', 'Z')
    assert payload['timestamp'] == expected_timestamp
