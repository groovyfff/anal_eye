from src.collectors.polygon_collector import PolygonCollector
from src.collectors.yahoo_finance_collector import YahooFinanceCollector

class _FakeResponse:

    def __init__(self, payload: dict, status_code: int=200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f'HTTP {self.status_code}')

    def json(self) -> dict:
        return self._payload

def _chart_payload() -> dict:
    return {'chart': {'result': [{'meta': {'regularMarketPrice': 182.5}, 'timestamp': [1705327800, 1705328100], 'indicators': {'quote': [{'open': [182.0, 182.2], 'high': [182.8, 182.9], 'low': [181.9, 182.1], 'close': [182.5, 182.7], 'volume': [1000, 1200]}]}}], 'error': None}}

def test_yahoo_include_prepost_flag_true(monkeypatch) -> None:
    collector = YahooFinanceCollector(include_prepost=True, request_delay_s=0.0, max_retries=1)
    captured: dict = {}

    def _fake_get(url: str, params: dict, timeout: int):
        _ = (url, timeout)
        captured.update(params)
        return _FakeResponse(_chart_payload())
    monkeypatch.setattr(collector._session, 'get', _fake_get)
    collector.fetch_ohlcv(symbol='AAPL', timeframe='5m', limit=2)
    assert captured['includePrePost'] == 'true'

def test_yahoo_quote_prefers_source_timestamp(monkeypatch) -> None:
    collector = YahooFinanceCollector(request_delay_s=0.0, max_retries=1)
    payload = _chart_payload()
    payload['chart']['result'][0]['meta']['regularMarketTime'] = 1767225600

    def _fake_get(url: str, params: dict, timeout: int):
        _ = (url, params, timeout)
        return _FakeResponse(payload)
    monkeypatch.setattr(collector._session, 'get', _fake_get)
    quote = collector.fetch_quote(symbol='AAPL')
    assert quote.timestamp_ms == 1767225600000

def test_polygon_quote_prefers_source_timestamp(monkeypatch) -> None:
    collector = PolygonCollector(api_key='test-key', request_delay_s=0.0, max_retries=1)

    def _fake_get(url: str, params: dict, timeout: int):
        _ = (url, params, timeout)
        return _FakeResponse({'results': {'p': 123.45, 't': 1767225600123456789}})
    monkeypatch.setattr('src.collectors.polygon_collector.requests.get', _fake_get)
    quote = collector.fetch_quote(symbol='AAPL')
    assert quote.timestamp_ms == 1767225600123
