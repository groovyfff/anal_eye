import pandas as pd
import pytest
from src.collectors.base_collector import BaseCollector, Quote, TooManyRequestsError

class _DummyCollector(BaseCollector):

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        _ = (symbol, timeframe, limit)
        raise NotImplementedError

    def fetch_quote(self, symbol: str) -> Quote:
        _ = symbol
        raise NotImplementedError

    def fetch_close_series(self, symbol: str, timeframe: str, limit: int) -> pd.Series:
        _ = (symbol, timeframe, limit)
        raise NotImplementedError

def test_backoff_uses_exponential_delays(monkeypatch) -> None:
    collector = _DummyCollector(request_delay_s=0.0, max_retries=5)
    delays: list[float] = []
    monkeypatch.setattr('src.collectors.base_collector.time.sleep', lambda value: delays.append(value))

    def _failing(symbol: str) -> None:
        _ = symbol
        raise TooManyRequestsError('rate limited')
    with pytest.raises(TooManyRequestsError):
        collector._with_backoff(_failing, 'AAPL')
    assert delays == [1.0, 2.0, 4.0, 4.0]
