from __future__ import annotations
import datetime as dt
import math
from typing import Any
import pandas as pd
import requests
from src.collectors.base_collector import BaseCollector, Quote, TooManyRequestsError
from src.utils.logger import get_logger

class YahooFinanceCollector(BaseCollector):
    BASE_URL = 'https://query1.finance.yahoo.com/v8/finance/chart'
    _INTERVAL_TO_RANGE = {'1m': '7d', '2m': '60d', '5m': '60d', '15m': '60d', '30m': '60d', '60m': '730d', '90m': '60d', '1h': '730d', '1d': '10y', '5d': '10y', '1wk': '10y', '1mo': '10y', '3mo': '10y', '4h': '730d'}

    def __init__(self, api_key: str | None=None, request_delay_s: float=0.25, quote_request_delay_s: float | None=None, max_retries: int=5, include_prepost: bool=False) -> None:
        super().__init__(api_key=api_key, request_delay_s=request_delay_s, quote_request_delay_s=quote_request_delay_s, max_retries=max_retries)
        self._logger = get_logger(__name__)
        self.include_prepost = include_prepost
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36', 'Accept': 'application/json,text/plain,*/*', 'Accept-Language': 'en-US,en;q=0.9', 'Connection': 'keep-alive', 'Referer': 'https://finance.yahoo.com/'})

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        interval = self._normalize_interval(timeframe)
        range_value = self._INTERVAL_TO_RANGE.get(interval, '60d')
        try:
            chart = self._with_backoff(self._request_chart, symbol, interval, range_value, throttle_key='ohlcv')
        except Exception as exc:
            if '429' in str(exc):
                raise TooManyRequestsError(str(exc)) from exc
            self._logger.error('Yahoo OHLCV fetch failed for %s: %s', symbol, exc, extra={'symbol': symbol})
            raise
        frame = self._chart_to_frame(chart, symbol)
        if timeframe == '4h':
            frame = self._resample_4h(frame, symbol=symbol)
        return frame.tail(limit)

    def fetch_quote(self, symbol: str) -> Quote:

        def _load() -> Quote:
            chart = self._request_chart(symbol=symbol, interval='1m', range_value='1d')
            meta: dict[str, Any] = chart.get('meta', {}) or {}
            last_price = self._num_or_none(meta.get('regularMarketPrice'))
            bid = self._num_or_none(meta.get('bid'))
            ask = self._num_or_none(meta.get('ask'))
            if last_price is None:
                frame = self._chart_to_frame(chart, symbol)
                closes = frame['close'].dropna()
                if closes.empty:
                    raise ValueError(f'No quote data for {symbol}')
                last_price = float(closes.iloc[-1])
            ts_ms = self._extract_quote_timestamp_ms(chart=chart, meta=meta)
            if ts_ms is None:
                ts_ms = int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000)
                self._logger.warning('Yahoo quote has no source timestamp, falling back to fetch time', extra={'symbol': symbol})
            return Quote(symbol=symbol, price=float(last_price), bid=bid, ask=ask, timestamp_ms=ts_ms)
        try:
            return self._with_backoff(_load, throttle_key='quote')
        except Exception as exc:
            if '429' in str(exc):
                raise TooManyRequestsError(str(exc)) from exc
            self._logger.error('Yahoo quote fetch failed for %s: %s', symbol, exc, extra={'symbol': symbol})
            raise

    @classmethod
    def _extract_quote_timestamp_ms(cls, chart: dict[str, Any], meta: dict[str, Any]) -> int | None:
        meta_ts_ms = cls._epoch_to_millis(meta.get('regularMarketTime'))
        if meta_ts_ms is not None:
            return meta_ts_ms
        timestamps = chart.get('timestamp') or []
        if timestamps:
            return cls._epoch_to_millis(timestamps[-1])
        return None

    def fetch_close_series(self, symbol: str, timeframe: str, limit: int) -> pd.Series:
        frame = self.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        return frame['close'].tail(limit)

    def _request_chart(self, symbol: str, interval: str, range_value: str) -> dict[str, Any]:
        url = f'{self.BASE_URL}/{symbol}'
        params = {'interval': interval, 'range': range_value, 'events': 'history', 'includePrePost': 'true' if self.include_prepost else 'false'}
        response = self._session.get(url, params=params, timeout=20)
        if response.status_code == 429:
            raise TooManyRequestsError('Yahoo rate limited with HTTP 429')
        response.raise_for_status()
        payload = response.json()
        chart = payload.get('chart', {}) or {}
        chart_error = chart.get('error')
        if chart_error:
            description = str(chart_error.get('description', 'Yahoo chart error'))
            code = str(chart_error.get('code', ''))
            if 'Too Many Requests' in description or 'Too Many Requests' in code:
                raise TooManyRequestsError(description)
            raise ValueError(f'Yahoo chart error for {symbol}: {description}')
        result = chart.get('result') or []
        if not result:
            raise ValueError(f'No chart result for {symbol}')
        return result[0]

    @staticmethod
    def _normalize_interval(timeframe: str) -> str:
        mapping = {'1h': '60m', '4h': '60m'}
        return mapping.get(timeframe, timeframe)

    @staticmethod
    def _chart_to_frame(chart: dict[str, Any], symbol: str) -> pd.DataFrame:
        timestamps = chart.get('timestamp') or []
        quote_block = (chart.get('indicators') or {}).get('quote') or []
        if not timestamps or not quote_block:
            raise ValueError(f'No OHLCV blocks for {symbol}')
        quote = quote_block[0]
        df = pd.DataFrame({'open': quote.get('open', []), 'high': quote.get('high', []), 'low': quote.get('low', []), 'close': quote.get('close', []), 'volume': quote.get('volume', [])}, index=pd.to_datetime(timestamps, unit='s', utc=True))
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = df['volume'].fillna(0.0)
        df = df.dropna(subset=['open', 'high', 'low', 'close'], how='any')
        df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
        if df.empty:
            raise ValueError(f'Empty OHLCV frame for {symbol}')
        return df

    @staticmethod
    def _num_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _epoch_to_millis(value: Any) -> int | None:
        if value is None:
            return None
        try:
            raw = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(raw) or raw <= 0:
            return None
        if raw >= 100000000000000.0:
            return int(raw / 1000000)
        if raw >= 100000000000.0:
            return int(raw)
        return int(raw * 1000)

    @staticmethod
    def _resample_4h(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if YahooFinanceCollector._is_equity_like_symbol(symbol):
            ny = frame.tz_convert('America/New_York')
            shifted = ny.copy()
            shifted.index = shifted.index - pd.Timedelta(minutes=90)
            aggregated = shifted.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            aggregated.index = aggregated.index + pd.Timedelta(minutes=90)
            aggregated = aggregated.tz_convert('UTC')
        else:
            aggregated = frame.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        aggregated = aggregated.dropna(subset=['open', 'high', 'low', 'close'])
        return aggregated

    @staticmethod
    def _is_equity_like_symbol(symbol: str) -> bool:
        normalized = symbol.upper().strip()
        return not normalized.endswith('=F') and (not normalized.endswith('=X'))
