from __future__ import annotations
import datetime as dt
import math
from typing import Any
import pandas as pd
import requests
from src.collectors.base_collector import BaseCollector, Quote, TooManyRequestsError
from src.utils.logger import get_logger

class PolygonCollector(BaseCollector):
    BASE_URL = 'https://api.polygon.io'

    def __init__(self, api_key: str, request_delay_s: float=0.25, quote_request_delay_s: float | None=None, max_retries: int=5) -> None:
        if not api_key:
            raise ValueError('Polygon collector requires api_key')
        super().__init__(api_key=api_key, request_delay_s=request_delay_s, quote_request_delay_s=quote_request_delay_s, max_retries=max_retries)
        self._logger = get_logger(__name__)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        multiplier, timespan = self._parse_timeframe(timeframe)
        end = dt.datetime.now(tz=dt.timezone.utc)
        start = end - dt.timedelta(days=60)
        url = f'{self.BASE_URL}/v2/aggs/ticker/{self._normalize_symbol(symbol)}/range/{multiplier}/{timespan}/{start.date()}/{end.date()}'
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': max(limit * 3, 500), 'apiKey': self.api_key}

        def _load() -> pd.DataFrame:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code == 429:
                raise TooManyRequestsError('Polygon rate limited')
            response.raise_for_status()
            payload = response.json()
            rows = payload.get('results', [])
            if not rows:
                raise ValueError(f'No Polygon OHLCV data for {symbol}')
            frame = pd.DataFrame(rows)
            frame = frame.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'ts'})
            frame['ts'] = pd.to_datetime(frame['ts'], unit='ms', utc=True)
            frame = frame.set_index('ts')
            return frame[['open', 'high', 'low', 'close', 'volume']]
        try:
            return self._with_backoff(_load, throttle_key='ohlcv').tail(limit)
        except Exception as exc:
            self._logger.error('Polygon OHLCV fetch failed for %s: %s', symbol, exc, extra={'symbol': symbol})
            raise

    def fetch_quote(self, symbol: str) -> Quote:
        ticker = self._normalize_symbol(symbol)
        trade_url = f'{self.BASE_URL}/v2/last/trade/{ticker}'
        nbbo_url = f'{self.BASE_URL}/v2/last/nbbo/{ticker}'
        params = {'apiKey': self.api_key}

        def _load() -> Quote:
            response = requests.get(trade_url, params=params, timeout=20)
            if response.status_code == 429:
                raise TooManyRequestsError('Polygon rate limited')
            response.raise_for_status()
            payload = response.json().get('results', {}) or {}
            price = self._to_float(payload.get('p'))
            bid = self._to_float(payload.get('b'))
            ask = self._to_float(payload.get('a'))
            if price is None:
                nbbo_response = requests.get(nbbo_url, params=params, timeout=20)
                if nbbo_response.status_code == 429:
                    raise TooManyRequestsError('Polygon rate limited')
                if nbbo_response.status_code < 400:
                    nbbo_payload = nbbo_response.json().get('results', {}) or {}
                    bid = bid if bid is not None else self._pick_float(nbbo_payload, ('bp', 'bid_price', 'b', 'bid'))
                    ask = ask if ask is not None else self._pick_float(nbbo_payload, ('ap', 'ask_price', 'a', 'ask'))
            if price is None and bid is not None and (ask is not None):
                price = (bid + ask) / 2.0
            if price is None:
                raise ValueError(f'No Polygon quote data for {symbol}')
            ts_ms = self._extract_quote_timestamp_ms(payload)
            if ts_ms is None:
                ts_ms = int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000)
                self._logger.warning('Polygon quote has no source timestamp, falling back to fetch time', extra={'symbol': symbol})
            return Quote(symbol=symbol, price=float(price), bid=bid, ask=ask, timestamp_ms=ts_ms)
        try:
            return self._with_backoff(_load, throttle_key='quote')
        except Exception as exc:
            self._logger.error('Polygon quote fetch failed for %s: %s', symbol, exc, extra={'symbol': symbol})
            raise

    def fetch_close_series(self, symbol: str, timeframe: str, limit: int) -> pd.Series:
        return self.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)['close']

    @staticmethod
    def _parse_timeframe(timeframe: str) -> tuple[int, str]:
        if timeframe.endswith('m'):
            return (int(timeframe[:-1]), 'minute')
        if timeframe.endswith('h'):
            return (int(timeframe[:-1]), 'hour')
        if timeframe.endswith('d'):
            return (int(timeframe[:-1]), 'day')
        raise ValueError(f'Unsupported timeframe for Polygon: {timeframe}')

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        normalized = symbol.strip().upper()
        if normalized.startswith('^'):
            return f'I:{normalized[1:]}'
        if normalized.endswith('=X'):
            pair = normalized[:-2]
            if len(pair) == 6 and pair.isalpha():
                return f'C:{pair}'
            return pair
        if normalized.endswith('=F'):
            return normalized[:-2]
        return normalized

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pick_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
        for key in keys:
            value = PolygonCollector._to_float(payload.get(key))
            if value is not None:
                return value
        return None

    @classmethod
    def _extract_quote_timestamp_ms(cls, payload: dict[str, Any]) -> int | None:
        for key in ('t', 'timestamp', 'sip_timestamp', 'participant_timestamp', 'trf_timestamp'):
            ts_ms = cls._epoch_to_millis(payload.get(key))
            if ts_ms is not None:
                return ts_ms
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
        if raw >= 1000000000000000.0:
            return int(raw / 1000000)
        if raw >= 1000000000000.0:
            return int(raw)
        if raw >= 1000000000.0:
            return int(raw * 1000)
        return None
