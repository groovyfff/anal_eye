from __future__ import annotations
import abc
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar
import pandas as pd
import requests
from src.utils.logger import get_logger
T = TypeVar('T')

class TooManyRequestsError(RuntimeError):
    pass

@dataclass(slots=True)
class Quote:
    symbol: str
    price: float
    bid: float | None
    ask: float | None
    timestamp_ms: int

class BaseCollector(abc.ABC):

    def __init__(self, api_key: str | None=None, request_delay_s: float=0.25, quote_request_delay_s: float | None=None, max_retries: int=5) -> None:
        self.api_key = api_key or ''
        self.request_delay_s = request_delay_s
        self.quote_request_delay_s = request_delay_s if quote_request_delay_s is None else float(quote_request_delay_s)
        self.max_retries = max(1, min(5, int(max_retries)))
        self._last_request_ts: dict[str, float] = {}
        self._throttle_lock = threading.Lock()
        self._logger = get_logger(__name__)

    def _throttle(self, throttle_key: str='default') -> None:
        request_delay_s = self.request_delay_s
        if throttle_key == 'quote':
            request_delay_s = self.quote_request_delay_s
        with self._throttle_lock:
            last_ts = self._last_request_ts.get(throttle_key, 0.0)
            elapsed = time.monotonic() - last_ts
            wait_s = request_delay_s - elapsed
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_request_ts[throttle_key] = time.monotonic()

    @staticmethod
    def _retry_delay_s(attempt: int) -> float:
        return float(min(2 ** (attempt - 1), 4))

    @staticmethod
    def _extract_symbol(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        if isinstance(kwargs.get('symbol'), str):
            return str(kwargs['symbol'])
        if args and isinstance(args[0], str):
            return str(args[0])
        return '-'

    def _with_backoff(self, fn: Callable[..., T], *args: Any, throttle_key: str='default', **kwargs: Any) -> T:
        symbol = self._extract_symbol(args, kwargs)
        for attempt in range(1, self.max_retries + 1):
            try:
                self._throttle(throttle_key=throttle_key)
                return fn(*args, **kwargs)
            except TooManyRequestsError as exc:
                if attempt == self.max_retries:
                    self._logger.error('Provider returned 429, max retries reached (%s/%s): %s', attempt, self.max_retries, exc, extra={'symbol': symbol})
                    raise
                delay = self._retry_delay_s(attempt)
                self._logger.error('Provider returned 429. Retrying in %.1fs (attempt %s/%s)', delay, attempt, self.max_retries, extra={'symbol': symbol})
                time.sleep(delay)
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code == 429:
                    if attempt == self.max_retries:
                        self._logger.error('Provider HTTP 429, max retries reached (%s/%s): %s', attempt, self.max_retries, exc, extra={'symbol': symbol})
                        raise TooManyRequestsError('Too many requests') from exc
                    delay = self._retry_delay_s(attempt)
                    self._logger.error('Provider HTTP 429. Retrying in %.1fs (attempt %s/%s)', delay, attempt, self.max_retries, extra={'symbol': symbol})
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError('Retry loop exited unexpectedly')

    @abc.abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def fetch_quote(self, symbol: str) -> Quote:
        pass

    @abc.abstractmethod
    def fetch_close_series(self, symbol: str, timeframe: str, limit: int) -> pd.Series:
        pass
