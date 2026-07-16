from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True
    close_time: int = 0
    quote_volume: float | None = None
    trades_count: int | None = None
    taker_buy_base_volume: float | None = None
    taker_buy_quote_volume: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades_count": self.trades_count,
            "taker_buy_base_volume": self.taker_buy_base_volume,
            "taker_buy_quote_volume": self.taker_buy_quote_volume,
        }


class CandleBuffer:
    """Rolling per-symbol candle history keyed by open timestamp."""

    def __init__(self, *, max_candles: int = 200) -> None:
        self._max_candles = max(1, max_candles)
        self._by_symbol: dict[str, dict[int, Candle]] = {}

    def load_bootstrap(self, symbol: str, candles: list[Candle]) -> int:
        store = self._by_symbol.setdefault(symbol.upper(), {})
        for candle in sorted(candles, key=lambda c: c.timestamp):
            store[candle.timestamp] = candle
        self._trim(symbol)
        return len(self._by_symbol[symbol.upper()])

    def upsert(self, symbol: str, candle: Candle) -> None:
        store = self._by_symbol.setdefault(symbol.upper(), {})
        store[candle.timestamp] = candle
        self._trim(symbol)

    def count(self, symbol: str) -> int:
        return len(self._by_symbol.get(symbol.upper(), {}))

    def candles(self, symbol: str) -> list[Candle]:
        store = self._by_symbol.get(symbol.upper(), {})
        return [store[ts] for ts in sorted(store)]

    def latest(self, symbol: str) -> Candle | None:
        rows = self.candles(symbol)
        return rows[-1] if rows else None

    def last_close_time(self, symbol: str) -> int | None:
        latest = self.latest(symbol)
        if latest is None:
            return None
        return latest.close_time or latest.timestamp

    def _trim(self, symbol: str) -> None:
        store = self._by_symbol.get(symbol.upper())
        if not store or len(store) <= self._max_candles:
            return
        keep_ts = sorted(store)[-self._max_candles :]
        self._by_symbol[symbol.upper()] = {ts: store[ts] for ts in keep_ts}
