from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    enabled: bool
    symbols: list[str]
    timeframe: str
    market: str
    wss_base_url: str
    reconnect_delay_sec: int
    min_candles: int
    bootstrap_limit: int
    rest_base_url: str
    publish_mode: str
    throttle_sec: int
    publish_on_candle_close: bool
    publish_on_every_update: bool
    max_candles: int = 200

    @property
    def stream_suffix(self) -> str:
        return f"kline_{self.timeframe}"

    def stream_name(self, symbol: str) -> str:
        return f"{symbol.lower()}@{self.stream_suffix}"

    def all_stream_names(self) -> list[str]:
        return [self.stream_name(symbol) for symbol in self.symbols]

    def build_wss_url(self) -> str:
        streams = self.all_stream_names()
        if len(streams) == 1:
            return f"{self.wss_base_url}/{streams[0]}"
        joined = "/".join(streams)
        return f"wss://fstream.binance.com/stream?streams={joined}"

    @classmethod
    def from_env(cls) -> ServiceConfig:
        symbols_raw = (os.environ.get("BINANCE_SYMBOLS") or "BTCUSDT").strip()
        symbols = [part.strip().upper() for part in symbols_raw.split(",") if part.strip()]
        if not symbols:
            symbols = ["BTCUSDT"]
        publish_on_every_update = _env_bool("BINANCE_CANDIDATE_PUBLISH_ON_EVERY_UPDATE", False)
        return cls(
            enabled=_env_bool("BINANCE_CANDIDATE_ENABLED", True),
            symbols=symbols,
            timeframe=(os.environ.get("BINANCE_TIMEFRAME") or "1h").strip(),
            market=(os.environ.get("BINANCE_MARKET") or "futures").strip(),
            wss_base_url=(os.environ.get("BINANCE_WSS_BASE_URL") or "wss://fstream.binance.com/ws").strip().rstrip("/"),
            reconnect_delay_sec=_env_int("BINANCE_RECONNECT_DELAY_SEC", 5),
            min_candles=_env_int("BINANCE_CANDIDATE_MIN_CANDLES", 100),
            bootstrap_limit=_env_int("BINANCE_BOOTSTRAP_LIMIT", 200),
            rest_base_url=(os.environ.get("BINANCE_REST_BASE_URL") or "https://fapi.binance.com").strip().rstrip("/"),
            publish_mode=(os.environ.get("BINANCE_CANDIDATE_PUBLISH_MODE") or "throttled").strip(),
            throttle_sec=_env_int("BINANCE_CANDIDATE_THROTTLE_SEC", 60),
            publish_on_candle_close=_env_bool("BINANCE_CANDIDATE_PUBLISH_ON_CANDLE_CLOSE", True),
            publish_on_every_update=publish_on_every_update,
        )
