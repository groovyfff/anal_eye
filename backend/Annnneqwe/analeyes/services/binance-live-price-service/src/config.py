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
    publish_raw_kline: bool
    log_every_n: int

    @property
    def stream_suffix(self) -> str:
        return f"kline_{self.timeframe}"

    def stream_name(self, symbol: str) -> str:
        return f"{symbol.lower()}@{self.stream_suffix}"

    def all_stream_names(self) -> list[str]:
        return [self.stream_name(symbol) for symbol in self.symbols]

    @classmethod
    def from_env(cls, *, symbols: list[str]) -> ServiceConfig:
        if not symbols:
            raise ValueError("symbols must not be empty — set SYMBOLS or enable auto-discovery")
        return cls(
            enabled=_env_bool("BINANCE_LIVE_ENABLED", True),
            symbols=[s.strip().upper() for s in symbols],
            timeframe=(os.environ.get("BINANCE_TIMEFRAME") or "1h").strip(),
            market=(os.environ.get("BINANCE_MARKET") or "futures").strip(),
            wss_base_url=(os.environ.get("BINANCE_WSS_BASE_URL") or "wss://fstream.binance.com/ws").strip().rstrip("/"),
            reconnect_delay_sec=_env_int("BINANCE_RECONNECT_DELAY_SEC", 5),
            publish_raw_kline=_env_bool("BINANCE_PUBLISH_RAW_KLINE", True),
            log_every_n=max(1, _env_int("BINANCE_LOG_EVERY_N", 20)),
        )

    def build_wss_url(self) -> str:
        streams = self.all_stream_names()
        if len(streams) == 1:
            return f"{self.wss_base_url}/{streams[0]}"
        joined = "/".join(streams)
        return f"wss://fstream.binance.com/stream?streams={joined}"
