from __future__ import annotations

import os
from dataclasses import dataclass

from shared.symbol_universe import default_allowed_symbols, resolve_production_symbols


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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    enabled: bool
    symbols: list[str]
    timeframe: str
    market: str
    wss_base_url: str
    reconnect_delay_sec: int
    window_candles: int
    rest_base_url: str
    closed_candles_only: bool
    dedup_enabled: bool
    min_interval_sec: int
    enable_legacy_parser: bool
    enable_high_frequency_test_parser: bool
    continuous_test_mode: bool
    publish_on_candle_close: bool
    publish_on_every_update: bool
    backfill_publish_historical: bool
    dedup_db_path: str
    app_env: str
    max_candles: int = 200
    ws_idle_timeout_sec: int = 60
    rest_fallback_enabled: bool = True
    rest_fallback_poll_sec: int = 60
    rest_fallback_always_on: bool = True

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
    def from_env(cls, *, symbols: list[str] | None = None) -> ServiceConfig:
        resolved = resolve_production_symbols(symbols)
        if not resolved:
            raise ValueError("allowed symbol universe is empty — set ANAL_EYES_ALLOWED_SYMBOLS")

        allowed = default_allowed_symbols()
        extras = [sym for sym in resolved if sym not in allowed]
        if extras:
            raise ValueError(f"unsupported symbols configured: {','.join(extras)}")

        timeframe = (os.environ.get("CANDIDATE_TIMEFRAME") or os.environ.get("BINANCE_TIMEFRAME") or "1h").strip()
        app_env = (os.environ.get("APP_ENV") or os.environ.get("ANAL_EYES_ENV") or "dev").strip().lower()
        closed_only = _env_bool("CANDIDATE_CLOSED_CANDLES_ONLY", True)
        dedup_enabled = _env_bool("CANDIDATE_DEDUP_ENABLED", True)
        legacy = _env_bool("ENABLE_LEGACY_PARSER", False)
        hf_test = _env_bool("ENABLE_HIGH_FREQUENCY_TEST_PARSER", False)
        continuous_test = _env_bool("CANDIDATE_CONTINUOUS_TEST_MODE", False)
        publish_on_close = _env_bool("BINANCE_CANDIDATE_PUBLISH_ON_CANDLE_CLOSE", True)
        publish_every_update = _env_bool("BINANCE_CANDIDATE_PUBLISH_ON_EVERY_UPDATE", False)

        if app_env in ("prod", "staging", "production"):
            if not closed_only:
                raise ValueError("CANDIDATE_CLOSED_CANDLES_ONLY=false is forbidden in prod/staging")
            if timeframe != "1h":
                raise ValueError(f"timeframe={timeframe!r} forbidden in prod/staging — model expects 1h")
            if legacy or hf_test:
                raise ValueError("legacy/high-frequency parsers are forbidden in prod/staging")
            if continuous_test:
                raise ValueError("CANDIDATE_CONTINUOUS_TEST_MODE=true is forbidden in prod/staging")
            if publish_every_update:
                raise ValueError("BINANCE_CANDIDATE_PUBLISH_ON_EVERY_UPDATE=true is forbidden in prod/staging")
            if not publish_on_close:
                raise ValueError("BINANCE_CANDIDATE_PUBLISH_ON_CANDLE_CLOSE=false is forbidden in prod/staging")
            skipped = os.environ.get("AEB_PUBLISH_SKIPPED_DECISIONS", "false").strip().lower()
            if skipped in {"1", "true", "yes", "on"}:
                raise ValueError("AEB_PUBLISH_SKIPPED_DECISIONS=true is forbidden in prod/staging")
            min_conf = _env_float("AEB_MIN_PUBLISH_CONFIDENCE", 0.70)
            if min_conf < 0.70:
                raise ValueError("AEB_MIN_PUBLISH_CONFIDENCE < 0.70 is forbidden in prod/staging")

        window = _env_int("CANDIDATE_WINDOW_CANDLES", 200)
        market_raw = (os.environ.get("BINANCE_MARKET") or "usdm_futures").strip()
        market = "usdm_futures" if market_raw in {"futures", "usdm_futures"} else market_raw

        return cls(
            enabled=_env_bool("BINANCE_CANDIDATE_ENABLED", True),
            symbols=resolved,
            timeframe=timeframe,
            market=market,
            wss_base_url=(os.environ.get("BINANCE_WSS_BASE_URL") or "wss://fstream.binance.com/ws").strip().rstrip("/"),
            reconnect_delay_sec=_env_int("BINANCE_RECONNECT_DELAY_SEC", 5),
            window_candles=window,
            rest_base_url=(os.environ.get("BINANCE_REST_BASE_URL") or "https://fapi.binance.com").strip().rstrip("/"),
            closed_candles_only=closed_only,
            dedup_enabled=dedup_enabled,
            min_interval_sec=_env_int("CANDIDATE_MIN_INTERVAL_SEC", 3600),
            enable_legacy_parser=legacy,
            enable_high_frequency_test_parser=hf_test,
            continuous_test_mode=continuous_test,
            publish_on_candle_close=publish_on_close,
            publish_on_every_update=publish_every_update,
            backfill_publish_historical=_env_bool("CANDIDATE_BACKFILL_PUBLISH_HISTORICAL", False),
            dedup_db_path=os.environ.get("CANDIDATE_DEDUP_DB_PATH", "/app/data/candidate_dedup.db"),
            app_env=app_env,
            max_candles=max(window, 200),
            ws_idle_timeout_sec=max(10, _env_int("CANDIDATE_WS_IDLE_TIMEOUT_SEC", 60)),
            rest_fallback_enabled=_env_bool("CANDIDATE_REST_FALLBACK_ENABLED", True),
            rest_fallback_poll_sec=max(5, _env_int("CANDIDATE_REST_FALLBACK_POLL_SEC", 60)),
            rest_fallback_always_on=_env_bool("CANDIDATE_REST_FALLBACK_ALWAYS_ON", True),
        )
