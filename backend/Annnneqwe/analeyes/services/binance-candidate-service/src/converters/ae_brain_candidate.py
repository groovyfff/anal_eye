"""Convert Binance USD-M Futures 1h candles into AE Brain candidate payloads."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from shared.symbol_universe import default_allowed_symbols, is_symbol_allowed

from src.candle_buffer import Candle
from src.indicators import compute_composite_score, compute_features, infer_market_state

logger = logging.getLogger(__name__)

_FORBIDDEN_KEYS = frozenset(
    {
        "decision",
        "signal_type",
        "side",
        "heuristic_signal_consensus",
        "reason_summary",
        "entry_price",
        "tp_price",
        "sl_price",
        "leverage",
        "tp",
        "sl",
    }
)


def ms_to_iso_utc(ms: int) -> str:
    return (
        datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _candle_row(candle: Candle) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": ms_to_iso_utc(candle.timestamp),
        "open": float(candle.open),
        "high": float(candle.high),
        "low": float(candle.low),
        "close": float(candle.close),
        "volume": float(candle.volume),
        "quote_volume": float(candle.quote_volume) if candle.quote_volume is not None else None,
        "trades_count": int(candle.trades_count) if candle.trades_count is not None else None,
        "taker_buy_base_volume": (
            float(candle.taker_buy_base_volume) if candle.taker_buy_base_volume is not None else None
        ),
        "taker_buy_quote_volume": (
            float(candle.taker_buy_quote_volume) if candle.taker_buy_quote_volume is not None else None
        ),
    }
    return row


def _dedupe_sort_candles(candles: list[Candle]) -> list[Candle]:
    by_ts: dict[int, Candle] = {}
    for candle in candles:
        by_ts[candle.timestamp] = candle
    return [by_ts[ts] for ts in sorted(by_ts)]


def build_ae_brain_candidate(
    *,
    symbol: str,
    timeframe: str,
    candles: list[Candle],
    closed_candle: Candle,
    market: str = "usdm_futures",
    optional_features: dict[str, Any] | None = None,
    window_candles: int = 200,
) -> dict[str, Any] | None:
    """Build AE Brain candidate or return None when window is not ready."""
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        logger.info("candidate_rejected_symbol symbol= reason=missing_symbol")
        return None
    if not is_symbol_allowed(normalized_symbol, default_allowed_symbols()):
        logger.info(
            "candidate_rejected_symbol symbol=%s allowed=%s",
            normalized_symbol,
            ",".join(sorted(default_allowed_symbols())),
        )
        return None

    sorted_candles = _dedupe_sort_candles(candles)
    if len(sorted_candles) < window_candles:
        logger.info(
            "candidate_window_not_ready symbol=%s timeframe=%s count=%s required=%s",
            normalized_symbol,
            timeframe,
            len(sorted_candles),
            window_candles,
        )
        return None

    window = sorted_candles[-window_candles:]
    tech_features = compute_features(window)
    current_price = float(tech_features["current_price"])
    market_state = infer_market_state(tech_features, current_price)
    composite_score = compute_composite_score(tech_features)

    optional = dict(optional_features or {})
    features: dict[str, Any] = {
        "funding_rate": float(optional.get("funding_rate", 0.0) or 0.0),
        "open_interest": float(optional.get("open_interest", 0.0) or 0.0),
        "mark_open": optional.get("mark_open"),
        "mark_high": optional.get("mark_high"),
        "mark_low": optional.get("mark_low"),
        "mark_close": optional.get("mark_close"),
        "index_open": optional.get("index_open"),
        "index_high": optional.get("index_high"),
        "index_low": optional.get("index_low"),
        "index_close": optional.get("index_close"),
        "spread_estimate": float(optional.get("spread_estimate", 0.0) or 0.0),
        "slippage_estimate": float(optional.get("slippage_estimate", 0.0) or 0.0),
        **tech_features,
    }

    close_time_ms = closed_candle.close_time or closed_candle.timestamp + 3_599_999
    payload: dict[str, Any] = {
        "source": "binance_kline_1h",
        "exchange": "binance",
        "market": market,
        "asset_class": "crypto",
        "symbol": normalized_symbol,
        "timeframe": timeframe,
        "interval": timeframe,
        "event_time": ms_to_iso_utc(close_time_ms),
        "candle_open_time": ms_to_iso_utc(closed_candle.timestamp),
        "candle_close_time": ms_to_iso_utc(close_time_ms),
        "candle_closed": True,
        "current_price": current_price,
        "market_state": market_state,
        "composite_score": composite_score,
        "candles_count": len(window),
        "candles": [_candle_row(c) for c in window],
        "features": features,
        "valid_json": True,
    }

    for key in _FORBIDDEN_KEYS:
        payload.pop(key, None)

    validate_candidate_payload(payload)
    return payload


def validate_candidate_payload(payload: dict[str, Any]) -> None:
    """Raise ValueError when payload is not safe to publish."""
    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol or not is_symbol_allowed(symbol):
        raise ValueError(f"symbol_not_allowed:{symbol}")

    if payload.get("candle_closed") is not True:
        raise ValueError("candle_not_closed")

    candles = payload.get("candles")
    if not isinstance(candles, list) or not candles:
        raise ValueError("missing_candles")

    for row in candles:
        if not isinstance(row, dict):
            raise ValueError("invalid_candle_row")
        for field in ("open", "high", "low", "close", "volume"):
            val = row.get(field)
            if val is None:
                raise ValueError(f"missing_{field}")
            float(val)

    if payload.get("composite_score") is None:
        raise ValueError("missing_composite_score")
    if not payload.get("features"):
        raise ValueError("missing_features")

    for key in _FORBIDDEN_KEYS:
        if key in payload:
            raise ValueError(f"forbidden_field:{key}")

    # Prove JSON serializable before RabbitMQ publish.
    json.dumps(payload, ensure_ascii=False)
