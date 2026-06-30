from __future__ import annotations

from typing import Any

from src.candle_buffer import Candle
from src.indicators import compute_composite_score, compute_features, infer_market_state

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


def build_candidate_payload(
    *,
    symbol: str,
    market: str,
    timeframe: str,
    event_time: int,
    candles: list[Candle],
) -> dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        raise ValueError("missing_symbol")

    features = compute_features(candles)
    current_price = float(features["current_price"])
    market_state = infer_market_state(features, current_price)
    composite_score = compute_composite_score(features)

    payload: dict[str, Any] = {
        "source": "binance",
        "exchange": "binance",
        "market": market,
        "asset_class": "crypto",
        "symbol": normalized_symbol,
        "timeframe": timeframe,
        "event_time": event_time,
        "current_price": current_price,
        "market_state": market_state,
        "composite_score": composite_score,
        "features": features,
        "candles": [c.to_dict() for c in candles],
    }

    for key in _FORBIDDEN_KEYS:
        payload.pop(key, None)
        payload.get("features", {}).pop(key, None)

    return payload


def assert_no_forbidden_fields(payload: dict[str, Any]) -> None:
    for key in _FORBIDDEN_KEYS:
        if key in payload:
            raise ValueError(f"forbidden field in candidate: {key}")
