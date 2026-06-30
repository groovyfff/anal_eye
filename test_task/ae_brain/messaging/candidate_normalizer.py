"""Normalize neutral market candidate payloads for AE Brain inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ae_brain.contracts import AssetClass

# Pre-filled decision fields are stripped; AE Brain must compute direction itself.
_STRIP_KEYS = frozenset(
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
    }
)

_CANDLE_LIST_KEYS = ("candles", "ohlcv", "klines", "kline_data", "historical_data", "price_history", "historical_ohlcv")
_SUPPORTED_ASSET_CLASSES = {a.value for a in AssetClass}


@dataclass(slots=True)
class NormalizeResult:
    payload: dict[str, Any] | None = None
    skip_reason: str | None = None
    direction_hint: str | None = None
    summary: str = ""
    symbol: str = ""


def _first(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _normalize_candle(row: dict[str, Any]) -> dict[str, Any]:
    candle = dict(row)
    ts = _first(candle.get("timestamp"), candle.get("time"), candle.get("open_time"), candle.get("ts"))
    if ts is not None:
        candle["timestamp"] = ts
        candle["ts"] = ts
    if "open" not in candle and "o" in candle:
        candle["open"] = candle["o"]
    if "high" not in candle and "h" in candle:
        candle["high"] = candle["h"]
    if "low" not in candle and "l" in candle:
        candle["low"] = candle["l"]
    if "close" not in candle and "c" in candle:
        candle["close"] = candle["c"]
    if "volume" not in candle and "v" in candle:
        candle["volume"] = candle["v"]
    return candle


def _summary(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "{}"
    return (
        f"symbol={payload.get('symbol')} asset_class={payload.get('asset_class')} "
        f"candles={len(payload.get('candles') or [])} "
        f"composite_score={payload.get('composite_score')}"
    )


def normalize_candidate(
    raw: Any,
    *,
    min_composite_score: float = 0.0,
) -> NormalizeResult:
    if isinstance(raw, list):
        return NormalizeResult(skip_reason="payload_is_raw_candles_array_expected_candidate_object")
    if not isinstance(raw, dict):
        return NormalizeResult(skip_reason="invalid_json")

    payload = dict(raw)
    direction_hint = None
    for key in ("decision", "signal_type", "side"):
        if key in raw and raw[key] not in (None, ""):
            direction_hint = str(raw[key]).upper()
    for key in _STRIP_KEYS:
        payload.pop(key, None)

    symbol = _first(payload.get("symbol"))
    if not symbol:
        return NormalizeResult(skip_reason="missing_symbol", summary=_summary(payload))

    asset_class_raw = _first(payload.get("asset_class"), payload.get("feat_asset_class"), "crypto")
    asset_class = str(asset_class_raw).lower()
    if not asset_class:
        return NormalizeResult(skip_reason="missing_asset_class", symbol=str(symbol), summary=_summary(payload))
    if asset_class not in _SUPPORTED_ASSET_CLASSES:
        return NormalizeResult(
            skip_reason="unsupported_asset_class",
            symbol=str(symbol),
            summary=_summary(payload),
        )

    features = dict(payload.get("features") or {})
    current_price = _first(
        payload.get("current_price"),
        payload.get("price"),
        features.get("current_price"),
        payload.get("feat_current_price"),
    )
    if current_price is None:
        return NormalizeResult(skip_reason="missing_current_price", symbol=str(symbol), summary=_summary(payload))
    try:
        current_price_f = float(current_price)
        if current_price_f <= 0:
            raise ValueError("non-positive")
    except (TypeError, ValueError):
        return NormalizeResult(skip_reason="invalid_current_price", symbol=str(symbol), summary=_summary(payload))

    composite_score = _first(
        payload.get("composite_score"),
        payload.get("composite_score_value"),
        payload.get("score"),
    )
    if composite_score is None:
        return NormalizeResult(skip_reason="missing_composite_score", symbol=str(symbol), summary=_summary(payload))
    try:
        composite_score_f = float(composite_score)
    except (TypeError, ValueError):
        return NormalizeResult(skip_reason="missing_composite_score", symbol=str(symbol), summary=_summary(payload))
    if composite_score_f < min_composite_score:
        return NormalizeResult(skip_reason="composite_below_threshold", symbol=str(symbol), summary=_summary(payload))

    if not features:
        return NormalizeResult(skip_reason="missing_features", symbol=str(symbol), summary=_summary(payload))
    features.setdefault("current_price", current_price_f)

    raw_candles = None
    for key in _CANDLE_LIST_KEYS:
        if payload.get(key):
            raw_candles = payload[key]
            break
    if not raw_candles:
        return NormalizeResult(skip_reason="missing_candles", symbol=str(symbol), summary=_summary(payload))
    candles = [_normalize_candle(dict(c)) for c in raw_candles if isinstance(c, dict)]
    if not candles:
        return NormalizeResult(skip_reason="missing_candles", symbol=str(symbol), summary=_summary(payload))

    market_state = _first(payload.get("market_state"), payload.get("feat_market_state"))
    meta = dict(payload.get("meta") or {})
    meta.update(
        {
            "current_price": current_price_f,
            "composite_score": composite_score_f,
            "features": features,
            "market_state": market_state,
            "direction_hint": direction_hint,
        }
    )

    normalized = {
        "symbol": str(symbol),
        "interval": str(_first(payload.get("timeframe"), payload.get("interval"), "5m")),
        "asset_class": asset_class,
        "candles": candles,
        "signal_log_db_id": int(payload["signal_log_db_id"]) if payload.get("signal_log_db_id") is not None else 0,
        "signal_id": str(payload.get("signal_id", "")),
        "correlation_id": str(payload.get("correlation_id", "")),
        "composite_score": composite_score_f,
        "current_price": current_price_f,
        "market_state": market_state,
        "features": features,
        "meta": meta,
        "source": payload.get("source"),
        "market": payload.get("market"),
    }
    return NormalizeResult(
        payload=normalized,
        direction_hint=direction_hint,
        symbol=str(symbol),
        summary=_summary(normalized),
    )
