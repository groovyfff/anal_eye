from __future__ import annotations

import datetime as dt
from typing import Any


def _iso_utc(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unwrap_kline_event(message: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if "data" in message and isinstance(message["data"], dict):
        stream = str(message.get("stream") or "")
        return (stream or None), message["data"]
    return None, message


def parse_kline_message(
    message: dict[str, Any],
    *,
    market: str,
    timeframe: str,
    default_stream: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return (live_price_payload, optional_raw_kline_payload)."""
    stream, event = _unwrap_kline_event(message)
    if event.get("e") != "kline":
        raise ValueError(f"unexpected event type={event.get('e')!r}")
    k = event.get("k")
    if not isinstance(k, dict):
        raise ValueError("missing kline object")

    symbol = str(event.get("s") or k.get("s") or "").strip().upper()
    if not symbol:
        raise ValueError("missing symbol")

    try:
        event_time = int(event.get("E") or k.get("T") or 0)
        open_time = int(k["t"])
        close_time = int(k["T"])
        price = float(k["c"])
        open_price = float(k["o"])
        high = float(k["h"])
        low = float(k["l"])
        volume = float(k["v"])
        is_closed = bool(k.get("x"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid kline fields: {exc}") from exc

    if event_time <= 0:
        event_time = close_time

    raw_stream = stream or default_stream or f"{symbol.lower()}@kline_{timeframe}"

    live_price = {
        "source": "binance",
        "market": market,
        "symbol": symbol,
        "asset_class": "crypto",
        "price": price,
        "bid": None,
        "ask": None,
        "ts": event_time,
        "timestamp": _iso_utc(event_time),
        "timeframe": timeframe,
        "candle_open_time": open_time,
        "candle_close_time": close_time,
        "is_candle_closed": is_closed,
        "raw_stream": raw_stream,
    }

    raw_kline = {
        "source": "binance",
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "event_time": event_time,
        "kline": {
            "open_time": open_time,
            "close_time": close_time,
            "open": open_price,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
            "is_closed": is_closed,
        },
        "raw": message,
    }
    return live_price, raw_kline
