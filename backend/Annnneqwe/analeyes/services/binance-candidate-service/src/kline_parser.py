from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParsedKline:
    symbol: str
    event_time: int
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool
    quote_volume: float | None = None
    trades_count: int | None = None
    taker_buy_base_volume: float | None = None
    taker_buy_quote_volume: float | None = None
    raw_stream: str | None = None


def _unwrap_kline_event(message: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if "data" in message and isinstance(message["data"], dict):
        stream = str(message.get("stream") or "")
        return (stream or None), message["data"]
    return None, message


def parse_kline_message(
    message: dict[str, Any],
    *,
    timeframe: str,
    default_stream: str | None = None,
) -> ParsedKline:
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
        open_price = float(k["o"])
        high = float(k["h"])
        low = float(k["l"])
        close = float(k["c"])
        volume = float(k["v"])
        is_closed = bool(k.get("x"))
        quote_volume = float(k["q"]) if k.get("q") is not None else None
        trades_count = int(k["n"]) if k.get("n") is not None else None
        taker_buy_base_volume = float(k["V"]) if k.get("V") is not None else None
        taker_buy_quote_volume = float(k["Q"]) if k.get("Q") is not None else None
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid kline fields: {exc}") from exc

    if event_time <= 0:
        event_time = close_time

    raw_stream = stream or default_stream or f"{symbol.lower()}@kline_{timeframe}"
    return ParsedKline(
        symbol=symbol,
        event_time=event_time,
        open_time=open_time,
        close_time=close_time,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        is_closed=is_closed,
        quote_volume=quote_volume,
        trades_count=trades_count,
        taker_buy_base_volume=taker_buy_base_volume,
        taker_buy_quote_volume=taker_buy_quote_volume,
        raw_stream=raw_stream,
    )
