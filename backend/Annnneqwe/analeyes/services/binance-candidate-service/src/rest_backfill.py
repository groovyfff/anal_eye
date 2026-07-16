"""Binance USD-M Futures REST backfill and gap recovery."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.candle_buffer import Candle, CandleBuffer
from src.converters.ae_brain_candidate import ms_to_iso_utc

logger = logging.getLogger(__name__)

_MIRROR_REST_BASE = "https://fapi.binancefuture.com"


def _parse_klines_response(raw: Any) -> list[Candle]:
    if not isinstance(raw, list):
        raise ValueError("klines response is not a list")
    candles: list[Candle] = []
    for row in raw:
        if not isinstance(row, list) or len(row) < 7:
            continue
        candles.append(
            Candle(
                timestamp=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                closed=True,
                close_time=int(row[6]) if len(row) > 6 else int(row[0]) + 3_599_999,
                quote_volume=float(row[7]) if len(row) > 7 else None,
                trades_count=int(row[8]) if len(row) > 8 else None,
                taker_buy_base_volume=float(row[9]) if len(row) > 9 else None,
                taker_buy_quote_volume=float(row[10]) if len(row) > 10 else None,
            )
        )
    if not candles:
        raise ValueError("no candles in bootstrap response")
    return candles


def _fetch_json(url: str, *, timeout: float = 20.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "analeyes-binance-candidate/2.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_klines(
    *,
    symbol: str,
    interval: str,
    limit: int,
    rest_base_url: str,
    start_time: int | None = None,
) -> list[Candle]:
    params: dict[str, str | int] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": max(1, min(limit, 1500)),
    }
    if start_time is not None:
        params["startTime"] = start_time
    query = urllib.parse.urlencode(params)
    last_exc: Exception | None = None
    for base in (rest_base_url, _MIRROR_REST_BASE):
        url = f"{base.rstrip('/')}/fapi/v1/klines?{query}"
        try:
            raw = _fetch_json(url)
            return _parse_klines_response(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("REST klines failed base=%s symbol=%s reason=%s", base, symbol, exc)
            last_exc = exc
    raise RuntimeError(f"Failed to fetch klines for {symbol}: {last_exc}") from last_exc


def fetch_optional_market_fields(
    *,
    symbol: str,
    rest_base_url: str,
) -> dict[str, Any]:
    """Best-effort optional futures fields; never raises."""
    out: dict[str, Any] = {}
    for path, key in (
        ("/fapi/v1/premiumIndex", "funding_rate"),
        ("/fapi/v1/openInterest", "open_interest"),
    ):
        try:
            url = f"{rest_base_url.rstrip('/')}{path}?{urllib.parse.urlencode({'symbol': symbol.upper()})}"
            data = _fetch_json(url)
            if isinstance(data, dict):
                if key == "funding_rate" and data.get("lastFundingRate") is not None:
                    out["funding_rate"] = float(data["lastFundingRate"])
                if key == "open_interest" and data.get("openInterest") is not None:
                    out["open_interest"] = float(data["openInterest"])
        except Exception as exc:  # noqa: BLE001 - optional fields
            logger.debug("optional_market_field_skip symbol=%s field=%s err=%s", symbol, key, exc)
    return out


def backfill_symbol(
    *,
    symbol: str,
    interval: str,
    limit: int,
    rest_base_url: str,
    buffer: CandleBuffer,
    publish_historical: bool = False,
) -> list[Candle]:
    """Load REST klines into buffer; never publishes unless publish_historical=True."""
    candles = fetch_klines(symbol=symbol, interval=interval, limit=limit, rest_base_url=rest_base_url)
    count = buffer.load_bootstrap(symbol, candles)
    first_ts = ms_to_iso_utc(candles[0].timestamp)
    last_ts = ms_to_iso_utc(candles[-1].close_time or candles[-1].timestamp)
    logger.info(
        "rest_backfill_loaded symbol=%s timeframe=%s candles_count=%s first_ts=%s last_ts=%s publish_historical=%s",
        symbol,
        interval,
        count,
        first_ts,
        last_ts,
        publish_historical,
    )
    return candles


def gap_recover_symbol(
    *,
    symbol: str,
    interval: str,
    rest_base_url: str,
    buffer: CandleBuffer,
    last_close_time: int | None,
    window_candles: int,
) -> int:
    """Fill missing closed candles after reconnect; does not publish."""
    if last_close_time is None:
        return backfill_symbol(
            symbol=symbol,
            interval=interval,
            limit=window_candles,
            rest_base_url=rest_base_url,
            buffer=buffer,
        ).__len__()

    candles = fetch_klines(
        symbol=symbol,
        interval=interval,
        limit=window_candles,
        rest_base_url=rest_base_url,
        start_time=last_close_time + 1,
    )
    added = 0
    for candle in candles:
        if candle.timestamp > last_close_time:
            buffer.upsert(symbol, candle)
            added += 1
    if added:
        logger.info(
            "rest_gap_recovered symbol=%s timeframe=%s added=%s",
            symbol,
            interval,
            added,
        )
    return added
