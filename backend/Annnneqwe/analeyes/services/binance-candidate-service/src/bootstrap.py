from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.candle_buffer import Candle

logger = logging.getLogger(__name__)

_MIRROR_REST_BASE = "https://fapi.binancefuture.com"


def fetch_bootstrap_klines(
    *,
    symbol: str,
    interval: str,
    limit: int,
    rest_base_url: str,
) -> list[Candle]:
    """One-time startup REST bootstrap — not a polling loop."""
    params = urllib.parse.urlencode({"symbol": symbol.upper(), "interval": interval, "limit": limit})
    for base in (rest_base_url, _MIRROR_REST_BASE):
        url = f"{base}/fapi/v1/klines?{params}"
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                raw = json.loads(response.read().decode("utf-8"))
            return _parse_klines_response(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Bootstrap REST failed base=%s symbol=%s reason=%s", base, symbol, exc)
    raise RuntimeError(f"Failed to bootstrap klines for {symbol}")


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
            )
        )
    if not candles:
        raise ValueError("no candles in bootstrap response")
    return candles
